from __future__ import annotations

import pytest

from lessons import lesson_28_guarded_rag as guarded
from lessons.lesson_23_search_vector_index import (
    IndexedChunk,
    LoadedVectorIndex,
    SearchResult,
)


def make_chunk(
    content: str,
    *,
    chunk_id: str = "section-test-chunk-001",
    section_id: str = "section-test",
    section_title: str = "Test Procedures",
) -> IndexedChunk:
    """Create a small valid indexed chunk for a test."""

    word_count = len(
        content.split()
    )

    return IndexedChunk(
        chunk_id=chunk_id,
        section_id=section_id,
        document_title="Test Fleet Manual",
        section_title=section_title,
        source="data/test_manual.txt",
        content=content,
        chunk_number=1,
        start_word=0,
        end_word_exclusive=word_count,
        embedding=[
            1.0,
            0.0,
        ],
    )


def make_vector_index(
    chunk: IndexedChunk,
) -> LoadedVectorIndex:
    """Create a minimal in-memory vector index."""

    return LoadedVectorIndex(
        embedding_model="test-embedding-model",
        embedding_dimension=2,
        distance_metric="cosine_similarity",
        chunks=[
            chunk,
        ],
    )


def install_fake_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: SearchResult,
) -> None:
    """Replace live vector retrieval with a fixed result."""

    def fake_retrieve_chunks(
        question: str,
        vector_index: LoadedVectorIndex,
    ) -> tuple[list[SearchResult], int]:
        assert question
        assert vector_index.chunks

        return (
            [result],
            7,
        )

    monkeypatch.setattr(
        guarded,
        "retrieve_chunks",
        fake_retrieve_chunks,
    )


def install_fake_generation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    answer: str,
) -> None:
    """Replace the Ollama generation call with a fixed answer."""

    def fake_generate_grounded_answer(
        user_message: str,
    ) -> tuple[str, int, int]:
        assert user_message

        return (
            answer,
            100,
            20,
        )

    monkeypatch.setattr(
        guarded,
        "generate_grounded_answer",
        fake_generate_grounded_answer,
    )


def test_low_similarity_abstains_without_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A low retrieval score must prevent an LLM call."""

    chunk = make_chunk(
        "The driver must park the vehicle safely."
    )

    vector_index = make_vector_index(
        chunk
    )

    install_fake_retrieval(
        monkeypatch,
        result=SearchResult(
            score=0.40,
            chunk=chunk,
        ),
    )

    def fail_if_generation_is_called(
        _user_message: str,
    ) -> tuple[str, int, int]:
        raise AssertionError(
            "Generation must not run after "
            "retrieval abstention."
        )

    monkeypatch.setattr(
        guarded,
        "generate_grounded_answer",
        fail_if_generation_is_called,
    )

    result = guarded.run_guarded_rag(
        question="What is the insurance number?",
        vector_index=vector_index,
    )

    assert result.decision == "ABSTAIN"
    assert (
        result.answer_strategy
        == "retrieval abstention"
    )
    assert result.answer == guarded.FALLBACK_ANSWER
    assert result.generation_called is False
    assert result.citation_repair_used is False
    assert result.extractive_fallback_used is False
    assert result.context_sources == []
    assert result.answer_passed_validation is True


def test_valid_grounded_answer_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supported cited answer should pass immediately."""

    chunk = make_chunk(
        "The driver must park the vehicle safely."
    )

    vector_index = make_vector_index(
        chunk
    )

    install_fake_retrieval(
        monkeypatch,
        result=SearchResult(
            score=0.70,
            chunk=chunk,
        ),
    )

    install_fake_generation(
        monkeypatch,
        answer=(
            "The driver must park the vehicle "
            "safely [S1]."
        ),
    )

    result = guarded.run_guarded_rag(
        question="What must the driver do?",
        vector_index=vector_index,
    )

    assert result.decision == "ANSWER"
    assert (
        result.answer_strategy
        == "grounded generation"
    )
    assert result.generation_called is True
    assert result.citation_repair_used is False
    assert result.extractive_fallback_used is False
    assert result.answer_passed_validation is True
    assert "[S1]" in result.answer
    assert len(result.citation_checks) == 1
    assert (
        result.citation_checks[0].support_passed
        is True
    )


def test_extractive_fallback_recovers_supported_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed paraphrase should fall back to source text."""

    source_sentence = (
        "A vehicle with repeated pressure loss must be "
        "removed from service until the tire has been "
        "inspected or replaced."
    )

    expected_answer = (
        source_sentence.removesuffix(".")
        + " [S1]."
    )

    chunk = make_chunk(
        source_sentence,
        section_id="section-tire",
        section_title="Tire Pressure Procedures",
    )

    vector_index = make_vector_index(
        chunk
    )

    install_fake_retrieval(
        monkeypatch,
        result=SearchResult(
            score=0.70,
            chunk=chunk,
        ),
    )

    install_fake_generation(
        monkeypatch,
        answer=(
            "No, the vehicle cannot remain in "
            "service [S1]."
        ),
    )

    result = guarded.run_guarded_rag(
        question=(
            "A tire keeps losing air repeatedly. "
            "Can the vehicle remain in service?"
        ),
        vector_index=vector_index,
    )

    assert result.decision == "ANSWER"
    assert (
        result.answer_strategy
        == "extractive fallback"
    )
    assert result.generation_called is True
    assert result.citation_repair_used is True
    assert result.extractive_fallback_used is True
    assert result.answer_passed_validation is True

    assert result.extractive_candidate is not None
    assert (
        result.extractive_candidate.sentence
        == source_sentence
    )
    assert (
        result.extractive_candidate.lexical_coverage
        == pytest.approx(1.0)
    )

    assert result.answer == expected_answer
    assert "[S1]" in result.answer
    assert len(result.citation_checks) == 1
    assert (
        result.citation_checks[0].support_passed
        is True
    )


def test_validation_failure_causes_final_abstention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported generation must not reach the user."""

    chunk = make_chunk(
        "The driver must park the vehicle safely."
    )

    vector_index = make_vector_index(
        chunk
    )

    install_fake_retrieval(
        monkeypatch,
        result=SearchResult(
            score=0.70,
            chunk=chunk,
        ),
    )

    install_fake_generation(
        monkeypatch,
        answer=(
            "The insurance policy number is "
            "12345 [S1]."
        ),
    )

    def return_no_extractive_candidate(
        **_arguments: object,
    ) -> None:
        return None

    monkeypatch.setattr(
        guarded,
        "select_extractive_candidate",
        return_no_extractive_candidate,
    )

    result = guarded.run_guarded_rag(
        question="What is the insurance policy number?",
        vector_index=vector_index,
    )

    assert (
        result.decision
        == "ABSTAIN_AFTER_VALIDATION"
    )
    assert (
        result.answer_strategy
        == "validation abstention"
    )
    assert result.answer == guarded.FALLBACK_ANSWER
    assert result.generation_called is True
    assert result.citation_repair_used is True
    assert result.extractive_fallback_used is True
    assert result.answer_passed_validation is False