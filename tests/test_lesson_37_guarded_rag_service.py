from __future__ import annotations

from collections.abc import Callable

import pytest

from lessons import lesson_28_guarded_rag as guarded
from lessons.lesson_23_search_vector_index import (
    IndexedChunk,
    LoadedVectorIndex,
    SearchResult,
)
from lessons.lesson_36_guarded_rag_service import (
    GuardedRAGService,
)


RetrieverFunction = Callable[
    [str, LoadedVectorIndex],
    tuple[list[SearchResult], int],
]


def make_chunk(
    content: str,
    *,
    chunk_id: str = "section-test-chunk-001",
    section_id: str = "section-test",
    section_title: str = "Test Procedures",
) -> IndexedChunk:
    """Create one small indexed chunk for service tests."""

    return IndexedChunk(
        chunk_id=chunk_id,
        section_id=section_id,
        document_title="Test Fleet Manual",
        section_title=section_title,
        source="data/test_manual.txt",
        content=content,
        chunk_number=1,
        start_word=0,
        end_word_exclusive=len(content.split()),
        embedding=[
            1.0,
            0.0,
        ],
    )


def make_vector_index(
    chunks: list[IndexedChunk],
) -> LoadedVectorIndex:
    """Create a minimal in-memory source vector index."""

    return LoadedVectorIndex(
        embedding_model="test-embedding-model",
        embedding_dimension=2,
        distance_metric="cosine_similarity",
        chunks=chunks,
    )


class FakeRetriever:
    """Controlled replacement for the real Qdrant retriever."""

    def __init__(
        self,
        *,
        results: list[SearchResult],
        token_count: int = 7,
        stored_point_count: int | None = None,
        collection_name: str = "test_collection",
        top_k: int = 3,
    ) -> None:
        self._results = results
        self._token_count = token_count
        self._stored_point_count = (
            stored_point_count
            if stored_point_count is not None
            else len(results)
        )
        self._collection_name = collection_name
        self._top_k = top_k
        self._is_closed = False
        self.questions: list[str] = []

    @property
    def collection_name(self) -> str:
        """Return the fake collection name."""

        return self._collection_name

    @property
    def top_k(self) -> int:
        """Return the fake retrieval limit."""

        return self._top_k

    @property
    def is_closed(self) -> bool:
        """Report whether the fake retriever was closed."""

        return self._is_closed

    def validate_collection(
        self,
        *,
        expected_point_count: int | None = None,
    ) -> int:
        """Validate the configured fake point count."""

        if self._is_closed:
            raise RuntimeError(
                "The fake retriever is closed."
            )

        if (
            expected_point_count is not None
            and expected_point_count
            != self._stored_point_count
        ):
            raise RuntimeError(
                "The fake collection and source index "
                "contain different numbers of chunks."
            )

        return self._stored_point_count

    def __call__(
        self,
        question: str,
        vector_index: LoadedVectorIndex,
    ) -> tuple[list[SearchResult], int]:
        """Return fixed retrieval results."""

        if self._is_closed:
            raise RuntimeError(
                "The fake retriever is closed."
            )

        assert question
        assert vector_index.chunks

        self.questions.append(question)

        return (
            list(self._results),
            self._token_count,
        )

    def close(self) -> None:
        """Mark the fake retriever as closed."""

        self._is_closed = True


def install_fixed_generation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    answer: str,
) -> None:
    """Replace Ollama generation with a fixed response."""

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


def test_service_returns_supported_answer_and_restores_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service should answer and restore the old retriever."""

    chunk = make_chunk(
        "The driver must park the vehicle safely."
    )

    vector_index = make_vector_index(
        [chunk]
    )

    retriever = FakeRetriever(
        results=[
            SearchResult(
                score=0.70,
                chunk=chunk,
            )
        ],
        stored_point_count=1,
    )

    def original_retriever(
        question: str,
        source_index: LoadedVectorIndex,
    ) -> tuple[list[SearchResult], int]:
        del question
        del source_index

        return (
            [],
            0,
        )

    monkeypatch.setattr(
        guarded,
        "retrieve_chunks",
        original_retriever,
    )

    install_fixed_generation(
        monkeypatch,
        answer=(
            "The driver must park the vehicle "
            "safely [S1]."
        ),
    )

    service = GuardedRAGService(
        vector_index=vector_index,
        retriever=retriever,  # type: ignore[arg-type]
        owns_retriever=False,
    )

    result = service.answer(
        "What must the driver do?"
    )

    assert result.decision == "ANSWER"
    assert (
        result.answer_strategy
        == "grounded generation"
    )
    assert result.generation_called is True
    assert result.answer_passed_validation is True
    assert "[S1]" in result.answer

    assert retriever.questions == [
        "What must the driver do?"
    ]

    assert (
        guarded.retrieve_chunks
        is original_retriever
    )

    service.close()


def test_low_similarity_abstains_without_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A low retriever score must prevent generation."""

    chunk = make_chunk(
        "The driver must park the vehicle safely."
    )

    vector_index = make_vector_index(
        [chunk]
    )

    retriever = FakeRetriever(
        results=[
            SearchResult(
                score=0.40,
                chunk=chunk,
            )
        ],
        stored_point_count=1,
    )

    def fail_if_generation_runs(
        _user_message: str,
    ) -> tuple[str, int, int]:
        raise AssertionError(
            "Generation must not run after "
            "retrieval abstention."
        )

    monkeypatch.setattr(
        guarded,
        "generate_grounded_answer",
        fail_if_generation_runs,
    )

    service = GuardedRAGService(
        vector_index=vector_index,
        retriever=retriever,  # type: ignore[arg-type]
        owns_retriever=False,
    )

    result = service.answer(
        "What is the account password?"
    )

    assert result.decision == "ABSTAIN"
    assert (
        result.answer_strategy
        == "retrieval abstention"
    )
    assert result.top_similarity == pytest.approx(
        0.40
    )
    assert result.generation_called is False
    assert result.answer == guarded.FALLBACK_ANSWER

    service.close()


def test_answer_many_processes_questions_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service should process a list of questions."""

    chunk = make_chunk(
        "The driver must park the vehicle safely."
    )

    vector_index = make_vector_index(
        [chunk]
    )

    retriever = FakeRetriever(
        results=[
            SearchResult(
                score=0.40,
                chunk=chunk,
            )
        ],
        stored_point_count=1,
    )

    def fail_if_generation_runs(
        _user_message: str,
    ) -> tuple[str, int, int]:
        raise AssertionError(
            "Generation must not run for these "
            "low-confidence questions."
        )

    monkeypatch.setattr(
        guarded,
        "generate_grounded_answer",
        fail_if_generation_runs,
    )

    service = GuardedRAGService(
        vector_index=vector_index,
        retriever=retriever,  # type: ignore[arg-type]
        owns_retriever=False,
    )

    questions = [
        "What is the password?",
        "What is the insurance number?",
    ]

    results = service.answer_many(
        questions
    )

    assert len(results) == 2
    assert [
        result.question
        for result in results
    ] == questions

    assert all(
        result.decision == "ABSTAIN"
        for result in results
    )

    assert retriever.questions == questions

    service.close()


def test_context_manager_closes_owned_retriever() -> None:
    """Leaving the service context should close owned resources."""

    chunk = make_chunk(
        "The driver must park the vehicle safely."
    )

    vector_index = make_vector_index(
        [chunk]
    )

    retriever = FakeRetriever(
        results=[
            SearchResult(
                score=0.70,
                chunk=chunk,
            )
        ],
        stored_point_count=1,
    )

    with GuardedRAGService(
        vector_index=vector_index,
        retriever=retriever,  # type: ignore[arg-type]
        owns_retriever=True,
    ) as service:
        assert service.is_closed is False
        assert retriever.is_closed is False
        assert service.validate() == 1
        assert service.collection_name == (
            "test_collection"
        )
        assert service.top_k == 3

    assert service.is_closed is True
    assert retriever.is_closed is True


def test_closed_service_rejects_new_answers() -> None:
    """A closed service must not accept new work."""

    chunk = make_chunk(
        "The driver must park the vehicle safely."
    )

    vector_index = make_vector_index(
        [chunk]
    )

    retriever = FakeRetriever(
        results=[
            SearchResult(
                score=0.70,
                chunk=chunk,
            )
        ],
        stored_point_count=1,
    )

    service = GuardedRAGService(
        vector_index=vector_index,
        retriever=retriever,  # type: ignore[arg-type]
        owns_retriever=False,
    )

    service.close()
    service.close()

    assert service.is_closed is True
    assert retriever.is_closed is False

    with pytest.raises(
        RuntimeError,
        match="already closed",
    ):
        service.answer(
            "What must the driver do?"
        )