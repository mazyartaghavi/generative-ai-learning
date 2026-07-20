from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lessons import lesson_28_guarded_rag as guarded
from lessons import lesson_32_qdrant_guarded_rag as qdrant_guarded
from lessons.lesson_23_search_vector_index import (
    IndexedChunk,
    LoadedVectorIndex,
)


def make_chunk(
    content: str,
    *,
    chunk_id: str = "section-test-chunk-001",
    section_id: str = "section-test",
    section_title: str = "Test Procedures",
) -> IndexedChunk:
    """Create one small indexed chunk for testing."""

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


class FakeQueryClient:
    """Small Qdrant client replacement for retrieval tests."""

    def __init__(
        self,
        points: list[SimpleNamespace],
    ) -> None:
        self.points = points
        self.last_query_arguments: dict[
            str,
            Any
        ] | None = None

    def query_points(
        self,
        **arguments: Any,
    ) -> SimpleNamespace:
        """Record the query and return fixed scored points."""

        self.last_query_arguments = arguments

        return SimpleNamespace(
            points=self.points
        )


class FakeCollectionClient:
    """Small Qdrant client replacement for collection tests."""

    def __init__(
        self,
        *,
        collection_exists: bool,
        stored_count: int,
    ) -> None:
        self._collection_exists = collection_exists
        self._stored_count = stored_count

    def collection_exists(
        self,
        *,
        collection_name: str,
    ) -> bool:
        """Return the configured collection state."""

        assert (
            collection_name
            == qdrant_guarded.COLLECTION_NAME
        )

        return self._collection_exists

    def count(
        self,
        *,
        collection_name: str,
        exact: bool,
    ) -> SimpleNamespace:
        """Return the configured stored-point count."""

        assert (
            collection_name
            == qdrant_guarded.COLLECTION_NAME
        )
        assert exact is True

        return SimpleNamespace(
            count=self._stored_count
        )


def install_fake_embedding(
    monkeypatch: pytest.MonkeyPatch,
    *,
    token_count: int = 7,
) -> None:
    """Prevent tests from calling the real Ollama API."""

    def fake_generate_query_embedding(
        *,
        query: str,
        embedding_model: str,
        expected_dimension: int,
    ) -> tuple[list[float], int]:
        assert query
        assert (
            embedding_model
            == "test-embedding-model"
        )
        assert expected_dimension == 2

        return (
            [
                0.9,
                0.1,
            ],
            token_count,
        )

    monkeypatch.setattr(
        qdrant_guarded,
        "generate_query_embedding",
        fake_generate_query_embedding,
    )


def test_qdrant_adapter_maps_points_to_search_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qdrant points should become Lesson 28 search results."""

    chunk = make_chunk(
        "The driver must park the vehicle safely."
    )

    vector_index = make_vector_index(
        [chunk]
    )

    client = FakeQueryClient(
        points=[
            SimpleNamespace(
                score=0.82,
                payload={
                    "chunk_id": chunk.chunk_id,
                },
            )
        ]
    )

    install_fake_embedding(
        monkeypatch,
        token_count=11,
    )

    retriever = (
        qdrant_guarded.create_qdrant_retriever(
            client  # type: ignore[arg-type]
        )
    )

    results, token_count = retriever(
        "What must the driver do?",
        vector_index,
    )

    assert token_count == 11
    assert len(results) == 1
    assert results[0].score == pytest.approx(
        0.82
    )
    assert results[0].chunk is chunk

    assert (
        client.last_query_arguments
        is not None
    )

    assert (
        client.last_query_arguments[
            "collection_name"
        ]
        == qdrant_guarded.COLLECTION_NAME
    )

    assert client.last_query_arguments["query"] == [
        0.9,
        0.1,
    ]

    assert (
        client.last_query_arguments["limit"]
        == qdrant_guarded.RETRIEVAL_TOP_K
    )

    assert (
        client.last_query_arguments[
            "with_payload"
        ]
        is True
    )

    assert (
        client.last_query_arguments[
            "with_vectors"
        ]
        is False
    )


def test_low_qdrant_similarity_causes_abstention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A low Qdrant score must block Llama generation."""

    chunk = make_chunk(
        "The driver must park the vehicle safely."
    )

    vector_index = make_vector_index(
        [chunk]
    )

    client = FakeQueryClient(
        points=[
            SimpleNamespace(
                score=0.40,
                payload={
                    "chunk_id": chunk.chunk_id,
                },
            )
        ]
    )

    install_fake_embedding(
        monkeypatch,
    )

    retriever = (
        qdrant_guarded.create_qdrant_retriever(
            client  # type: ignore[arg-type]
        )
    )

    monkeypatch.setattr(
        guarded,
        "retrieve_chunks",
        retriever,
    )

    def fail_if_generation_runs(
        _user_message: str,
    ) -> tuple[str, int, int]:
        raise AssertionError(
            "Generation must not run after "
            "a low Qdrant retrieval score."
        )

    monkeypatch.setattr(
        guarded,
        "generate_grounded_answer",
        fail_if_generation_runs,
    )

    result = guarded.run_guarded_rag(
        question="What is the account password?",
        vector_index=vector_index,
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


def test_unknown_qdrant_chunk_id_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qdrant must not return an unknown source chunk."""

    chunk = make_chunk(
        "The driver must park the vehicle safely."
    )

    vector_index = make_vector_index(
        [chunk]
    )

    client = FakeQueryClient(
        points=[
            SimpleNamespace(
                score=0.90,
                payload={
                    "chunk_id": (
                        "unknown-section-chunk-999"
                    ),
                },
            )
        ]
    )

    install_fake_embedding(
        monkeypatch,
    )

    retriever = (
        qdrant_guarded.create_qdrant_retriever(
            client  # type: ignore[arg-type]
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "missing from the source vector index"
        ),
    ):
        retriever(
            "What must the driver do?",
            vector_index,
        )


def test_collection_count_mismatch_is_rejected() -> None:
    """Qdrant and the source index must remain synchronized."""

    client = FakeCollectionClient(
        collection_exists=True,
        stored_count=8,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "contain different numbers of chunks"
        ),
    ):
        qdrant_guarded.validate_qdrant_database(
            client=client,  # type: ignore[arg-type]
            expected_point_count=9,
        )