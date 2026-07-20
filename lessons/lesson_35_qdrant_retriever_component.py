from __future__ import annotations

import collections.abc

from pathlib import Path
from types import TracebackType
from typing import Any

from qdrant_client import QdrantClient

from lessons.lesson_23_search_vector_index import (
    INDEX_PATH,
    LoadedVectorIndex,
    SearchResult,
    generate_query_embedding,
    load_vector_index,
)
from lessons.lesson_30_qdrant_local_index import (
    COLLECTION_NAME,
    QDRANT_PATH,
)


DEFAULT_TOP_K = 3

QueryEmbeddingProvider = collections.abc.Callable[
    [str, str, int],
    tuple[list[float], int | None],
]

TEST_QUESTIONS = [
    (
        "What should a driver do when an engine "
        "warning light appears?"
    ),
    (
        "Can a vehicle remain in service after "
        "repeated tire-pressure loss?"
    ),
]


class QdrantRetriever:
    """
    Retrieve indexed document chunks from Qdrant.

    The component accepts an existing Qdrant client so it
    can be used with local storage, a Docker server, or a
    remote Qdrant deployment.
    """

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str = COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
        owns_client: bool = False,
        query_embedding_provider: (
            QueryEmbeddingProvider | None
        ) = None,
    ) -> None:
        """Initialize the reusable retriever component."""

        cleaned_collection_name = (
            collection_name.strip()
        )

        if not cleaned_collection_name:
            raise ValueError(
                "The Qdrant collection name cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "The retrieval top_k value must be "
                "greater than zero."
            )

        if (
            query_embedding_provider is not None
            and not callable(query_embedding_provider)
        ):
            raise TypeError(
                "The query embedding provider must be "
                "callable."
            )

        self._client = client
        self._collection_name = (
            cleaned_collection_name
        )
        self._top_k = top_k
        self._owns_client = owns_client
        self._query_embedding_provider = (
            query_embedding_provider
        )
        self._closed = False

    @classmethod
    def from_local_path(
        cls,
        path: Path,
        *,
        collection_name: str = COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
        query_embedding_provider: (
            QueryEmbeddingProvider | None
        ) = None,
    ) -> QdrantRetriever:
        """Create a retriever for persistent local storage."""

        if not isinstance(path, Path):
            raise TypeError(
                "The local Qdrant path must be a "
                "pathlib.Path object."
            )

        if not path.exists():
            raise FileNotFoundError(
                "The local Qdrant storage directory "
                f"does not exist: {path}"
            )

        client = QdrantClient(
            path=str(path)
        )

        return cls(
            client=client,
            collection_name=collection_name,
            top_k=top_k,
            owns_client=True,
            query_embedding_provider=(
                query_embedding_provider
            ),
        )

    @property
    def collection_name(self) -> str:
        """Return the configured Qdrant collection name."""

        return self._collection_name

    @property
    def top_k(self) -> int:
        """Return the number of requested search results."""

        return self._top_k

    @property
    def is_closed(self) -> bool:
        """Report whether this retriever has been closed."""

        return self._closed

    @property
    def uses_default_query_embedding_provider(
        self,
    ) -> bool:
        """Report whether legacy query embedding is active."""

        return self._query_embedding_provider is None

    def set_query_embedding_provider(
        self,
        provider: QueryEmbeddingProvider | None,
    ) -> None:
        """
        Configure or restore the query embedding provider.

        Passing None restores Lesson 23's original standalone
        Ollama embedding function.
        """

        self._require_open()

        if provider is not None and not callable(provider):
            raise TypeError(
                "The query embedding provider must be "
                "callable."
            )

        self._query_embedding_provider = provider

    def validate_collection(
        self,
        *,
        expected_point_count: int | None = None,
    ) -> int:
        """
        Validate the collection and return its point count.

        When expected_point_count is supplied, Qdrant and
        the source index must contain the same number of
        document chunks.
        """

        self._require_open()

        if (
            expected_point_count is not None
            and expected_point_count <= 0
        ):
            raise ValueError(
                "The expected point count must be "
                "greater than zero."
            )

        collection_exists = (
            self._client.collection_exists(
                collection_name=(
                    self._collection_name
                ),
            )
        )

        if not collection_exists:
            raise RuntimeError(
                "The expected Qdrant collection does not "
                f"exist: {self._collection_name!r}."
            )

        count_result = self._client.count(
            collection_name=self._collection_name,
            exact=True,
        )

        stored_point_count = int(
            count_result.count
        )

        if stored_point_count <= 0:
            raise RuntimeError(
                "The Qdrant collection contains no points."
            )

        if (
            expected_point_count is not None
            and stored_point_count
            != expected_point_count
        ):
            raise RuntimeError(
                "The Qdrant collection and source vector "
                "index contain different numbers of "
                "chunks. "
                f"Qdrant contains {stored_point_count}, "
                "while the source index contains "
                f"{expected_point_count}."
            )

        return stored_point_count

    def retrieve(
        self,
        question: str,
        vector_index: LoadedVectorIndex,
    ) -> tuple[list[SearchResult], int]:
        """
        Retrieve the nearest document chunks from Qdrant.

        The return type matches the retrieval interface
        expected by the guarded-RAG pipeline.
        """

        self._require_open()

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "The retrieval question cannot be empty."
            )

        if not vector_index.chunks:
            raise ValueError(
                "The source vector index contains no chunks."
            )

        active_embedding_provider = (
            generate_query_embedding
            if self._query_embedding_provider is None
            else self._query_embedding_provider
        )

        (
            query_embedding,
            query_token_count,
        ) = active_embedding_provider(
            cleaned_question,
            vector_index.embedding_model,
            vector_index.embedding_dimension,
        )

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=self._top_k,
            with_payload=True,
            with_vectors=False,
        )

        chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in vector_index.chunks
        }

        search_results: list[SearchResult] = []

        for point in response.points:
            raw_payload = point.payload

            if raw_payload is None:
                raise RuntimeError(
                    "A Qdrant search result has no payload."
                )

            payload = dict(
                raw_payload
            )

            chunk_id = self._require_payload_string(
                payload=payload,
                field_name="chunk_id",
            )

            matching_chunk = chunks_by_id.get(
                chunk_id
            )

            if matching_chunk is None:
                raise RuntimeError(
                    "Qdrant returned a chunk ID that is "
                    "missing from the source vector index: "
                    f"{chunk_id!r}."
                )

            search_results.append(
                SearchResult(
                    score=float(point.score),
                    chunk=matching_chunk,
                )
            )

        token_count = (
            int(query_token_count)
            if query_token_count is not None
            else 0
        )

        return (
            search_results,
            token_count,
        )

    def __call__(
        self,
        question: str,
        vector_index: LoadedVectorIndex,
    ) -> tuple[list[SearchResult], int]:
        """
        Allow the retriever object to behave like a function.

        This permits:

            results, tokens = retriever(
                question,
                vector_index,
            )
        """

        return self.retrieve(
            question=question,
            vector_index=vector_index,
        )

    def close(self) -> None:
        """Close the owned Qdrant client safely."""

        if self._closed:
            return

        if self._owns_client:
            self._client.close()

        self._closed = True

    def __enter__(self) -> QdrantRetriever:
        """Enter a context-manager block."""

        self._require_open()

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the retriever when leaving the block."""

        del exception_type
        del exception_value
        del traceback

        self.close()

    def _require_open(self) -> None:
        """Prevent operations after the retriever is closed."""

        if self._closed:
            raise RuntimeError(
                "The Qdrant retriever is already closed."
            )

    @staticmethod
    def _require_payload_string(
        *,
        payload: dict[str, Any],
        field_name: str,
    ) -> str:
        """Read and validate one Qdrant payload string."""

        value = payload.get(
            field_name
        )

        if not isinstance(value, str):
            raise RuntimeError(
                "A Qdrant result contains an invalid "
                f"{field_name!r} payload field."
            )

        cleaned_value = value.strip()

        if not cleaned_value:
            raise RuntimeError(
                "A Qdrant result contains an empty "
                f"{field_name!r} payload field."
            )

        return cleaned_value


def print_search_results(
    *,
    question: str,
    search_results: list[SearchResult],
    query_token_count: int,
) -> None:
    """Print one reusable-retriever demonstration."""

    print("QUESTION")
    print("--------")
    print(question)
    print()

    print(
        "Query input tokens:",
        query_token_count,
    )
    print()

    print("QDRANT RESULTS")
    print("---------------")

    if not search_results:
        print(
            "No document chunks were returned."
        )
        print()
        return

    for rank, result in enumerate(
        search_results,
        start=1,
    ):
        print(
            f"Rank: {rank}"
        )
        print(
            "Similarity:",
            f"{result.score:.4f}",
        )
        print(
            "Chunk ID:",
            result.chunk.chunk_id,
        )
        print(
            "Section:",
            result.chunk.section_title,
        )
        print(
            "Content:"
        )
        print(
            result.chunk.content
        )
        print()


def main() -> None:
    """Demonstrate the reusable Qdrant retriever."""

    vector_index = load_vector_index(
        INDEX_PATH
    )

    print("REUSABLE QDRANT RETRIEVER")
    print("==========================")
    print()
    print(
        "Storage path:",
        QDRANT_PATH,
    )
    print(
        "Collection:",
        COLLECTION_NAME,
    )
    print(
        "Embedding model:",
        vector_index.embedding_model,
    )
    print(
        "Embedding dimension:",
        vector_index.embedding_dimension,
    )
    print(
        "Source chunks:",
        len(vector_index.chunks),
    )
    print()

    with QdrantRetriever.from_local_path(
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        top_k=DEFAULT_TOP_K,
    ) as retriever:
        print(
            "1. Validating the persistent collection..."
        )

        stored_point_count = (
            retriever.validate_collection(
                expected_point_count=len(
                    vector_index.chunks
                ),
            )
        )

        print(
            "   Collection validated."
        )
        print(
            "   Stored points:",
            stored_point_count,
        )
        print(
            "   Retrieval top K:",
            retriever.top_k,
        )
        print()

        print(
            "2. Running reusable retrieval..."
        )
        print()

        for question_number, question in enumerate(
            TEST_QUESTIONS,
            start=1,
        ):
            print("=" * 70)
            print(
                f"RETRIEVAL EXAMPLE {question_number}"
            )
            print("=" * 70)

            (
                search_results,
                query_token_count,
            ) = retriever(
                question,
                vector_index,
            )

            print_search_results(
                question=question,
                search_results=search_results,
                query_token_count=query_token_count,
            )

        print("COMPONENT STATUS")
        print("----------------")
        print(
            "Retriever closed inside block:",
            retriever.is_closed,
        )

    print(
        "Retriever closed after block:",
        retriever.is_closed,
    )
    print()

    print("STATUS")
    print("------")
    print(
        "Qdrant retrieval is now packaged as a "
        "reusable application component."
    )
    print(
        "The component can be passed into future RAG "
        "pipelines without duplicating retrieval logic."
    )


if __name__ == "__main__":
    main()
