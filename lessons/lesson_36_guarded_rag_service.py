from __future__ import annotations

from pathlib import Path
from types import TracebackType

from lessons import lesson_28_guarded_rag as guarded
from lessons.lesson_23_search_vector_index import (
    INDEX_PATH,
    LoadedVectorIndex,
    load_vector_index,
)
from lessons.lesson_30_qdrant_local_index import (
    COLLECTION_NAME,
    QDRANT_PATH,
)
from lessons.lesson_35_qdrant_retriever_component import (
    DEFAULT_TOP_K,
    QdrantRetriever,
    QueryEmbeddingProvider,
)


QUESTIONS = [
    (
        "What should a driver do when an engine "
        "warning light appears?"
    ),
    (
        "Can a vehicle remain in service after "
        "repeated tire-pressure loss?"
    ),
    "What is the driver's account password?",
    "What is the fleet insurance policy number?",
]


class GuardedRAGService:
    """
    Provide a reusable guarded-RAG application service.

    The service combines:

    - A source vector index containing chunk metadata.
    - A reusable Qdrant retriever.
    - Lesson 28's retrieval threshold.
    - Grounded generation.
    - Citation validation.
    - Deterministic citation repair.
    - Extractive fallback.
    - Final abstention.
    """

    def __init__(
        self,
        *,
        vector_index: LoadedVectorIndex,
        retriever: QdrantRetriever,
        owns_retriever: bool = False,
    ) -> None:
        """Initialize the guarded-RAG service."""

        if not vector_index.chunks:
            raise ValueError(
                "The source vector index contains no chunks."
            )

        if retriever.is_closed:
            raise ValueError(
                "The supplied Qdrant retriever is closed."
            )

        self._vector_index = vector_index
        self._retriever = retriever
        self._owns_retriever = owns_retriever
        self._generation_provider: (
            guarded.GenerationFunction | None
        ) = None
        self._model_providers_configured = False
        self._closed = False
        self._answer_in_progress = False

    @classmethod
    def from_local_qdrant(
        cls,
        *,
        index_path: Path = INDEX_PATH,
        qdrant_path: Path = QDRANT_PATH,
        collection_name: str = COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
    ) -> GuardedRAGService:
        """
        Build the complete service from persistent storage.

        The resulting service owns the retriever and closes
        it automatically when the service is closed.
        """

        if not isinstance(index_path, Path):
            raise TypeError(
                "The source index path must be a "
                "pathlib.Path object."
            )

        if not isinstance(qdrant_path, Path):
            raise TypeError(
                "The Qdrant path must be a "
                "pathlib.Path object."
            )

        vector_index = load_vector_index(
            index_path
        )

        retriever = QdrantRetriever.from_local_path(
            path=qdrant_path,
            collection_name=collection_name,
            top_k=top_k,
        )

        try:
            retriever.validate_collection(
                expected_point_count=len(
                    vector_index.chunks
                )
            )
        except Exception:
            retriever.close()
            raise

        return cls(
            vector_index=vector_index,
            retriever=retriever,
            owns_retriever=True,
        )

    @property
    def is_closed(self) -> bool:
        """Report whether the service has been closed."""

        return self._closed

    @property
    def source_chunk_count(self) -> int:
        """Return the number of source chunks."""

        return len(
            self._vector_index.chunks
        )

    @property
    def collection_name(self) -> str:
        """Return the active Qdrant collection name."""

        return self._retriever.collection_name

    @property
    def top_k(self) -> int:
        """Return the configured retrieval result limit."""

        return self._retriever.top_k

    @property
    def model_providers_configured(self) -> bool:
        """Report whether shared model providers are active."""

        return self._model_providers_configured

    def configure_model_providers(
        self,
        *,
        query_embedding_provider: QueryEmbeddingProvider,
        generation_provider: guarded.GenerationFunction,
    ) -> None:
        """
        Bind shared embedding and generation providers.

        The service does not own the provider resources. Their
        lifecycle remains the responsibility of the application
        runtime that supplied them.
        """

        self._require_open()

        if self._answer_in_progress:
            raise RuntimeError(
                "Model providers cannot be changed while an "
                "answer is being generated."
            )

        if not callable(query_embedding_provider):
            raise TypeError(
                "The query embedding provider must be "
                "callable."
            )

        if not callable(generation_provider):
            raise TypeError(
                "The generation provider must be callable."
            )

        self._retriever.set_query_embedding_provider(
            query_embedding_provider
        )

        self._generation_provider = generation_provider
        self._model_providers_configured = True

    def validate(self) -> int:
        """Validate the retriever and source index."""

        self._require_open()

        return self._retriever.validate_collection(
            expected_point_count=len(
                self._vector_index.chunks
            )
        )

    def answer(
        self,
        question: str,
    ) -> guarded.GuardedRAGResult:
        """
        Answer one question through the guarded pipeline.

        The Qdrant retriever is supplied explicitly, so the
        service does not modify Lesson 28's module-level
        retrieval dependency.
        """

        self._require_open()

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "The question cannot be empty."
            )

        if self._answer_in_progress:
            raise RuntimeError(
                "A guarded-RAG answer is already in progress "
                "for this service."
            )

        self._answer_in_progress = True

        try:
            return guarded.run_guarded_rag(
                question=cleaned_question,
                vector_index=self._vector_index,
                retriever=self._retriever,
                generator=self._generation_provider,
            )
        finally:
            self._answer_in_progress = False

    def answer_many(
        self,
        questions: list[str],
    ) -> list[guarded.GuardedRAGResult]:
        """Answer several questions sequentially."""

        self._require_open()

        if not questions:
            raise ValueError(
                "At least one question is required."
            )

        results: list[
            guarded.GuardedRAGResult
        ] = []

        for question in questions:
            result = self.answer(
                question
            )

            results.append(
                result
            )

        return results

    def close(self) -> None:
        """Close resources owned by the service."""

        if self._closed:
            return

        if self._answer_in_progress:
            raise RuntimeError(
                "The service cannot be closed while an "
                "answer is being generated."
            )

        if self._owns_retriever:
            self._retriever.close()

        self._closed = True

    def __enter__(self) -> GuardedRAGService:
        """Enter a context-manager block."""

        self._require_open()

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the service when leaving the block."""

        del exception_type
        del exception_value
        del traceback

        self.close()

    def _require_open(self) -> None:
        """Prevent operations after closing the service."""

        if self._closed:
            raise RuntimeError(
                "The guarded-RAG service is already closed."
            )


def print_result(
    *,
    question_number: int,
    result: guarded.GuardedRAGResult,
) -> None:
    """Print one service response."""

    print("=" * 70)
    print(
        f"QUESTION {question_number}"
    )
    print("=" * 70)
    print(result.question)
    print()

    print("RETRIEVAL")
    print("---------")
    print(
        "Top similarity:",
        f"{result.top_similarity:.4f}",
    )
    print(
        "Threshold:",
        f"{result.threshold:.4f}",
    )

    if result.retrieval_results:
        print(
            "Top section:",
            (
                result.retrieval_results[
                    0
                ].chunk.section_title
            ),
        )
        print(
            "Top chunk:",
            (
                result.retrieval_results[
                    0
                ].chunk.chunk_id
            ),
        )
    else:
        print(
            "Top section:",
            "none",
        )
        print(
            "Top chunk:",
            "none",
        )

    print()

    print("DECISION")
    print("--------")
    print(
        "Decision:",
        result.decision,
    )
    print(
        "Strategy:",
        result.answer_strategy,
    )
    print(
        "Generation called:",
        result.generation_called,
    )
    print(
        "Citation repair used:",
        result.citation_repair_used,
    )
    print(
        "Extractive fallback used:",
        result.extractive_fallback_used,
    )
    print(
        "Validation passed:",
        result.answer_passed_validation,
    )
    print()

    print("ANSWER")
    print("------")
    print(result.answer)
    print()


def print_summary(
    results: list[guarded.GuardedRAGResult],
) -> None:
    """Print aggregate service results."""

    answer_count = sum(
        result.decision == "ANSWER"
        for result in results
    )

    retrieval_abstention_count = sum(
        result.answer_strategy
        == "retrieval abstention"
        for result in results
    )

    validation_abstention_count = sum(
        result.answer_strategy
        == "validation abstention"
        for result in results
    )

    generation_count = sum(
        result.generation_called
        for result in results
    )

    repair_count = sum(
        result.citation_repair_used
        for result in results
    )

    fallback_answer_count = sum(
        result.extractive_fallback_used
        and result.decision == "ANSWER"
        for result in results
    )

    print("=" * 70)
    print("GUARDED-RAG SERVICE SUMMARY")
    print("=" * 70)
    print(
        "Questions:",
        len(results),
    )
    print(
        "Answers:",
        answer_count,
    )
    print(
        "Retrieval abstentions:",
        retrieval_abstention_count,
    )
    print(
        "Validation abstentions:",
        validation_abstention_count,
    )
    print(
        "Generation calls:",
        generation_count,
    )
    print(
        "Citation repairs:",
        repair_count,
    )
    print(
        "Extractive fallback answers:",
        fallback_answer_count,
    )
    print()


def main() -> None:
    """Demonstrate the reusable guarded-RAG service."""

    print("REUSABLE GUARDED-RAG SERVICE")
    print("============================")
    print()
    print(
        "Source index:",
        INDEX_PATH,
    )
    print(
        "Qdrant storage:",
        QDRANT_PATH,
    )
    print(
        "Collection:",
        COLLECTION_NAME,
    )
    print()

    with GuardedRAGService.from_local_qdrant(
        index_path=INDEX_PATH,
        qdrant_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        top_k=DEFAULT_TOP_K,
    ) as service:
        print(
            "1. Validating the complete service..."
        )

        stored_point_count = service.validate()

        print(
            "   Service validated."
        )
        print(
            "   Source chunks:",
            service.source_chunk_count,
        )
        print(
            "   Stored Qdrant points:",
            stored_point_count,
        )
        print(
            "   Retrieval top K:",
            service.top_k,
        )
        print()

        print(
            "2. Answering questions through the service..."
        )
        print()

        results = service.answer_many(
            QUESTIONS
        )

        for question_number, result in enumerate(
            results,
            start=1,
        ):
            print_result(
                question_number=question_number,
                result=result,
            )

        print_summary(
            results
        )

        print("SERVICE STATE")
        print("-------------")
        print(
            "Closed inside context:",
            service.is_closed,
        )
        print()

    print(
        "Closed after context:",
        service.is_closed,
    )
    print()

    print("STATUS")
    print("------")
    print(
        "The application now has one reusable service "
        "for Qdrant retrieval and guarded answer generation."
    )
    print(
        "Callers no longer need to configure or restore "
        "the retrieval dependency manually."
    )


if __name__ == "__main__":
    main()
