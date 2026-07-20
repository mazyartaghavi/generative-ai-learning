from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qdrant_client import QdrantClient

from lessons import lesson_28_guarded_rag as guarded
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


RETRIEVAL_TOP_K = 3

QUESTIONS = [
    (
        "What should a driver do when an engine "
        "warning icon appears?"
    ),
    (
        "A tire keeps losing air repeatedly. "
        "Can the vehicle remain in service?"
    ),
    "What is the driver's account password?",
    "What is the fleet insurance policy number?",
]


RetrieverFunction = Callable[
    [str, LoadedVectorIndex],
    tuple[list[SearchResult], int],
]


def require_payload_string(
    payload: dict[str, Any],
    field_name: str,
) -> str:
    """Read and validate a required Qdrant payload string."""

    value = payload.get(field_name)

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


def create_qdrant_retriever(
    client: QdrantClient,
) -> RetrieverFunction:
    """
    Create a Qdrant retriever compatible with Lesson 28.

    Lesson 28 expects a retrieval function with this form:

        retrieve(question, vector_index)
            -> (search_results, query_token_count)
    """

    def retrieve_from_qdrant(
        question: str,
        vector_index: LoadedVectorIndex,
    ) -> tuple[list[SearchResult], int]:
        """Embed one question and retrieve its nearest chunks."""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "The retrieval question cannot be empty."
            )

        if not vector_index.chunks:
            raise ValueError(
                "The source vector index contains no chunks."
            )

        (
            query_embedding,
            query_token_count,
        ) = generate_query_embedding(
            query=cleaned_question,
            embedding_model=vector_index.embedding_model,
            expected_dimension=(
                vector_index.embedding_dimension
            ),
        )

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=RETRIEVAL_TOP_K,
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

            payload = dict(raw_payload)

            chunk_id = require_payload_string(
                payload,
                "chunk_id",
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
            query_token_count
            if query_token_count is not None
            else 0
        )

        return (
            search_results,
            token_count,
        )

    return retrieve_from_qdrant


def print_retrieval_results(
    results: list[SearchResult],
) -> None:
    """Print the Qdrant results used by guarded RAG."""

    print("QDRANT RETRIEVAL")
    print("-----------------")

    if not results:
        print("No chunks were retrieved.")
        print()
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"Rank {rank}: "
            f"{result.chunk.section_title}"
        )
        print(
            "  Chunk ID:",
            result.chunk.chunk_id,
        )
        print(
            "  Similarity:",
            f"{result.score:.4f}",
        )

    print()


def print_citation_checks(
    result: guarded.GuardedRAGResult,
) -> None:
    """Print citation-validation information."""

    print("CITATION VALIDATION")
    print("-------------------")

    if not result.citation_checks:
        print(
            "No citation checks were required."
        )
        print()
        return

    for check in result.citation_checks:
        cited_labels = (
            ", ".join(check.cited_labels)
            if check.cited_labels
            else "none"
        )

        print(
            "Sentence:",
            check.sentence_number,
        )
        print(
            "  Citations:",
            cited_labels,
        )
        print(
            "  Lexical support:",
            f"{check.lexical_coverage:.2%}",
        )
        print(
            "  Passed:",
            check.support_passed,
        )

        if check.invalid_labels:
            print(
                "  Invalid labels:",
                ", ".join(check.invalid_labels),
            )

        if check.missing_support_terms:
            print(
                "  Missing support terms:",
                ", ".join(
                    check.missing_support_terms
                ),
            )

    print()


def print_pipeline_actions(
    result: guarded.GuardedRAGResult,
) -> None:
    """Print which guarded-RAG stages were used."""

    print("PIPELINE ACTIONS")
    print("----------------")
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
        "Final validation passed:",
        result.answer_passed_validation,
    )
    print(
        "Generation input tokens:",
        result.generation_input_tokens,
    )
    print(
        "Generation output tokens:",
        result.generation_output_tokens,
    )
    print()


def print_guarded_result(
    question_number: int,
    result: guarded.GuardedRAGResult,
) -> None:
    """Print one complete guarded-RAG result."""

    print("=" * 70)
    print(
        f"QUESTION {question_number}"
    )
    print("=" * 70)
    print(result.question)
    print()

    print_retrieval_results(
        result.retrieval_results
    )

    print("GUARD DECISION")
    print("--------------")
    print(
        "Top similarity:",
        f"{result.top_similarity:.4f}",
    )
    print(
        "Required threshold:",
        f"{result.threshold:.4f}",
    )
    print(
        "Decision:",
        result.decision,
    )
    print(
        "Answer strategy:",
        result.answer_strategy,
    )
    print()

    print_pipeline_actions(
        result
    )

    print("FINAL ANSWER")
    print("------------")
    print(result.answer)
    print()

    print_citation_checks(
        result
    )


def validate_qdrant_database(
    client: QdrantClient,
    expected_point_count: int,
) -> int:
    """Validate the persistent collection before retrieval."""

    if expected_point_count <= 0:
        raise ValueError(
            "The expected point count must be positive."
        )

    if not client.collection_exists(
        collection_name=COLLECTION_NAME,
    ):
        raise RuntimeError(
            "The Qdrant collection does not exist. "
            "Run Lesson 30 first."
        )

    count_result = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )

    stored_point_count = count_result.count

    if stored_point_count <= 0:
        raise RuntimeError(
            "The Qdrant collection contains no points."
        )

    if stored_point_count != expected_point_count:
        raise RuntimeError(
            "The Qdrant collection and source vector index "
            "contain different numbers of chunks. "
            f"Qdrant contains {stored_point_count}, while "
            f"the source index contains "
            f"{expected_point_count}."
        )

    return stored_point_count


def print_summary(
    results: list[guarded.GuardedRAGResult],
) -> None:
    """Print aggregate guarded-RAG results."""

    if not results:
        raise ValueError(
            "At least one guarded-RAG result is required."
        )

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

    generation_call_count = sum(
        result.generation_called
        for result in results
    )

    citation_repair_count = sum(
        result.citation_repair_used
        for result in results
    )

    extractive_fallback_answer_count = sum(
        result.extractive_fallback_used
        and result.decision == "ANSWER"
        for result in results
    )

    thresholds = {
        result.threshold
        for result in results
    }

    print("=" * 70)
    print("QDRANT GUARDED-RAG SUMMARY")
    print("=" * 70)
    print(
        "Questions:",
        len(results),
    )
    print(
        "Final answers:",
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
        generation_call_count,
    )
    print(
        "Citation repairs attempted:",
        citation_repair_count,
    )
    print(
        "Extractive fallback answers:",
        extractive_fallback_answer_count,
    )

    if len(thresholds) == 1:
        threshold = next(
            iter(thresholds)
        )

        print(
            "Guard threshold:",
            f"{threshold:.4f}",
        )
    else:
        formatted_thresholds = sorted(
            f"{threshold:.4f}"
            for threshold in thresholds
        )

        print(
            "Guard thresholds:",
            ", ".join(formatted_thresholds),
        )

    print()


def main() -> None:
    """Run guarded RAG using persistent Qdrant retrieval."""

    vector_index = load_vector_index(
        INDEX_PATH
    )

    print("GUARDED RAG WITH QDRANT RETRIEVAL")
    print("=================================")
    print()
    print(
        "Qdrant storage path:",
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
    print(
        "Retrieval top K:",
        RETRIEVAL_TOP_K,
    )
    print(
        "Guard threshold:",
        "reported by Lesson 28 after retrieval",
    )
    print()

    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            "The Qdrant storage directory was not found. "
            "Run Lesson 30 first."
        )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    original_retriever = guarded.retrieve_chunks

    try:
        print(
            "1. Opening the persistent Qdrant database..."
        )

        stored_point_count = validate_qdrant_database(
            client=client,
            expected_point_count=len(
                vector_index.chunks
            ),
        )

        print(
            "   Existing collection found."
        )
        print(
            "   Stored points:",
            stored_point_count,
        )
        print()

        print(
            "2. Installing the Qdrant retrieval adapter..."
        )

        qdrant_retriever = create_qdrant_retriever(
            client
        )

        guarded.retrieve_chunks = qdrant_retriever

        print(
            "   Lesson 28 will now retrieve from Qdrant."
        )
        print()

        print(
            "3. Running guarded RAG questions..."
        )
        print()

        results: list[
            guarded.GuardedRAGResult
        ] = []

        for question_number, question in enumerate(
            QUESTIONS,
            start=1,
        ):
            result = guarded.run_guarded_rag(
                question=question,
                vector_index=vector_index,
            )

            results.append(result)

            print_guarded_result(
                question_number=question_number,
                result=result,
            )

        print_summary(
            results
        )

        print("STATUS")
        print("------")
        print(
            "The guarded-RAG pipeline used Qdrant for "
            "nearest-vector retrieval."
        )
        print(
            "Low-confidence questions were rejected "
            "before Llama generation."
        )
        print(
            "Accepted questions remained protected by "
            "citation validation, deterministic repair, "
            "extractive fallback, and final abstention."
        )

    finally:
        guarded.retrieve_chunks = original_retriever
        client.close()


if __name__ == "__main__":
    main()