from __future__ import annotations

from dataclasses import dataclass

from lessons.lesson_23_search_vector_index import (
    INDEX_PATH,
    SearchResult,
    generate_query_embedding,
    load_vector_index,
    search_index,
)


DISPLAY_TOP_K = 3


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """One question with its expected relevant sections."""

    case_id: str
    question: str
    expected_section_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """Metrics and rankings for one evaluation question."""

    case: RetrievalEvaluationCase
    ranked_results: list[SearchResult]
    first_relevant_rank: int | None
    hit_at_1: bool
    hit_at_3: bool
    reciprocal_rank: float


EVALUATION_CASES = [
    RetrievalEvaluationCase(
        case_id="engine-warning",
        question=(
            "What should a driver do when an engine "
            "warning light appears?"
        ),
        expected_section_ids=(
            "section-002",
        ),
    ),
    RetrievalEvaluationCase(
        case_id="tire-pressure",
        question=(
            "What action is required when a vehicle "
            "has severely low tire pressure?"
        ),
        expected_section_ids=(
            "section-003",
        ),
    ),
    RetrievalEvaluationCase(
        case_id="route-completion",
        question=(
            "What information must a driver record "
            "after completing a delivery route?"
        ),
        expected_section_ids=(
            "section-004",
        ),
    ),
    RetrievalEvaluationCase(
        case_id="emergency-escalation",
        question=(
            "When must a driver contact the emergency "
            "coordinator?"
        ),
        expected_section_ids=(
            "section-005",
        ),
    ),
    RetrievalEvaluationCase(
        case_id="manual-overview",
        question=(
            "What does the manual say about reporting "
            "vehicle faults?"
        ),
        expected_section_ids=(
            "section-001",
        ),
    ),
]


def find_first_relevant_rank(
    ranked_results: list[SearchResult],
    expected_section_ids: tuple[str, ...],
) -> int | None:
    """Find the one-based rank of the first relevant result."""

    expected_sections = set(
        expected_section_ids
    )

    for rank, result in enumerate(
        ranked_results,
        start=1,
    ):
        if (
            result.chunk.section_id
            in expected_sections
        ):
            return rank

    return None


def evaluate_case(
    case: RetrievalEvaluationCase,
    ranked_results: list[SearchResult],
) -> RetrievalEvaluationResult:
    """Calculate retrieval metrics for one question."""

    first_relevant_rank = find_first_relevant_rank(
        ranked_results=ranked_results,
        expected_section_ids=(
            case.expected_section_ids
        ),
    )

    hit_at_1 = (
        first_relevant_rank is not None
        and first_relevant_rank <= 1
    )

    hit_at_3 = (
        first_relevant_rank is not None
        and first_relevant_rank <= 3
    )

    if first_relevant_rank is None:
        reciprocal_rank = 0.0
    else:
        reciprocal_rank = (
            1.0 / first_relevant_rank
        )

    return RetrievalEvaluationResult(
        case=case,
        ranked_results=ranked_results,
        first_relevant_rank=first_relevant_rank,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        reciprocal_rank=reciprocal_rank,
    )


def format_expected_sections(
    section_ids: tuple[str, ...],
) -> str:
    """Format expected section identifiers for printing."""

    return ", ".join(
        section_ids
    )


def print_case_result(
    result: RetrievalEvaluationResult,
) -> None:
    """Print detailed rankings and metrics for one case."""

    print(
        f"CASE: {result.case.case_id}"
    )
    print(
        "-" * (
            6 + len(result.case.case_id)
        )
    )
    print(
        "Question:",
        result.case.question,
    )
    print(
        "Expected sections:",
        format_expected_sections(
            result.case.expected_section_ids
        ),
    )
    print()

    print(
        f"TOP {DISPLAY_TOP_K} RESULTS"
    )

    for rank, search_result in enumerate(
        result.ranked_results[
            :DISPLAY_TOP_K
        ],
        start=1,
    ):
        chunk = search_result.chunk

        is_relevant = (
            chunk.section_id
            in result.case.expected_section_ids
        )

        print(
            f"{rank}. "
            f"{chunk.section_id} | "
            f"{chunk.section_title} | "
            f"{search_result.score:.4f} | "
            f"Relevant: {is_relevant}"
        )
        print(
            "   Chunk:",
            chunk.chunk_id,
        )

    print()
    print(
        "First relevant rank:",
        result.first_relevant_rank,
    )
    print(
        "Hit@1:",
        result.hit_at_1,
    )
    print(
        "Hit@3:",
        result.hit_at_3,
    )
    print(
        "Reciprocal rank:",
        f"{result.reciprocal_rank:.4f}",
    )
    print()


def main() -> None:
    """Evaluate vector retrieval across labelled questions."""

    vector_index = load_vector_index(
        INDEX_PATH
    )

    print("RETRIEVAL EVALUATION")
    print("====================")
    print()
    print("Index:", INDEX_PATH)
    print(
        "Embedding model:",
        vector_index.embedding_model,
    )
    print(
        "Indexed chunks:",
        len(vector_index.chunks),
    )
    print(
        "Evaluation cases:",
        len(EVALUATION_CASES),
    )
    print()

    evaluation_results: list[
        RetrievalEvaluationResult
    ] = []

    for case_number, case in enumerate(
        EVALUATION_CASES,
        start=1,
    ):
        print(
            f"Embedding query "
            f"{case_number}/{len(EVALUATION_CASES)}..."
        )

        (
            query_embedding,
            _query_token_count,
        ) = generate_query_embedding(
            query=case.question,
            embedding_model=(
                vector_index.embedding_model
            ),
            expected_dimension=(
                vector_index.embedding_dimension
            ),
        )

        ranked_results = search_index(
            query_embedding=query_embedding,
            chunks=vector_index.chunks,
            top_k=len(
                vector_index.chunks
            ),
        )

        evaluation_result = evaluate_case(
            case=case,
            ranked_results=ranked_results,
        )

        evaluation_results.append(
            evaluation_result
        )

    print("All evaluation queries completed.")
    print()

    print("DETAILED RESULTS")
    print("================")
    print()

    for result in evaluation_results:
        print_case_result(
            result
        )

    case_count = len(
        evaluation_results
    )

    hit_at_1_count = sum(
        result.hit_at_1
        for result in evaluation_results
    )

    hit_at_3_count = sum(
        result.hit_at_3
        for result in evaluation_results
    )

    mean_reciprocal_rank = (
        sum(
            result.reciprocal_rank
            for result in evaluation_results
        )
        / case_count
    )

    hit_at_1_rate = (
        hit_at_1_count / case_count
    )

    hit_at_3_rate = (
        hit_at_3_count / case_count
    )

    print("AGGREGATE METRICS")
    print("=================")
    print(
        "Cases evaluated:",
        case_count,
    )
    print(
        "Hit@1 count:",
        f"{hit_at_1_count}/{case_count}",
    )
    print(
        "Hit@1 rate:",
        f"{hit_at_1_rate:.2%}",
    )
    print(
        "Hit@3 count:",
        f"{hit_at_3_count}/{case_count}",
    )
    print(
        "Hit@3 rate:",
        f"{hit_at_3_rate:.2%}",
    )
    print(
        "Mean Reciprocal Rank:",
        f"{mean_reciprocal_rank:.4f}",
    )
    print()

    print("INTERPRETATION")
    print("==============")

    if hit_at_1_rate == 1.0:
        print(
            "Every evaluation question retrieved a "
            "relevant section at rank 1."
        )
    else:
        print(
            "At least one evaluation question did not "
            "retrieve a relevant section at rank 1."
        )

    if hit_at_3_rate == 1.0:
        print(
            "Every evaluation question retrieved a "
            "relevant section within the top three results."
        )
    else:
        print(
            "At least one evaluation question failed to "
            "retrieve a relevant section within the top "
            "three results."
        )

    print(
        "These results apply only to this small labelled "
        "evaluation set, not to every possible question."
    )


if __name__ == "__main__":
    main()