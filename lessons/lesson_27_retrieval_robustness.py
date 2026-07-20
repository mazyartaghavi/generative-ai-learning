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
class RobustnessCase:
    """One answerable or unanswerable retrieval test."""

    case_id: str
    question: str
    answerable: bool
    expected_section_ids: tuple[str, ...]


@dataclass(frozen=True)
class CaseResult:
    """Retrieval results and metrics for one test case."""

    case: RobustnessCase
    ranked_results: list[SearchResult]
    top_score: float
    first_relevant_rank: int | None
    hit_at_1: bool
    hit_at_3: bool


@dataclass(frozen=True)
class ThresholdMetrics:
    """Binary answerability metrics at one score threshold."""

    threshold: float
    true_accepts: int
    false_accepts: int
    true_rejects: int
    false_rejects: int
    true_accept_rate: float
    true_reject_rate: float
    balanced_accuracy: float


ROBUSTNESS_CASES = [
    RobustnessCase(
        case_id="engine-icon-paraphrase",
        question=(
            "The dashboard has illuminated an engine icon. "
            "What is the driver expected to do?"
        ),
        answerable=True,
        expected_section_ids=(
            "section-002",
        ),
    ),
    RobustnessCase(
        case_id="repeated-air-loss",
        question=(
            "A tire keeps losing air repeatedly. "
            "Can the vehicle remain in service?"
        ),
        answerable=True,
        expected_section_ids=(
            "section-003",
        ),
    ),
    RobustnessCase(
        case_id="end-of-trip-records",
        question=(
            "Which details must be logged when a delivery "
            "trip has finished?"
        ),
        answerable=True,
        expected_section_ids=(
            "section-004",
        ),
    ),
    RobustnessCase(
        case_id="crash-or-fire",
        question=(
            "Who must be contacted after a vehicle crash "
            "or fire?"
        ),
        answerable=True,
        expected_section_ids=(
            "section-005",
        ),
    ),
    RobustnessCase(
        case_id="manual-purpose",
        question=(
            "What is the general purpose of this fleet manual?"
        ),
        answerable=True,
        expected_section_ids=(
            "section-001",
        ),
    ),
    RobustnessCase(
        case_id="smoke-and-power-loss",
        question=(
            "The vehicle is producing smoke and losing power. "
            "Which procedures are relevant?"
        ),
        answerable=True,
        expected_section_ids=(
            "section-002",
            "section-005",
        ),
    ),
    RobustnessCase(
        case_id="oil-change-interval",
        question=(
            "How many kilometres may the vehicle travel "
            "between engine oil changes?"
        ),
        answerable=False,
        expected_section_ids=(),
    ),
    RobustnessCase(
        case_id="maximum-speed",
        question=(
            "What is the maximum permitted speed for "
            "company vehicles?"
        ),
        answerable=False,
        expected_section_ids=(),
    ),
    RobustnessCase(
        case_id="fuel-payment",
        question=(
            "Which employee is responsible for paying "
            "for vehicle fuel?"
        ),
        answerable=False,
        expected_section_ids=(),
    ),
    RobustnessCase(
        case_id="driver-password",
        question=(
            "What password should a driver use to sign in "
            "to the fleet management system?"
        ),
        answerable=False,
        expected_section_ids=(),
    ),
    RobustnessCase(
        case_id="insurance-number",
        question=(
            "What is the fleet insurance policy number?"
        ),
        answerable=False,
        expected_section_ids=(),
    ),
]


def find_first_relevant_rank(
    ranked_results: list[SearchResult],
    expected_section_ids: tuple[str, ...],
) -> int | None:
    """Return the first rank belonging to an expected section."""

    expected_sections = set(
        expected_section_ids
    )

    if not expected_sections:
        return None

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
    case: RobustnessCase,
    ranked_results: list[SearchResult],
) -> CaseResult:
    """Calculate retrieval metrics for one case."""

    if not ranked_results:
        raise ValueError(
            "At least one ranked result is required."
        )

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

    return CaseResult(
        case=case,
        ranked_results=ranked_results,
        top_score=ranked_results[0].score,
        first_relevant_rank=first_relevant_rank,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
    )


def build_threshold_candidates(
    results: list[CaseResult],
) -> list[float]:
    """Build thresholds from scores and score midpoints."""

    if not results:
        raise ValueError(
            "At least one case result is required."
        )

    unique_scores = sorted(
        {
            result.top_score
            for result in results
        }
    )

    candidates = {
        unique_scores[0] - 0.000001,
        unique_scores[-1] + 0.000001,
    }

    candidates.update(
        unique_scores
    )

    # The second sequence is intentionally one item shorter.
    # Therefore, strict=True must not be used here.
    for lower_score, upper_score in zip(
        unique_scores,
        unique_scores[1:],
    ):
        midpoint = (
            lower_score + upper_score
        ) / 2.0

        candidates.add(
            midpoint
        )

    return sorted(
        candidates
    )


def evaluate_threshold(
    results: list[CaseResult],
    threshold: float,
) -> ThresholdMetrics:
    """Evaluate answerability classification at one threshold."""

    true_accepts = 0
    false_accepts = 0
    true_rejects = 0
    false_rejects = 0

    for result in results:
        accepted = (
            result.top_score >= threshold
        )

        if result.case.answerable and accepted:
            true_accepts += 1

        elif result.case.answerable and not accepted:
            false_rejects += 1

        elif not result.case.answerable and accepted:
            false_accepts += 1

        else:
            true_rejects += 1

    answerable_count = (
        true_accepts + false_rejects
    )

    unanswerable_count = (
        true_rejects + false_accepts
    )

    if answerable_count == 0:
        raise ValueError(
            "The dataset contains no answerable cases."
        )

    if unanswerable_count == 0:
        raise ValueError(
            "The dataset contains no unanswerable cases."
        )

    true_accept_rate = (
        true_accepts / answerable_count
    )

    true_reject_rate = (
        true_rejects / unanswerable_count
    )

    balanced_accuracy = (
        true_accept_rate + true_reject_rate
    ) / 2.0

    return ThresholdMetrics(
        threshold=threshold,
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        true_rejects=true_rejects,
        false_rejects=false_rejects,
        true_accept_rate=true_accept_rate,
        true_reject_rate=true_reject_rate,
        balanced_accuracy=balanced_accuracy,
    )


def select_best_threshold(
    metrics: list[ThresholdMetrics],
) -> ThresholdMetrics:
    """Select the best threshold using balanced accuracy."""

    if not metrics:
        raise ValueError(
            "At least one threshold result is required."
        )

    return max(
        metrics,
        key=lambda item: (
            item.balanced_accuracy,
            -item.false_accepts,
            item.threshold,
        ),
    )


def print_case_result(
    result: CaseResult,
) -> None:
    """Print the detailed result for one query."""

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
        "Expected answerability:",
        result.case.answerable,
    )

    if result.case.expected_section_ids:
        print(
            "Expected sections:",
            ", ".join(
                result.case.expected_section_ids
            ),
        )
    else:
        print(
            "Expected sections: none"
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
        "Top similarity score:",
        f"{result.top_score:.4f}",
    )

    if result.case.answerable:
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
    else:
        print(
            "This question is labelled unanswerable."
        )
        print(
            "The nearest result is therefore a distractor, "
            "not a correct answer."
        )

    print()


def print_threshold_metrics(
    metrics: ThresholdMetrics,
    selected: bool,
) -> None:
    """Print one threshold-evaluation row."""

    marker = (
        " <-- selected"
        if selected
        else ""
    )

    print(
        f"Threshold {metrics.threshold:.4f} | "
        f"TA: {metrics.true_accepts} | "
        f"FA: {metrics.false_accepts} | "
        f"TR: {metrics.true_rejects} | "
        f"FR: {metrics.false_rejects} | "
        f"Balanced accuracy: "
        f"{metrics.balanced_accuracy:.2%}"
        f"{marker}"
    )


def main() -> None:
    """Evaluate robust retrieval and calibrate abstention."""

    vector_index = load_vector_index(
        INDEX_PATH
    )

    answerable_case_count = sum(
        case.answerable
        for case in ROBUSTNESS_CASES
    )

    unanswerable_case_count = sum(
        not case.answerable
        for case in ROBUSTNESS_CASES
    )

    print("RETRIEVAL ROBUSTNESS EVALUATION")
    print("===============================")
    print()
    print(
        "Index:",
        INDEX_PATH,
    )
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
        len(ROBUSTNESS_CASES),
    )
    print(
        "Answerable cases:",
        answerable_case_count,
    )
    print(
        "Unanswerable cases:",
        unanswerable_case_count,
    )
    print()

    results: list[CaseResult] = []

    for case_number, case in enumerate(
        ROBUSTNESS_CASES,
        start=1,
    ):
        print(
            f"Embedding query "
            f"{case_number}/{len(ROBUSTNESS_CASES)}..."
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

        results.append(
            evaluate_case(
                case=case,
                ranked_results=ranked_results,
            )
        )

    print(
        "All robustness queries completed."
    )
    print()
    print(
        "DETAILED RESULTS"
    )
    print(
        "================"
    )
    print()

    for result in results:
        print_case_result(
            result
        )

    answerable_results = [
        result
        for result in results
        if result.case.answerable
    ]

    answerable_count = len(
        answerable_results
    )

    if answerable_count == 0:
        raise ValueError(
            "The evaluation contains no answerable cases."
        )

    hit_at_1_count = sum(
        result.hit_at_1
        for result in answerable_results
    )

    hit_at_3_count = sum(
        result.hit_at_3
        for result in answerable_results
    )

    hit_at_1_rate = (
        hit_at_1_count / answerable_count
    )

    hit_at_3_rate = (
        hit_at_3_count / answerable_count
    )

    print(
        "ANSWERABLE RETRIEVAL METRICS"
    )
    print(
        "============================"
    )
    print(
        "Answerable cases:",
        answerable_count,
    )
    print(
        "Hit@1:",
        f"{hit_at_1_count}/{answerable_count} "
        f"({hit_at_1_rate:.2%})",
    )
    print(
        "Hit@3:",
        f"{hit_at_3_count}/{answerable_count} "
        f"({hit_at_3_rate:.2%})",
    )
    print()

    threshold_candidates = build_threshold_candidates(
        results
    )

    threshold_metrics = [
        evaluate_threshold(
            results=results,
            threshold=threshold,
        )
        for threshold in threshold_candidates
    ]

    selected_metrics = select_best_threshold(
        threshold_metrics
    )

    print(
        "THRESHOLD CALIBRATION"
    )
    print(
        "====================="
    )
    print(
        "TA = true accept, FA = false accept, "
        "TR = true reject, FR = false reject"
    )
    print()

    for metrics in threshold_metrics:
        print_threshold_metrics(
            metrics=metrics,
            selected=(
                metrics == selected_metrics
            ),
        )

    print()
    print(
        "SELECTED THRESHOLD"
    )
    print(
        "=================="
    )
    print(
        "Threshold:",
        f"{selected_metrics.threshold:.4f}",
    )
    print(
        "True accept rate:",
        f"{selected_metrics.true_accept_rate:.2%}",
    )
    print(
        "True reject rate:",
        f"{selected_metrics.true_reject_rate:.2%}",
    )
    print(
        "Balanced accuracy:",
        f"{selected_metrics.balanced_accuracy:.2%}",
    )
    print(
        "False acceptances:",
        selected_metrics.false_accepts,
    )
    print(
        "False rejections:",
        selected_metrics.false_rejects,
    )
    print()

    print(
        "CLASSIFICATIONS AT SELECTED THRESHOLD"
    )
    print(
        "====================================="
    )

    for result in results:
        accepted = (
            result.top_score
            >= selected_metrics.threshold
        )

        expected_action = (
            "ACCEPT"
            if result.case.answerable
            else "ABSTAIN"
        )

        actual_action = (
            "ACCEPT"
            if accepted
            else "ABSTAIN"
        )

        classification_correct = (
            expected_action == actual_action
        )

        print(
            f"{result.case.case_id}: "
            f"score={result.top_score:.4f} | "
            f"expected={expected_action} | "
            f"actual={actual_action} | "
            f"correct={classification_correct}"
        )

    print()
    print(
        "IMPORTANT LIMITATION"
    )
    print(
        "===================="
    )
    print(
        "The selected threshold was calibrated on this "
        "small dataset. It is not a universal confidence "
        "threshold and may overfit these examples."
    )
    print(
        "A production system should calibrate and validate "
        "the threshold on larger, independent datasets."
    )


if __name__ == "__main__":
    main()