from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

from lessons.lesson_23_search_vector_index import (
    INDEX_PATH,
    generate_query_embedding,
    load_vector_index,
    search_index,
)
from lessons.lesson_24_grounded_rag import (
    INITIAL_TOP_K,
    QUESTION,
    ContextSource,
    build_context,
    build_user_message,
    expand_results_with_neighbours,
    generate_grounded_answer,
)
from lessons.lesson_25_citation_evaluation import (
    MINIMUM_LEXICAL_COVERAGE,
    build_repair_message,
    calculate_lexical_support,
    evaluate_answer,
    generate_repaired_answer,
    print_evaluation,
    split_sentences,
)


@dataclass(frozen=True)
class CitationDecision:
    """One deterministic citation-selection decision."""

    sentence_number: int
    original_sentence: str
    selected_labels: list[str]
    lexical_coverage: float
    missing_support_terms: list[str]
    threshold_met: bool


def build_source_map(
    sources: list[ContextSource],
) -> dict[str, str]:
    """Map every citation label to its source content."""

    return {
        source.label: source.chunk.content
        for source in sources
    }


def find_minimal_supporting_labels(
    sentence: str,
    sources: list[ContextSource],
) -> tuple[list[str], float, list[str]]:
    """Find the smallest source combination supporting a sentence."""

    source_map = build_source_map(
        sources
    )

    available_labels = list(
        source_map
    )

    if not available_labels:
        raise ValueError(
            "At least one context source is required."
        )

    best_labels: list[str] = []
    best_coverage = -1.0
    best_missing_terms: list[str] = []

    for group_size in range(
        1,
        len(available_labels) + 1,
    ):
        passing_candidates: list[
            tuple[float, list[str], list[str]]
        ] = []

        for candidate_tuple in combinations(
            available_labels,
            group_size,
        ):
            candidate_labels = list(
                candidate_tuple
            )

            (
                coverage,
                missing_terms,
            ) = calculate_lexical_support(
                sentence=sentence,
                cited_labels=candidate_labels,
                source_map=source_map,
            )

            if coverage > best_coverage:
                best_labels = candidate_labels
                best_coverage = coverage
                best_missing_terms = missing_terms

            if coverage >= MINIMUM_LEXICAL_COVERAGE:
                passing_candidates.append(
                    (
                        coverage,
                        candidate_labels,
                        missing_terms,
                    )
                )

        if passing_candidates:
            (
                selected_coverage,
                selected_labels,
                selected_missing_terms,
            ) = max(
                passing_candidates,
                key=lambda candidate: candidate[0],
            )

            return (
                selected_labels,
                selected_coverage,
                selected_missing_terms,
            )

    return (
        best_labels,
        best_coverage,
        best_missing_terms,
    )


def replace_sentence_citations(
    sentence: str,
    selected_labels: list[str],
) -> str:
    """Replace existing citations with selected source labels."""

    sentence_without_citations = re.sub(
        r"\s*\[S\d+\]",
        "",
        sentence,
    ).strip()

    punctuation = ""

    if (
        sentence_without_citations
        and sentence_without_citations[-1]
        in ".!?"
    ):
        punctuation = sentence_without_citations[-1]
        sentence_without_citations = (
            sentence_without_citations[:-1].rstrip()
        )

    citation_text = "".join(
        f"[{label}]"
        for label in selected_labels
    )

    if not citation_text:
        return (
            sentence_without_citations
            + punctuation
        )

    return (
        f"{sentence_without_citations} "
        f"{citation_text}{punctuation}"
    )


def deterministically_repair_citations(
    answer: str,
    sources: list[ContextSource],
) -> tuple[str, list[CitationDecision]]:
    """Select source combinations and rebuild citations."""

    sentences = split_sentences(
        answer
    )

    if not sentences:
        raise ValueError(
            "The answer contains no sentences to repair."
        )

    repaired_sentences: list[str] = []
    decisions: list[CitationDecision] = []

    for sentence_number, sentence in enumerate(
        sentences,
        start=1,
    ):
        (
            selected_labels,
            lexical_coverage,
            missing_support_terms,
        ) = find_minimal_supporting_labels(
            sentence=sentence,
            sources=sources,
        )

        threshold_met = (
            lexical_coverage
            >= MINIMUM_LEXICAL_COVERAGE
        )

        repaired_sentence = replace_sentence_citations(
            sentence=sentence,
            selected_labels=selected_labels,
        )

        repaired_sentences.append(
            repaired_sentence
        )

        decisions.append(
            CitationDecision(
                sentence_number=sentence_number,
                original_sentence=sentence,
                selected_labels=selected_labels,
                lexical_coverage=lexical_coverage,
                missing_support_terms=(
                    missing_support_terms
                ),
                threshold_met=threshold_met,
            )
        )

    repaired_answer = " ".join(
        repaired_sentences
    )

    return repaired_answer, decisions


def print_decisions(
    decisions: list[CitationDecision],
) -> None:
    """Print deterministic citation-search decisions."""

    print("DETERMINISTIC CITATION DECISIONS")
    print("--------------------------------")

    for decision in decisions:
        print(
            f"Sentence {decision.sentence_number}:"
        )
        print(
            "  Original sentence:",
            decision.original_sentence,
        )
        print(
            "  Selected labels:",
            decision.selected_labels,
        )
        print(
            "  Lexical coverage:",
            f"{decision.lexical_coverage:.2%}",
        )
        print(
            "  Missing support terms:",
            decision.missing_support_terms,
        )
        print(
            "  Threshold met:",
            decision.threshold_met,
        )
        print()


def main() -> None:
    """Run generation, LLM repair, and deterministic repair."""

    vector_index = load_vector_index(
        INDEX_PATH
    )

    print("DETERMINISTIC CITATION REPAIR")
    print("=============================")
    print()
    print("Question:")
    print(QUESTION)
    print()
    print(
        "Minimum lexical coverage:",
        f"{MINIMUM_LEXICAL_COVERAGE:.0%}",
    )
    print()

    print("1. Retrieving context...")

    (
        query_embedding,
        _query_token_count,
    ) = generate_query_embedding(
        query=QUESTION,
        embedding_model=(
            vector_index.embedding_model
        ),
        expected_dimension=(
            vector_index.embedding_dimension
        ),
    )

    retrieval_results = search_index(
        query_embedding=query_embedding,
        chunks=vector_index.chunks,
        top_k=INITIAL_TOP_K,
    )

    context_sources = expand_results_with_neighbours(
        results=retrieval_results,
        all_chunks=vector_index.chunks,
    )

    context = build_context(
        context_sources
    )

    print("   Completed.")
    print(
        "   Context sources:",
        len(context_sources),
    )
    print()

    print("2. Generating original answer...")

    original_user_message = build_user_message(
        question=QUESTION,
        context=context,
    )

    (
        original_answer,
        _original_prompt_tokens,
        _original_output_tokens,
    ) = generate_grounded_answer(
        original_user_message
    )

    print("   Completed.")
    print()

    (
        original_checks,
        original_passed,
    ) = evaluate_answer(
        answer=original_answer,
        sources=context_sources,
    )

    print_evaluation(
        title="ORIGINAL ANSWER EVALUATION",
        answer=original_answer,
        checks=original_checks,
        answer_passed=original_passed,
    )

    if original_passed:
        print("STATUS")
        print("------")
        print(
            "The original answer passed all citation checks. "
            "No repair was required."
        )
        return

    print("3. Requesting LLM citation repair...")

    repair_message = build_repair_message(
        question=QUESTION,
        context=context,
        original_answer=original_answer,
        checks=original_checks,
    )

    (
        llm_repaired_answer,
        _repair_prompt_tokens,
        _repair_output_tokens,
    ) = generate_repaired_answer(
        repair_message
    )

    print("   Completed.")
    print()

    (
        llm_repaired_checks,
        llm_repaired_passed,
    ) = evaluate_answer(
        answer=llm_repaired_answer,
        sources=context_sources,
    )

    print_evaluation(
        title="LLM-REPAIRED ANSWER EVALUATION",
        answer=llm_repaired_answer,
        checks=llm_repaired_checks,
        answer_passed=llm_repaired_passed,
    )

    if llm_repaired_passed:
        print("STATUS")
        print("------")
        print(
            "The LLM-repaired answer passed all "
            "citation checks."
        )
        return

    print(
        "4. Searching deterministic source combinations..."
    )

    (
        deterministic_answer,
        citation_decisions,
    ) = deterministically_repair_citations(
        answer=llm_repaired_answer,
        sources=context_sources,
    )

    print("   Completed.")
    print()

    print_decisions(
        citation_decisions
    )

    (
        deterministic_checks,
        deterministic_passed,
    ) = evaluate_answer(
        answer=deterministic_answer,
        sources=context_sources,
    )

    print_evaluation(
        title="DETERMINISTICALLY REPAIRED ANSWER",
        answer=deterministic_answer,
        checks=deterministic_checks,
        answer_passed=deterministic_passed,
    )

    print("STATUS")
    print("------")

    if deterministic_passed:
        print(
            "The deterministic citation search found "
            "supporting source combinations, and the final "
            "answer passed evaluation."
        )
    else:
        print(
            "At least one sentence could not reach the "
            "required lexical-support threshold."
        )


if __name__ == "__main__":
    main()