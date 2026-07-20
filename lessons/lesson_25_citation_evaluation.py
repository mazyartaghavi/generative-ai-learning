from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

from lessons.lesson_23_search_vector_index import (
    INDEX_PATH,
    generate_query_embedding,
    load_vector_index,
    search_index,
)
from lessons.lesson_24_grounded_rag import (
    GENERATION_MODEL,
    INITIAL_TOP_K,
    OLLAMA_CHAT_URL,
    QUESTION,
    ContextSource,
    build_context,
    build_user_message,
    expand_results_with_neighbours,
    generate_grounded_answer,
)


MINIMUM_LEXICAL_COVERAGE = 0.90

FALLBACK_ANSWER = (
    "The supplied sources do not contain enough information."
)

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "when",
    "that",
    "this",
    "these",
    "those",
}


@dataclass(frozen=True)
class SentenceCitationCheck:
    """Evaluation results for one generated sentence."""

    sentence_number: int
    sentence: str
    cited_labels: list[str]
    invalid_labels: list[str]
    missing_citation: bool
    lexical_coverage: float | None
    missing_support_terms: list[str]
    support_passed: bool


def split_sentences(answer: str) -> list[str]:
    """Split an answer into non-empty sentences."""

    normalized_answer = " ".join(
        answer.split()
    )

    if not normalized_answer:
        return []

    raw_sentences = re.split(
        r"(?<=[.!?])\s+",
        normalized_answer,
    )

    return [
        sentence.strip()
        for sentence in raw_sentences
        if sentence.strip()
    ]


def extract_citation_labels(
    sentence: str,
) -> list[str]:
    """Extract unique labels such as S1 and S2."""

    labels = re.findall(
        r"\[(S\d+)\]",
        sentence,
    )

    return list(
        dict.fromkeys(labels)
    )


def extract_claim_terms(
    sentence: str,
) -> list[str]:
    """Extract important lowercase terms from a sentence."""

    sentence_without_citations = re.sub(
        r"\[S\d+\]",
        "",
        sentence,
    )

    tokens = re.findall(
        r"[A-Za-z0-9]+",
        sentence_without_citations.casefold(),
    )

    return list(
        dict.fromkeys(
            token
            for token in tokens
            if (
                token not in STOPWORDS
                and len(token) > 1
            )
        )
    )


def build_source_map(
    sources: list[ContextSource],
) -> dict[str, str]:
    """Map citation labels to their passage contents."""

    return {
        source.label: source.chunk.content
        for source in sources
    }


def calculate_lexical_support(
    sentence: str,
    cited_labels: list[str],
    source_map: dict[str, str],
) -> tuple[float, list[str]]:
    """Measure claim-term coverage in cited passages."""

    claim_terms = extract_claim_terms(
        sentence
    )

    if not claim_terms:
        return 1.0, []

    combined_source_text = " ".join(
        source_map[label]
        for label in cited_labels
        if label in source_map
    )

    source_terms = set(
        re.findall(
            r"[A-Za-z0-9]+",
            combined_source_text.casefold(),
        )
    )

    missing_terms = [
        term
        for term in claim_terms
        if term not in source_terms
    ]

    matched_term_count = (
        len(claim_terms) - len(missing_terms)
    )

    coverage = (
        matched_term_count / len(claim_terms)
    )

    return coverage, missing_terms


def evaluate_answer(
    answer: str,
    sources: list[ContextSource],
) -> tuple[list[SentenceCitationCheck], bool]:
    """Evaluate citation coverage and lexical support."""

    if answer.strip() == FALLBACK_ANSWER:
        return [], True

    sentences = split_sentences(
        answer
    )

    if not sentences:
        return [], False

    source_map = build_source_map(
        sources
    )

    allowed_labels = set(
        source_map
    )

    checks: list[SentenceCitationCheck] = []

    for sentence_number, sentence in enumerate(
        sentences,
        start=1,
    ):
        cited_labels = extract_citation_labels(
            sentence
        )

        invalid_labels = [
            label
            for label in cited_labels
            if label not in allowed_labels
        ]

        missing_citation = not cited_labels

        if missing_citation or invalid_labels:
            lexical_coverage = None
            missing_support_terms: list[str] = []
            support_passed = False
        else:
            (
                lexical_coverage,
                missing_support_terms,
            ) = calculate_lexical_support(
                sentence=sentence,
                cited_labels=cited_labels,
                source_map=source_map,
            )

            support_passed = (
                lexical_coverage
                >= MINIMUM_LEXICAL_COVERAGE
            )

        checks.append(
            SentenceCitationCheck(
                sentence_number=sentence_number,
                sentence=sentence,
                cited_labels=cited_labels,
                invalid_labels=invalid_labels,
                missing_citation=missing_citation,
                lexical_coverage=lexical_coverage,
                missing_support_terms=(
                    missing_support_terms
                ),
                support_passed=support_passed,
            )
        )

    answer_passed = all(
        (
            not check.missing_citation
            and not check.invalid_labels
            and check.support_passed
        )
        for check in checks
    )

    return checks, answer_passed


def build_repair_message(
    question: str,
    context: str,
    original_answer: str,
    checks: list[SentenceCitationCheck],
) -> str:
    """Build a prompt requesting citation repair."""

    issue_descriptions: list[str] = []

    for check in checks:
        issues: list[str] = []

        if check.missing_citation:
            issues.append(
                "has no citation"
            )

        if check.invalid_labels:
            issues.append(
                "uses invalid labels "
                + ", ".join(check.invalid_labels)
            )

        if (
            check.lexical_coverage is not None
            and not check.support_passed
        ):
            missing_terms = ", ".join(
                check.missing_support_terms
            )

            issues.append(
                "is not fully supported by its cited "
                f"passages; missing terms: {missing_terms}"
            )

        if issues:
            issue_descriptions.append(
                f"Sentence {check.sentence_number}: "
                + "; ".join(issues)
            )

    issue_block = "\n".join(
        issue_descriptions
    )

    return (
        "SOURCE PASSAGES\n"
        "===============\n\n"
        f"{context}\n\n"
        "QUESTION\n"
        "========\n"
        f"{question}\n\n"
        "DRAFT ANSWER\n"
        "============\n"
        f"{original_answer}\n\n"
        "DETECTED CITATION PROBLEMS\n"
        "==========================\n"
        f"{issue_block}\n\n"
        "REPAIR REQUIREMENTS\n"
        "===================\n"
        "1. Rewrite the answer using only the source passages.\n"
        "2. Every factual sentence must end with one or more "
        "citations before its final punctuation.\n"
        "3. Cite every source needed to support the complete "
        "sentence.\n"
        "4. When a claim spans adjacent chunks, cite both labels.\n"
        "5. Do not cite irrelevant passages.\n"
        "6. Return only the repaired answer."
    )


def generate_repaired_answer(
    repair_message: str,
) -> tuple[str, int | None, int | None]:
    """Ask the LLM to repair citation problems."""

    system_message = (
        "You are a citation-repair system for a "
        "retrieval-augmented application. Use only the supplied "
        "source passages. Every factual sentence must end with "
        "all source labels required to support that complete "
        "sentence. Place citations before the final punctuation."
    )

    request_body = {
        "model": GENERATION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": repair_message,
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 4096,
            "num_predict": 180,
        },
    }

    try:
        response = httpx.post(
            OLLAMA_CHAT_URL,
            json=request_body,
            timeout=180.0,
        )
        response.raise_for_status()
    except httpx.ConnectError as error:
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Confirm that Ollama is running."
        ) from error
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Ollama did not repair the answer "
            "before the timeout."
        ) from error
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "Ollama returned HTTP status "
            f"{error.response.status_code}: "
            f"{error.response.text}"
        ) from error

    try:
        response_body = response.json()
    except ValueError as error:
        raise RuntimeError(
            "Ollama returned invalid JSON."
        ) from error

    message = response_body.get("message")

    if not isinstance(message, dict):
        raise RuntimeError(
            "Ollama did not return a message object."
        )

    answer = message.get("content")

    if not isinstance(answer, str):
        raise RuntimeError(
            "Ollama did not return an answer string."
        )

    prompt_token_count = response_body.get(
        "prompt_eval_count"
    )
    output_token_count = response_body.get(
        "eval_count"
    )

    if (
        prompt_token_count is not None
        and not isinstance(prompt_token_count, int)
    ):
        raise RuntimeError(
            "Invalid repair input-token count."
        )

    if (
        output_token_count is not None
        and not isinstance(output_token_count, int)
    ):
        raise RuntimeError(
            "Invalid repair output-token count."
        )

    return (
        answer.strip(),
        prompt_token_count,
        output_token_count,
    )


def print_evaluation(
    title: str,
    answer: str,
    checks: list[SentenceCitationCheck],
    answer_passed: bool,
) -> None:
    """Print detailed citation evaluation results."""

    print(title)
    print("-" * len(title))
    print(answer)
    print()

    if not checks and answer == FALLBACK_ANSWER:
        print(
            "Fallback answer detected: no citations required."
        )
        print("Overall result: PASS")
        print()
        return

    for check in checks:
        print(
            f"Sentence {check.sentence_number}:"
        )
        print(
            "  Text:",
            check.sentence,
        )
        print(
            "  Cited labels:",
            check.cited_labels,
        )
        print(
            "  Invalid labels:",
            check.invalid_labels,
        )
        print(
            "  Missing citation:",
            check.missing_citation,
        )

        if check.lexical_coverage is None:
            print(
                "  Lexical support coverage: not evaluated"
            )
        else:
            print(
                "  Lexical support coverage:",
                f"{check.lexical_coverage:.2%}",
            )

        print(
            "  Missing support terms:",
            check.missing_support_terms,
        )
        print(
            "  Support passed:",
            check.support_passed,
        )
        print()

    print(
        "Overall result:",
        "PASS" if answer_passed else "FAIL",
    )
    print()


def main() -> None:
    """Generate, evaluate, and repair a grounded answer."""

    vector_index = load_vector_index(
        INDEX_PATH
    )

    print("RAG CITATION EVALUATION")
    print("=======================")
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

    print("2. Generating the original answer...")

    original_user_message = build_user_message(
        question=QUESTION,
        context=context,
    )

    (
        original_answer,
        original_prompt_tokens,
        original_output_tokens,
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

    if original_prompt_tokens is not None:
        print(
            "Original generation input tokens:",
            original_prompt_tokens,
        )

    if original_output_tokens is not None:
        print(
            "Original generation output tokens:",
            original_output_tokens,
        )

    print()

    if original_passed:
        print("REPAIR")
        print("------")
        print(
            "The original answer passed, so no repair "
            "request was needed."
        )
        return

    print("3. Requesting citation repair...")

    repair_message = build_repair_message(
        question=QUESTION,
        context=context,
        original_answer=original_answer,
        checks=original_checks,
    )

    (
        repaired_answer,
        repair_prompt_tokens,
        repair_output_tokens,
    ) = generate_repaired_answer(
        repair_message
    )

    print("   Completed.")
    print()

    (
        repaired_checks,
        repaired_passed,
    ) = evaluate_answer(
        answer=repaired_answer,
        sources=context_sources,
    )

    print_evaluation(
        title="REPAIRED ANSWER EVALUATION",
        answer=repaired_answer,
        checks=repaired_checks,
        answer_passed=repaired_passed,
    )

    if repair_prompt_tokens is not None:
        print(
            "Repair input tokens:",
            repair_prompt_tokens,
        )

    if repair_output_tokens is not None:
        print(
            "Repair output tokens:",
            repair_output_tokens,
        )

    print()
    print("STATUS")
    print("------")

    if repaired_passed:
        print(
            "The original citation problems were detected "
            "and the repaired answer passed evaluation."
        )
    else:
        print(
            "The repaired answer still failed at least one "
            "citation check and requires further handling."
        )


if __name__ == "__main__":
    main()