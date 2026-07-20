from __future__ import annotations

import collections.abc

import re
from dataclasses import dataclass

from lessons.lesson_23_search_vector_index import (
    INDEX_PATH,
    IndexedChunk,
    LoadedVectorIndex,
    SearchResult,
    generate_query_embedding,
    load_vector_index,
    search_index,
)
from lessons.lesson_24_grounded_rag import (
    ContextSource,
    build_context,
    build_user_message,
    expand_results_with_neighbours,
    generate_grounded_answer,
)
from lessons.lesson_25_citation_evaluation import (
    FALLBACK_ANSWER,
    MINIMUM_LEXICAL_COVERAGE,
    SentenceCitationCheck,
    evaluate_answer,
    extract_claim_terms,
    split_sentences,
)
from lessons.lesson_25b_deterministic_citation_repair import (
    CitationDecision,
    deterministically_repair_citations,
    find_minimal_supporting_labels,
    replace_sentence_citations,
)


ABSTENTION_THRESHOLD = 0.5413
RETRIEVAL_TOP_K = 3


DEMO_QUESTIONS = [
    (
        "The dashboard has illuminated an engine icon. "
        "What is the driver expected to do?"
    ),
    (
        "A tire keeps losing air repeatedly. "
        "Can the vehicle remain in service?"
    ),
    (
        "What password should a driver use to sign in "
        "to the fleet management system?"
    ),
    "What is the fleet insurance policy number?",
]


TERM_ALIASES = {
    "accidents": "accident",
    "completed": "complete",
    "completing": "complete",
    "contacted": "contact",
    "contacting": "contact",
    "crash": "accident",
    "crashes": "accident",
    "details": "information",
    "finished": "complete",
    "finishes": "complete",
    "icon": "warning",
    "illuminated": "warning",
    "logged": "record",
    "logging": "record",
    "loses": "loss",
    "losing": "loss",
    "repeated": "repeat",
    "repeatedly": "repeat",
    "recorded": "record",
    "recording": "record",
    "trips": "route",
    "trip": "route",
}


@dataclass(frozen=True)
class ExtractiveCandidate:
    """One source sentence considered for fallback answering."""

    sentence_number: int
    sentence: str
    selected_labels: list[str]
    lexical_coverage: float
    missing_support_terms: list[str]
    question_overlap_count: int
    question_overlap_ratio: float


@dataclass(frozen=True)
class GuardedRAGResult:
    """Final outcome of one guarded RAG request."""

    question: str
    decision: str
    answer_strategy: str
    top_similarity: float
    threshold: float
    retrieval_results: list[SearchResult]
    context_sources: list[ContextSource]
    generated_answer: str | None
    answer: str
    generation_called: bool
    citation_repair_used: bool
    extractive_fallback_used: bool
    citation_checks: list[SentenceCitationCheck]
    citation_decisions: list[CitationDecision]
    extractive_candidate: ExtractiveCandidate | None
    answer_passed_validation: bool
    generation_input_tokens: int | None
    generation_output_tokens: int | None


def normalize_citation_format(
    answer: str,
) -> str:
    """Normalize spacing and move trailing citations before punctuation."""

    normalized_answer = " ".join(
        answer.split()
    )

    normalized_answer = re.sub(
        r"([.!?])\s*((?:\[S\d+\])+)",
        r" \2\1",
        normalized_answer,
    )

    normalized_answer = re.sub(
        r"(?<=\w)(\[S\d+\])",
        r" \1",
        normalized_answer,
    )

    return normalized_answer.strip()


def normalize_term(
    term: str,
) -> str:
    """Normalize selected vocabulary variants."""

    return TERM_ALIASES.get(
        term,
        term,
    )


def normalized_content_terms(
    text: str,
) -> set[str]:
    """Extract and normalize important terms from text."""

    return {
        normalize_term(term)
        for term in extract_claim_terms(text)
    }


def retrieve_chunks(
    question: str,
    vector_index: LoadedVectorIndex,
) -> tuple[list[SearchResult], int | None]:
    """Embed a question and retrieve its highest-ranked chunks."""

    (
        query_embedding,
        query_token_count,
    ) = generate_query_embedding(
        query=question,
        embedding_model=vector_index.embedding_model,
        expected_dimension=(
            vector_index.embedding_dimension
        ),
    )

    retrieval_results = search_index(
        query_embedding=query_embedding,
        chunks=vector_index.chunks,
        top_k=RETRIEVAL_TOP_K,
    )

    if not retrieval_results:
        raise RuntimeError(
            "Vector search returned no retrieval results."
        )

    return (
        retrieval_results,
        query_token_count,
    )


def complete_top_section_context(
    initial_sources: list[ContextSource],
    top_section_id: str,
    all_chunks: list[IndexedChunk],
) -> list[ContextSource]:
    """Ensure every chunk from the top-ranked section is available."""

    selected_sources = list(
        initial_sources
    )

    included_chunk_ids = {
        source.chunk.chunk_id
        for source in selected_sources
    }

    top_section_chunks = sorted(
        (
            chunk
            for chunk in all_chunks
            if chunk.section_id == top_section_id
        ),
        key=lambda chunk: (
            chunk.start_word,
            chunk.chunk_number,
        ),
    )

    for chunk in top_section_chunks:
        if chunk.chunk_id in included_chunk_ids:
            continue

        selected_sources.append(
            ContextSource(
                label="",
                chunk=chunk,
                retrieval_score=None,
                inclusion_reason=(
                    "complete top-ranked section"
                ),
            )
        )

        included_chunk_ids.add(
            chunk.chunk_id
        )

    relabelled_sources: list[ContextSource] = []

    for source_number, source in enumerate(
        selected_sources,
        start=1,
    ):
        relabelled_sources.append(
            ContextSource(
                label=f"S{source_number}",
                chunk=source.chunk,
                retrieval_score=source.retrieval_score,
                inclusion_reason=source.inclusion_reason,
            )
        )

    return relabelled_sources


def reconstruct_section_text(
    section_id: str,
    all_chunks: list[IndexedChunk],
) -> str:
    """Reconstruct a section from its overlapping chunks."""

    section_chunks = sorted(
        (
            chunk
            for chunk in all_chunks
            if chunk.section_id == section_id
        ),
        key=lambda chunk: (
            chunk.start_word,
            chunk.chunk_number,
        ),
    )

    if not section_chunks:
        raise ValueError(
            f"No chunks were found for section {section_id}."
        )

    section_start = min(
        chunk.start_word
        for chunk in section_chunks
    )

    section_end = max(
        chunk.end_word_exclusive
        for chunk in section_chunks
    )

    word_slots: list[str | None] = [
        None
    ] * (
        section_end - section_start
    )

    for chunk in section_chunks:
        chunk_words = chunk.content.split()

        expected_word_count = (
            chunk.end_word_exclusive
            - chunk.start_word
        )

        if len(chunk_words) != expected_word_count:
            raise ValueError(
                f"{chunk.chunk_id} has an inconsistent "
                "word range."
            )

        for word_offset, word in enumerate(
            chunk_words
        ):
            absolute_word_index = (
                chunk.start_word + word_offset
            )

            slot_index = (
                absolute_word_index - section_start
            )

            existing_word = word_slots[
                slot_index
            ]

            if (
                existing_word is not None
                and existing_word != word
            ):
                raise ValueError(
                    "Overlapping chunks contain different "
                    f"words at index {absolute_word_index}."
                )

            word_slots[
                slot_index
            ] = word

    if any(
        word is None
        for word in word_slots
    ):
        raise ValueError(
            f"Section {section_id} contains a gap between "
            "its stored chunks."
        )

    complete_words = [
        word
        for word in word_slots
        if word is not None
    ]

    return " ".join(
        complete_words
    )


def build_extractive_candidates(
    question: str,
    top_section_id: str,
    context_sources: list[ContextSource],
    all_chunks: list[IndexedChunk],
) -> list[ExtractiveCandidate]:
    """Create source-supported extractive answer candidates."""

    section_text = reconstruct_section_text(
        section_id=top_section_id,
        all_chunks=all_chunks,
    )

    source_sentences = split_sentences(
        section_text
    )

    question_terms = normalized_content_terms(
        question
    )

    candidates: list[ExtractiveCandidate] = []

    for sentence_number, sentence in enumerate(
        source_sentences,
        start=1,
    ):
        (
            selected_labels,
            lexical_coverage,
            missing_support_terms,
        ) = find_minimal_supporting_labels(
            sentence=sentence,
            sources=context_sources,
        )

        sentence_terms = normalized_content_terms(
            sentence
        )

        overlapping_terms = (
            question_terms
            & sentence_terms
        )

        overlap_count = len(
            overlapping_terms
        )

        if question_terms:
            overlap_ratio = (
                overlap_count
                / len(question_terms)
            )
        else:
            overlap_ratio = 0.0

        candidates.append(
            ExtractiveCandidate(
                sentence_number=sentence_number,
                sentence=sentence,
                selected_labels=selected_labels,
                lexical_coverage=lexical_coverage,
                missing_support_terms=(
                    missing_support_terms
                ),
                question_overlap_count=overlap_count,
                question_overlap_ratio=overlap_ratio,
            )
        )

    return candidates


def select_extractive_candidate(
    question: str,
    top_section_id: str,
    context_sources: list[ContextSource],
    all_chunks: list[IndexedChunk],
) -> ExtractiveCandidate | None:
    """Select the most question-relevant supported sentence."""

    candidates = build_extractive_candidates(
        question=question,
        top_section_id=top_section_id,
        context_sources=context_sources,
        all_chunks=all_chunks,
    )

    passing_candidates = [
        candidate
        for candidate in candidates
        if (
            candidate.lexical_coverage
            >= MINIMUM_LEXICAL_COVERAGE
            and candidate.selected_labels
        )
    ]

    if not passing_candidates:
        return None

    return max(
        passing_candidates,
        key=lambda candidate: (
            candidate.question_overlap_count,
            candidate.question_overlap_ratio,
            -candidate.sentence_number,
        ),
    )


def build_extractive_answer(
    candidate: ExtractiveCandidate,
) -> str:
    """Attach validated citations to one source sentence."""

    extractive_answer = replace_sentence_citations(
        sentence=candidate.sentence,
        selected_labels=candidate.selected_labels,
    )

    return normalize_citation_format(
        extractive_answer
    )


RetrieverFunction = collections.abc.Callable[
    [str, LoadedVectorIndex],
    tuple[list[SearchResult], int | None],
]

GenerationFunction = collections.abc.Callable[
    [str],
    tuple[str, int | None, int | None],
]


def run_guarded_rag(
    question: str,
    vector_index: LoadedVectorIndex,
    *,
    retriever: RetrieverFunction | None = None,
    generator: GenerationFunction | None = None,
) -> GuardedRAGResult:
    """Run retrieval, abstention, generation, and validation."""

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "The question cannot be empty."
        )

    active_retriever = (
        retrieve_chunks
        if retriever is None
        else retriever
    )

    active_generator = (
        generate_grounded_answer
        if generator is None
        else generator
    )

    if not callable(active_retriever):
        raise TypeError(
            "The retriever must be callable."
        )

    if not callable(active_generator):
        raise TypeError(
            "The generator must be callable."
        )

    (
        retrieval_results,
        _query_token_count,
    ) = active_retriever(
        question=cleaned_question,
        vector_index=vector_index,
    )

    top_similarity = (
        retrieval_results[0].score
    )

    if top_similarity < ABSTENTION_THRESHOLD:
        return GuardedRAGResult(
            question=cleaned_question,
            decision="ABSTAIN",
            answer_strategy="retrieval abstention",
            top_similarity=top_similarity,
            threshold=ABSTENTION_THRESHOLD,
            retrieval_results=retrieval_results,
            context_sources=[],
            generated_answer=None,
            answer=FALLBACK_ANSWER,
            generation_called=False,
            citation_repair_used=False,
            extractive_fallback_used=False,
            citation_checks=[],
            citation_decisions=[],
            extractive_candidate=None,
            answer_passed_validation=True,
            generation_input_tokens=None,
            generation_output_tokens=None,
        )

    initial_context_sources = (
        expand_results_with_neighbours(
            results=retrieval_results,
            all_chunks=vector_index.chunks,
        )
    )

    top_section_id = (
        retrieval_results[0].chunk.section_id
    )

    context_sources = complete_top_section_context(
        initial_sources=initial_context_sources,
        top_section_id=top_section_id,
        all_chunks=vector_index.chunks,
    )

    context = build_context(
        context_sources
    )

    user_message = build_user_message(
        question=cleaned_question,
        context=context,
    )

    (
        raw_generated_answer,
        generation_input_tokens,
        generation_output_tokens,
    ) = active_generator(
        user_message
    )

    generated_answer = normalize_citation_format(
        raw_generated_answer
    )

    (
        generated_checks,
        generated_answer_passed,
    ) = evaluate_answer(
        answer=generated_answer,
        sources=context_sources,
    )

    if generated_answer_passed:
        return GuardedRAGResult(
            question=cleaned_question,
            decision="ANSWER",
            answer_strategy="grounded generation",
            top_similarity=top_similarity,
            threshold=ABSTENTION_THRESHOLD,
            retrieval_results=retrieval_results,
            context_sources=context_sources,
            generated_answer=generated_answer,
            answer=generated_answer,
            generation_called=True,
            citation_repair_used=False,
            extractive_fallback_used=False,
            citation_checks=generated_checks,
            citation_decisions=[],
            extractive_candidate=None,
            answer_passed_validation=True,
            generation_input_tokens=(
                generation_input_tokens
            ),
            generation_output_tokens=(
                generation_output_tokens
            ),
        )

    (
        citation_repaired_answer,
        citation_decisions,
    ) = deterministically_repair_citations(
        answer=generated_answer,
        sources=context_sources,
    )

    citation_repaired_answer = (
        normalize_citation_format(
            citation_repaired_answer
        )
    )

    (
        citation_repaired_checks,
        citation_repaired_passed,
    ) = evaluate_answer(
        answer=citation_repaired_answer,
        sources=context_sources,
    )

    if citation_repaired_passed:
        return GuardedRAGResult(
            question=cleaned_question,
            decision="ANSWER",
            answer_strategy=(
                "deterministic citation repair"
            ),
            top_similarity=top_similarity,
            threshold=ABSTENTION_THRESHOLD,
            retrieval_results=retrieval_results,
            context_sources=context_sources,
            generated_answer=generated_answer,
            answer=citation_repaired_answer,
            generation_called=True,
            citation_repair_used=True,
            extractive_fallback_used=False,
            citation_checks=citation_repaired_checks,
            citation_decisions=citation_decisions,
            extractive_candidate=None,
            answer_passed_validation=True,
            generation_input_tokens=(
                generation_input_tokens
            ),
            generation_output_tokens=(
                generation_output_tokens
            ),
        )

    extractive_candidate = (
        select_extractive_candidate(
            question=cleaned_question,
            top_section_id=top_section_id,
            context_sources=context_sources,
            all_chunks=vector_index.chunks,
        )
    )

    if extractive_candidate is not None:
        extractive_answer = (
            build_extractive_answer(
                extractive_candidate
            )
        )

        (
            extractive_checks,
            extractive_answer_passed,
        ) = evaluate_answer(
            answer=extractive_answer,
            sources=context_sources,
        )

        if extractive_answer_passed:
            return GuardedRAGResult(
                question=cleaned_question,
                decision="ANSWER",
                answer_strategy="extractive fallback",
                top_similarity=top_similarity,
                threshold=ABSTENTION_THRESHOLD,
                retrieval_results=retrieval_results,
                context_sources=context_sources,
                generated_answer=generated_answer,
                answer=extractive_answer,
                generation_called=True,
                citation_repair_used=True,
                extractive_fallback_used=True,
                citation_checks=extractive_checks,
                citation_decisions=citation_decisions,
                extractive_candidate=(
                    extractive_candidate
                ),
                answer_passed_validation=True,
                generation_input_tokens=(
                    generation_input_tokens
                ),
                generation_output_tokens=(
                    generation_output_tokens
                ),
            )

    return GuardedRAGResult(
        question=cleaned_question,
        decision="ABSTAIN_AFTER_VALIDATION",
        answer_strategy="validation abstention",
        top_similarity=top_similarity,
        threshold=ABSTENTION_THRESHOLD,
        retrieval_results=retrieval_results,
        context_sources=context_sources,
        generated_answer=generated_answer,
        answer=FALLBACK_ANSWER,
        generation_called=True,
        citation_repair_used=True,
        extractive_fallback_used=True,
        citation_checks=citation_repaired_checks,
        citation_decisions=citation_decisions,
        extractive_candidate=extractive_candidate,
        answer_passed_validation=False,
        generation_input_tokens=(
            generation_input_tokens
        ),
        generation_output_tokens=(
            generation_output_tokens
        ),
    )


def print_retrieval_results(
    retrieval_results: list[SearchResult],
) -> None:
    """Print ranked vector-search results."""

    print("RETRIEVAL RESULTS")
    print("-----------------")

    for rank, result in enumerate(
        retrieval_results,
        start=1,
    ):
        chunk = result.chunk

        print(
            f"{rank}. "
            f"{chunk.section_title} | "
            f"{result.score:.4f}"
        )
        print(
            "   Chunk:",
            chunk.chunk_id,
        )

    print()


def print_context_sources(
    sources: list[ContextSource],
) -> None:
    """Print context sources used for generation."""

    print("CONTEXT SOURCES")
    print("---------------")

    if not sources:
        print(
            "No context was sent to the generation model."
        )
        print()
        return

    for source in sources:
        print(
            f"[{source.label}] "
            f"{source.chunk.section_title} | "
            f"{source.chunk.chunk_id}"
        )
        print(
            "   Inclusion reason:",
            source.inclusion_reason,
        )

        if source.retrieval_score is not None:
            print(
                "   Retrieval score:",
                f"{source.retrieval_score:.4f}",
            )

    print()


def print_citation_checks(
    checks: list[SentenceCitationCheck],
) -> None:
    """Print concise sentence-level citation results."""

    print("CITATION VALIDATION")
    print("-------------------")

    if not checks:
        print(
            "Citation validation was not required."
        )
        print()
        return

    for check in checks:
        print(
            f"Sentence {check.sentence_number}:"
        )
        print(
            "  Labels:",
            check.cited_labels,
        )
        print(
            "  Missing citation:",
            check.missing_citation,
        )
        print(
            "  Invalid labels:",
            check.invalid_labels,
        )

        if check.lexical_coverage is None:
            print(
                "  Lexical coverage: not evaluated"
            )
        else:
            print(
                "  Lexical coverage:",
                f"{check.lexical_coverage:.2%}",
            )

        print(
            "  Support passed:",
            check.support_passed,
        )

    print()


def print_citation_decisions(
    decisions: list[CitationDecision],
) -> None:
    """Print deterministic citation-repair decisions."""

    print("DETERMINISTIC CITATION REPAIR")
    print("-----------------------------")

    if not decisions:
        print(
            "Deterministic citation repair was not used."
        )
        print()
        return

    for decision in decisions:
        print(
            f"Sentence {decision.sentence_number}:"
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
            "  Threshold met:",
            decision.threshold_met,
        )

    print()


def print_extractive_candidate(
    candidate: ExtractiveCandidate | None,
) -> None:
    """Print extractive-fallback details."""

    print("EXTRACTIVE FALLBACK")
    print("-------------------")

    if candidate is None:
        print(
            "Extractive fallback was not used."
        )
        print()
        return

    print(
        "Selected source sentence:",
        candidate.sentence_number,
    )
    print(
        "Selected labels:",
        candidate.selected_labels,
    )
    print(
        "Lexical support:",
        f"{candidate.lexical_coverage:.2%}",
    )
    print(
        "Question-term overlap:",
        candidate.question_overlap_count,
    )
    print(
        "Question overlap ratio:",
        f"{candidate.question_overlap_ratio:.2%}",
    )
    print()


def print_guarded_result(
    result_number: int,
    result: GuardedRAGResult,
) -> None:
    """Print one complete guarded-RAG outcome."""

    print(
        f"QUESTION {result_number}"
    )
    print(
        "=" * (
            9 + len(str(result_number))
        )
    )
    print(result.question)
    print()

    print(
        "Top similarity:",
        f"{result.top_similarity:.4f}",
    )
    print(
        "Abstention threshold:",
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
    print(
        "Generation model called:",
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
    print()

    print_retrieval_results(
        result.retrieval_results
    )

    print_context_sources(
        result.context_sources
    )

    if (
        result.generated_answer is not None
        and result.generated_answer != result.answer
    ):
        print("INITIAL GENERATED ANSWER")
        print("------------------------")
        print(result.generated_answer)
        print()

    print("FINAL ANSWER")
    print("------------")
    print(result.answer)
    print()

    print_citation_checks(
        result.citation_checks
    )

    print_citation_decisions(
        result.citation_decisions
    )

    print_extractive_candidate(
        result.extractive_candidate
    )

    print(
        "Answer passed validation:",
        result.answer_passed_validation,
    )

    if result.generation_input_tokens is not None:
        print(
            "Generation input tokens:",
            result.generation_input_tokens,
        )

    if result.generation_output_tokens is not None:
        print(
            "Generation output tokens:",
            result.generation_output_tokens,
        )

    print()
    print()


def main() -> None:
    """Demonstrate guarded RAG with extractive fallback."""

    vector_index = load_vector_index(
        INDEX_PATH
    )

    print(
        "GUARDED RAG WITH EXTRACTIVE FALLBACK"
    )
    print(
        "===================================="
    )
    print()
    print(
        "Vector index:",
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
        "Retrieval top-k:",
        RETRIEVAL_TOP_K,
    )
    print(
        "Abstention threshold:",
        f"{ABSTENTION_THRESHOLD:.4f}",
    )
    print(
        "Questions:",
        len(DEMO_QUESTIONS),
    )
    print()

    results: list[GuardedRAGResult] = []

    for question_number, question in enumerate(
        DEMO_QUESTIONS,
        start=1,
    ):
        print(
            f"Processing question "
            f"{question_number}/{len(DEMO_QUESTIONS)}..."
        )

        result = run_guarded_rag(
            question=question,
            vector_index=vector_index,
        )

        results.append(
            result
        )

    print()
    print(
        "All guarded RAG requests completed."
    )
    print()
    print()

    for result_number, result in enumerate(
        results,
        start=1,
    ):
        print_guarded_result(
            result_number=result_number,
            result=result,
        )

    answer_count = sum(
        result.decision == "ANSWER"
        for result in results
    )

    direct_abstention_count = sum(
        result.decision == "ABSTAIN"
        for result in results
    )

    validation_abstention_count = sum(
        result.decision
        == "ABSTAIN_AFTER_VALIDATION"
        for result in results
    )

    generation_call_count = sum(
        result.generation_called
        for result in results
    )

    repair_count = sum(
        result.citation_repair_used
        for result in results
    )

    extractive_fallback_count = sum(
        result.extractive_fallback_used
        and result.decision == "ANSWER"
        for result in results
    )

    print("SUMMARY")
    print("=======")
    print(
        "Questions processed:",
        len(results),
    )
    print(
        "Answers returned:",
        answer_count,
    )
    print(
        "Retrieval-based abstentions:",
        direct_abstention_count,
    )
    print(
        "Validation-based abstentions:",
        validation_abstention_count,
    )
    print(
        "Generation model calls:",
        generation_call_count,
    )
    print(
        "Citation repairs attempted:",
        repair_count,
    )
    print(
        "Extractive fallback answers:",
        extractive_fallback_count,
    )
    print()

    print("SAFETY INTERPRETATION")
    print("=====================")
    print(
        "Questions below the retrieval threshold were "
        "rejected before any context was sent to Llama."
    )
    print(
        "Generated answers were returned only after "
        "citation validation."
    )
    print(
        "When a supported paraphrase failed strict lexical "
        "validation, the system attempted a source-verbatim "
        "extractive fallback."
    )
    print(
        "The retrieval threshold and lexical rules remain "
        "specific to this educational dataset."
    )


if __name__ == "__main__":
    main()
