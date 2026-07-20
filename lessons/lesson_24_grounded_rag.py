from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from lessons.lesson_23_search_vector_index import (
    INDEX_PATH,
    IndexedChunk,
    SearchResult,
    generate_query_embedding,
    load_vector_index,
    search_index,
)


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
GENERATION_MODEL = "llama3.2:3b"

QUESTION = (
    "What should a driver do when an engine warning "
    "light appears?"
)

INITIAL_TOP_K = 3

SYSTEM_MESSAGE = (
    "You are a precise retrieval-augmented question-answering "
    "assistant. Answer using only the supplied source passages. "
    "Do not use outside knowledge. Cite supporting passages "
    "using labels such as [S1] and [S2]. Every factual statement "
    "must have a citation. If the supplied passages do not contain "
    "enough information, say exactly: "
    "'The supplied sources do not contain enough information.'"
)


@dataclass(frozen=True)
class ContextSource:
    """One labelled source included in the LLM context."""

    label: str
    chunk: IndexedChunk
    retrieval_score: float | None
    inclusion_reason: str


def find_next_chunk(
    source_chunk: IndexedChunk,
    all_chunks: list[IndexedChunk],
) -> IndexedChunk | None:
    """Find the next chunk from the same document section."""

    expected_chunk_number = (
        source_chunk.chunk_number + 1
    )

    for candidate in all_chunks:
        if (
            candidate.section_id
            == source_chunk.section_id
            and candidate.chunk_number
            == expected_chunk_number
        ):
            return candidate

    return None


def expand_results_with_neighbours(
    results: list[SearchResult],
    all_chunks: list[IndexedChunk],
) -> list[ContextSource]:
    """Add the next chunk after each retrieved result."""

    selected_sources: list[ContextSource] = []
    included_chunk_ids: set[str] = set()

    for result in results:
        retrieved_chunk = result.chunk

        if retrieved_chunk.chunk_id not in included_chunk_ids:
            selected_sources.append(
                ContextSource(
                    label="",
                    chunk=retrieved_chunk,
                    retrieval_score=result.score,
                    inclusion_reason="vector retrieval",
                )
            )
            included_chunk_ids.add(
                retrieved_chunk.chunk_id
            )

        next_chunk = find_next_chunk(
            source_chunk=retrieved_chunk,
            all_chunks=all_chunks,
        )

        if (
            next_chunk is not None
            and next_chunk.chunk_id
            not in included_chunk_ids
        ):
            selected_sources.append(
                ContextSource(
                    label="",
                    chunk=next_chunk,
                    retrieval_score=None,
                    inclusion_reason=(
                        "adjacent-chunk expansion"
                    ),
                )
            )
            included_chunk_ids.add(
                next_chunk.chunk_id
            )

    labelled_sources: list[ContextSource] = []

    for source_number, source in enumerate(
        selected_sources,
        start=1,
    ):
        labelled_sources.append(
            ContextSource(
                label=f"S{source_number}",
                chunk=source.chunk,
                retrieval_score=source.retrieval_score,
                inclusion_reason=source.inclusion_reason,
            )
        )

    return labelled_sources


def build_context(
    sources: list[ContextSource],
) -> str:
    """Convert labelled chunks into an LLM context block."""

    context_parts: list[str] = []

    for source in sources:
        chunk = source.chunk

        source_block = (
            f"[{source.label}]\n"
            f"Document: {chunk.document_title}\n"
            f"Section: {chunk.section_title}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Source file: {chunk.source}\n"
            f"Word range: "
            f"{chunk.start_word}:"
            f"{chunk.end_word_exclusive}\n"
            f"Content:\n{chunk.content}"
        )

        context_parts.append(source_block)

    return "\n\n".join(context_parts)


def build_user_message(
    question: str,
    context: str,
) -> str:
    """Build the complete grounded user message."""

    return (
        "SOURCE PASSAGES\n"
        "===============\n\n"
        f"{context}\n\n"
        "QUESTION\n"
        "========\n"
        f"{question}\n\n"
        "ANSWER REQUIREMENTS\n"
        "===================\n"
        "1. Use only the source passages.\n"
        "2. Give a concise direct answer.\n"
        "3. Put a citation such as [S1] after every "
        "factual sentence.\n"
        "4. Do not cite a source that does not support "
        "the statement.\n"
        "5. Do not include a separate bibliography."
    )


def generate_grounded_answer(
    user_message: str,
) -> tuple[str, int | None, int | None]:
    """Generate a grounded answer from the supplied context."""

    request_body = {
        "model": GENERATION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": user_message,
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
            "Ollama did not generate an answer "
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
        and (
            isinstance(prompt_token_count, bool)
            or not isinstance(prompt_token_count, int)
        )
    ):
        raise RuntimeError(
            "Ollama returned an invalid input-token count."
        )

    if (
        output_token_count is not None
        and (
            isinstance(output_token_count, bool)
            or not isinstance(output_token_count, int)
        )
    ):
        raise RuntimeError(
            "Ollama returned an invalid output-token count."
        )

    return (
        answer.strip(),
        prompt_token_count,
        output_token_count,
    )


def extract_citation_labels(
    answer: str,
) -> list[str]:
    """Extract citation labels such as S1 and S2."""

    return re.findall(
        r"\[(S\d+)\]",
        answer,
    )


def validate_citations(
    answer: str,
    sources: list[ContextSource],
) -> tuple[list[str], list[str], bool]:
    """Check whether the answer cites only supplied labels."""

    cited_labels = extract_citation_labels(
        answer
    )

    unique_cited_labels = list(
        dict.fromkeys(cited_labels)
    )

    allowed_labels = {
        source.label
        for source in sources
    }

    invalid_labels = [
        label
        for label in unique_cited_labels
        if label not in allowed_labels
    ]

    fallback_answer = (
        "The supplied sources do not contain "
        "enough information."
    )

    citation_present_when_required = (
        answer == fallback_answer
        or bool(unique_cited_labels)
    )

    citations_valid = (
        not invalid_labels
        and citation_present_when_required
    )

    return (
        unique_cited_labels,
        invalid_labels,
        citations_valid,
    )


def main() -> None:
    """Run retrieval, context construction, and generation."""

    vector_index = load_vector_index(
        INDEX_PATH
    )

    print("GROUNDED RAG PIPELINE")
    print("=====================")
    print()
    print("Question:")
    print(QUESTION)
    print()
    print(
        "Embedding model:",
        vector_index.embedding_model,
    )
    print(
        "Generation model:",
        GENERATION_MODEL,
    )
    print(
        "Indexed chunks:",
        len(vector_index.chunks),
    )
    print(
        "Initial top-k:",
        INITIAL_TOP_K,
    )
    print()

    print("1. Generating query embedding...")

    (
        query_embedding,
        query_token_count,
    ) = generate_query_embedding(
        query=QUESTION,
        embedding_model=(
            vector_index.embedding_model
        ),
        expected_dimension=(
            vector_index.embedding_dimension
        ),
    )

    print("   Completed.")

    if query_token_count is not None:
        print(
            "   Query input tokens:",
            query_token_count,
        )

    print()
    print("2. Retrieving relevant chunks...")

    retrieval_results = search_index(
        query_embedding=query_embedding,
        chunks=vector_index.chunks,
        top_k=INITIAL_TOP_K,
    )

    print("   Completed.")
    print()

    print("INITIAL RETRIEVAL RESULTS")
    print("-------------------------")

    for rank, result in enumerate(
        retrieval_results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result.chunk.chunk_id} | "
            f"{result.chunk.section_title} | "
            f"{result.score:.4f}"
        )

    print()
    print("3. Expanding context with adjacent chunks...")

    context_sources = expand_results_with_neighbours(
        results=retrieval_results,
        all_chunks=vector_index.chunks,
    )

    print("   Completed.")
    print(
        "   Context chunks:",
        len(context_sources),
    )
    print()

    print("CONTEXT SOURCES")
    print("---------------")

    for source in context_sources:
        print(
            f"[{source.label}] "
            f"{source.chunk.chunk_id}"
        )
        print(
            "  Section:",
            source.chunk.section_title,
        )
        print(
            "  Inclusion reason:",
            source.inclusion_reason,
        )

        if source.retrieval_score is not None:
            print(
                "  Similarity:",
                f"{source.retrieval_score:.4f}",
            )

        print(
            "  Word range:",
            f"{source.chunk.start_word}:"
            f"{source.chunk.end_word_exclusive}",
        )
        print()

    context = build_context(
        context_sources
    )

    user_message = build_user_message(
        question=QUESTION,
        context=context,
    )

    print("4. Generating grounded answer...")

    (
        answer,
        prompt_token_count,
        output_token_count,
    ) = generate_grounded_answer(
        user_message
    )

    print("   Completed.")
    print()

    print("GROUNDED ANSWER")
    print("---------------")
    print(answer)
    print()

    (
        cited_labels,
        invalid_labels,
        citations_valid,
    ) = validate_citations(
        answer=answer,
        sources=context_sources,
    )

    print("CITATION VALIDATION")
    print("-------------------")
    print(
        "Cited labels:",
        cited_labels,
    )
    print(
        "Invalid labels:",
        invalid_labels,
    )
    print(
        "Citation structure valid:",
        citations_valid,
    )

    if prompt_token_count is not None:
        print(
            "Generation input tokens:",
            prompt_token_count,
        )

    if output_token_count is not None:
        print(
            "Generation output tokens:",
            output_token_count,
        )

    print()
    print("SOURCE DIRECTORY")
    print("----------------")

    for source in context_sources:
        print(f"[{source.label}]")
        print(
            "  Document:",
            source.chunk.document_title,
        )
        print(
            "  Section:",
            source.chunk.section_title,
        )
        print(
            "  Chunk:",
            source.chunk.chunk_id,
        )
        print(
            "  Source:",
            source.chunk.source,
        )
        print(
            "  Word range:",
            f"{source.chunk.start_word}:"
            f"{source.chunk.end_word_exclusive}",
        )

    print()
    print("STATUS")
    print("------")
    print(
        "Retrieval, context construction, grounded "
        "generation, and citation validation completed."
    )


if __name__ == "__main__":
    main()