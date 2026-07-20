from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


INPUT_PATH = Path(
    "data/fleet_chunks.json"
)

OUTPUT_PATH = Path(
    "data/fleet_vector_index.json"
)

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "embeddinggemma"


@dataclass(frozen=True)
class SourceChunk:
    """One validated document chunk loaded from JSON."""

    chunk_id: str
    section_id: str
    document_title: str
    section_title: str
    source: str
    content: str
    chunk_number: int
    start_word: int
    end_word_exclusive: int
    word_count: int
    character_count: int


@dataclass(frozen=True)
class IndexedChunk:
    """One document chunk together with its embedding."""

    chunk_id: str
    section_id: str
    document_title: str
    section_title: str
    source: str
    content: str
    chunk_number: int
    start_word: int
    end_word_exclusive: int
    word_count: int
    character_count: int
    embedding: list[float]


def require_string(
    record: dict[str, Any],
    field_name: str,
    record_number: int,
) -> str:
    """Read and validate a required non-empty string."""

    value = record.get(field_name)

    if not isinstance(value, str):
        raise ValueError(
            f"Chunk record {record_number} has an invalid "
            f"{field_name!r} field."
        )

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(
            f"Chunk record {record_number} has an empty "
            f"{field_name!r} field."
        )

    return cleaned_value


def require_integer(
    record: dict[str, Any],
    field_name: str,
    record_number: int,
) -> int:
    """Read and validate a required integer."""

    value = record.get(field_name)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Chunk record {record_number} has an invalid "
            f"{field_name!r} field."
        )

    return value


def load_chunks(
    input_path: Path,
) -> list[SourceChunk]:
    """Load and validate document chunks from JSON."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Chunk file was not found: {input_path}"
        )

    if not input_path.is_file():
        raise ValueError(
            f"Chunk path is not a file: {input_path}"
        )

    try:
        payload = json.loads(
            input_path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Chunk file contains invalid JSON: {input_path}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            "The top-level chunk JSON value must be an object."
        )

    raw_chunks = payload.get("chunks")

    if not isinstance(raw_chunks, list):
        raise ValueError(
            "The chunk JSON must contain a chunks list."
        )

    chunks: list[SourceChunk] = []
    seen_chunk_ids: set[str] = set()

    for record_number, raw_chunk in enumerate(
        raw_chunks,
        start=1,
    ):
        if not isinstance(raw_chunk, dict):
            raise ValueError(
                f"Chunk record {record_number} "
                "must be a JSON object."
            )

        chunk = SourceChunk(
            chunk_id=require_string(
                raw_chunk,
                "chunk_id",
                record_number,
            ),
            section_id=require_string(
                raw_chunk,
                "section_id",
                record_number,
            ),
            document_title=require_string(
                raw_chunk,
                "document_title",
                record_number,
            ),
            section_title=require_string(
                raw_chunk,
                "section_title",
                record_number,
            ),
            source=require_string(
                raw_chunk,
                "source",
                record_number,
            ),
            content=require_string(
                raw_chunk,
                "content",
                record_number,
            ),
            chunk_number=require_integer(
                raw_chunk,
                "chunk_number",
                record_number,
            ),
            start_word=require_integer(
                raw_chunk,
                "start_word",
                record_number,
            ),
            end_word_exclusive=require_integer(
                raw_chunk,
                "end_word_exclusive",
                record_number,
            ),
            word_count=require_integer(
                raw_chunk,
                "word_count",
                record_number,
            ),
            character_count=require_integer(
                raw_chunk,
                "character_count",
                record_number,
            ),
        )

        if chunk.chunk_id in seen_chunk_ids:
            raise ValueError(
                f"Duplicate chunk ID found: {chunk.chunk_id}"
            )

        if chunk.chunk_number <= 0:
            raise ValueError(
                f"{chunk.chunk_id} has an invalid chunk number."
            )

        if chunk.start_word < 0:
            raise ValueError(
                f"{chunk.chunk_id} has a negative start index."
            )

        if (
            chunk.end_word_exclusive
            <= chunk.start_word
        ):
            raise ValueError(
                f"{chunk.chunk_id} has an invalid word range."
            )

        if chunk.word_count != len(chunk.content.split()):
            raise ValueError(
                f"{chunk.chunk_id} has an incorrect word count."
            )

        if chunk.character_count != len(chunk.content):
            raise ValueError(
                f"{chunk.chunk_id} has an incorrect "
                "character count."
            )

        seen_chunk_ids.add(chunk.chunk_id)
        chunks.append(chunk)

    if not chunks:
        raise ValueError(
            "The input file contains no chunks."
        )

    return chunks


def generate_embeddings(
    texts: list[str],
) -> tuple[str, list[list[float]], int | None]:
    """Generate and validate a batch of Ollama embeddings."""

    if not texts:
        raise ValueError(
            "At least one text is required for embedding."
        )

    request_body = {
        "model": EMBEDDING_MODEL,
        "input": texts,
        "truncate": False,
    }

    try:
        response = httpx.post(
            OLLAMA_EMBED_URL,
            json=request_body,
            timeout=180.0,
        )
        response.raise_for_status()
    except httpx.ConnectError as error:
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Confirm that the Ollama application is running."
        ) from error
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Ollama did not generate embeddings "
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
            "Ollama returned a response that was not valid JSON."
        ) from error

    returned_model = response_body.get("model")
    raw_embeddings = response_body.get("embeddings")
    prompt_token_count = response_body.get(
        "prompt_eval_count"
    )

    if not isinstance(returned_model, str):
        raise RuntimeError(
            "Ollama did not return a valid model name."
        )

    if not isinstance(raw_embeddings, list):
        raise RuntimeError(
            "Ollama did not return an embeddings list."
        )

    if len(raw_embeddings) != len(texts):
        raise RuntimeError(
            "The number of returned embeddings does not "
            "match the number of input texts."
        )

    if (
        prompt_token_count is not None
        and (
            isinstance(prompt_token_count, bool)
            or not isinstance(prompt_token_count, int)
        )
    ):
        raise RuntimeError(
            "Ollama returned an invalid prompt token count."
        )

    embeddings: list[list[float]] = []

    for embedding_number, raw_embedding in enumerate(
        raw_embeddings,
        start=1,
    ):
        if not isinstance(raw_embedding, list):
            raise RuntimeError(
                f"Embedding {embedding_number} is not a list."
            )

        if not raw_embedding:
            raise RuntimeError(
                f"Embedding {embedding_number} is empty."
            )

        embedding: list[float] = []

        for value in raw_embedding:
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
            ):
                raise RuntimeError(
                    f"Embedding {embedding_number} "
                    "contains a non-numeric value."
                )

            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise RuntimeError(
                    f"Embedding {embedding_number} "
                    "contains a non-finite value."
                )

            embedding.append(numeric_value)

        embeddings.append(embedding)

    dimensions = {
        len(embedding)
        for embedding in embeddings
    }

    if len(dimensions) != 1:
        raise RuntimeError(
            "Returned embeddings have inconsistent dimensions."
        )

    return (
        returned_model,
        embeddings,
        prompt_token_count,
    )


def vector_norm(vector: list[float]) -> float:
    """Calculate a vector's Euclidean, or L2, norm."""

    return math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )


def build_indexed_chunks(
    chunks: list[SourceChunk],
    embeddings: list[list[float]],
) -> list[IndexedChunk]:
    """Attach each embedding to its source chunk."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Every chunk must have exactly one embedding."
        )

    indexed_chunks: list[IndexedChunk] = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
        strict=True,
    ):
        indexed_chunks.append(
            IndexedChunk(
                chunk_id=chunk.chunk_id,
                section_id=chunk.section_id,
                document_title=chunk.document_title,
                section_title=chunk.section_title,
                source=chunk.source,
                content=chunk.content,
                chunk_number=chunk.chunk_number,
                start_word=chunk.start_word,
                end_word_exclusive=(
                    chunk.end_word_exclusive
                ),
                word_count=chunk.word_count,
                character_count=chunk.character_count,
                embedding=embedding,
            )
        )

    return indexed_chunks


def save_vector_index(
    indexed_chunks: list[IndexedChunk],
    returned_model: str,
    embedding_dimension: int,
    prompt_token_count: int | None,
    output_path: Path,
) -> None:
    """Save the complete persistent vector index."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "index_format": "local_json_vector_index_v1",
        "requested_embedding_model": EMBEDDING_MODEL,
        "returned_embedding_model": returned_model,
        "embedding_dimension": embedding_dimension,
        "distance_metric": "cosine_similarity",
        "chunk_count": len(indexed_chunks),
        "embedding_prompt_token_count": prompt_token_count,
        "chunks": [
            asdict(chunk)
            for chunk in indexed_chunks
        ],
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def create_preview(
    content: str,
    maximum_length: int = 85,
) -> str:
    """Create a short one-line content preview."""

    one_line_content = " ".join(
        content.split()
    )

    if len(one_line_content) <= maximum_length:
        return one_line_content

    return (
        one_line_content[
            : maximum_length - 3
        ]
        + "..."
    )


def main() -> None:
    """Generate embeddings and save a local vector index."""

    chunks = load_chunks(
        INPUT_PATH
    )

    chunk_texts = [
        chunk.content
        for chunk in chunks
    ]

    print("BUILDING LOCAL VECTOR INDEX")
    print("===========================")
    print()
    print("Input:", INPUT_PATH)
    print("Output:", OUTPUT_PATH)
    print("Embedding model:", EMBEDDING_MODEL)
    print("Chunks to embed:", len(chunks))
    print()
    print("Sending batch embedding request to Ollama...")

    (
        returned_model,
        embeddings,
        prompt_token_count,
    ) = generate_embeddings(
        chunk_texts
    )

    print("Embedding request completed.")
    print()

    embedding_dimension = len(
        embeddings[0]
    )

    indexed_chunks = build_indexed_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    vector_norms = [
        vector_norm(embedding)
        for embedding in embeddings
    ]

    save_vector_index(
        indexed_chunks=indexed_chunks,
        returned_model=returned_model,
        embedding_dimension=embedding_dimension,
        prompt_token_count=prompt_token_count,
        output_path=OUTPUT_PATH,
    )

    print("VECTOR INDEX REPORT")
    print("-------------------")
    print("Requested model:", EMBEDDING_MODEL)
    print("Returned model:", returned_model)
    print(
        "Chunks indexed:",
        len(indexed_chunks),
    )
    print(
        "Embedding dimension:",
        embedding_dimension,
    )

    if prompt_token_count is not None:
        print(
            "Input tokens embedded:",
            prompt_token_count,
        )

    print(
        "Minimum vector norm:",
        f"{min(vector_norms):.6f}",
    )
    print(
        "Maximum vector norm:",
        f"{max(vector_norms):.6f}",
    )
    print(
        "Index file size:",
        f"{OUTPUT_PATH.stat().st_size:,} bytes",
    )
    print()

    print("INDEXED CHUNKS")
    print("--------------")

    for indexed_chunk in indexed_chunks:
        print(
            f"Chunk: {indexed_chunk.chunk_id}"
        )
        print(
            f"Section: {indexed_chunk.section_title}"
        )
        print(
            "Vector values:",
            len(indexed_chunk.embedding),
        )
        print(
            "Vector norm:",
            f"{vector_norm(indexed_chunk.embedding):.6f}",
        )
        print(
            "Preview:",
            create_preview(indexed_chunk.content),
        )
        print()

    print("STATUS")
    print("------")
    print(
        "All chunks were embedded and saved in a "
        "persistent local vector index."
    )


if __name__ == "__main__":
    main()