from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


INDEX_PATH = Path(
    "data/fleet_vector_index.json"
)

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

QUERY = (
    "What should a driver do when an engine warning "
    "light appears?"
)

TOP_K = 3


@dataclass(frozen=True)
class IndexedChunk:
    """One validated chunk from the saved vector index."""

    chunk_id: str
    section_id: str
    document_title: str
    section_title: str
    source: str
    content: str
    chunk_number: int
    start_word: int
    end_word_exclusive: int
    embedding: list[float]


@dataclass(frozen=True)
class LoadedVectorIndex:
    """Validated index configuration and indexed chunks."""

    embedding_model: str
    embedding_dimension: int
    distance_metric: str
    chunks: list[IndexedChunk]


@dataclass(frozen=True)
class SearchResult:
    """One chunk and its similarity to the query."""

    score: float
    chunk: IndexedChunk


def require_string(
    record: dict[str, Any],
    field_name: str,
    record_description: str,
) -> str:
    """Read and validate a required non-empty string."""

    value = record.get(field_name)

    if not isinstance(value, str):
        raise ValueError(
            f"{record_description} has an invalid "
            f"{field_name!r} field."
        )

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(
            f"{record_description} has an empty "
            f"{field_name!r} field."
        )

    return cleaned_value


def require_integer(
    record: dict[str, Any],
    field_name: str,
    record_description: str,
) -> int:
    """Read and validate a required integer."""

    value = record.get(field_name)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{record_description} has an invalid "
            f"{field_name!r} field."
        )

    return value


def require_embedding(
    record: dict[str, Any],
    record_description: str,
    expected_dimension: int,
) -> list[float]:
    """Read and validate one embedding vector."""

    raw_embedding = record.get("embedding")

    if not isinstance(raw_embedding, list):
        raise ValueError(
            f"{record_description} has no valid embedding list."
        )

    if len(raw_embedding) != expected_dimension:
        raise ValueError(
            f"{record_description} has embedding dimension "
            f"{len(raw_embedding)}, expected "
            f"{expected_dimension}."
        )

    embedding: list[float] = []

    for value in raw_embedding:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
        ):
            raise ValueError(
                f"{record_description} contains a "
                "non-numeric embedding value."
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"{record_description} contains a "
                "non-finite embedding value."
            )

        embedding.append(numeric_value)

    return embedding


def load_vector_index(
    index_path: Path,
) -> LoadedVectorIndex:
    """Load and validate the persistent vector index."""

    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector index was not found: {index_path}"
        )

    if not index_path.is_file():
        raise ValueError(
            f"Vector index path is not a file: {index_path}"
        )

    try:
        payload = json.loads(
            index_path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Vector index contains invalid JSON: {index_path}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            "The vector index must contain a JSON object."
        )

    embedding_model = require_string(
        payload,
        "requested_embedding_model",
        "Vector index",
    )

    distance_metric = require_string(
        payload,
        "distance_metric",
        "Vector index",
    )

    if distance_metric != "cosine_similarity":
        raise ValueError(
            "This lesson supports only cosine similarity."
        )

    embedding_dimension = require_integer(
        payload,
        "embedding_dimension",
        "Vector index",
    )

    if embedding_dimension <= 0:
        raise ValueError(
            "Embedding dimension must be greater than zero."
        )

    declared_chunk_count = require_integer(
        payload,
        "chunk_count",
        "Vector index",
    )

    raw_chunks = payload.get("chunks")

    if not isinstance(raw_chunks, list):
        raise ValueError(
            "The vector index must contain a chunks list."
        )

    if declared_chunk_count != len(raw_chunks):
        raise ValueError(
            "The declared chunk count does not match "
            "the stored chunks."
        )

    chunks: list[IndexedChunk] = []
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

        description = (
            f"Chunk record {record_number}"
        )

        chunk_id = require_string(
            raw_chunk,
            "chunk_id",
            description,
        )

        if chunk_id in seen_chunk_ids:
            raise ValueError(
                f"Duplicate chunk ID found: {chunk_id}"
            )

        chunk = IndexedChunk(
            chunk_id=chunk_id,
            section_id=require_string(
                raw_chunk,
                "section_id",
                description,
            ),
            document_title=require_string(
                raw_chunk,
                "document_title",
                description,
            ),
            section_title=require_string(
                raw_chunk,
                "section_title",
                description,
            ),
            source=require_string(
                raw_chunk,
                "source",
                description,
            ),
            content=require_string(
                raw_chunk,
                "content",
                description,
            ),
            chunk_number=require_integer(
                raw_chunk,
                "chunk_number",
                description,
            ),
            start_word=require_integer(
                raw_chunk,
                "start_word",
                description,
            ),
            end_word_exclusive=require_integer(
                raw_chunk,
                "end_word_exclusive",
                description,
            ),
            embedding=require_embedding(
                raw_chunk,
                description,
                embedding_dimension,
            ),
        )

        seen_chunk_ids.add(chunk_id)
        chunks.append(chunk)

    if not chunks:
        raise ValueError(
            "The vector index contains no chunks."
        )

    return LoadedVectorIndex(
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        distance_metric=distance_metric,
        chunks=chunks,
    )


def generate_query_embedding(
    query: str,
    embedding_model: str,
    expected_dimension: int,
) -> tuple[list[float], int | None]:
    """Generate and validate an embedding for one query."""

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "The search query cannot be empty."
        )

    request_body = {
        "model": embedding_model,
        "input": cleaned_query,
        "truncate": False,
    }

    try:
        response = httpx.post(
            OLLAMA_EMBED_URL,
            json=request_body,
            timeout=120.0,
        )
        response.raise_for_status()
    except httpx.ConnectError as error:
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Confirm that Ollama is running."
        ) from error
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Ollama did not generate the query embedding "
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
            "Ollama returned a response that was not "
            "valid JSON."
        ) from error

    raw_embeddings = response_body.get("embeddings")
    prompt_token_count = response_body.get(
        "prompt_eval_count"
    )

    if (
        not isinstance(raw_embeddings, list)
        or len(raw_embeddings) != 1
    ):
        raise RuntimeError(
            "Ollama did not return exactly one "
            "query embedding."
        )

    raw_embedding = raw_embeddings[0]

    if not isinstance(raw_embedding, list):
        raise RuntimeError(
            "The query embedding is not a list."
        )

    if len(raw_embedding) != expected_dimension:
        raise RuntimeError(
            "Query embedding dimension does not match "
            "the vector index."
        )

    query_embedding: list[float] = []

    for value in raw_embedding:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
        ):
            raise RuntimeError(
                "The query embedding contains a "
                "non-numeric value."
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise RuntimeError(
                "The query embedding contains a "
                "non-finite value."
            )

        query_embedding.append(numeric_value)

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

    return query_embedding, prompt_token_count


def dot_product(
    first_vector: list[float],
    second_vector: list[float],
) -> float:
    """Calculate the dot product of two vectors."""

    if len(first_vector) != len(second_vector):
        raise ValueError(
            "Vectors must have the same dimension."
        )

    return sum(
        first_value * second_value
        for first_value, second_value in zip(
            first_vector,
            second_vector,
            strict=True,
        )
    )


def vector_norm(
    vector: list[float],
) -> float:
    """Calculate the Euclidean length of a vector."""

    return math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )


def cosine_similarity(
    first_vector: list[float],
    second_vector: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    numerator = dot_product(
        first_vector,
        second_vector,
    )

    denominator = (
        vector_norm(first_vector)
        * vector_norm(second_vector)
    )

    if denominator == 0:
        raise ValueError(
            "Cosine similarity is undefined for "
            "a zero-length vector."
        )

    return numerator / denominator


def search_index(
    query_embedding: list[float],
    chunks: list[IndexedChunk],
    top_k: int,
) -> list[SearchResult]:
    """Rank all chunks and return the top-k results."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    scored_results = [
        SearchResult(
            score=cosine_similarity(
                query_embedding,
                chunk.embedding,
            ),
            chunk=chunk,
        )
        for chunk in chunks
    ]

    scored_results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return scored_results[
        : min(top_k, len(scored_results))
    ]


def main() -> None:
    """Load the index and retrieve relevant chunks."""

    vector_index = load_vector_index(
        INDEX_PATH
    )

    print("PERSISTENT VECTOR SEARCH")
    print("========================")
    print()
    print("Index:", INDEX_PATH)
    print(
        "Embedding model:",
        vector_index.embedding_model,
    )
    print(
        "Embedding dimension:",
        vector_index.embedding_dimension,
    )
    print(
        "Chunks available:",
        len(vector_index.chunks),
    )
    print("Top-k:", TOP_K)
    print()

    print("QUERY")
    print("-----")
    print(QUERY)
    print()
    print("Generating query embedding...")

    (
        query_embedding,
        prompt_token_count,
    ) = generate_query_embedding(
        query=QUERY,
        embedding_model=vector_index.embedding_model,
        expected_dimension=(
            vector_index.embedding_dimension
        ),
    )

    print("Query embedding generated.")

    if prompt_token_count is not None:
        print(
            "Query input tokens:",
            prompt_token_count,
        )

    print(
        "Query vector norm:",
        f"{vector_norm(query_embedding):.6f}",
    )
    print()

    results = search_index(
        query_embedding=query_embedding,
        chunks=vector_index.chunks,
        top_k=TOP_K,
    )

    print("TOP RETRIEVAL RESULTS")
    print("---------------------")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        chunk = result.chunk

        print(f"Rank: {rank}")
        print(
            "Similarity:",
            f"{result.score:.4f}",
        )
        print(
            "Chunk ID:",
            chunk.chunk_id,
        )
        print(
            "Document:",
            chunk.document_title,
        )
        print(
            "Section:",
            chunk.section_title,
        )
        print(
            "Source:",
            chunk.source,
        )
        print(
            "Word range:",
            f"{chunk.start_word}:"
            f"{chunk.end_word_exclusive}",
        )
        print("Content:")
        print(chunk.content)
        print()

    print("STATUS")
    print("------")
    print(
        "The query was embedded and compared against "
        "all stored chunk vectors."
    )


if __name__ == "__main__":
    main()