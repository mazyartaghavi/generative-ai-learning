from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from lessons.lesson_23_search_vector_index import (
    INDEX_PATH,
    generate_query_embedding,
    load_vector_index,
)
from lessons.lesson_30_qdrant_local_index import (
    COLLECTION_NAME,
    QDRANT_PATH,
)


QUERY = (
    "What should a driver do when a vehicle "
    "warning appears?"
)

FILTER_SECTION_ID = "section-003"

UNFILTERED_TOP_K = 3
FILTERED_TOP_K = 2


def require_payload_string(
    payload: dict[str, Any],
    field_name: str,
) -> str:
    """Read and validate one required payload string."""

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


def require_payload_integer(
    payload: dict[str, Any],
    field_name: str,
) -> int:
    """Read and validate one required payload integer."""

    value = payload.get(field_name)

    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(
            "A Qdrant result contains an invalid "
            f"{field_name!r} payload field."
        )

    return value


def build_section_filter(
    section_id: str,
) -> models.Filter:
    """Create an exact-match filter for one section ID."""

    cleaned_section_id = section_id.strip()

    if not cleaned_section_id:
        raise ValueError(
            "The section filter cannot be empty."
        )

    return models.Filter(
        must=[
            models.FieldCondition(
                key="section_id",
                match=models.MatchValue(
                    value=cleaned_section_id,
                ),
            )
        ]
    )


def query_qdrant(
    client: QdrantClient,
    query_embedding: list[float],
    *,
    limit: int,
    query_filter: models.Filter | None = None,
) -> list[models.ScoredPoint]:
    """Run an unfiltered or filtered Qdrant search."""

    if not query_embedding:
        raise ValueError(
            "The query embedding cannot be empty."
        )

    if limit <= 0:
        raise ValueError(
            "The result limit must be greater than zero."
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return list(response.points)


def print_result(
    rank: int,
    result: models.ScoredPoint,
) -> None:
    """Print one Qdrant search result."""

    raw_payload = result.payload

    if raw_payload is None:
        raise RuntimeError(
            "A Qdrant result has no payload."
        )

    payload = dict(raw_payload)

    chunk_id = require_payload_string(
        payload,
        "chunk_id",
    )

    section_id = require_payload_string(
        payload,
        "section_id",
    )

    section_title = require_payload_string(
        payload,
        "section_title",
    )

    source = require_payload_string(
        payload,
        "source",
    )

    content = require_payload_string(
        payload,
        "content",
    )

    chunk_number = require_payload_integer(
        payload,
        "chunk_number",
    )

    start_word = require_payload_integer(
        payload,
        "start_word",
    )

    end_word_exclusive = require_payload_integer(
        payload,
        "end_word_exclusive",
    )

    print(f"Rank: {rank}")
    print("Point ID:", result.id)
    print(
        "Similarity:",
        f"{result.score:.4f}",
    )
    print("Chunk ID:", chunk_id)
    print("Section ID:", section_id)
    print("Section:", section_title)
    print("Chunk number:", chunk_number)
    print(
        "Word range:",
        f"{start_word}:{end_word_exclusive}",
    )
    print("Source:", source)
    print("Content:")
    print(content)
    print()


def print_results(
    title: str,
    results: list[models.ScoredPoint],
) -> None:
    """Print a complete ranked-result group."""

    print(title)
    print("-" * len(title))

    if not results:
        print(
            "No matching points were returned."
        )
        print()
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print_result(
            rank=rank,
            result=result,
        )


def validate_filtered_results(
    results: list[models.ScoredPoint],
    expected_section_id: str,
) -> None:
    """Confirm every filtered result belongs to one section."""

    if not results:
        raise RuntimeError(
            "The filtered search returned no points."
        )

    for result in results:
        raw_payload = result.payload

        if raw_payload is None:
            raise RuntimeError(
                "A filtered result has no payload."
            )

        payload = dict(raw_payload)

        returned_section_id = require_payload_string(
            payload,
            "section_id",
        )

        if returned_section_id != expected_section_id:
            raise RuntimeError(
                "Qdrant returned a point outside the "
                "requested section filter."
            )


def main() -> None:
    """Reopen persistent Qdrant storage and filter results."""

    vector_index = load_vector_index(
        INDEX_PATH
    )

    print("QDRANT PERSISTENT SEARCH AND FILTERING")
    print("======================================")
    print()
    print(
        "Source vector index:",
        INDEX_PATH,
    )
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
        "Query:",
        QUERY,
    )
    print(
        "Section filter:",
        FILTER_SECTION_ID,
    )
    print()

    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            "The persistent Qdrant directory was not found. "
            "Run Lesson 30 first."
        )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    try:
        print(
            "1. Opening the existing Qdrant database..."
        )

        if not client.collection_exists(
            collection_name=COLLECTION_NAME,
        ):
            raise RuntimeError(
                "The expected Qdrant collection does not "
                "exist. Run Lesson 30 first."
            )

        print(
            "   Existing collection found."
        )
        print()

        count_result = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        )

        print("PERSISTENCE REPORT")
        print("------------------")
        print(
            "Stored points:",
            count_result.count,
        )
        print(
            "Collection recreated:",
            False,
        )
        print(
            "Points uploaded:",
            0,
        )
        print()

        if count_result.count <= 0:
            raise RuntimeError(
                "The Qdrant collection contains no points."
            )

        print(
            "2. Generating one query embedding..."
        )

        (
            query_embedding,
            query_token_count,
        ) = generate_query_embedding(
            query=QUERY,
            embedding_model=(
                vector_index.embedding_model
            ),
            expected_dimension=(
                vector_index.embedding_dimension
            ),
        )

        print(
            "   Query embedding generated."
        )

        if query_token_count is not None:
            print(
                "   Query input tokens:",
                query_token_count,
            )

        print()

        print(
            "3. Running an unfiltered search..."
        )

        unfiltered_results = query_qdrant(
            client=client,
            query_embedding=query_embedding,
            limit=UNFILTERED_TOP_K,
        )

        print(
            "   Unfiltered search completed."
        )
        print()

        print_results(
            title="UNFILTERED RESULTS",
            results=unfiltered_results,
        )

        print(
            "4. Building an exact section filter..."
        )

        section_filter = build_section_filter(
            FILTER_SECTION_ID
        )

        print(
            "   Filter created."
        )
        print()

        print(
            "5. Searching only the filtered section..."
        )

        filtered_results = query_qdrant(
            client=client,
            query_embedding=query_embedding,
            limit=FILTERED_TOP_K,
            query_filter=section_filter,
        )

        print(
            "   Filtered search completed."
        )
        print()

        validate_filtered_results(
            results=filtered_results,
            expected_section_id=FILTER_SECTION_ID,
        )

        print_results(
            title=(
                "FILTERED RESULTS "
                f"({FILTER_SECTION_ID})"
            ),
            results=filtered_results,
        )

        print("COMPARISON")
        print("----------")

        if not unfiltered_results:
            raise RuntimeError(
                "The unfiltered search returned no points."
            )

        if not filtered_results:
            raise RuntimeError(
                "The filtered search returned no points."
            )

        unfiltered_payload = (
            unfiltered_results[0].payload
        )

        filtered_payload = (
            filtered_results[0].payload
        )

        if (
            unfiltered_payload is None
            or filtered_payload is None
        ):
            raise RuntimeError(
                "A top-ranked result has no payload."
            )

        unfiltered_section = require_payload_string(
            dict(unfiltered_payload),
            "section_title",
        )

        filtered_section = require_payload_string(
            dict(filtered_payload),
            "section_title",
        )

        print(
            "Unfiltered top section:",
            unfiltered_section,
        )
        print(
            "Filtered top section:",
            filtered_section,
        )
        print()

        print("STATUS")
        print("------")
        print(
            "The existing Qdrant database was reopened "
            "without rebuilding it."
        )
        print(
            "The unfiltered search considered all stored "
            "points, while the filtered search considered "
            f"only points from {FILTER_SECTION_ID}."
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()