from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient, models

from lessons.lesson_23_search_vector_index import (
    INDEX_PATH,
    IndexedChunk,
    generate_query_embedding,
    load_vector_index,
)


QDRANT_PATH = Path(
    "data/qdrant_local"
)

COLLECTION_NAME = "fleet_manual_chunks"

QUERY = (
    "What should a driver do when an engine warning "
    "light appears?"
)

TOP_K = 3


def build_payload(
    chunk: IndexedChunk,
) -> dict[str, str | int]:
    """Convert one indexed chunk into Qdrant payload data."""

    return {
        "chunk_id": chunk.chunk_id,
        "section_id": chunk.section_id,
        "document_title": chunk.document_title,
        "section_title": chunk.section_title,
        "source": chunk.source,
        "content": chunk.content,
        "chunk_number": chunk.chunk_number,
        "start_word": chunk.start_word,
        "end_word_exclusive": (
            chunk.end_word_exclusive
        ),
    }


def build_qdrant_points(
    chunks: list[IndexedChunk],
) -> list[models.PointStruct]:
    """Convert indexed chunks into Qdrant points."""

    points: list[models.PointStruct] = []

    for point_id, chunk in enumerate(
        chunks,
        start=1,
    ):
        point = models.PointStruct(
            id=point_id,
            vector=chunk.embedding,
            payload=build_payload(
                chunk
            ),
        )

        points.append(
            point
        )

    return points


def recreate_collection(
    client: QdrantClient,
    embedding_dimension: int,
) -> None:
    """Delete an old collection and create a fresh one."""

    if embedding_dimension <= 0:
        raise ValueError(
            "Embedding dimension must be greater than zero."
        )

    if client.collection_exists(
        collection_name=COLLECTION_NAME,
    ):
        client.delete_collection(
            collection_name=COLLECTION_NAME,
        )

    created = client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embedding_dimension,
            distance=models.Distance.COSINE,
        ),
    )

    if not created:
        raise RuntimeError(
            "Qdrant did not create the collection."
        )


def upload_points(
    client: QdrantClient,
    points: list[models.PointStruct],
) -> None:
    """Upload all vector points to the collection."""

    if not points:
        raise ValueError(
            "At least one Qdrant point is required."
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )


def require_payload_string(
    payload: dict[str, object],
    field_name: str,
) -> str:
    """Read and validate one string from Qdrant payload."""

    value = payload.get(
        field_name
    )

    if not isinstance(value, str):
        raise RuntimeError(
            f"Qdrant result contains an invalid "
            f"{field_name!r} payload field."
        )

    cleaned_value = value.strip()

    if not cleaned_value:
        raise RuntimeError(
            f"Qdrant result contains an empty "
            f"{field_name!r} payload field."
        )

    return cleaned_value


def query_collection(
    client: QdrantClient,
    query_embedding: list[float],
) -> list[models.ScoredPoint]:
    """Search Qdrant using one query embedding."""

    if not query_embedding:
        raise ValueError(
            "The query embedding cannot be empty."
        )

    query_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    )

    return list(
        query_response.points
    )


def print_search_results(
    results: list[models.ScoredPoint],
) -> None:
    """Print ranked Qdrant search results."""

    print("QDRANT SEARCH RESULTS")
    print("---------------------")

    if not results:
        print(
            "No search results were returned."
        )
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        raw_payload = result.payload

        if raw_payload is None:
            raise RuntimeError(
                "A Qdrant result has no payload."
            )

        payload = dict(
            raw_payload
        )

        chunk_id = require_payload_string(
            payload,
            "chunk_id",
        )

        section_title = require_payload_string(
            payload,
            "section_title",
        )

        document_title = require_payload_string(
            payload,
            "document_title",
        )

        source = require_payload_string(
            payload,
            "source",
        )

        content = require_payload_string(
            payload,
            "content",
        )

        print(
            f"Rank: {rank}"
        )
        print(
            "Point ID:",
            result.id,
        )
        print(
            "Similarity:",
            f"{result.score:.4f}",
        )
        print(
            "Chunk ID:",
            chunk_id,
        )
        print(
            "Document:",
            document_title,
        )
        print(
            "Section:",
            section_title,
        )
        print(
            "Source:",
            source,
        )
        print(
            "Content:"
        )
        print(
            content
        )
        print()


def main() -> None:
    """Build and query a persistent local Qdrant database."""

    vector_index = load_vector_index(
        INDEX_PATH
    )

    print("QDRANT LOCAL VECTOR DATABASE")
    print("============================")
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
        "Chunks available:",
        len(vector_index.chunks),
    )
    print()

    QDRANT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    try:
        print(
            "1. Recreating the Qdrant collection..."
        )

        recreate_collection(
            client=client,
            embedding_dimension=(
                vector_index.embedding_dimension
            ),
        )

        print(
            "   Collection created."
        )
        print()

        print(
            "2. Converting chunks into Qdrant points..."
        )

        points = build_qdrant_points(
            vector_index.chunks
        )

        print(
            "   Points prepared:",
            len(points),
        )
        print()

        print(
            "3. Uploading points to Qdrant..."
        )

        upload_points(
            client=client,
            points=points,
        )

        print(
            "   Upload completed."
        )
        print()

        count_result = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        )

        stored_point_count = (
            count_result.count
        )

        print("COLLECTION REPORT")
        print("-----------------")
        print(
            "Collection:",
            COLLECTION_NAME,
        )
        print(
            "Expected points:",
            len(points),
        )
        print(
            "Stored points:",
            stored_point_count,
        )
        print(
            "Embedding dimension:",
            vector_index.embedding_dimension,
        )
        print(
            "Distance metric:",
            "cosine",
        )
        print()

        if stored_point_count != len(points):
            raise RuntimeError(
                "The Qdrant point count does not match "
                "the number of uploaded chunks."
            )

        print("TEST QUERY")
        print("----------")
        print(
            QUERY
        )
        print()

        print(
            "4. Generating the query embedding..."
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
            "5. Searching the Qdrant collection..."
        )

        search_results = query_collection(
            client=client,
            query_embedding=query_embedding,
        )

        print(
            "   Search completed."
        )
        print()

        print_search_results(
            search_results
        )

        if not search_results:
            raise RuntimeError(
                "Qdrant returned no results."
            )

        first_payload = (
            search_results[0].payload
        )

        if first_payload is None:
            raise RuntimeError(
                "The first Qdrant result has no payload."
            )

        first_section_id = (
            first_payload.get(
                "section_id"
            )
        )

        print("STATUS")
        print("------")

        if first_section_id == "section-002":
            print(
                "The JSON vectors were migrated into "
                "persistent Qdrant storage, and the engine "
                "warning section ranked first."
            )
        else:
            print(
                "The Qdrant database was created "
                "successfully, but the engine warning "
                "section did not rank first."
            )

    finally:
        client.close()


if __name__ == "__main__":
    main()