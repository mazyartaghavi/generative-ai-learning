from __future__ import annotations

import math

import httpx


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "embeddinggemma"

DOCUMENTS = [
    (
        "Vehicle A has an engine warning and should be "
        "inspected before its next trip."
    ),
    (
        "Vehicle B completed its delivery route and "
        "is operating normally."
    ),
    (
        "The fleet office ordered new uniforms for "
        "the drivers."
    ),
    (
        "Vehicle C has low tire pressure and requires "
        "tire service."
    ),
]

QUERY = "Which vehicle needs inspection because of an engine problem?"


def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """Generate one embedding vector for every supplied text."""

    request_body = {
        "model": EMBEDDING_MODEL,
        "input": texts,
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
            f"{error.response.status_code}."
        ) from error

    response_body = response.json()
    raw_embeddings = response_body.get("embeddings")

    if not isinstance(raw_embeddings, list):
        raise RuntimeError(
            "Ollama did not return an embeddings list."
        )

    if len(raw_embeddings) != len(texts):
        raise RuntimeError(
            "The number of returned embeddings does not "
            "match the number of input texts."
        )

    embeddings: list[list[float]] = []

    for index, raw_vector in enumerate(
        raw_embeddings,
        start=1,
    ):
        if not isinstance(raw_vector, list):
            raise RuntimeError(
                f"Embedding {index} is not a list."
            )

        vector: list[float] = []

        for value in raw_vector:
            if not isinstance(value, int | float):
                raise RuntimeError(
                    f"Embedding {index} contains "
                    "a non-numeric value."
                )

            vector.append(float(value))

        embeddings.append(vector)

    dimensions = {
        len(vector)
        for vector in embeddings
    }

    if len(dimensions) != 1:
        raise RuntimeError(
            "The returned embedding vectors have "
            "different dimensions."
        )

    return embeddings


def dot_product(
    first_vector: list[float],
    second_vector: list[float],
) -> float:
    """Calculate the dot product of two vectors."""

    if len(first_vector) != len(second_vector):
        raise ValueError(
            "Vectors must have the same dimension."
        )

    total = 0.0

    for first_value, second_value in zip(
        first_vector,
        second_vector,
        strict=True,
    ):
        total = total + first_value * second_value

    return total


def vector_magnitude(vector: list[float]) -> float:
    """Calculate the magnitude, or length, of a vector."""

    squared_values = [
        value * value
        for value in vector
    ]

    return math.sqrt(sum(squared_values))


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
        vector_magnitude(first_vector)
        * vector_magnitude(second_vector)
    )

    if denominator == 0:
        raise ValueError(
            "Cosine similarity is undefined "
            "for a zero-length vector."
        )

    return numerator / denominator


def main() -> None:
    """Rank documents by their similarity to the query."""

    all_texts = DOCUMENTS + [QUERY]
    all_embeddings = generate_embeddings(all_texts)

    document_embeddings = all_embeddings[:-1]
    query_embedding = all_embeddings[-1]

    print("EMBEDDING INFORMATION")
    print("---------------------")
    print("Model:", EMBEDDING_MODEL)
    print("Number of documents:", len(DOCUMENTS))
    print("Embedding dimension:", len(query_embedding))
    print()

    scored_documents: list[tuple[float, str]] = []

    for document, document_embedding in zip(
        DOCUMENTS,
        document_embeddings,
        strict=True,
    ):
        score = cosine_similarity(
            query_embedding,
            document_embedding,
        )

        scored_documents.append(
            (score, document)
        )

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    print("QUERY")
    print("-----")
    print(QUERY)
    print()

    print("SEMANTIC SEARCH RESULTS")
    print("-----------------------")

    for rank, (score, document) in enumerate(
        scored_documents,
        start=1,
    ):
        print(f"{rank}. Similarity: {score:.4f}")
        print(f"   {document}")
        print()


if __name__ == "__main__":
    main()