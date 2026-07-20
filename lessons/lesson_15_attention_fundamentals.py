from __future__ import annotations

import math


TOKENS = [
    "vehicle",
    "engine",
    "warning",
]

# This query represents what the token "warning"
# is looking for in this educational example.
QUERY = [0.1, 1.0]

KEYS = {
    "vehicle": [1.0, 0.0],
    "engine": [0.0, 1.0],
    "warning": [0.2, 0.8],
}

# The three value-vector positions represent:
# 0: vehicle identity
# 1: engine context
# 2: alert context
VALUES = {
    "vehicle": [1.0, 0.0, 0.0],
    "engine": [0.0, 1.0, 0.0],
    "warning": [0.0, 0.0, 1.0],
}

VALUE_FEATURES = [
    "vehicle identity",
    "engine context",
    "alert context",
]


def dot_product(
    first_vector: list[float],
    second_vector: list[float],
) -> float:
    """Calculate the dot product of two equal-length vectors."""

    if len(first_vector) != len(second_vector):
        raise ValueError(
            "The vectors must have the same dimension."
        )

    total = 0.0

    for first_value, second_value in zip(
        first_vector,
        second_vector,
        strict=True,
    ):
        total = total + first_value * second_value

    return total


def softmax(scores: list[float]) -> list[float]:
    """Convert arbitrary scores into positive weights summing to one."""

    if not scores:
        raise ValueError("Softmax requires at least one score.")

    maximum_score = max(scores)

    exponentials = [
        math.exp(score - maximum_score)
        for score in scores
    ]

    exponential_sum = sum(exponentials)

    return [
        exponential / exponential_sum
        for exponential in exponentials
    ]


def weighted_sum(
    weights: list[float],
    vectors: list[list[float]],
) -> list[float]:
    """Combine vectors according to their attention weights."""

    if len(weights) != len(vectors):
        raise ValueError(
            "There must be one weight for every vector."
        )

    if not vectors:
        raise ValueError("At least one vector is required.")

    vector_dimension = len(vectors[0])
    output = [0.0] * vector_dimension

    for weight, vector in zip(
        weights,
        vectors,
        strict=True,
    ):
        if len(vector) != vector_dimension:
            raise ValueError(
                "All value vectors must have the same dimension."
            )

        for index, value in enumerate(vector):
            output[index] = output[index] + weight * value

    return output


def main() -> None:
    """Demonstrate one simplified attention calculation."""

    print("ATTENTION INPUT")
    print("---------------")
    print("Tokens:", ", ".join(TOKENS))
    print("Query token: warning")
    print("Query vector:", QUERY)
    print()

    raw_scores: list[float] = []

    for token in TOKENS:
        score = dot_product(
            QUERY,
            KEYS[token],
        )
        raw_scores.append(score)

    key_dimension = len(QUERY)
    scaling_factor = math.sqrt(key_dimension)

    scaled_scores = [
        score / scaling_factor
        for score in raw_scores
    ]

    attention_weights = softmax(scaled_scores)

    print("ATTENTION SCORE CALCULATION")
    print("---------------------------")

    for token, raw_score, scaled_score in zip(
        TOKENS,
        raw_scores,
        scaled_scores,
        strict=True,
    ):
        print(f"Token: {token}")
        print(f"  Raw dot-product score: {raw_score:.4f}")
        print(f"  Scaled score: {scaled_score:.4f}")

    print()
    print("ATTENTION WEIGHTS")
    print("-----------------")

    for token, weight in zip(
        TOKENS,
        attention_weights,
        strict=True,
    ):
        print(f"{token}: {weight:.4f}")

    print(
        "Weight total:",
        f"{sum(attention_weights):.4f}",
    )
    print()

    value_vectors = [
        VALUES[token]
        for token in TOKENS
    ]

    context_vector = weighted_sum(
        attention_weights,
        value_vectors,
    )

    print("CONTEXT VECTOR")
    print("--------------")

    for feature, value in zip(
        VALUE_FEATURES,
        context_vector,
        strict=True,
    ):
        print(f"{feature}: {value:.4f}")

    highest_weight_index = attention_weights.index(
        max(attention_weights)
    )

    print()
    print("INTERPRETATION")
    print("--------------")
    print(
        "The query assigned its greatest attention "
        f"weight to: {TOKENS[highest_weight_index]}"
    )
    print(
        "The final context vector is a weighted "
        "combination of all three value vectors."
    )


if __name__ == "__main__":
    main()