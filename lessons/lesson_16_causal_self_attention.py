from __future__ import annotations

import math


TOKENS = [
    "Vehicle",
    "A",
    "needs",
    "inspection",
]

# Every token has its own query vector.
QUERIES = [
    [1.0, 0.0],
    [0.8, 0.2],
    [0.2, 0.9],
    [0.1, 1.0],
]

# Every token also has its own key vector.
KEYS = [
    [1.0, 0.0],
    [0.9, 0.1],
    [0.2, 0.8],
    [0.0, 1.0],
]

# One-hot values make the contribution from each
# source token easy to inspect in this lesson.
VALUES = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def dot_product(
    first_vector: list[float],
    second_vector: list[float],
) -> float:
    """Calculate the dot product of equal-length vectors."""

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


def softmax(scores: list[float]) -> list[float]:
    """Convert scores into positive weights summing to one."""

    if not scores:
        raise ValueError(
            "Softmax requires at least one score."
        )

    maximum_score = max(scores)

    exponentials = [
        math.exp(score - maximum_score)
        for score in scores
    ]

    exponential_sum = sum(exponentials)

    return [
        value / exponential_sum
        for value in exponentials
    ]


def calculate_causal_attention() -> list[list[float]]:
    """Calculate one causally masked attention row per token."""

    key_dimension = len(KEYS[0])
    scaling_factor = math.sqrt(key_dimension)

    attention_matrix: list[list[float]] = []

    for query_index, query in enumerate(QUERIES):
        allowed_key_indexes: list[int] = []
        allowed_scores: list[float] = []

        for key_index, key in enumerate(KEYS):
            # A token may access only itself and earlier tokens.
            if key_index <= query_index:
                raw_score = dot_product(query, key)
                scaled_score = raw_score / scaling_factor

                allowed_key_indexes.append(key_index)
                allowed_scores.append(scaled_score)

        allowed_weights = softmax(allowed_scores)

        complete_row = [0.0] * len(TOKENS)

        for key_index, weight in zip(
            allowed_key_indexes,
            allowed_weights,
            strict=True,
        ):
            complete_row[key_index] = weight

        attention_matrix.append(complete_row)

    return attention_matrix


def weighted_sum(
    weights: list[float],
    vectors: list[list[float]],
) -> list[float]:
    """Combine value vectors according to attention weights."""

    if len(weights) != len(vectors):
        raise ValueError(
            "There must be one weight per value vector."
        )

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


def print_attention_matrix(
    attention_matrix: list[list[float]],
) -> None:
    """Print the attention matrix as a readable table."""

    column_width = 13

    header = "Query \\ Key".ljust(column_width)

    for token in TOKENS:
        header = header + token.rjust(column_width)

    print(header)
    print("-" * len(header))

    for query_token, row in zip(
        TOKENS,
        attention_matrix,
        strict=True,
    ):
        line = query_token.ljust(column_width)

        for weight in row:
            line = line + f"{weight:.4f}".rjust(column_width)

        print(line)


def main() -> None:
    """Demonstrate causal self-attention over a sequence."""

    print("TOKEN SEQUENCE")
    print("--------------")
    print(" ".join(TOKENS))
    print()

    attention_matrix = calculate_causal_attention()

    print("CAUSAL ATTENTION MATRIX")
    print("-----------------------")
    print_attention_matrix(attention_matrix)
    print()

    print("ROW TOTALS")
    print("----------")

    for token, row in zip(
        TOKENS,
        attention_matrix,
        strict=True,
    ):
        print(
            f"{token}: {sum(row):.4f}"
        )

    print()
    print("CONTEXT VECTORS")
    print("---------------")

    for query_token, weights in zip(
        TOKENS,
        attention_matrix,
        strict=True,
    ):
        context_vector = weighted_sum(
            weights,
            VALUES,
        )

        print(f"Query token: {query_token}")

        for source_token, contribution in zip(
            TOKENS,
            context_vector,
            strict=True,
        ):
            print(
                f"  Contribution from "
                f"{source_token}: {contribution:.4f}"
            )

        print()

    print("INTERPRETATION")
    print("--------------")
    print(
        "Every attention row sums to one."
    )
    print(
        "Every position above the causal boundary "
        "has weight zero."
    )
    print(
        "The final token may attend to all tokens, "
        "but the first token may attend only to itself."
    )


if __name__ == "__main__":
    main()