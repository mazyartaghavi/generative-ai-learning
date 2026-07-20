from __future__ import annotations

import math


TOKENS = [
    "Vehicle",
    "A",
    "needs",
    "inspection",
]

# Simplified contextual input vectors.
INPUT_VECTORS = [
    [1.0, 0.0, 0.2, 0.1],
    [0.8, 0.2, 0.0, 0.1],
    [0.1, 0.9, 0.3, 0.0],
    [0.0, 1.0, 0.8, 0.2],
]

# Causal attention weights obtained from Lesson 16.
ATTENTION_WEIGHTS = [
    [1.0000, 0.0000, 0.0000, 0.0000],
    [0.5106, 0.4894, 0.0000, 0.0000],
    [0.2828, 0.2971, 0.4201, 0.0000],
    [0.1780, 0.1897, 0.2961, 0.3363],
]

# First feed-forward transformation:
# four input values become six hidden values.
HIDDEN_WEIGHTS = [
    [0.5, -0.2, 0.1, 0.3],
    [-0.1, 0.6, 0.2, 0.0],
    [0.3, 0.1, 0.5, -0.2],
    [0.0, 0.4, -0.3, 0.6],
    [0.2, 0.2, 0.2, 0.2],
    [-0.4, 0.1, 0.3, 0.5],
]

HIDDEN_BIAS = [
    0.1,
    0.0,
    0.05,
    -0.05,
    0.0,
    0.1,
]

# Second feed-forward transformation:
# six hidden values become four output values.
OUTPUT_WEIGHTS = [
    [0.4, 0.0, 0.2, -0.1, 0.3, 0.1],
    [-0.2, 0.5, 0.1, 0.3, 0.0, 0.2],
    [0.1, 0.2, 0.4, -0.2, 0.2, 0.0],
    [0.3, -0.1, 0.0, 0.4, 0.1, 0.2],
]

OUTPUT_BIAS = [
    0.0,
    0.05,
    -0.05,
    0.0,
]


def add_vectors(
    first_vector: list[float],
    second_vector: list[float],
) -> list[float]:
    """Add two equal-length vectors element by element."""

    if len(first_vector) != len(second_vector):
        raise ValueError(
            "Vectors must have the same dimension."
        )

    return [
        first_value + second_value
        for first_value, second_value in zip(
            first_vector,
            second_vector,
            strict=True,
        )
    ]


def weighted_sum(
    weights: list[float],
    vectors: list[list[float]],
) -> list[float]:
    """Combine vectors according to supplied weights."""

    if len(weights) != len(vectors):
        raise ValueError(
            "There must be one weight for every vector."
        )

    if not vectors:
        raise ValueError(
            "At least one vector is required."
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
                "All vectors must have the same dimension."
            )

        for index, value in enumerate(vector):
            output[index] = output[index] + weight * value

    return output


def layer_normalize(
    vector: list[float],
    epsilon: float = 1e-5,
) -> list[float]:
    """Normalize one token vector across its dimensions."""

    if not vector:
        raise ValueError(
            "A vector is required for normalization."
        )

    mean = sum(vector) / len(vector)

    variance = sum(
        (value - mean) ** 2
        for value in vector
    ) / len(vector)

    denominator = math.sqrt(
        variance + epsilon
    )

    return [
        (value - mean) / denominator
        for value in vector
    ]


def matrix_vector_product(
    matrix: list[list[float]],
    vector: list[float],
    bias: list[float],
) -> list[float]:
    """Apply a linear transformation to a vector."""

    if len(matrix) != len(bias):
        raise ValueError(
            "Every matrix row requires one bias value."
        )

    output: list[float] = []

    for row, bias_value in zip(
        matrix,
        bias,
        strict=True,
    ):
        if len(row) != len(vector):
            raise ValueError(
                "Matrix row and vector dimensions do not match."
            )

        transformed_value = sum(
            weight * value
            for weight, value in zip(
                row,
                vector,
                strict=True,
            )
        )

        output.append(
            transformed_value + bias_value
        )

    return output


def relu(vector: list[float]) -> list[float]:
    """Apply the ReLU activation function."""

    return [
        max(0.0, value)
        for value in vector
    ]


def feed_forward(
    vector: list[float],
) -> list[float]:
    """Apply a simplified two-layer feed-forward network."""

    hidden_values = matrix_vector_product(
        HIDDEN_WEIGHTS,
        vector,
        HIDDEN_BIAS,
    )

    activated_values = relu(
        hidden_values
    )

    return matrix_vector_product(
        OUTPUT_WEIGHTS,
        activated_values,
        OUTPUT_BIAS,
    )


def format_vector(
    vector: list[float],
) -> str:
    """Format a vector using four decimal places."""

    formatted_values = [
        f"{value:.4f}"
        for value in vector
    ]

    return "[" + ", ".join(formatted_values) + "]"


def main() -> None:
    """Run one simplified transformer decoder block."""

    print("SIMPLIFIED TRANSFORMER DECODER BLOCK")
    print("====================================")
    print()

    final_outputs: list[list[float]] = []

    for token, input_vector, attention_row in zip(
        TOKENS,
        INPUT_VECTORS,
        ATTENTION_WEIGHTS,
        strict=True,
    ):
        attention_output = weighted_sum(
            attention_row,
            INPUT_VECTORS,
        )

        first_residual = add_vectors(
            input_vector,
            attention_output,
        )

        first_normalized = layer_normalize(
            first_residual
        )

        feed_forward_output = feed_forward(
            first_normalized
        )

        second_residual = add_vectors(
            first_normalized,
            feed_forward_output,
        )

        final_output = layer_normalize(
            second_residual
        )

        final_outputs.append(final_output)

        print(f"TOKEN: {token}")
        print("-" * (7 + len(token)))
        print(
            "Input vector:              ",
            format_vector(input_vector),
        )
        print(
            "Attention output:          ",
            format_vector(attention_output),
        )
        print(
            "After residual + norm:     ",
            format_vector(first_normalized),
        )
        print(
            "Feed-forward output:       ",
            format_vector(feed_forward_output),
        )
        print(
            "Final block output:        ",
            format_vector(final_output),
        )
        print()

    print("OUTPUT SEQUENCE")
    print("---------------")

    for token, output_vector in zip(
        TOKENS,
        final_outputs,
        strict=True,
    ):
        print(
            f"{token}: {format_vector(output_vector)}"
        )

    print()
    print("INTERPRETATION")
    print("--------------")
    print(
        "Attention mixes information across permitted tokens."
    )
    print(
        "Residual connections preserve earlier representations."
    )
    print(
        "Normalization stabilizes the numerical scale."
    )
    print(
        "The feed-forward network transforms each token "
        "independently."
    )


if __name__ == "__main__":
    main()