from __future__ import annotations

import math
import random


FINAL_TOKEN_VECTOR = [
    -0.8447,
    1.3956,
    0.4950,
    -1.0458,
]

VOCABULARY = [
    ".",
    "before",
    "today",
    "because",
    "immediately",
]

# There is one output-weight row for every vocabulary token.
# Each row has four values because the final token vector
# has four dimensions.
OUTPUT_WEIGHTS = [
    [-0.2, 0.8, 0.3, -0.4],   # .
    [0.1, 0.6, 0.2, -0.1],    # before
    [0.2, 0.3, 0.4, -0.2],    # today
    [-0.1, 0.2, 0.1, 0.2],    # because
    [0.3, 0.1, 0.5, -0.3],    # immediately
]

OUTPUT_BIASES = [
    0.2,
    0.1,
    0.0,
    0.0,
    -0.1,
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

    return sum(
        first_value * second_value
        for first_value, second_value in zip(
            first_vector,
            second_vector,
            strict=True,
        )
    )


def calculate_logits(
    hidden_vector: list[float],
) -> list[float]:
    """Project a hidden vector into vocabulary logits."""

    if len(OUTPUT_WEIGHTS) != len(VOCABULARY):
        raise ValueError(
            "Every vocabulary token requires one weight row."
        )

    if len(OUTPUT_BIASES) != len(VOCABULARY):
        raise ValueError(
            "Every vocabulary token requires one bias."
        )

    logits: list[float] = []

    for weight_row, bias in zip(
        OUTPUT_WEIGHTS,
        OUTPUT_BIASES,
        strict=True,
    ):
        logit = dot_product(
            weight_row,
            hidden_vector,
        ) + bias

        logits.append(logit)

    return logits


def softmax_with_temperature(
    logits: list[float],
    temperature: float,
) -> list[float]:
    """Convert logits into probabilities using temperature."""

    if not logits:
        raise ValueError(
            "At least one logit is required."
        )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    adjusted_logits = [
        logit / temperature
        for logit in logits
    ]

    maximum_logit = max(adjusted_logits)

    exponentials = [
        math.exp(logit - maximum_logit)
        for logit in adjusted_logits
    ]

    exponential_sum = sum(exponentials)

    return [
        value / exponential_sum
        for value in exponentials
    ]


def greedy_decode(
    probabilities: list[float],
) -> str:
    """Select the token with the highest probability."""

    highest_probability_index = probabilities.index(
        max(probabilities)
    )

    return VOCABULARY[highest_probability_index]


def sample_token(
    probabilities: list[float],
    random_generator: random.Random,
) -> str:
    """Randomly sample a token according to its probability."""

    selected_tokens = random_generator.choices(
        population=VOCABULARY,
        weights=probabilities,
        k=1,
    )

    return selected_tokens[0]


def print_distribution(
    logits: list[float],
    temperature: float,
) -> list[float]:
    """Print the probability distribution for one temperature."""

    probabilities = softmax_with_temperature(
        logits,
        temperature,
    )

    print(f"TEMPERATURE: {temperature}")
    print("-" * 16)

    for token, probability in zip(
        VOCABULARY,
        probabilities,
        strict=True,
    ):
        print(
            f"{token:<12} "
            f"{probability:.4f} "
            f"({probability * 100:.2f}%)"
        )

    print(
        "Probability total:",
        f"{sum(probabilities):.4f}",
    )
    print(
        "Greedy token:",
        greedy_decode(probabilities),
    )
    print()

    return probabilities


def main() -> None:
    """Demonstrate logits, temperature, and token sampling."""

    print("FINAL TOKEN REPRESENTATION")
    print("--------------------------")
    print(FINAL_TOKEN_VECTOR)
    print()

    logits = calculate_logits(
        FINAL_TOKEN_VECTOR
    )

    print("VOCABULARY LOGITS")
    print("-----------------")

    for token, logit in zip(
        VOCABULARY,
        logits,
        strict=True,
    ):
        print(f"{token:<12} {logit:.4f}")

    print()

    temperatures = [
        0.5,
        1.0,
        2.0,
    ]

    distributions: dict[float, list[float]] = {}

    for temperature in temperatures:
        distributions[temperature] = print_distribution(
            logits,
            temperature,
        )

    print("REPEATABLE SAMPLING")
    print("-------------------")

    for temperature in temperatures:
        random_generator = random.Random(42)

        sampled_tokens = [
            sample_token(
                distributions[temperature],
                random_generator,
            )
            for _ in range(10)
        ]

        print(
            f"Temperature {temperature}: "
            + " | ".join(sampled_tokens)
        )

    print()
    print("INTERPRETATION")
    print("--------------")
    print(
        "A lower temperature concentrates probability "
        "on the strongest token."
    )
    print(
        "A higher temperature produces a flatter, "
        "more varied distribution."
    )
    print(
        "Greedy decoding always chooses the token with "
        "the highest probability."
    )
    print(
        "Sampling can choose lower-probability tokens."
    )


if __name__ == "__main__":
    main()