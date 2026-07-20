from __future__ import annotations

from typing import Any

import httpx


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

SECRET_CODE = "MADEIRA-7319"
CONTEXT_SIZE = 4096
FILLER_REPETITIONS = 100

FILLER_SENTENCE = (
    "This routine fleet record discusses vehicle schedules, "
    "ordinary maintenance, driver assignments, and deliveries."
)

POSITIONS = [
    "beginning",
    "middle",
    "end",
]


def build_filler(
    first_index: int,
    final_index: int,
) -> str:
    """Create numbered irrelevant fleet records."""

    return "\n".join(
        (
            f"Routine record {index}: "
            f"{FILLER_SENTENCE}"
        )
        for index in range(
            first_index,
            final_index + 1,
        )
    )


def build_reference_record() -> str:
    """Create the unique record that contains the code."""

    return (
        "IMPORTANT REFERENCE RECORD\n"
        f"ACCESS_CODE={SECRET_CODE}\n"
        "END IMPORTANT REFERENCE RECORD"
    )


def build_prompt(position: str) -> str:
    """Place the reference record at a selected position."""

    reference_record = build_reference_record()

    if position == "beginning":
        body = (
            f"{reference_record}\n\n"
            f"{build_filler(1, FILLER_REPETITIONS)}"
        )

    elif position == "middle":
        midpoint = FILLER_REPETITIONS // 2

        body = (
            f"{build_filler(1, midpoint)}\n\n"
            f"{reference_record}\n\n"
            f"{build_filler(midpoint + 1, FILLER_REPETITIONS)}"
        )

    elif position == "end":
        body = (
            f"{build_filler(1, FILLER_REPETITIONS)}\n\n"
            f"{reference_record}"
        )

    else:
        raise ValueError(
            f"Unsupported position: {position}"
        )

    return (
        "Read the records below and answer the final question.\n\n"
        f"{body}\n\n"
        "FINAL QUESTION\n"
        "What is the value of ACCESS_CODE in the important "
        "reference record?\n"
        "Return only the exact code. If you cannot find an "
        "ACCESS_CODE record, return UNKNOWN."
    )


def generate_answer(
    prompt: str,
) -> dict[str, Any]:
    """Ask Ollama to extract the code from one prompt."""

    request_body = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 20,
            "num_ctx": CONTEXT_SIZE,
        },
    }

    try:
        response = httpx.post(
            OLLAMA_GENERATE_URL,
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
            "Ollama did not respond before the timeout."
        ) from error
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "Ollama returned HTTP status "
            f"{error.response.status_code}: "
            f"{error.response.text}"
        ) from error

    response_body = response.json()

    answer = response_body.get("response")
    prompt_token_count = response_body.get(
        "prompt_eval_count"
    )
    output_token_count = response_body.get(
        "eval_count"
    )

    if not isinstance(answer, str):
        raise RuntimeError(
            "Ollama did not return a response string."
        )

    if not isinstance(prompt_token_count, int):
        raise RuntimeError(
            "Ollama did not return an input-token count."
        )

    if not isinstance(output_token_count, int):
        raise RuntimeError(
            "Ollama did not return an output-token count."
        )

    return {
        "answer": answer.strip(),
        "prompt_token_count": prompt_token_count,
        "output_token_count": output_token_count,
    }


def main() -> None:
    """Compare extraction at three prompt positions."""

    print("CONTEXT POSITION EXPERIMENT")
    print("===========================")
    print()
    print("Model:", MODEL_NAME)
    print("Context size:", CONTEXT_SIZE)
    print("Expected code:", SECRET_CODE)
    print("Filler records:", FILLER_REPETITIONS)
    print()

    results: dict[str, dict[str, Any]] = {}

    for position in POSITIONS:
        prompt = build_prompt(position)

        print(
            f"Testing code at the {position}..."
        )

        result = generate_answer(prompt)
        result["prompt_character_count"] = len(prompt)

        results[position] = result

        print("Completed.")
        print()

    print("RESULTS")
    print("-------")

    for position in POSITIONS:
        result = results[position]

        answer = result["answer"]
        prompt_token_count = result[
            "prompt_token_count"
        ]
        output_token_count = result[
            "output_token_count"
        ]
        prompt_character_count = result[
            "prompt_character_count"
        ]

        code_was_recalled = (
            SECRET_CODE.casefold()
            in answer.casefold()
        )

        print(
            f"Code position: {position}"
        )
        print(
            "Prompt characters:",
            prompt_character_count,
        )
        print(
            "Input tokens processed:",
            prompt_token_count,
        )
        print(
            "Output tokens generated:",
            output_token_count,
        )
        print(
            "Model answer:",
            repr(answer),
        )
        print(
            "Correct code recalled:",
            code_was_recalled,
        )
        print()

    print("INTERPRETATION")
    print("--------------")
    print(
        "All requests use the same context capacity."
    )
    print(
    "Compare the outcomes before deciding whether "
    "position affected retrieval."
    )
    print(
        "Information fitting inside the context window "
        "does not guarantee successful retrieval."
    )


if __name__ == "__main__":
    main()