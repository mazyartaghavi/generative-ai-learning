from __future__ import annotations

from typing import Any

import httpx


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

SECRET_CODE = "MADEIRA-7319"

FILLER_SENTENCE = (
    "The fleet report contains routine operational notes "
    "about vehicles, schedules, maintenance, and deliveries."
)

FILLER_REPETITIONS = 100

CONTEXT_SIZES = [
    512,
    2048,
    4096,
]


def build_long_prompt() -> str:
    """Build a prompt with important information near the beginning."""

    filler_text = "\n".join(
        f"{index}. {FILLER_SENTENCE}"
        for index in range(
            1,
            FILLER_REPETITIONS + 1,
        )
    )

    return (
        f"ACCESS CODE: {SECRET_CODE}\n"
        "Remember the access code written above.\n\n"
        f"{filler_text}\n\n"
        "QUESTION:\n"
        "What was the access code at the very beginning?\n"
        "Return only the code. If it is unavailable in your "
        "current context, return UNKNOWN."
    )


def generate_answer(
    prompt: str,
    context_size: int,
) -> dict[str, Any]:
    """Generate an answer using a specified context size."""

    request_body = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 20,
            "num_ctx": context_size,
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
            "Ollama did not return a valid response string."
        )

    if not isinstance(prompt_token_count, int):
        raise RuntimeError(
            "Ollama did not return a valid input-token count."
        )

    if not isinstance(output_token_count, int):
        raise RuntimeError(
            "Ollama did not return a valid output-token count."
        )

    return {
        "answer": answer.strip(),
        "prompt_token_count": prompt_token_count,
        "output_token_count": output_token_count,
    }


def main() -> None:
    """Compare two allocated context-window sizes."""

    prompt = build_long_prompt()

    print("CONTEXT-WINDOW EXPERIMENT")
    print("=========================")
    print()
    print("Important code:", SECRET_CODE)
    print(
        "Filler repetitions:",
        FILLER_REPETITIONS,
    )
    print(
        "Prompt character count:",
        len(prompt),
    )
    print()

    results: dict[int, dict[str, Any]] = {}

    for context_size in CONTEXT_SIZES:
        print(
            f"Sending request with num_ctx={context_size}..."
        )

        result = generate_answer(
            prompt=prompt,
            context_size=context_size,
        )

        results[context_size] = result

        print("Completed.")
        print()

    print("RESULTS")
    print("-------")

    for context_size, result in results.items():
        answer = result["answer"]
        prompt_token_count = result[
            "prompt_token_count"
        ]
        output_token_count = result[
            "output_token_count"
        ]

        code_was_recalled = (
            SECRET_CODE.casefold()
            in answer.casefold()
        )

        print(f"Allocated context: {context_size}")
        print(
            "Input tokens processed:",
            prompt_token_count,
        )
        print(
            "Output tokens generated:",
            output_token_count,
        )
        print("Model answer:", repr(answer))
        print(
            "Correct code recalled:",
            code_was_recalled,
        )
        print()

    print("INTERPRETATION")
    print("--------------")
    print(
        "A context window limits how much tokenized "
        "information the model can use."
    )
    print(
        "When a prompt exceeds the allocated context, "
        "some information may no longer be available."
    )
    print(
        "RAG systems must select and compress context "
        "rather than inserting unlimited documents."
    )


if __name__ == "__main__":
    main()