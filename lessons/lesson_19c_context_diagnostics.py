from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

EXPECTED_VALUE = "MADEIRA-7319"
CONTEXT_SIZE = 4096
FILLER_RECORDS = 100

SYSTEM_MESSAGE = (
    "You are a precise information-extraction system. "
    "Copy the requested field value exactly from the user's "
    "records. Return only the value, with no explanation, "
    "quotation marks, or additional punctuation."
)

FILLER_SENTENCE = (
    "This routine fleet record contains ordinary information "
    "about vehicle schedules, maintenance, drivers, and deliveries."
)


@dataclass(frozen=True)
class DiagnosticCase:
    """One controlled information-extraction test."""

    name: str
    purpose: str
    prompt: str


def build_filler(record_count: int) -> str:
    """Create numbered irrelevant records."""

    return "\n".join(
        (
            f"Routine record {index}: "
            f"{FILLER_SENTENCE}"
        )
        for index in range(
            1,
            record_count + 1,
        )
    )


def build_diagnostic_cases() -> list[DiagnosticCase]:
    """Build four prompts that isolate different factors."""

    filler = build_filler(FILLER_RECORDS)

    minimal_access_prompt = (
        "Read the following record:\n\n"
        f"ACCESS_CODE={EXPECTED_VALUE}\n\n"
        "What is the value of ACCESS_CODE?\n"
        "Return only the exact value."
    )

    minimal_reference_prompt = (
        "Read the following record:\n\n"
        f"REFERENCE_ID={EXPECTED_VALUE}\n\n"
        "What is the value of REFERENCE_ID?\n"
        "Return only the exact value."
    )

    long_reference_at_end_prompt = (
        "Read all records and answer the final question.\n\n"
        f"{filler}\n\n"
        "IMPORTANT REFERENCE RECORD\n"
        f"REFERENCE_ID={EXPECTED_VALUE}\n"
        "END IMPORTANT REFERENCE RECORD\n\n"
        "FINAL QUESTION\n"
        "What is the value of REFERENCE_ID?\n"
        "Return only the exact value."
    )

    long_reference_at_beginning_prompt = (
        "Read all records and answer the final question.\n\n"
        "IMPORTANT REFERENCE RECORD\n"
        f"REFERENCE_ID={EXPECTED_VALUE}\n"
        "END IMPORTANT REFERENCE RECORD\n\n"
        f"{filler}\n\n"
        "FINAL QUESTION\n"
        "What is the value of REFERENCE_ID?\n"
        "Return only the exact value."
    )

    return [
        DiagnosticCase(
            name="MINIMAL ACCESS CODE",
            purpose=(
                "Tests whether the original ACCESS_CODE "
                "wording works without filler."
            ),
            prompt=minimal_access_prompt,
        ),
        DiagnosticCase(
            name="MINIMAL REFERENCE ID",
            purpose=(
                "Tests equivalent extraction using neutral "
                "REFERENCE_ID wording."
            ),
            prompt=minimal_reference_prompt,
        ),
        DiagnosticCase(
            name="LONG REFERENCE AT END",
            purpose=(
                "Tests long-context extraction when the value "
                "is immediately before the question."
            ),
            prompt=long_reference_at_end_prompt,
        ),
        DiagnosticCase(
            name="LONG REFERENCE AT BEGINNING",
            purpose=(
                "Tests long-context extraction when the value "
                "is far from the final question."
            ),
            prompt=long_reference_at_beginning_prompt,
        ),
    ]


def extract_value(
    prompt: str,
) -> dict[str, Any]:
    """Ask Ollama to extract a field value."""

    request_body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
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
            OLLAMA_CHAT_URL,
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

    message = response_body.get("message")

    if not isinstance(message, dict):
        raise RuntimeError(
            "Ollama did not return a message object."
        )

    answer = message.get("content")
    prompt_token_count = response_body.get(
        "prompt_eval_count"
    )
    output_token_count = response_body.get(
        "eval_count"
    )

    if not isinstance(answer, str):
        raise RuntimeError(
            "Ollama did not return an answer string."
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


def normalize_answer(answer: str) -> str:
    """Remove simple wrappers around an extracted value."""

    return answer.strip().strip(
        " \t\r\n`'\".,:;"
    )


def main() -> None:
    """Run controlled extraction diagnostics."""

    cases = build_diagnostic_cases()

    print("CONTEXT EXTRACTION DIAGNOSTICS")
    print("==============================")
    print()
    print("Model:", MODEL_NAME)
    print("Allocated context:", CONTEXT_SIZE)
    print("Expected value:", EXPECTED_VALUE)
    print("Long-prompt filler records:", FILLER_RECORDS)
    print()

    results: dict[str, dict[str, Any]] = {}

    for case in cases:
        print(f"Running: {case.name}")
        print("Purpose:", case.purpose)

        result = extract_value(case.prompt)
        result["prompt_character_count"] = len(case.prompt)

        results[case.name] = result

        print("Completed.")
        print()

    print("RESULTS")
    print("-------")

    for case in cases:
        result = results[case.name]

        answer = result["answer"]
        normalized_answer = normalize_answer(answer)

        exact_match = (
            normalized_answer.casefold()
            == EXPECTED_VALUE.casefold()
        )

        contains_expected_value = (
            EXPECTED_VALUE.casefold()
            in answer.casefold()
        )

        print(case.name)
        print("-" * len(case.name))
        print(
            "Prompt characters:",
            result["prompt_character_count"],
        )
        print(
            "Input tokens processed:",
            result["prompt_token_count"],
        )
        print(
            "Output tokens generated:",
            result["output_token_count"],
        )
        print("Raw answer:", repr(answer))
        print(
            "Normalized answer:",
            repr(normalized_answer),
        )
        print(
            "Contains expected value:",
            contains_expected_value,
        )
        print(
            "Exact normalized match:",
            exact_match,
        )
        print()

    print("HOW TO INTERPRET THE PATTERN")
    print("----------------------------")
    print(
        "If both minimal tests fail, investigate the "
        "basic extraction prompt or model behavior."
    )
    print(
        "If ACCESS_CODE fails but REFERENCE_ID passes, "
        "the terminology affected the response."
    )
    print(
        "If minimal extraction passes but both long tests "
        "fail, long irrelevant context is the main problem."
    )
    print(
        "If the long end test passes but the long beginning "
        "test fails, distance or position affected retrieval."
    )
    print(
        "If all tests pass, the earlier UNKNOWN fallback "
        "likely encouraged the previous failures."
    )


if __name__ == "__main__":
    main()