from __future__ import annotations

import json
from typing import Any

import httpx


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

EXPECTED_PAIRS = [
    ("customer", "client application"),
    ("waiter", "API"),
    ("menu", "API documentation"),
    ("order", "request"),
    ("kitchen", "backend server"),
    ("prepared dish", "response"),
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "mappings": {
            "type": "array",
            "minItems": 6,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "restaurant_term": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "software_term": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "explanation": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "required": [
                    "restaurant_term",
                    "software_term",
                    "explanation",
                ],
                "additionalProperties": False,
            },
        },
        "definition": {
            "type": "string",
            "minLength": 1,
        },
    },
    "required": [
        "mappings",
        "definition",
    ],
    "additionalProperties": False,
}

PROMPT = """
Explain an API using a restaurant analogy.

Return the following six exact mappings:
1. customer = client application
2. waiter = API
3. menu = API documentation
4. order = request
5. kitchen = backend server
6. prepared dish = response

For each mapping, provide one concise explanation.
Also provide one concise technical definition of an API.
Do not add any other mappings.
""".strip()


def ask_ollama_for_json(prompt: str) -> dict[str, Any]:
    """Request structured JSON data from Ollama."""

    request_body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful software-engineering tutor. "
                    "Follow the supplied JSON schema exactly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    try:
        response = httpx.post(
            OLLAMA_CHAT_URL,
            json=request_body,
            timeout=120.0,
        )
        response.raise_for_status()
    except httpx.ConnectError as error:
        raise RuntimeError(
            "Could not connect to Ollama. Confirm that Ollama is running."
        ) from error
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Ollama did not respond before the timeout."
        ) from error
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            f"Ollama returned HTTP status "
            f"{error.response.status_code}."
        ) from error

    response_body = response.json()

    message = response_body.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(
            "Ollama returned an unexpected message structure."
        )

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(
            "Ollama did not return textual JSON content."
        )

    try:
        parsed_data = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The model response was not valid JSON."
        ) from error

    if not isinstance(parsed_data, dict):
        raise RuntimeError(
            "The top-level JSON value was not an object."
        )

    return parsed_data


def normalize_text(value: str) -> str:
    """Normalize text for case-insensitive comparisons."""

    return " ".join(value.lower().split())


def validate_structured_answer(
    data: dict[str, Any],
) -> list[str]:
    """Return a list describing validation errors."""

    errors: list[str] = []

    mappings = data.get("mappings")

    if not isinstance(mappings, list):
        return ["The 'mappings' field is not a list."]

    if len(mappings) != 6:
        errors.append(
            f"Expected 6 mappings but received {len(mappings)}."
        )

    actual_pairs: list[tuple[str, str]] = []

    for index, mapping in enumerate(mappings, start=1):
        if not isinstance(mapping, dict):
            errors.append(
                f"Mapping {index} is not a JSON object."
            )
            continue

        restaurant_term = mapping.get("restaurant_term")
        software_term = mapping.get("software_term")
        explanation = mapping.get("explanation")

        if not isinstance(restaurant_term, str):
            errors.append(
                f"Mapping {index} has an invalid restaurant term."
            )
            continue

        if not isinstance(software_term, str):
            errors.append(
                f"Mapping {index} has an invalid software term."
            )
            continue

        if not isinstance(explanation, str):
            errors.append(
                f"Mapping {index} has an invalid explanation."
            )
            continue

        if not explanation.strip():
            errors.append(
                f"Mapping {index} has an empty explanation."
            )

        actual_pairs.append(
            (
                normalize_text(restaurant_term),
                normalize_text(software_term),
            )
        )

    expected_pairs = {
        (
            normalize_text(restaurant_term),
            normalize_text(software_term),
        )
        for restaurant_term, software_term in EXPECTED_PAIRS
    }

    actual_pair_set = set(actual_pairs)

    missing_pairs = expected_pairs - actual_pair_set
    unexpected_pairs = actual_pair_set - expected_pairs

    for restaurant_term, software_term in sorted(missing_pairs):
        errors.append(
            f"Missing mapping: "
            f"{restaurant_term} = {software_term}"
        )

    for restaurant_term, software_term in sorted(unexpected_pairs):
        errors.append(
            f"Unexpected mapping: "
            f"{restaurant_term} = {software_term}"
        )

    if len(actual_pairs) != len(actual_pair_set):
        errors.append("The response contains a duplicate mapping.")

    definition = data.get("definition")

    if not isinstance(definition, str):
        errors.append(
            "The 'definition' field is not a string."
        )
    elif not definition.strip():
        errors.append(
            "The API definition is empty."
        )

    return errors


def main() -> None:
    print("Requesting structured JSON from Llama...")
    print()

    structured_answer = ask_ollama_for_json(PROMPT)

    print("PARSED JSON RESPONSE")
    print("--------------------")
    print(
        json.dumps(
            structured_answer,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()

    validation_errors = validate_structured_answer(
        structured_answer
    )

    print("SEMANTIC VALIDATION")
    print("-------------------")

    if not validation_errors:
        print("PASS: All structured mappings are correct.")
    else:
        print(
            f"FAIL: Found {len(validation_errors)} "
            "validation problem(s)."
        )

        for error in validation_errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()