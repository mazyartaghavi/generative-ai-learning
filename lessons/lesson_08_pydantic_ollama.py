from __future__ import annotations

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"


class ApiMapping(BaseModel):
    """Represent one restaurant-to-software mapping."""

    model_config = ConfigDict(extra="forbid")

    restaurant_term: str = Field(min_length=1)
    software_term: str = Field(min_length=1)
    explanation: str = Field(
        min_length=10,
        max_length=250,
    )


class ApiExplanation(BaseModel):
    """Represent the complete structured API explanation."""

    model_config = ConfigDict(extra="forbid")

    mappings: list[ApiMapping] = Field(
        min_length=6,
        max_length=6,
    )
    definition: str = Field(
        min_length=20,
        max_length=300,
    )


EXPECTED_PAIRS = {
    ("customer", "client application"),
    ("waiter", "api"),
    ("menu", "api documentation"),
    ("order", "request"),
    ("kitchen", "backend server"),
    ("prepared dish", "response"),
}


PROMPT = """
Explain an API using a restaurant analogy.

Return these six exact mappings:
1. customer = client application
2. waiter = API
3. menu = API documentation
4. order = request
5. kitchen = backend server
6. prepared dish = response

For each mapping, provide one concise explanation.
Also provide one concise technical definition of an API.
Do not return additional mappings.
""".strip()


def request_structured_explanation() -> ApiExplanation:
    """Request and validate structured output from Ollama."""

    response_schema = ApiExplanation.model_json_schema()

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
                "content": PROMPT,
            },
        ],
        "format": response_schema,
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
            "Could not connect to Ollama. "
            "Confirm that the Ollama application is running."
        ) from error
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Ollama did not respond before the timeout."
        ) from error
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "Ollama returned HTTP status "
            f"{error.response.status_code}."
        ) from error

    response_body = response.json()

    message = response_body.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(
            "Ollama returned an invalid message structure."
        )

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(
            "Ollama did not return textual JSON content."
        )

    try:
        return ApiExplanation.model_validate_json(content)
    except ValidationError as error:
        raise RuntimeError(
            "Ollama returned JSON that did not satisfy "
            "the Pydantic model."
        ) from error


def normalize_text(text: str) -> str:
    """Normalize text for case-insensitive comparison."""

    return " ".join(text.lower().split())


def get_actual_pairs(
    explanation: ApiExplanation,
) -> list[tuple[str, str]]:
    """Extract normalized mapping pairs from the validated object."""

    actual_pairs: list[tuple[str, str]] = []

    for mapping in explanation.mappings:
        pair = (
            normalize_text(mapping.restaurant_term),
            normalize_text(mapping.software_term),
        )
        actual_pairs.append(pair)

    return actual_pairs


def validate_semantics(
    explanation: ApiExplanation,
) -> list[str]:
    """Check whether the exact expected mappings were returned."""

    errors: list[str] = []
    actual_pairs = get_actual_pairs(explanation)
    actual_pair_set = set(actual_pairs)

    missing_pairs = EXPECTED_PAIRS - actual_pair_set
    unexpected_pairs = actual_pair_set - EXPECTED_PAIRS

    for restaurant_term, software_term in sorted(missing_pairs):
        errors.append(
            "Missing mapping: "
            f"{restaurant_term} = {software_term}"
        )

    for restaurant_term, software_term in sorted(unexpected_pairs):
        errors.append(
            "Unexpected mapping: "
            f"{restaurant_term} = {software_term}"
        )

    if len(actual_pairs) != len(actual_pair_set):
        errors.append("A duplicate mapping was returned.")

    return errors


def print_mappings(explanation: ApiExplanation) -> None:
    """Print the validated mappings in a readable format."""

    print("VALIDATED MAPPINGS")
    print("------------------")

    for number, mapping in enumerate(
        explanation.mappings,
        start=1,
    ):
        print(
            f"{number}. "
            f"{mapping.restaurant_term} = "
            f"{mapping.software_term}"
        )
        print(f"   {mapping.explanation}")
        print()


def main() -> None:
    print("Generating a JSON Schema from Pydantic...")
    print()

    schema = ApiExplanation.model_json_schema()

    print("SCHEMA INFORMATION")
    print("------------------")
    print("Top-level type:", schema.get("type"))
    print("Required fields:", schema.get("required"))
    print()

    print("Requesting structured output from Ollama...")
    print()

    explanation = request_structured_explanation()

    print("VALIDATED PYDANTIC JSON")
    print("-----------------------")
    print(explanation.model_dump_json(indent=2))
    print()

    print_mappings(explanation)

    print("API DEFINITION")
    print("--------------")
    print(explanation.definition)
    print()

    semantic_errors = validate_semantics(explanation)

    print("SEMANTIC VALIDATION")
    print("-------------------")

    if not semantic_errors:
        print("PASS: All six exact mappings are correct.")
    else:
        print(
            f"FAIL: Found {len(semantic_errors)} "
            "semantic validation problem(s)."
        )

        for error in semantic_errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()