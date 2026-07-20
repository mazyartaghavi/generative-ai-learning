from __future__ import annotations

import json
from typing import Any

import httpx


API_BASE_URL = "http://127.0.0.1:8000"


def print_response(
    title: str,
    response: httpx.Response,
) -> None:
    """Print an HTTP response in a readable format."""

    print(title)
    print("-" * len(title))
    print("Status code:", response.status_code)
    print("Reason phrase:", response.reason_phrase)

    try:
        response_body: Any = response.json()
    except ValueError:
        print("Response body is not valid JSON:")
        print(response.text)
    else:
        print("Response body:")
        print(
            json.dumps(
                response_body,
                indent=2,
                ensure_ascii=False,
            )
        )

    print()


def main() -> None:
    """Call the FastAPI service with valid and invalid requests."""

    with httpx.Client(
        base_url=API_BASE_URL,
        timeout=150.0,
    ) as client:
        health_response = client.get("/health")

        print_response(
            title="1. HEALTH CHECK",
            response=health_response,
        )

        valid_chat_response = client.post(
            "/chat",
            json={
                "prompt": (
                    "Explain the difference between model training "
                    "and inference in exactly three simple sentences."
                ),
            },
        )

        print_response(
            title="2. VALID CHAT REQUEST",
            response=valid_chat_response,
        )

        empty_prompt_response = client.post(
            "/chat",
            json={
                "prompt": "",
            },
        )

        print_response(
            title="3. EMPTY PROMPT REQUEST",
            response=empty_prompt_response,
        )

        extra_field_response = client.post(
            "/chat",
            json={
                "prompt": "Explain what an API is.",
                "temperature": 0.5,
            },
        )

        print_response(
            title="4. EXTRA FIELD REQUEST",
            response=extra_field_response,
        )


if __name__ == "__main__":
    main()