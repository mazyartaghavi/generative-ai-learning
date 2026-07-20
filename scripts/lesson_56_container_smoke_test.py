from __future__ import annotations

import json
import sys
import time
from typing import Any

import httpx


BASE_URL = "http://127.0.0.1:8000"

HEALTH_ATTEMPTS = 45
HEALTH_DELAY_SECONDS = 2.0


def require(
    condition: bool,
    message: str,
) -> None:
    """Raise a clear smoke-test failure."""

    if not condition:
        raise RuntimeError(
            message
        )


def print_payload(
    title: str,
    payload: dict[str, Any],
) -> None:
    """Print one response payload readably."""

    print(title)
    print("-" * len(title))
    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()


def wait_for_health(
    client: httpx.Client,
) -> dict[str, Any]:
    """Wait until the container reports a healthy runtime."""

    last_error: Exception | None = None

    for attempt in range(
        1,
        HEALTH_ATTEMPTS + 1,
    ):
        try:
            response = client.get(
                "/health",
                headers={
                    "X-Request-ID": (
                        "lesson-56-health"
                    )
                },
            )

            if response.status_code == 200:
                payload = response.json()

                if not isinstance(
                    payload,
                    dict,
                ):
                    raise RuntimeError(
                        "Health response was not a JSON object."
                    )

                return payload
        except (
            httpx.HTTPError,
            ValueError,
            RuntimeError,
        ) as error:
            last_error = error

        print(
            "Waiting for container health "
            f"({attempt}/{HEALTH_ATTEMPTS})..."
        )

        time.sleep(
            HEALTH_DELAY_SECONDS
        )

    raise RuntimeError(
        "The container did not become healthy. "
        "Run `docker compose logs api` to inspect startup. "
        f"Last observed error: {last_error}"
    )


def main() -> None:
    """Validate the containerized guarded-RAG API."""

    print("LESSON 56 — CONTAINER SMOKE TEST")
    print("================================")
    print()

    with httpx.Client(
        base_url=BASE_URL,
        timeout=180.0,
    ) as client:
        health = wait_for_health(
            client
        )

        print_payload(
            "HEALTH",
            health,
        )

        require(
            health.get("status") == "ok",
            "The health status was not 'ok'.",
        )

        require(
            health.get(
                "shared_model_providers_configured"
            )
            is True,
            (
                "The shared Ollama providers were "
                "not configured."
            ),
        )

        supported_response = client.post(
            "/rag/answer",
            headers={
                "X-Request-ID": (
                    "lesson-56-supported"
                )
            },
            json={
                "question": (
                    "What should a driver do when an "
                    "engine warning light appears?"
                )
            },
        )

        require(
            supported_response.status_code == 200,
            (
                "Supported request failed with HTTP "
                f"{supported_response.status_code}."
            ),
        )

        require(
            supported_response.headers.get(
                "X-Request-ID"
            )
            == "lesson-56-supported",
            (
                "Supported request ID was not "
                "correlated."
            ),
        )

        supported = supported_response.json()

        require(
            isinstance(
                supported,
                dict,
            ),
            (
                "Supported response was not a "
                "JSON object."
            ),
        )

        print_payload(
            "SUPPORTED QUESTION",
            supported,
        )

        require(
            supported.get("decision") == "ANSWER",
            (
                "Supported question did not produce "
                "an ANSWER decision."
            ),
        )

        require(
            supported.get(
                "answer_passed_validation"
            )
            is True,
            (
                "Supported answer did not pass "
                "validation."
            ),
        )

        unsupported_response = client.post(
            "/rag/answer",
            headers={
                "X-Request-ID": (
                    "lesson-56-unsupported"
                )
            },
            json={
                "question": (
                    "What is the fleet insurance "
                    "policy number?"
                )
            },
        )

        require(
            unsupported_response.status_code == 200,
            (
                "Unsupported request failed with HTTP "
                f"{unsupported_response.status_code}."
            ),
        )

        require(
            unsupported_response.headers.get(
                "X-Request-ID"
            )
            == "lesson-56-unsupported",
            (
                "Unsupported request ID was not "
                "correlated."
            ),
        )

        unsupported = (
            unsupported_response.json()
        )

        require(
            isinstance(
                unsupported,
                dict,
            ),
            (
                "Unsupported response was not a "
                "JSON object."
            ),
        )

        print_payload(
            "UNSUPPORTED QUESTION",
            unsupported,
        )

        require(
            unsupported.get("decision") == "ABSTAIN",
            (
                "Unsupported question did not "
                "abstain."
            ),
        )

        require(
            unsupported.get(
                "answer_strategy"
            )
            == "retrieval abstention",
            (
                "Unsupported question used the "
                "wrong strategy."
            ),
        )

        require(
            unsupported.get(
                "generation_called"
            )
            is False,
            (
                "Unsupported question incorrectly "
                "called generation."
            ),
        )

    print("STATUS")
    print("------")
    print(
        "The Docker container passed health, "
        "supported-answer, abstention, citation, "
        "and request-correlation checks."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"SMOKE TEST FAILED: {error}",
            file=sys.stderr,
        )
        raise
