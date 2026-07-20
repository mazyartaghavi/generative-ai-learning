from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from lessons.lesson_10_fastapi_ollama_chat import app


client = TestClient(app)


def test_health_endpoint() -> None:
    """Verify that the health endpoint returns service information."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "local-llm-chat-api",
        "model": "llama3.2:3b",
    }


def test_valid_chat_request() -> None:
    """Verify that a valid chat request returns an answer."""

    fake_answer = (
        "Training adjusts model parameters using examples. "
        "Inference uses the trained parameters to make predictions. "
        "Training learns, while inference applies what was learned."
    )

    with patch(
        "lessons.lesson_10_fastapi_ollama_chat.ask_ollama",
        new=AsyncMock(return_value=fake_answer),
    ):
        response = client.post(
            "/chat",
            json={
                "prompt": (
                    "Explain the difference between training "
                    "and inference."
                ),
            },
        )

    response_body = response.json()

    assert response.status_code == 200
    assert response_body["answer"] == fake_answer
    assert response_body["model"] == "llama3.2:3b"
    assert response_body["elapsed_seconds"] >= 0


def test_empty_prompt_is_rejected() -> None:
    """Verify that an empty prompt produces a validation error."""

    response = client.post(
        "/chat",
        json={
            "prompt": "",
        },
    )

    response_body = response.json()
    first_error = response_body["detail"][0]

    assert response.status_code == 422
    assert first_error["type"] == "string_too_short"
    assert first_error["loc"] == ["body", "prompt"]


def test_extra_field_is_rejected() -> None:
    """Verify that undeclared request fields are prohibited."""

    response = client.post(
        "/chat",
        json={
            "prompt": "Explain what an API is.",
            "temperature": 0.5,
        },
    )

    response_body = response.json()
    first_error = response_body["detail"][0]

    assert response.status_code == 422
    assert first_error["type"] == "extra_forbidden"
    assert first_error["loc"] == ["body", "temperature"]