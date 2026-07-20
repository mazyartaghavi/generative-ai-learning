from __future__ import annotations

import json
from typing import NoReturn

import httpx
import pytest

from lessons.lesson_42_application_settings import (
    AppSettings,
)
from lessons.lesson_45_configured_ollama_client import (
    ConfiguredOllamaClient,
)


def make_settings() -> AppSettings:
    """Create isolated settings for Ollama client tests."""

    return AppSettings(
        ollama_base_url="http://ollama.test:11434",
        generation_model="test-generation-model",
        embedding_model="test-embedding-model",
        ollama_timeout_seconds=15.0,
        _env_file=None,
    )


def decode_request_json(
    request: httpx.Request,
) -> dict[str, object]:
    """Decode a mock HTTP request body as a JSON object."""

    payload = json.loads(
        request.content.decode("utf-8")
    )

    if not isinstance(payload, dict):
        raise AssertionError(
            "The request body was not a JSON object."
        )

    return payload


def test_health_check_returns_available_model_names() -> None:
    """The tags endpoint should return model names."""

    observed_requests: list[
        tuple[str, str]
    ] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        observed_requests.append(
            (
                request.method,
                request.url.path,
            )
        )

        return httpx.Response(
            status_code=200,
            json={
                "models": [
                    {
                        "name": "test-generation-model",
                    },
                    {
                        "name": "test-embedding-model",
                    },
                    {
                        "invalid": "ignored",
                    },
                ]
            },
            request=request,
        )

    settings = make_settings()

    with httpx.Client(
        base_url=settings.ollama_base_url,
        transport=httpx.MockTransport(
            handler
        ),
    ) as http_client:
        with ConfiguredOllamaClient(
            settings,
            http_client=http_client,
        ) as client:
            model_names = client.health_check()

    assert model_names == [
        "test-generation-model",
        "test-embedding-model",
    ]

    assert observed_requests == [
        (
            "GET",
            "/api/tags",
        )
    ]


def test_generate_sends_configured_request_and_parses_result() -> None:
    """Generation should use configured model and options."""

    observed_payload: dict[
        str,
        object,
    ] = {}

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/generate"

        observed_payload.update(
            decode_request_json(
                request
            )
        )

        return httpx.Response(
            status_code=200,
            json={
                "model": "test-generation-model",
                "response": (
                    " Qdrant retrieves relevant chunks. "
                ),
                "prompt_eval_count": 40,
                "eval_count": 9,
                "total_duration": 123456,
            },
            request=request,
        )

    settings = make_settings()

    with httpx.Client(
        base_url=settings.ollama_base_url,
        transport=httpx.MockTransport(
            handler
        ),
    ) as http_client:
        client = ConfiguredOllamaClient(
            settings,
            http_client=http_client,
        )

        result = client.generate(
            "  Explain Qdrant.  ",
            system_prompt=(
                "  Preserve technical names.  "
            ),
            temperature=0.2,
        )

    assert observed_payload == {
        "model": "test-generation-model",
        "prompt": "Explain Qdrant.",
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
        "system": "Preserve technical names.",
    }

    assert result.model == (
        "test-generation-model"
    )

    assert result.response == (
        "Qdrant retrieves relevant chunks."
    )

    assert result.prompt_tokens == 40
    assert result.generated_tokens == 9

    assert (
        result.total_duration_nanoseconds
        == 123456
    )


def test_embed_single_text_uses_string_input() -> None:
    """A single string should remain a string in JSON."""

    observed_payload: dict[
        str,
        object,
    ] = {}

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/embed"

        observed_payload.update(
            decode_request_json(
                request
            )
        )

        return httpx.Response(
            status_code=200,
            json={
                "model": "test-embedding-model",
                "embeddings": [
                    [
                        0.1,
                        0.2,
                        0.3,
                    ]
                ],
                "prompt_eval_count": 5,
                "total_duration": 500,
            },
            request=request,
        )

    settings = make_settings()

    with httpx.Client(
        base_url=settings.ollama_base_url,
        transport=httpx.MockTransport(
            handler
        ),
    ) as http_client:
        client = ConfiguredOllamaClient(
            settings,
            http_client=http_client,
        )

        result = client.embed(
            "  Fleet maintenance policy.  "
        )

    assert observed_payload == {
        "model": "test-embedding-model",
        "input": "Fleet maintenance policy.",
    }

    assert result.model == (
        "test-embedding-model"
    )

    assert result.embedding_count == 1
    assert result.embedding_dimension == 3

    assert result.embeddings == [
        [
            0.1,
            0.2,
            0.3,
        ]
    ]

    assert result.input_tokens == 5

    assert (
        result.total_duration_nanoseconds
        == 500
    )


def test_embed_multiple_texts_returns_matching_vectors() -> None:
    """Batch embedding should preserve input order."""

    observed_payload: dict[
        str,
        object,
    ] = {}

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/embed"

        observed_payload.update(
            decode_request_json(
                request
            )
        )

        return httpx.Response(
            status_code=200,
            json={
                "model": "test-embedding-model",
                "embeddings": [
                    [
                        1,
                        2,
                    ],
                    [
                        3,
                        4,
                    ],
                ],
                "prompt_eval_count": 8,
            },
            request=request,
        )

    settings = make_settings()

    with httpx.Client(
        base_url=settings.ollama_base_url,
        transport=httpx.MockTransport(
            handler
        ),
    ) as http_client:
        client = ConfiguredOllamaClient(
            settings,
            http_client=http_client,
        )

        result = client.embed(
            [
                "First text",
                "Second text",
            ]
        )

    assert observed_payload == {
        "model": "test-embedding-model",
        "input": [
            "First text",
            "Second text",
        ],
    }

    assert result.embedding_count == 2
    assert result.embedding_dimension == 2

    assert result.embeddings == [
        [
            1.0,
            2.0,
        ],
        [
            3.0,
            4.0,
        ],
    ]


def test_invalid_inputs_fail_before_http_request() -> None:
    """Invalid input should fail before an HTTP request."""

    def unexpected_request(
        request: httpx.Request,
    ) -> NoReturn:
        del request

        raise AssertionError(
            "An HTTP request should not have been made."
        )

    settings = make_settings()

    with httpx.Client(
        base_url=settings.ollama_base_url,
        transport=httpx.MockTransport(
            unexpected_request
        ),
    ) as http_client:
        client = ConfiguredOllamaClient(
            settings,
            http_client=http_client,
        )

        with pytest.raises(
            ValueError,
            match="generation prompt",
        ):
            client.generate(
                "   "
            )

        with pytest.raises(
            ValueError,
            match="Temperature",
        ):
            client.generate(
                "Valid prompt",
                temperature=-0.1,
            )

        with pytest.raises(
            ValueError,
            match="system prompt",
        ):
            client.generate(
                "Valid prompt",
                system_prompt="   ",
            )

        with pytest.raises(
            ValueError,
            match="At least one",
        ):
            client.embed(
                []
            )

        with pytest.raises(
            ValueError,
            match="cannot be blank",
        ):
            client.embed(
                [
                    "Valid text",
                    "   ",
                ]
            )

        with pytest.raises(
            TypeError,
            match="must be a string",
        ):
            client.embed(
                [
                    "Valid text",
                    42,  # type: ignore[list-item]
                ]
            )


def test_inconsistent_embedding_dimensions_are_rejected() -> None:
    """Every embedding must have the same dimension."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "model": "test-embedding-model",
                "embeddings": [
                    [
                        0.1,
                        0.2,
                    ],
                    [
                        0.3,
                    ],
                ],
            },
            request=request,
        )

    settings = make_settings()

    with httpx.Client(
        base_url=settings.ollama_base_url,
        transport=httpx.MockTransport(
            handler
        ),
    ) as http_client:
        client = ConfiguredOllamaClient(
            settings,
            http_client=http_client,
        )

        with pytest.raises(
            RuntimeError,
            match="inconsistent dimensions",
        ):
            client.embed(
                [
                    "First text",
                    "Second text",
                ]
            )


def test_http_failure_becomes_runtime_error() -> None:
    """An Ollama HTTP error should be translated clearly."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=500,
            json={
                "error": "model failure",
            },
            request=request,
        )

    settings = make_settings()

    with httpx.Client(
        base_url=settings.ollama_base_url,
        transport=httpx.MockTransport(
            handler
        ),
    ) as http_client:
        client = ConfiguredOllamaClient(
            settings,
            http_client=http_client,
        )

        with pytest.raises(
            RuntimeError,
            match="HTTP status 500",
        ):
            client.generate(
                "Valid prompt"
            )


def test_connection_failure_becomes_runtime_error() -> None:
    """A connection failure should mention the server URL."""

    def handler(
        request: httpx.Request,
    ) -> NoReturn:
        raise httpx.ConnectError(
            "Connection refused.",
            request=request,
        )

    settings = make_settings()

    with httpx.Client(
        base_url=settings.ollama_base_url,
        transport=httpx.MockTransport(
            handler
        ),
    ) as http_client:
        client = ConfiguredOllamaClient(
            settings,
            http_client=http_client,
        )

        with pytest.raises(
            RuntimeError,
            match="Could not connect",
        ) as captured_error:
            client.health_check()

    assert settings.ollama_base_url in str(
        captured_error.value
    )


def test_context_manager_closes_wrapper_not_injected_client() -> None:
    """The wrapper should not close an injected HTTPX client."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "models": [],
            },
            request=request,
        )

    settings = make_settings()

    with httpx.Client(
        base_url=settings.ollama_base_url,
        transport=httpx.MockTransport(
            handler
        ),
    ) as http_client:
        client = ConfiguredOllamaClient(
            settings,
            http_client=http_client,
        )

        with client as active_client:
            assert active_client is client
            assert client.is_closed is False
            assert http_client.is_closed is False

        assert client.is_closed is True
        assert http_client.is_closed is False

        with pytest.raises(
            RuntimeError,
            match="client is closed",
        ):
            client.health_check()

    assert http_client.is_closed is True