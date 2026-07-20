from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lessons.lesson_28_guarded_rag import (
    GuardedRAGResult,
)
from lessons.lesson_42_application_settings import (
    AppSettings,
)
from lessons.lesson_47_application_runtime import (
    ApplicationRuntime,
)
from lessons.lesson_52_runtime_fastapi_app import (
    create_app,
)


class FakeRAGService:
    """Controlled guarded-RAG service for API tests."""

    def __init__(
        self,
        *,
        collection_name: str,
        top_k: int,
        source_chunk_count: int = 9,
        stored_point_count: int = 9,
    ) -> None:
        self._collection_name = collection_name
        self._top_k = top_k
        self._source_chunk_count = source_chunk_count
        self._stored_point_count = stored_point_count

        self._closed = False

        self.validate_calls = 0
        self.answer_calls = 0
        self.close_calls = 0

        self.received_questions: list[str] = []

        self.validation_error: Exception | None = None
        self.answer_error: Exception | None = None

    @property
    def is_closed(self) -> bool:
        """Report whether the fake service is closed."""

        return self._closed

    @property
    def source_chunk_count(self) -> int:
        """Return the configured source-chunk count."""

        return self._source_chunk_count

    @property
    def collection_name(self) -> str:
        """Return the configured collection name."""

        return self._collection_name

    @property
    def top_k(self) -> int:
        """Return the configured retrieval limit."""

        return self._top_k

    @property
    def model_providers_configured(self) -> bool:
        """Report that shared model providers are configured."""

        return True

    def validate(self) -> int:
        """Validate the fake Qdrant state."""

        if self._closed:
            raise RuntimeError(
                "The fake RAG service is closed."
            )

        self.validate_calls += 1

        if self.validation_error is not None:
            raise self.validation_error

        return self._stored_point_count

    def answer(
        self,
        question: str,
    ) -> GuardedRAGResult:
        """Return a controlled guarded-RAG response."""

        if self._closed:
            raise RuntimeError(
                "The fake RAG service is closed."
            )

        self.answer_calls += 1
        self.received_questions.append(
            question
        )

        if self.answer_error is not None:
            raise self.answer_error

        if "insurance" in question.lower():
            return make_result(
                question=question,
                decision="ABSTAIN",
                answer_strategy=(
                    "retrieval abstention"
                ),
                answer=(
                    "The supplied sources do not contain "
                    "enough information."
                ),
                top_similarity=0.4392,
                generation_called=False,
                extractive_fallback_used=False,
                generation_input_tokens=None,
                generation_output_tokens=None,
            )

        return make_result(
            question=question,
            decision="ANSWER",
            answer_strategy="extractive fallback",
            answer=(
                "The driver must park the vehicle safely "
                "and record the warning [S1]."
            ),
            top_similarity=0.6547,
            generation_called=True,
            extractive_fallback_used=True,
            generation_input_tokens=658,
            generation_output_tokens=61,
        )

    def close(self) -> None:
        """Close the fake service."""

        if self._closed:
            return

        self.close_calls += 1
        self._closed = True


class FakeOllamaClient:
    """Controlled Ollama client for API tests."""

    def __init__(
        self,
        *,
        generation_model: str,
        embedding_model: str,
    ) -> None:
        self._generation_model = generation_model
        self._embedding_model = embedding_model
        self._closed = False

        self.health_check_calls = 0
        self.close_calls = 0

    @property
    def is_closed(self) -> bool:
        """Report whether the fake client is closed."""

        return self._closed

    def health_check(self) -> list[str]:
        """Return the configured model names."""

        if self._closed:
            raise RuntimeError(
                "The fake Ollama client is closed."
            )

        self.health_check_calls += 1

        return [
            self._generation_model,
            (
                self._embedding_model
                if ":" in self._embedding_model
                else f"{self._embedding_model}:latest"
            ),
        ]

    def close(self) -> None:
        """Close the fake Ollama client."""

        if self._closed:
            return

        self.close_calls += 1
        self._closed = True


def make_result(
    *,
    question: str,
    decision: str,
    answer_strategy: str,
    answer: str,
    top_similarity: float,
    generation_called: bool,
    extractive_fallback_used: bool,
    generation_input_tokens: int | None,
    generation_output_tokens: int | None,
) -> GuardedRAGResult:
    """Create one serializable guarded-RAG result."""

    return GuardedRAGResult(
        question=question,
        decision=decision,
        answer_strategy=answer_strategy,
        top_similarity=top_similarity,
        threshold=0.5413,
        retrieval_results=[],
        context_sources=[],
        generated_answer=(
            answer
            if generation_called
            else None
        ),
        answer=answer,
        generation_called=generation_called,
        citation_repair_used=(
            extractive_fallback_used
        ),
        extractive_fallback_used=(
            extractive_fallback_used
        ),
        citation_checks=[],
        citation_decisions=[],
        extractive_candidate=None,
        answer_passed_validation=True,
        generation_input_tokens=(
            generation_input_tokens
        ),
        generation_output_tokens=(
            generation_output_tokens
        ),
    )


def make_settings(
    temporary_directory: Path,
) -> AppSettings:
    """Create isolated application settings."""

    return AppSettings(
        api_title="Runtime API Test",
        api_version="2.0.0",
        api_host="127.0.0.1",
        api_port=8123,
        vector_index_path=(
            temporary_directory
            / "vector_index.json"
        ),
        qdrant_path=(
            temporary_directory
            / "qdrant"
        ),
        qdrant_collection=(
            "runtime_api_test_collection"
        ),
        retrieval_top_k=5,
        ollama_base_url=(
            "http://ollama.test:11434"
        ),
        generation_model="llama3.2:3b",
        embedding_model="embeddinggemma",
        ollama_timeout_seconds=20.0,
        _env_file=None,
    )


def make_runtime(
    settings: AppSettings,
) -> tuple[
    ApplicationRuntime,
    FakeRAGService,
    FakeOllamaClient,
]:
    """Create one runtime with controlled dependencies."""

    rag_service = FakeRAGService(
        collection_name=(
            settings.qdrant_collection
        ),
        top_k=settings.retrieval_top_k,
    )

    ollama_client = FakeOllamaClient(
        generation_model=(
            settings.generation_model
        ),
        embedding_model=(
            settings.embedding_model
        ),
    )

    runtime = ApplicationRuntime(
        settings=settings,
        rag_service=rag_service,  # type: ignore[arg-type]
        ollama_client=ollama_client,  # type: ignore[arg-type]
        owns_rag_service=True,
        owns_ollama_client=True,
    )

    return (
        runtime,
        rag_service,
        ollama_client,
    )


def make_runtime_factory(
    runtime: ApplicationRuntime,
    received_settings: list[AppSettings],
) -> Any:
    """Create a recording runtime factory."""

    def factory(
        settings: AppSettings,
    ) -> ApplicationRuntime:
        received_settings.append(
            settings
        )

        return runtime

    return factory


def test_application_metadata_and_discovery(
    tmp_path: Path,
) -> None:
    """The app factory should preserve configured metadata."""

    settings = make_settings(
        tmp_path
    )

    runtime, _, _ = make_runtime(
        settings
    )

    app = create_app(
        settings,
        runtime_factory=(
            make_runtime_factory(
                runtime,
                [],
            )
        ),
    )

    assert app.title == settings.api_title
    assert app.version == settings.api_version

    with TestClient(app) as client:
        response = client.get(
            "/"
        )

    assert response.status_code == 200

    assert response.json() == {
        "service": settings.api_title,
        "version": settings.api_version,
        "health_endpoint": "/health",
        "configuration_endpoint": "/config",
        "answer_endpoint": "/rag/answer",
        "documentation_endpoint": "/docs",
    }


def test_lifespan_validates_and_closes_one_runtime(
    tmp_path: Path,
) -> None:
    """Startup should validate and shutdown should close."""

    settings = make_settings(
        tmp_path
    )

    (
        runtime,
        rag_service,
        ollama_client,
    ) = make_runtime(
        settings
    )

    received_settings: list[
        AppSettings
    ] = []

    app = create_app(
        settings,
        runtime_factory=(
            make_runtime_factory(
                runtime,
                received_settings,
            )
        ),
    )

    with TestClient(app) as client:
        response = client.get(
            "/health"
        )

        assert runtime.is_closed is False
        assert rag_service.is_closed is False
        assert ollama_client.is_closed is False

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "api_title": settings.api_title,
        "api_version": settings.api_version,
        "qdrant_collection": (
            settings.qdrant_collection
        ),
        "source_chunk_count": 9,
        "stored_point_count": 9,
        "retrieval_top_k": (
            settings.retrieval_top_k
        ),
        "generation_model": (
            settings.generation_model
        ),
        "embedding_model": (
            settings.embedding_model
        ),
        "shared_model_providers_configured": True,
    }

    assert received_settings == [
        settings
    ]

    assert rag_service.validate_calls == 1
    assert ollama_client.health_check_calls == 1

    assert runtime.is_closed is True
    assert rag_service.is_closed is True
    assert ollama_client.is_closed is True

    assert rag_service.close_calls == 1
    assert ollama_client.close_calls == 1


def test_configuration_endpoint_reports_settings(
    tmp_path: Path,
) -> None:
    """The configuration endpoint should expose safe settings."""

    settings = make_settings(
        tmp_path
    )

    runtime, _, _ = make_runtime(
        settings
    )

    app = create_app(
        settings,
        runtime_factory=(
            make_runtime_factory(
                runtime,
                [],
            )
        ),
    )

    with TestClient(app) as client:
        response = client.get(
            "/config"
        )

    assert response.status_code == 200

    assert response.json() == {
        "api_title": settings.api_title,
        "api_version": settings.api_version,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "vector_index_path": str(
            settings.vector_index_path
        ),
        "qdrant_path": str(
            settings.qdrant_path
        ),
        "qdrant_collection": (
            settings.qdrant_collection
        ),
        "retrieval_top_k": (
            settings.retrieval_top_k
        ),
        "ollama_base_url": (
            settings.ollama_base_url
        ),
        "generation_model": (
            settings.generation_model
        ),
        "embedding_model": (
            settings.embedding_model
        ),
        "ollama_timeout_seconds": (
            settings.ollama_timeout_seconds
        ),
    }


def test_supported_answer_is_serialized(
    tmp_path: Path,
) -> None:
    """A supported answer should preserve guarded metadata."""

    settings = make_settings(
        tmp_path
    )

    runtime, rag_service, _ = make_runtime(
        settings
    )

    app = create_app(
        settings,
        runtime_factory=(
            make_runtime_factory(
                runtime,
                [],
            )
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/rag/answer",
            json={
                "question": (
                    "  What should a driver do when "
                    "an engine warning appears?  "
                )
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["decision"] == "ANSWER"

    assert (
        payload["answer_strategy"]
        == "extractive fallback"
    )

    assert payload["generation_called"] is True

    assert (
        payload["extractive_fallback_used"]
        is True
    )

    assert (
        payload["answer_passed_validation"]
        is True
    )

    assert payload["generation_input_tokens"] == 658
    assert payload["generation_output_tokens"] == 61
    assert payload["sources"] == []

    assert rag_service.received_questions == [
        (
            "What should a driver do when "
            "an engine warning appears?"
        )
    ]


def test_unsupported_answer_abstains_without_generation(
    tmp_path: Path,
) -> None:
    """Unsupported questions should expose safe abstention."""

    settings = make_settings(
        tmp_path
    )

    runtime, _, _ = make_runtime(
        settings
    )

    app = create_app(
        settings,
        runtime_factory=(
            make_runtime_factory(
                runtime,
                [],
            )
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/rag/answer",
            json={
                "question": (
                    "What is the fleet insurance "
                    "policy number?"
                )
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["decision"] == "ABSTAIN"

    assert (
        payload["answer_strategy"]
        == "retrieval abstention"
    )

    assert payload["generation_called"] is False
    assert payload["generation_input_tokens"] is None
    assert payload["generation_output_tokens"] is None
    assert payload["top_similarity"] == 0.4392


def test_multiple_requests_share_one_runtime(
    tmp_path: Path,
) -> None:
    """All requests should reuse the startup runtime."""

    settings = make_settings(
        tmp_path
    )

    runtime, rag_service, _ = make_runtime(
        settings
    )

    received_settings: list[
        AppSettings
    ] = []

    app = create_app(
        settings,
        runtime_factory=(
            make_runtime_factory(
                runtime,
                received_settings,
            )
        ),
    )

    with TestClient(app) as client:
        first_response = client.post(
            "/rag/answer",
            json={
                "question": (
                    "What should a driver do when "
                    "an engine warning appears?"
                )
            },
        )

        second_response = client.post(
            "/rag/answer",
            json={
                "question": (
                    "What is the fleet insurance "
                    "policy number?"
                )
            },
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert received_settings == [
        settings
    ]

    assert rag_service.validate_calls == 1
    assert rag_service.answer_calls == 2


@pytest.mark.parametrize(
    "request_body",
    [
        {},
        {
            "question": "",
        },
        {
            "question": "   ",
        },
        {
            "question": "valid",
            "unexpected": "field",
        },
    ],
)
def test_invalid_request_bodies_are_rejected(
    tmp_path: Path,
    request_body: dict[str, object],
) -> None:
    """Pydantic should reject malformed requests."""

    settings = make_settings(
        tmp_path
    )

    runtime, rag_service, _ = make_runtime(
        settings
    )

    app = create_app(
        settings,
        runtime_factory=(
            make_runtime_factory(
                runtime,
                [],
            )
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/rag/answer",
            json=request_body,
        )

    assert response.status_code == 422
    assert rag_service.answer_calls == 0


def test_service_runtime_failure_becomes_http_503(
    tmp_path: Path,
) -> None:
    """Runtime service failures should become HTTP 503."""

    settings = make_settings(
        tmp_path
    )

    runtime, rag_service, _ = make_runtime(
        settings
    )

    rag_service.answer_error = RuntimeError(
        "Simulated guarded-RAG failure."
    )

    app = create_app(
        settings,
        runtime_factory=(
            make_runtime_factory(
                runtime,
                [],
            )
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/rag/answer",
            json={
                "question": (
                    "What should a driver do?"
                )
            },
        )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "Simulated guarded-RAG failure."
        )
    }


def test_startup_validation_failure_closes_runtime(
    tmp_path: Path,
) -> None:
    """Failed startup validation should clean up resources."""

    settings = make_settings(
        tmp_path
    )

    (
        runtime,
        rag_service,
        ollama_client,
    ) = make_runtime(
        settings
    )

    rag_service.validation_error = RuntimeError(
        "Simulated startup validation failure."
    )

    app = create_app(
        settings,
        runtime_factory=(
            make_runtime_factory(
                runtime,
                [],
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="startup validation failure",
    ):
        with TestClient(app):
            pass

    assert runtime.is_closed is True
    assert rag_service.is_closed is True
    assert ollama_client.is_closed is True

    assert rag_service.close_calls == 1
    assert ollama_client.close_calls == 1
