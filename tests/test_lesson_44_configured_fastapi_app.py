from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from lessons import lesson_40_fastapi_guarded_rag_api as legacy_api
from lessons import lesson_43_configured_fastapi_app as configured
from lessons.lesson_42_application_settings import (
    AppSettings,
)


class FakeGuardedRAGService:
    """Controlled replacement for the real guarded-RAG service."""

    last_instance: FakeGuardedRAGService | None = None

    last_factory_arguments: dict[
        str,
        Any,
    ] | None = None

    def __init__(
        self,
        *,
        collection_name: str,
        top_k: int,
    ) -> None:
        self._collection_name = collection_name
        self._top_k = top_k
        self._closed = False

        self.validate_calls = 0
        self.questions: list[str] = []

        self.answer_result: Any | None = None
        self.answer_error: Exception | None = None

    @classmethod
    def from_local_qdrant(
        cls,
        *,
        index_path: Path,
        qdrant_path: Path,
        collection_name: str,
        top_k: int,
    ) -> FakeGuardedRAGService:
        """Create a fake service and record factory arguments."""

        cls.last_factory_arguments = {
            "index_path": index_path,
            "qdrant_path": qdrant_path,
            "collection_name": collection_name,
            "top_k": top_k,
        }

        instance = cls(
            collection_name=collection_name,
            top_k=top_k,
        )

        cls.last_instance = instance

        return instance

    @property
    def is_closed(self) -> bool:
        """Report whether the fake service is closed."""

        return self._closed

    @property
    def source_chunk_count(self) -> int:
        """Return a fixed source-index size."""

        return 9

    @property
    def collection_name(self) -> str:
        """Return the configured collection name."""

        return self._collection_name

    @property
    def top_k(self) -> int:
        """Return the configured retrieval limit."""

        return self._top_k

    def validate(self) -> int:
        """Return a fixed Qdrant point count."""

        if self._closed:
            raise RuntimeError(
                "The fake service is closed."
            )

        self.validate_calls += 1

        return 9

    def answer(
        self,
        question: str,
    ) -> Any:
        """Return a configured result or raise an error."""

        if self._closed:
            raise RuntimeError(
                "The fake service is closed."
            )

        self.questions.append(
            question
        )

        if self.answer_error is not None:
            raise self.answer_error

        if self.answer_result is None:
            raise RuntimeError(
                "The test did not configure an answer."
            )

        return self.answer_result

    def close(self) -> None:
        """Close the fake service."""

        self._closed = True


def make_settings(
    temporary_directory: Path,
) -> AppSettings:
    """Create explicit settings for an isolated API test."""

    return AppSettings(
        api_title="Configured Fleet Test API",
        api_version="2.5.0",
        api_host="127.0.0.1",
        api_port=8100,
        vector_index_path=(
            temporary_directory
            / "test_vector_index.json"
        ),
        qdrant_path=(
            temporary_directory
            / "test_qdrant"
        ),
        qdrant_collection=(
            "configured_test_collection"
        ),
        retrieval_top_k=5,
        ollama_base_url=(
            "http://127.0.0.1:11434"
        ),
        generation_model="llama3.2:3b",
        embedding_model="embeddinggemma",
        ollama_timeout_seconds=90.0,
        _env_file=None,
    )


def make_answer_result(
    question: str,
) -> SimpleNamespace:
    """Create a supported result for response serialization."""

    direct_chunk = SimpleNamespace(
        chunk_id="section-002-chunk-001",
        section_id="section-002",
        section_title=(
            "Engine Warning Procedures"
        ),
    )

    expanded_chunk = SimpleNamespace(
        chunk_id="section-002-chunk-002",
        section_id="section-002",
        section_title=(
            "Engine Warning Procedures"
        ),
    )

    return SimpleNamespace(
        question=question,
        decision="ANSWER",
        answer_strategy="grounded generation",
        answer=(
            "The driver must park the vehicle "
            "safely [S1]."
        ),
        top_similarity=0.70,
        threshold=0.5413,
        generation_called=True,
        citation_repair_used=False,
        extractive_fallback_used=False,
        answer_passed_validation=True,
        generation_input_tokens=100,
        generation_output_tokens=15,
        retrieval_results=[
            SimpleNamespace(
                score=0.70,
                chunk=direct_chunk,
            )
        ],
        context_sources=[
            SimpleNamespace(
                label="S1",
                chunk=direct_chunk,
                retrieval_score=0.70,
                inclusion_reason=(
                    "vector retrieval"
                ),
            ),
            SimpleNamespace(
                label="S2",
                chunk=expanded_chunk,
                retrieval_score=None,
                inclusion_reason=(
                    "adjacent-chunk expansion"
                ),
            ),
        ],
        citation_checks=[
            SimpleNamespace(
                sentence_number=1,
                cited_labels=["S1"],
                invalid_labels=[],
                missing_citation=False,
                lexical_coverage=1.0,
                missing_support_terms=[],
                support_passed=True,
            )
        ],
    )


def install_fake_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Replace the service in both modules that reference it.

    Lesson 43 creates the service, while the imported
    Lesson 40 helper validates its type.
    """

    FakeGuardedRAGService.last_instance = None
    FakeGuardedRAGService.last_factory_arguments = None

    monkeypatch.setattr(
        configured,
        "GuardedRAGService",
        FakeGuardedRAGService,
    )

    monkeypatch.setattr(
        legacy_api,
        "GuardedRAGService",
        FakeGuardedRAGService,
    )


def require_fake_service() -> FakeGuardedRAGService:
    """Return the fake service created during lifespan startup."""

    service = FakeGuardedRAGService.last_instance

    if service is None:
        raise AssertionError(
            "The application lifespan did not create "
            "the fake service."
        )

    return service


@pytest.fixture
def configured_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[
    tuple[
        TestClient,
        AppSettings,
    ]
]:
    """Create a settings-driven application and run its lifespan."""

    install_fake_service(
        monkeypatch
    )

    settings = make_settings(
        tmp_path
    )

    application = configured.create_app(
        settings
    )

    with TestClient(
        application
    ) as client:
        yield (
            client,
            settings,
        )


def test_application_factory_uses_explicit_metadata(
    tmp_path: Path,
) -> None:
    """The application title and version should use settings."""

    settings = make_settings(
        tmp_path
    )

    application = configured.create_app(
        settings
    )

    assert (
        application.title
        == "Configured Fleet Test API"
    )

    assert application.version == "2.5.0"


def test_lifespan_passes_settings_to_service_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Startup should use settings and shutdown should clean up."""

    install_fake_service(
        monkeypatch
    )

    settings = make_settings(
        tmp_path
    )

    application = configured.create_app(
        settings
    )

    assert (
        FakeGuardedRAGService.last_instance
        is None
    )

    with TestClient(
        application
    ):
        service = require_fake_service()

        assert (
            FakeGuardedRAGService.last_factory_arguments
            == {
                "index_path": (
                    settings.vector_index_path
                ),
                "qdrant_path": (
                    settings.qdrant_path
                ),
                "collection_name": (
                    settings.qdrant_collection
                ),
                "top_k": (
                    settings.retrieval_top_k
                ),
            }
        )

        assert service.validate_calls == 1
        assert service.is_closed is False

    assert service.is_closed is True


def test_configuration_endpoint_reports_active_settings(
    configured_client: tuple[
        TestClient,
        AppSettings,
    ],
) -> None:
    """The configuration endpoint should expose active values."""

    client, settings = configured_client

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
    }


def test_health_endpoint_reports_configured_service(
    configured_client: tuple[
        TestClient,
        AppSettings,
    ],
) -> None:
    """Health should report the configured collection and top K."""

    client, settings = configured_client

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "service_closed": False,
        "source_chunk_count": 9,
        "stored_point_count": 9,
        "collection_name": (
            settings.qdrant_collection
        ),
        "retrieval_top_k": (
            settings.retrieval_top_k
        ),
    }


def test_root_endpoint_uses_configured_discovery_values(
    configured_client: tuple[
        TestClient,
        AppSettings,
    ],
) -> None:
    """The root endpoint should advertise configured values."""

    client, settings = configured_client

    response = client.get(
        "/"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["name"] == settings.api_title
    assert payload["version"] == settings.api_version

    assert (
        payload["configuration_endpoint"]
        == "/config"
    )

    assert (
        payload["collection"]
        == settings.qdrant_collection
    )

    assert (
        payload["retrieval_top_k"]
        == settings.retrieval_top_k
    )


def test_answer_endpoint_uses_configured_service(
    configured_client: tuple[
        TestClient,
        AppSettings,
    ],
) -> None:
    """The configured application should return guarded answers."""

    client, _settings = configured_client

    service = require_fake_service()

    question = (
        "What should a driver do when an engine "
        "warning light appears?"
    )

    service.answer_result = (
        make_answer_result(
            question
        )
    )

    response = client.post(
        "/rag/answer",
        json={
            "question": question,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["decision"] == "ANSWER"

    assert (
        payload["validation_passed"]
        is True
    )

    assert (
        payload["retrieval_results"][0][
            "similarity"
        ]
        == pytest.approx(0.70)
    )

    assert (
        payload["answer_sources"][1][
            "retrieval_score"
        ]
        is None
    )

    assert service.questions == [
        question
    ]


def test_invalid_settings_are_rejected() -> None:
    """Pydantic should reject invalid ports and top-K values."""

    with pytest.raises(
        ValidationError
    ) as captured_error:
        AppSettings(
            api_port=70000,
            retrieval_top_k=0,
            _env_file=None,
        )

    error_message = str(
        captured_error.value
    )

    assert "api_port" in error_message
    assert "retrieval_top_k" in error_message