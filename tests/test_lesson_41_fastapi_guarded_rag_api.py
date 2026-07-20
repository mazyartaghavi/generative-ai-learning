from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lessons import lesson_40_fastapi_guarded_rag_api as api


class FakeGuardedRAGService:
    """Controlled service replacement for API tests."""

    last_instance: FakeGuardedRAGService | None = None

    def __init__(self) -> None:
        self._closed = False
        self.answer_result: Any | None = None
        self.answer_error: Exception | None = None
        self.questions: list[str] = []

    @classmethod
    def from_local_qdrant(
        cls,
    ) -> FakeGuardedRAGService:
        """Create the fake service used by the API lifespan."""

        instance = cls()
        cls.last_instance = instance

        return instance

    @property
    def is_closed(self) -> bool:
        """Report whether the fake service is closed."""

        return self._closed

    @property
    def source_chunk_count(self) -> int:
        """Return a fixed source-chunk count."""

        return 9

    @property
    def collection_name(self) -> str:
        """Return the expected collection name."""

        return "fleet_manual_chunks"

    @property
    def top_k(self) -> int:
        """Return the configured retrieval limit."""

        return 3

    def validate(self) -> int:
        """Return a fixed stored-point count."""

        if self._closed:
            raise RuntimeError(
                "The fake service is closed."
            )

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
                "The test did not configure an "
                "answer result."
            )

        return self.answer_result

    def close(self) -> None:
        """Mark the fake service as closed."""

        self._closed = True


def make_chunk(
    *,
    chunk_id: str,
    section_id: str,
    section_title: str,
) -> SimpleNamespace:
    """Create the chunk fields used by API serialization."""

    return SimpleNamespace(
        chunk_id=chunk_id,
        section_id=section_id,
        section_title=section_title,
    )


def make_answer_result(
    question: str,
) -> SimpleNamespace:
    """Create one supported guarded-RAG result."""

    direct_chunk = make_chunk(
        chunk_id="section-002-chunk-001",
        section_id="section-002",
        section_title="Engine Warning Procedures",
    )

    adjacent_chunk = make_chunk(
        chunk_id="section-002-chunk-002",
        section_id="section-002",
        section_title="Engine Warning Procedures",
    )

    answer = (
        "The driver must park the vehicle safely and "
        "record the warning [S1]."
    )

    return SimpleNamespace(
        question=question,
        decision="ANSWER",
        answer_strategy=(
            "deterministic citation repair"
        ),
        answer=answer,
        top_similarity=0.6547,
        threshold=0.5413,
        generation_called=True,
        citation_repair_used=True,
        extractive_fallback_used=False,
        answer_passed_validation=True,
        generation_input_tokens=120,
        generation_output_tokens=18,
        retrieval_results=[
            SimpleNamespace(
                score=0.6547,
                chunk=direct_chunk,
            )
        ],
        context_sources=[
            SimpleNamespace(
                label="S1",
                chunk=direct_chunk,
                retrieval_score=0.6547,
                inclusion_reason=(
                    "vector retrieval"
                ),
            ),
            SimpleNamespace(
                label="S2",
                chunk=adjacent_chunk,
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


def make_abstention_result(
    question: str,
) -> SimpleNamespace:
    """Create one retrieval-abstention result."""

    overview_chunk = make_chunk(
        chunk_id="section-001-chunk-001",
        section_id="section-001",
        section_title="Document Overview",
    )

    return SimpleNamespace(
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
        threshold=0.5413,
        generation_called=False,
        citation_repair_used=False,
        extractive_fallback_used=False,
        answer_passed_validation=True,
        generation_input_tokens=None,
        generation_output_tokens=None,
        retrieval_results=[
            SimpleNamespace(
                score=0.4392,
                chunk=overview_chunk,
            )
        ],
        context_sources=[],
        citation_checks=[],
    )


def install_fake_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make the API lifespan create the fake service."""

    FakeGuardedRAGService.last_instance = None

    monkeypatch.setattr(
        api,
        "GuardedRAGService",
        FakeGuardedRAGService,
    )


def require_fake_service() -> FakeGuardedRAGService:
    """Return the service created during API startup."""

    service = (
        FakeGuardedRAGService.last_instance
    )

    if service is None:
        raise AssertionError(
            "The API lifespan did not create "
            "the fake service."
        )

    return service


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """Start the API using a controlled fake service."""

    install_fake_service(
        monkeypatch
    )

    with TestClient(
        api.app
    ) as test_client:
        yield test_client


def test_health_endpoint_runs_lifespan_and_reports_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The health route should report startup state."""

    install_fake_service(
        monkeypatch
    )

    with TestClient(
        api.app
    ) as test_client:
        service = require_fake_service()

        response = test_client.get(
            "/health"
        )

        assert response.status_code == 200

        assert response.json() == {
            "status": "healthy",
            "service_closed": False,
            "source_chunk_count": 9,
            "stored_point_count": 9,
            "collection_name": (
                "fleet_manual_chunks"
            ),
            "retrieval_top_k": 3,
        }

        assert service.is_closed is False

    assert service.is_closed is True


def test_answerable_question_serializes_nullable_source_score(
    client: TestClient,
) -> None:
    """An expanded source should serialize its score as null."""

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
        payload["generation_called"]
        is True
    )

    assert (
        payload["validation_passed"]
        is True
    )

    assert (
        payload["retrieval_results"][0]["rank"]
        == 1
    )

    assert (
        payload["retrieval_results"][0][
            "similarity"
        ]
        == pytest.approx(0.6547)
    )

    assert (
        payload["answer_sources"][0][
            "retrieval_score"
        ]
        == pytest.approx(0.6547)
    )

    assert (
        payload["answer_sources"][1][
            "retrieval_score"
        ]
        is None
    )

    assert (
        payload["answer_sources"][1][
            "inclusion_reason"
        ]
        == "adjacent-chunk expansion"
    )

    assert (
        payload["citation_checks"][0][
            "support_passed"
        ]
        is True
    )

    assert service.questions == [
        question
    ]


def test_unsupported_question_returns_retrieval_abstention(
    client: TestClient,
) -> None:
    """An unsupported question should abstain."""

    service = require_fake_service()

    question = (
        "What is the fleet insurance policy number?"
    )

    service.answer_result = (
        make_abstention_result(
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

    assert payload["decision"] == "ABSTAIN"

    assert (
        payload["answer_strategy"]
        == "retrieval abstention"
    )

    assert (
        payload["generation_called"]
        is False
    )

    assert (
        payload["generation_input_tokens"]
        is None
    )

    assert (
        payload["generation_output_tokens"]
        is None
    )

    assert payload["answer_sources"] == []
    assert payload["citation_checks"] == []

    assert service.questions == [
        question
    ]


@pytest.mark.parametrize(
    "request_body",
    [
        {},
        {
            "question": "  ",
        },
        {
            "question": "What is the policy?",
            "unexpected": True,
        },
    ],
)
def test_invalid_request_bodies_are_rejected(
    client: TestClient,
    request_body: dict[str, Any],
) -> None:
    """Missing, empty, and extra fields should fail."""

    response = client.post(
        "/rag/answer",
        json=request_body,
    )

    assert response.status_code == 422
    assert "detail" in response.json()

    assert (
        require_fake_service().questions
        == []
    )


def test_service_runtime_failure_becomes_http_503(
    client: TestClient,
) -> None:
    """A service runtime error should become HTTP 503."""

    service = require_fake_service()

    service.answer_error = RuntimeError(
        "The vector store is unavailable."
    )

    response = client.post(
        "/rag/answer",
        json={
            "question": (
                "What should the driver do?"
            ),
        },
    )

    assert response.status_code == 503

    detail = response.json()["detail"]

    assert (
        "could not complete the request"
        in detail
    )

    assert (
        "vector store is unavailable"
        in detail
    )


def test_root_endpoint_exposes_api_discovery(
    client: TestClient,
) -> None:
    """The root route should advertise API endpoints."""

    response = client.get(
        "/"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["name"] == api.API_TITLE
    assert payload["version"] == api.API_VERSION

    assert (
        payload["health_endpoint"]
        == "/health"
    )

    assert (
        payload["answer_endpoint"]
        == "/rag/answer"
    )

    assert (
        payload["documentation_endpoint"]
        == "/docs"
    )

    assert (
        payload["collection"]
        == api.COLLECTION_NAME
    )