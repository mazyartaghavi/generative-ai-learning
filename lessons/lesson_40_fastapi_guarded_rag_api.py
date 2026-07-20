from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from lessons import lesson_28_guarded_rag as guarded
from lessons.lesson_30_qdrant_local_index import (
    COLLECTION_NAME,
)
from lessons.lesson_36_guarded_rag_service import (
    GuardedRAGService,
)


API_TITLE = "Fleet Guarded-RAG API"
API_VERSION = "1.0.0"


class StrictModel(BaseModel):
    """Base model that rejects unexpected JSON fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class RAGQuestionRequest(StrictModel):
    """Request body for one guarded-RAG question."""

    question: str = Field(
        min_length=3,
        max_length=500,
        description=(
            "A question about the indexed fleet "
            "operations manual."
        ),
        examples=[
            (
                "What should a driver do when an engine "
                "warning light appears?"
            )
        ],
    )


class RetrievalResultResponse(StrictModel):
    """One ranked vector-retrieval result."""

    rank: int
    chunk_id: str
    section_id: str
    section_title: str
    similarity: float


class AnswerSourceResponse(StrictModel):
    """One source included in the answer context."""

    label: str
    chunk_id: str
    section_id: str
    section_title: str

    retrieval_score: float | None = Field(
        description=(
            "The Qdrant similarity score when the source "
            "was retrieved directly. The value is null "
            "when the source was added through section "
            "reconstruction or context expansion."
        )
    )

    inclusion_reason: str


class CitationCheckResponse(StrictModel):
    """Citation-validation result for one answer sentence."""

    sentence_number: int
    cited_labels: list[str]
    invalid_labels: list[str]
    missing_citation: bool
    lexical_coverage: float
    missing_support_terms: list[str]
    support_passed: bool


class RAGAnswerResponse(StrictModel):
    """Structured response from the guarded-RAG pipeline."""

    question: str
    decision: str
    answer_strategy: str
    answer: str

    top_similarity: float
    threshold: float

    generation_called: bool
    citation_repair_used: bool
    extractive_fallback_used: bool
    validation_passed: bool

    generation_input_tokens: int | None
    generation_output_tokens: int | None

    retrieval_results: list[
        RetrievalResultResponse
    ]

    answer_sources: list[
        AnswerSourceResponse
    ]

    citation_checks: list[
        CitationCheckResponse
    ]


class HealthResponse(StrictModel):
    """Report whether the API service is operational."""

    status: str
    service_closed: bool
    source_chunk_count: int
    stored_point_count: int
    collection_name: str
    retrieval_top_k: int


def get_rag_service(
    request: Request,
) -> GuardedRAGService:
    """Read the initialized service from application state."""

    service = getattr(
        request.app.state,
        "rag_service",
        None,
    )

    if not isinstance(
        service,
        GuardedRAGService,
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "The guarded-RAG service is not initialized."
            ),
        )

    if service.is_closed:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "The guarded-RAG service is closed."
            ),
        )

    return service


def get_service_lock(
    request: Request,
) -> Lock:
    """Read the service synchronization lock."""

    service_lock = getattr(
        request.app.state,
        "rag_lock",
        None,
    )

    if service_lock is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "The guarded-RAG request lock is "
                "not initialized."
            ),
        )

    return service_lock


def build_retrieval_responses(
    result: guarded.GuardedRAGResult,
) -> list[RetrievalResultResponse]:
    """Convert internal retrieval results to API models."""

    responses: list[
        RetrievalResultResponse
    ] = []

    for rank, search_result in enumerate(
        result.retrieval_results,
        start=1,
    ):
        responses.append(
            RetrievalResultResponse(
                rank=rank,
                chunk_id=(
                    search_result.chunk.chunk_id
                ),
                section_id=(
                    search_result.chunk.section_id
                ),
                section_title=(
                    search_result.chunk.section_title
                ),
                similarity=float(
                    search_result.score
                ),
            )
        )

    return responses


def optional_float(
    value: float | int | None,
) -> float | None:
    """Convert a numeric value to float while preserving None."""

    if value is None:
        return None

    return float(value)


def build_source_responses(
    result: guarded.GuardedRAGResult,
) -> list[AnswerSourceResponse]:
    """Convert answer-context sources to API models."""

    responses: list[
        AnswerSourceResponse
    ] = []

    for source in result.context_sources:
        responses.append(
            AnswerSourceResponse(
                label=source.label,
                chunk_id=source.chunk.chunk_id,
                section_id=source.chunk.section_id,
                section_title=(
                    source.chunk.section_title
                ),
                retrieval_score=optional_float(
                    source.retrieval_score
                ),
                inclusion_reason=(
                    source.inclusion_reason
                ),
            )
        )

    return responses


def build_citation_responses(
    result: guarded.GuardedRAGResult,
) -> list[CitationCheckResponse]:
    """Convert internal citation checks to API models."""

    return [
        CitationCheckResponse(
            sentence_number=(
                check.sentence_number
            ),
            cited_labels=list(
                check.cited_labels
            ),
            invalid_labels=list(
                check.invalid_labels
            ),
            missing_citation=(
                check.missing_citation
            ),
            lexical_coverage=float(
                check.lexical_coverage
            ),
            missing_support_terms=list(
                check.missing_support_terms
            ),
            support_passed=(
                check.support_passed
            ),
        )
        for check in result.citation_checks
    ]


def build_answer_response(
    result: guarded.GuardedRAGResult,
) -> RAGAnswerResponse:
    """Convert an internal guarded result to API output."""

    return RAGAnswerResponse(
        question=result.question,
        decision=result.decision,
        answer_strategy=(
            result.answer_strategy
        ),
        answer=result.answer,
        top_similarity=float(
            result.top_similarity
        ),
        threshold=float(
            result.threshold
        ),
        generation_called=(
            result.generation_called
        ),
        citation_repair_used=(
            result.citation_repair_used
        ),
        extractive_fallback_used=(
            result.extractive_fallback_used
        ),
        validation_passed=(
            result.answer_passed_validation
        ),
        generation_input_tokens=(
            result.generation_input_tokens
        ),
        generation_output_tokens=(
            result.generation_output_tokens
        ),
        retrieval_results=(
            build_retrieval_responses(
                result
            )
        ),
        answer_sources=(
            build_source_responses(
                result
            )
        ),
        citation_checks=(
            build_citation_responses(
                result
            )
        ),
    )


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """Open shared RAG resources and close them on shutdown."""

    service = (
        GuardedRAGService.from_local_qdrant()
    )

    stored_point_count = service.validate()

    app.state.rag_service = service
    app.state.rag_lock = Lock()
    app.state.stored_point_count = (
        stored_point_count
    )

    try:
        yield
    finally:
        service.close()


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "A local API for Qdrant-backed guarded retrieval-"
        "augmented generation over the Fleet Operations "
        "Manual."
    ),
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health(
    request: Request,
) -> HealthResponse:
    """Return API, service, and Qdrant health information."""

    service = get_rag_service(
        request
    )

    stored_point_count = getattr(
        request.app.state,
        "stored_point_count",
        0,
    )

    return HealthResponse(
        status="healthy",
        service_closed=service.is_closed,
        source_chunk_count=(
            service.source_chunk_count
        ),
        stored_point_count=int(
            stored_point_count
        ),
        collection_name=(
            service.collection_name
        ),
        retrieval_top_k=(
            service.top_k
        ),
    )


@app.post(
    "/rag/answer",
    response_model=RAGAnswerResponse,
    tags=["Guarded RAG"],
)
def answer_question(
    payload: RAGQuestionRequest,
    request: Request,
) -> RAGAnswerResponse:
    """Answer one question through guarded Qdrant RAG."""

    service = get_rag_service(
        request
    )

    service_lock = get_service_lock(
        request
    )

    try:
        with service_lock:
            result = service.answer(
                payload.question
            )
    except ValueError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail=str(error),
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "The guarded-RAG service could not "
                f"complete the request: {error}"
            ),
        ) from error

    return build_answer_response(
        result
    )


@app.get(
    "/",
    tags=["System"],
)
def root() -> dict[str, Any]:
    """Return basic API discovery information."""

    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "health_endpoint": "/health",
        "answer_endpoint": "/rag/answer",
        "documentation_endpoint": "/docs",
        "collection": COLLECTION_NAME,
    }