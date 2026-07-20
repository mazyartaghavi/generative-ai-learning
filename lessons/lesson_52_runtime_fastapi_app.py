from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from _thread import LockType
from threading import Lock
from typing import Annotated, AsyncIterator, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from lessons.lesson_28_guarded_rag import GuardedRAGResult
from lessons.lesson_42_application_settings import (
    AppSettings,
    get_settings,
)
from lessons.lesson_47_application_runtime import (
    ApplicationRuntime,
    RuntimeValidationReport,
)


QuestionText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=1000,
    ),
]

RuntimeFactory = Callable[
    [AppSettings],
    ApplicationRuntime,
]


class HealthResponse(BaseModel):
    """Runtime health information returned by the API."""

    model_config = ConfigDict(
        extra="forbid",
    )

    status: Literal["ok"]
    api_title: str
    api_version: str
    qdrant_collection: str
    source_chunk_count: int
    stored_point_count: int
    retrieval_top_k: int
    generation_model: str
    embedding_model: str
    shared_model_providers_configured: bool


class ConfigurationResponse(BaseModel):
    """Non-secret active application configuration."""

    model_config = ConfigDict(
        extra="forbid",
    )

    api_title: str
    api_version: str
    api_host: str
    api_port: int
    vector_index_path: str
    qdrant_path: str
    qdrant_collection: str
    retrieval_top_k: int
    ollama_base_url: str
    generation_model: str
    embedding_model: str
    ollama_timeout_seconds: float


class RAGAnswerRequest(BaseModel):
    """Validated question accepted by the guarded-RAG API."""

    model_config = ConfigDict(
        extra="forbid",
    )

    question: QuestionText


class SourceResponse(BaseModel):
    """One source supplied to the guarded answer pipeline."""

    model_config = ConfigDict(
        extra="forbid",
    )

    label: str
    chunk_id: str
    section_id: str
    section_title: str
    retrieval_score: float | None
    inclusion_reason: str


class RAGAnswerResponse(BaseModel):
    """Serializable guarded-RAG result."""

    model_config = ConfigDict(
        extra="forbid",
    )

    question: str
    decision: str
    answer_strategy: str
    answer: str

    top_similarity: float
    threshold: float

    generation_called: bool
    citation_repair_used: bool
    extractive_fallback_used: bool
    answer_passed_validation: bool

    generation_input_tokens: int | None
    generation_output_tokens: int | None

    sources: list[SourceResponse] = Field(
        default_factory=list
    )


class DiscoveryResponse(BaseModel):
    """Simple API discovery document."""

    model_config = ConfigDict(
        extra="forbid",
    )

    service: str
    version: str
    health_endpoint: str
    configuration_endpoint: str
    answer_endpoint: str
    documentation_endpoint: str


def default_runtime_factory(
    settings: AppSettings,
) -> ApplicationRuntime:
    """Create the production application runtime."""

    return ApplicationRuntime.from_settings(
        settings
    )


def require_runtime(
    request: Request,
) -> ApplicationRuntime:
    """Return the active runtime or report startup failure."""

    runtime = getattr(
        request.app.state,
        "runtime",
        None,
    )

    if not isinstance(
        runtime,
        ApplicationRuntime,
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "The application runtime is not available."
            ),
        )

    if runtime.is_closed:
        raise HTTPException(
            status_code=503,
            detail=(
                "The application runtime is closed."
            ),
        )

    return runtime


def require_validation_report(
    request: Request,
) -> RuntimeValidationReport:
    """Return the startup validation report."""

    report = getattr(
        request.app.state,
        "validation_report",
        None,
    )

    if not isinstance(
        report,
        RuntimeValidationReport,
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "The runtime validation report is "
                "not available."
            ),
        )

    return report


def serialize_answer(
    result: GuardedRAGResult,
) -> RAGAnswerResponse:
    """Convert one guarded result into an API response."""

    sources = [
        SourceResponse(
            label=source.label,
            chunk_id=source.chunk.chunk_id,
            section_id=source.chunk.section_id,
            section_title=source.chunk.section_title,
            retrieval_score=(
                source.retrieval_score
            ),
            inclusion_reason=(
                source.inclusion_reason
            ),
        )
        for source in result.context_sources
    ]

    return RAGAnswerResponse(
        question=result.question,
        decision=result.decision,
        answer_strategy=result.answer_strategy,
        answer=result.answer,
        top_similarity=result.top_similarity,
        threshold=result.threshold,
        generation_called=result.generation_called,
        citation_repair_used=(
            result.citation_repair_used
        ),
        extractive_fallback_used=(
            result.extractive_fallback_used
        ),
        answer_passed_validation=(
            result.answer_passed_validation
        ),
        generation_input_tokens=(
            result.generation_input_tokens
        ),
        generation_output_tokens=(
            result.generation_output_tokens
        ),
        sources=sources,
    )


def create_app(
    settings: AppSettings,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> FastAPI:
    """Create a FastAPI app around ApplicationRuntime."""

    active_runtime_factory = (
        default_runtime_factory
        if runtime_factory is None
        else runtime_factory
    )

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        """
        Create, validate, and close one application runtime.

        FastAPI executes this function once during startup and
        once during shutdown. Every request between those two
        events shares the same Qdrant and Ollama dependencies.
        """

        runtime = active_runtime_factory(
            settings
        )

        try:
            validation_report = (
                runtime.validate()
            )
        except Exception:
            runtime.close()
            raise

        app.state.runtime = runtime
        app.state.validation_report = (
            validation_report
        )
        app.state.answer_lock = Lock()

        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )

    app.state.settings = settings

    @app.get(
        "/",
        response_model=DiscoveryResponse,
        tags=["Discovery"],
    )
    def root() -> DiscoveryResponse:
        """Return API endpoint discovery information."""

        return DiscoveryResponse(
            service=settings.api_title,
            version=settings.api_version,
            health_endpoint="/health",
            configuration_endpoint="/config",
            answer_endpoint="/rag/answer",
            documentation_endpoint="/docs",
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Operations"],
    )
    def health(
        request: Request,
    ) -> HealthResponse:
        """Return validated runtime health information."""

        runtime = require_runtime(
            request
        )

        report = require_validation_report(
            request
        )

        return HealthResponse(
            status="ok",
            api_title=settings.api_title,
            api_version=settings.api_version,
            qdrant_collection=(
                report.collection_name
            ),
            source_chunk_count=(
                report.source_chunk_count
            ),
            stored_point_count=(
                report.stored_point_count
            ),
            retrieval_top_k=(
                report.retrieval_top_k
            ),
            generation_model=(
                settings.generation_model
            ),
            embedding_model=(
                settings.embedding_model
            ),
            shared_model_providers_configured=(
                runtime.rag_service
                .model_providers_configured
            ),
        )

    @app.get(
        "/config",
        response_model=ConfigurationResponse,
        tags=["Operations"],
    )
    def configuration() -> ConfigurationResponse:
        """Return active non-secret application settings."""

        return ConfigurationResponse(
            api_title=settings.api_title,
            api_version=settings.api_version,
            api_host=settings.api_host,
            api_port=settings.api_port,
            vector_index_path=str(
                settings.vector_index_path
            ),
            qdrant_path=str(
                settings.qdrant_path
            ),
            qdrant_collection=(
                settings.qdrant_collection
            ),
            retrieval_top_k=(
                settings.retrieval_top_k
            ),
            ollama_base_url=(
                settings.ollama_base_url
            ),
            generation_model=(
                settings.generation_model
            ),
            embedding_model=(
                settings.embedding_model
            ),
            ollama_timeout_seconds=(
                settings.ollama_timeout_seconds
            ),
        )

    @app.post(
        "/rag/answer",
        response_model=RAGAnswerResponse,
        tags=["Guarded RAG"],
    )
    def answer(
        payload: RAGAnswerRequest,
        request: Request,
    ) -> RAGAnswerResponse:
        """Answer one question through the shared runtime."""

        runtime = require_runtime(
            request
        )

        answer_lock = getattr(
            request.app.state,
            "answer_lock",
            None,
        )

        if not isinstance(
            answer_lock,
            LockType,
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "The guarded-RAG request lock is "
                    "not available."
                ),
            )

        try:
            with answer_lock:
                result = (
                    runtime.rag_service.answer(
                        payload.question
                    )
                )
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=503,
                detail=str(error),
            ) from error

        return serialize_answer(
            result
        )

    return app


settings = get_settings()

app = create_app(
    settings
)


def main() -> None:
    """Run the configured FastAPI application."""

    print("RUNTIME FASTAPI APPLICATION")
    print("===========================")
    print()
    print(
        "API title:",
        settings.api_title,
    )
    print(
        "Host:",
        settings.api_host,
    )
    print(
        "Port:",
        settings.api_port,
    )
    print()
    print(
        "Open the interactive documentation at:"
    )
    print(
        f"http://{settings.api_host}:"
        f"{settings.api_port}/docs"
    )
    print()

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
