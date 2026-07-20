from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)

from lessons.lesson_36_guarded_rag_service import (
    GuardedRAGService,
)
from lessons.lesson_40_fastapi_guarded_rag_api import (
    HealthResponse,
    RAGAnswerResponse,
    RAGQuestionRequest,
    StrictModel,
    build_answer_response,
    get_rag_service,
    get_service_lock,
)
from lessons.lesson_42_application_settings import (
    AppSettings,
    get_settings,
)


class RuntimeConfigurationResponse(StrictModel):
    """Non-secret runtime configuration used by the API."""

    api_title: str
    api_version: str
    api_host: str
    api_port: int
    vector_index_path: str
    qdrant_path: str
    qdrant_collection: str
    retrieval_top_k: int


def get_app_settings(
    request: Request,
) -> AppSettings:
    """Read validated settings from application state."""

    settings = getattr(
        request.app.state,
        "settings",
        None,
    )

    if not isinstance(
        settings,
        AppSettings,
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "The application settings are not "
                "initialized."
            ),
        )

    return settings


def create_app(
    settings: AppSettings | None = None,
) -> FastAPI:
    """
    Create a configured FastAPI application.

    Passing settings explicitly makes the application
    factory easy to test. When no settings are supplied,
    validated environment and .env settings are loaded.
    """

    active_settings = (
        settings
        if settings is not None
        else get_settings()
    )

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        """Initialize and close the configured RAG service."""

        service = (
            GuardedRAGService.from_local_qdrant(
                index_path=(
                    active_settings.vector_index_path
                ),
                qdrant_path=(
                    active_settings.qdrant_path
                ),
                collection_name=(
                    active_settings.qdrant_collection
                ),
                top_k=(
                    active_settings.retrieval_top_k
                ),
            )
        )

        try:
            stored_point_count = (
                service.validate()
            )
        except Exception:
            service.close()
            raise

        application.state.settings = (
            active_settings
        )

        application.state.rag_service = (
            service
        )

        application.state.rag_lock = Lock()

        application.state.stored_point_count = (
            stored_point_count
        )

        try:
            yield
        finally:
            service.close()

    configured_app = FastAPI(
        title=active_settings.api_title,
        version=active_settings.api_version,
        description=(
            "A settings-driven API for Qdrant-backed "
            "guarded retrieval-augmented generation over "
            "the Fleet Operations Manual."
        ),
        lifespan=lifespan,
    )

    @configured_app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
    )
    def health(
        request: Request,
    ) -> HealthResponse:
        """Return configured service-health information."""

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
            service_closed=(
                service.is_closed
            ),
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

    @configured_app.get(
        "/config",
        response_model=(
            RuntimeConfigurationResponse
        ),
        tags=["System"],
    )
    def runtime_configuration(
        request: Request,
    ) -> RuntimeConfigurationResponse:
        """Return the non-secret active configuration."""

        current_settings = get_app_settings(
            request
        )

        return RuntimeConfigurationResponse(
            api_title=(
                current_settings.api_title
            ),
            api_version=(
                current_settings.api_version
            ),
            api_host=(
                current_settings.api_host
            ),
            api_port=(
                current_settings.api_port
            ),
            vector_index_path=str(
                current_settings.vector_index_path
            ),
            qdrant_path=str(
                current_settings.qdrant_path
            ),
            qdrant_collection=(
                current_settings.qdrant_collection
            ),
            retrieval_top_k=(
                current_settings.retrieval_top_k
            ),
        )

    @configured_app.post(
        "/rag/answer",
        response_model=RAGAnswerResponse,
        tags=["Guarded RAG"],
    )
    def answer_question(
        payload: RAGQuestionRequest,
        request: Request,
    ) -> RAGAnswerResponse:
        """Answer a question through configured guarded RAG."""

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

    @configured_app.get(
        "/",
        tags=["System"],
    )
    def root(
        request: Request,
    ) -> dict[str, Any]:
        """Return configured API discovery information."""

        current_settings = get_app_settings(
            request
        )

        return {
            "name": (
                current_settings.api_title
            ),
            "version": (
                current_settings.api_version
            ),
            "health_endpoint": "/health",
            "configuration_endpoint": "/config",
            "answer_endpoint": "/rag/answer",
            "documentation_endpoint": "/docs",
            "collection": (
                current_settings.qdrant_collection
            ),
            "retrieval_top_k": (
                current_settings.retrieval_top_k
            ),
        }

    return configured_app


settings = get_settings()

app = create_app(
    settings
)


def main() -> None:
    """Run Uvicorn using validated application settings."""

    print("CONFIGURED GUARDED-RAG API")
    print("==========================")
    print()
    print(
        "API title:",
        settings.api_title,
    )
    print(
        "API version:",
        settings.api_version,
    )
    print(
        "API host:",
        settings.api_host,
    )
    print(
        "API port:",
        settings.api_port,
    )
    print(
        "Vector index:",
        settings.vector_index_path,
    )
    print(
        "Qdrant path:",
        settings.qdrant_path,
    )
    print(
        "Qdrant collection:",
        settings.qdrant_collection,
    )
    print(
        "Retrieval top K:",
        settings.retrieval_top_k,
    )
    print()

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()