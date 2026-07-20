from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lessons import lesson_47_application_runtime as runtime_module
from lessons.lesson_42_application_settings import (
    AppSettings,
)
from lessons.lesson_47_application_runtime import (
    ApplicationRuntime,
    RuntimeValidationReport,
    model_is_available,
)


class FakeRAGService:
    """Controlled guarded-RAG dependency for runtime tests."""

    def __init__(
        self,
        *,
        collection_name: str = "test_collection",
        top_k: int = 3,
        source_chunk_count: int = 9,
        stored_point_count: int = 9,
    ) -> None:
        self._collection_name = collection_name
        self._top_k = top_k
        self._source_chunk_count = source_chunk_count
        self._stored_point_count = stored_point_count

        self._closed = False

        self.validate_calls = 0
        self.close_calls = 0

        self.validation_error: Exception | None = None
        self.close_error: Exception | None = None

    @property
    def is_closed(self) -> bool:
        """Report whether the fake dependency is closed."""

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

    def validate(self) -> int:
        """Return the configured stored-point count."""

        if self._closed:
            raise RuntimeError(
                "The fake RAG service is closed."
            )

        self.validate_calls += 1

        if self.validation_error is not None:
            raise self.validation_error

        return self._stored_point_count

    def close(self) -> None:
        """Close the fake dependency."""

        self.close_calls += 1
        self._closed = True

        if self.close_error is not None:
            raise self.close_error


class FakeOllamaClient:
    """Controlled Ollama dependency for runtime tests."""

    def __init__(
        self,
        *,
        available_models: list[str] | None = None,
    ) -> None:
        self._available_models = (
            list(available_models)
            if available_models is not None
            else [
                "llama3.2:3b",
                "embeddinggemma:latest",
            ]
        )

        self._closed = False

        self.health_check_calls = 0
        self.close_calls = 0

        self.health_check_error: Exception | None = None
        self.close_error: Exception | None = None

    @property
    def is_closed(self) -> bool:
        """Report whether the fake client is closed."""

        return self._closed

    def health_check(self) -> list[str]:
        """Return controlled available-model data."""

        if self._closed:
            raise RuntimeError(
                "The fake Ollama client is closed."
            )

        self.health_check_calls += 1

        if self.health_check_error is not None:
            raise self.health_check_error

        return list(
            self._available_models
        )

    def close(self) -> None:
        """Close the fake Ollama dependency."""

        self.close_calls += 1
        self._closed = True

        if self.close_error is not None:
            raise self.close_error


class FactoryRAGService(FakeRAGService):
    """Fake class-level RAG service factory."""

    last_factory_arguments: dict[
        str,
        Any,
    ] | None = None

    last_instance: FactoryRAGService | None = None

    @classmethod
    def reset(cls) -> None:
        """Reset recorded factory state."""

        cls.last_factory_arguments = None
        cls.last_instance = None

    @classmethod
    def from_local_qdrant(
        cls,
        *,
        index_path: Path,
        qdrant_path: Path,
        collection_name: str,
        top_k: int,
    ) -> FactoryRAGService:
        """Create and record one fake RAG service."""

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


class FactoryOllamaClient(FakeOllamaClient):
    """Fake settings-driven Ollama client factory."""

    last_settings: AppSettings | None = None
    last_instance: FactoryOllamaClient | None = None

    @classmethod
    def reset(cls) -> None:
        """Reset recorded constructor state."""

        cls.last_settings = None
        cls.last_instance = None

    def __init__(
        self,
        settings: AppSettings,
    ) -> None:
        super().__init__()

        type(self).last_settings = settings
        type(self).last_instance = self


class FailingOllamaClient:
    """Ollama constructor that always fails."""

    def __init__(
        self,
        settings: AppSettings,
    ) -> None:
        del settings

        raise RuntimeError(
            "Simulated Ollama-client startup failure."
        )


def make_settings(
    temporary_directory: Path,
) -> AppSettings:
    """Create isolated application settings."""

    return AppSettings(
        api_title="Runtime Test API",
        api_version="1.2.3",
        api_host="127.0.0.1",
        api_port=8100,
        vector_index_path=(
            temporary_directory
            / "vector_index.json"
        ),
        qdrant_path=(
            temporary_directory
            / "qdrant"
        ),
        qdrant_collection=(
            "runtime_test_collection"
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


@pytest.mark.parametrize(
    (
        "configured_model",
        "available_models",
        "expected",
    ),
    [
        (
            "llama3.2:3b",
            (
                "llama3.2:3b",
                "embeddinggemma:latest",
            ),
            True,
        ),
        (
            "embeddinggemma",
            (
                "embeddinggemma:latest",
            ),
            True,
        ),
        (
            "embeddinggemma:latest",
            (
                "embeddinggemma",
            ),
            True,
        ),
        (
            "missing-model",
            (
                "llama3.2:3b",
                "embeddinggemma:latest",
            ),
            False,
        ),
        (
            "model:v1",
            (
                "model:latest",
            ),
            False,
        ),
    ],
)
def test_model_is_available_handles_tags(
    configured_model: str,
    available_models: tuple[str, ...],
    expected: bool,
) -> None:
    """Model matching should handle exact and latest tags."""

    assert (
        model_is_available(
            configured_model=configured_model,
            available_models=available_models,
        )
        is expected
    )


def test_validate_returns_and_stores_complete_report(
    tmp_path: Path,
) -> None:
    """Successful validation should create a status report."""

    settings = make_settings(
        tmp_path
    )

    rag_service = FakeRAGService(
        collection_name=(
            settings.qdrant_collection
        ),
        top_k=settings.retrieval_top_k,
        source_chunk_count=9,
        stored_point_count=9,
    )

    ollama_client = FakeOllamaClient(
        available_models=[
            "llama3.2:3b",
            "embeddinggemma:latest",
        ]
    )

    runtime = ApplicationRuntime(
        settings=settings,
        rag_service=rag_service,
        ollama_client=ollama_client,
        owns_rag_service=False,
        owns_ollama_client=False,
    )

    report = runtime.validate()

    assert isinstance(
        report,
        RuntimeValidationReport,
    )

    assert report.stored_point_count == 9
    assert report.source_chunk_count == 9

    assert (
        report.collection_name
        == settings.qdrant_collection
    )

    assert (
        report.retrieval_top_k
        == settings.retrieval_top_k
    )

    assert report.available_models == (
        "llama3.2:3b",
        "embeddinggemma:latest",
    )

    assert (
        report.generation_model_available
        is True
    )

    assert (
        report.embedding_model_available
        is True
    )

    assert (
        report.all_required_models_available
        is True
    )

    assert runtime.validation_report is report
    assert rag_service.validate_calls == 1
    assert ollama_client.health_check_calls == 1


def test_validate_rejects_missing_model(
    tmp_path: Path,
) -> None:
    """Runtime validation should identify missing models."""

    settings = make_settings(
        tmp_path
    )

    rag_service = FakeRAGService(
        collection_name=(
            settings.qdrant_collection
        ),
        top_k=settings.retrieval_top_k,
    )

    ollama_client = FakeOllamaClient(
        available_models=[
            "embeddinggemma:latest",
        ]
    )

    runtime = ApplicationRuntime(
        settings=settings,
        rag_service=rag_service,
        ollama_client=ollama_client,
        owns_rag_service=False,
        owns_ollama_client=False,
    )

    with pytest.raises(
        RuntimeError,
        match="llama3.2:3b",
    ):
        runtime.validate()

    assert runtime.validation_report is None
    assert rag_service.validate_calls == 1
    assert ollama_client.health_check_calls == 1


def test_context_manager_closes_owned_dependencies(
    tmp_path: Path,
) -> None:
    """Leaving the context should close owned dependencies."""

    settings = make_settings(
        tmp_path
    )

    rag_service = FakeRAGService()
    ollama_client = FakeOllamaClient()

    runtime = ApplicationRuntime(
        settings=settings,
        rag_service=rag_service,
        ollama_client=ollama_client,
        owns_rag_service=True,
        owns_ollama_client=True,
    )

    with runtime as active_runtime:
        assert active_runtime is runtime

        assert runtime.is_closed is False
        assert rag_service.is_closed is False
        assert ollama_client.is_closed is False

    assert runtime.is_closed is True
    assert rag_service.is_closed is True
    assert ollama_client.is_closed is True

    assert rag_service.close_calls == 1
    assert ollama_client.close_calls == 1

    runtime.close()

    assert rag_service.close_calls == 1
    assert ollama_client.close_calls == 1


def test_non_owned_dependencies_are_not_closed(
    tmp_path: Path,
) -> None:
    """Closing the runtime should preserve injected dependencies."""

    settings = make_settings(
        tmp_path
    )

    rag_service = FakeRAGService()
    ollama_client = FakeOllamaClient()

    runtime = ApplicationRuntime(
        settings=settings,
        rag_service=rag_service,
        ollama_client=ollama_client,
        owns_rag_service=False,
        owns_ollama_client=False,
    )

    runtime.close()

    assert runtime.is_closed is True

    assert rag_service.is_closed is False
    assert ollama_client.is_closed is False

    assert rag_service.close_calls == 0
    assert ollama_client.close_calls == 0


def test_closed_runtime_rejects_dependency_access_and_validation(
    tmp_path: Path,
) -> None:
    """Closed runtimes should reject further operations."""

    settings = make_settings(
        tmp_path
    )

    runtime = ApplicationRuntime(
        settings=settings,
        rag_service=FakeRAGService(),
        ollama_client=FakeOllamaClient(),
        owns_rag_service=False,
        owns_ollama_client=False,
    )

    runtime.close()

    with pytest.raises(
        RuntimeError,
        match="runtime is closed",
    ):
        _ = runtime.rag_service

    with pytest.raises(
        RuntimeError,
        match="runtime is closed",
    ):
        _ = runtime.ollama_client

    with pytest.raises(
        RuntimeError,
        match="runtime is closed",
    ):
        runtime.validate()


def test_from_settings_passes_configuration_to_factories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Factory construction should receive validated settings."""

    FactoryRAGService.reset()
    FactoryOllamaClient.reset()

    monkeypatch.setattr(
        runtime_module,
        "GuardedRAGService",
        FactoryRAGService,
    )

    monkeypatch.setattr(
        runtime_module,
        "ConfiguredOllamaClient",
        FactoryOllamaClient,
    )

    settings = make_settings(
        tmp_path
    )

    runtime = ApplicationRuntime.from_settings(
        settings
    )

    assert (
        FactoryRAGService.last_factory_arguments
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

    assert (
        FactoryOllamaClient.last_settings
        is settings
    )

    assert (
        runtime.rag_service
        is FactoryRAGService.last_instance
    )

    assert (
        runtime.ollama_client
        is FactoryOllamaClient.last_instance
    )

    runtime.close()

    assert (
        FactoryRAGService.last_instance
        is not None
    )

    assert (
        FactoryRAGService.last_instance.is_closed
        is True
    )

    assert (
        FactoryOllamaClient.last_instance
        is not None
    )

    assert (
        FactoryOllamaClient.last_instance.is_closed
        is True
    )


def test_partial_startup_failure_closes_created_rag_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A later constructor failure should clean up earlier work."""

    FactoryRAGService.reset()

    monkeypatch.setattr(
        runtime_module,
        "GuardedRAGService",
        FactoryRAGService,
    )

    monkeypatch.setattr(
        runtime_module,
        "ConfiguredOllamaClient",
        FailingOllamaClient,
    )

    settings = make_settings(
        tmp_path
    )

    with pytest.raises(
        RuntimeError,
        match="startup failure",
    ):
        ApplicationRuntime.from_settings(
            settings
        )

    created_service = (
        FactoryRAGService.last_instance
    )

    assert created_service is not None
    assert created_service.is_closed is True
    assert created_service.close_calls == 1