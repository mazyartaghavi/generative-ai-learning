from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lessons import lesson_36_guarded_rag_service as service_module
from lessons import lesson_47_application_runtime as runtime_module
from lessons.lesson_28_guarded_rag import (
    ABSTENTION_THRESHOLD,
    run_guarded_rag,
)
from lessons.lesson_35_qdrant_retriever_component import (
    QdrantRetriever,
)
from lessons.lesson_42_application_settings import (
    AppSettings,
)
from lessons.lesson_45_configured_ollama_client import (
    EmbeddingResult,
    GenerationResult,
)
from lessons.lesson_47_application_runtime import (
    GROUNDING_SYSTEM_PROMPT,
    ApplicationRuntime,
    ConfiguredOllamaRAGAdapter,
    model_names_equivalent,
)


class FakeOllamaClient:
    """Controlled shared Ollama client for adapter tests."""

    def __init__(
        self,
        *,
        embedding_result: EmbeddingResult | None = None,
        generation_result: GenerationResult | None = None,
    ) -> None:
        self.embedding_model = "test-embedding"
        self.generation_model = "test-generation"
        self.is_closed = False

        self.embedding_result = (
            embedding_result
            if embedding_result is not None
            else EmbeddingResult(
                model="test-embedding:latest",
                embeddings=[
                    [
                        0.1,
                        0.2,
                    ]
                ],
                input_tokens=7,
                total_duration_nanoseconds=100,
            )
        )

        self.generation_result = (
            generation_result
            if generation_result is not None
            else GenerationResult(
                model="test-generation:latest",
                response="A grounded answer [S1].",
                prompt_tokens=30,
                generated_tokens=8,
                total_duration_nanoseconds=200,
            )
        )

        self.embed_calls: list[str] = []
        self.generate_calls: list[
            dict[str, object]
        ] = []
        self.close_calls = 0

    def embed(
        self,
        inputs: str,
    ) -> EmbeddingResult:
        """Return the configured embedding result."""

        self.embed_calls.append(
            inputs
        )

        return self.embedding_result

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Return the configured generation result."""

        self.generate_calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
            }
        )

        return self.generation_result

    def health_check(self) -> list[str]:
        """Return both configured model names."""

        return [
            "test-generation:latest",
            "test-embedding:latest",
        ]

    def close(self) -> None:
        """Close the fake client."""

        self.close_calls += 1
        self.is_closed = True


class FakeQdrantClient:
    """Minimal Qdrant client used to test provider injection."""

    def __init__(self) -> None:
        self.query_arguments: dict[
            str,
            object,
        ] | None = None

    def query_points(
        self,
        **arguments: object,
    ) -> SimpleNamespace:
        """Record a query and return one controlled point."""

        self.query_arguments = dict(
            arguments
        )

        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload={
                        "chunk_id": "chunk-001",
                    },
                    score=0.91,
                )
            ]
        )


class FakeServiceRetriever:
    """Retriever double for GuardedRAGService tests."""

    def __init__(self) -> None:
        self.is_closed = False
        self.collection_name = "test_collection"
        self.top_k = 3
        self.embedding_provider: object | None = None

    def set_query_embedding_provider(
        self,
        provider: object,
    ) -> None:
        """Record the configured query-embedding provider."""

        self.embedding_provider = provider

    def validate_collection(
        self,
        *,
        expected_point_count: int | None = None,
    ) -> int:
        """Return the expected point count."""

        return (
            expected_point_count
            if expected_point_count is not None
            else 1
        )

    def close(self) -> None:
        """Close the fake retriever."""

        self.is_closed = True


class FactoryRAGService:
    """Runtime service factory that records shared providers."""

    last_instance: FactoryRAGService | None = None

    def __init__(self) -> None:
        self.is_closed = False
        self.source_chunk_count = 9
        self.collection_name = "runtime_collection"
        self.top_k = 4
        self.model_providers_configured = False

        self.query_embedding_provider: object | None = None
        self.generation_provider: object | None = None
        self.close_calls = 0

    @classmethod
    def from_local_qdrant(
        cls,
        **arguments: object,
    ) -> FactoryRAGService:
        """Create one controlled runtime service."""

        del arguments

        instance = cls()
        cls.last_instance = instance

        return instance

    def configure_model_providers(
        self,
        *,
        query_embedding_provider: object,
        generation_provider: object,
    ) -> None:
        """Record providers supplied by ApplicationRuntime."""

        self.query_embedding_provider = (
            query_embedding_provider
        )
        self.generation_provider = (
            generation_provider
        )
        self.model_providers_configured = True

    def validate(self) -> int:
        """Return a controlled collection count."""

        return 9

    def close(self) -> None:
        """Close the controlled service."""

        self.close_calls += 1
        self.is_closed = True


class FactoryOllamaClient(FakeOllamaClient):
    """Settings-compatible fake Ollama client factory."""

    last_instance: FactoryOllamaClient | None = None

    def __init__(
        self,
        settings: AppSettings,
    ) -> None:
        del settings

        super().__init__()

        type(self).last_instance = self


def make_settings(
    temporary_directory: Path,
) -> AppSettings:
    """Create isolated runtime settings."""

    return AppSettings(
        vector_index_path=(
            temporary_directory
            / "vector_index.json"
        ),
        qdrant_path=(
            temporary_directory
            / "qdrant"
        ),
        qdrant_collection="runtime_collection",
        retrieval_top_k=4,
        ollama_base_url="http://ollama.test:11434",
        generation_model="test-generation",
        embedding_model="test-embedding",
        ollama_timeout_seconds=15.0,
        _env_file=None,
    )


@pytest.mark.parametrize(
    (
        "first_model",
        "second_model",
        "expected",
    ),
    [
        (
            "embeddinggemma",
            "embeddinggemma:latest",
            True,
        ),
        (
            "llama3.2:3b",
            "llama3.2:3b",
            True,
        ),
        (
            "model:v1",
            "model:latest",
            False,
        ),
        (
            "first-model",
            "second-model",
            False,
        ),
    ],
)
def test_model_names_equivalent_normalizes_latest_tag(
    first_model: str,
    second_model: str,
    expected: bool,
) -> None:
    """Model comparison should normalize only ':latest'."""

    assert (
        model_names_equivalent(
            first_model,
            second_model,
        )
        is expected
    )


def test_adapter_embeds_query_through_shared_client() -> None:
    """The adapter should return one validated query vector."""

    client = FakeOllamaClient()
    adapter = ConfiguredOllamaRAGAdapter(
        client  # type: ignore[arg-type]
    )

    vector, token_count = adapter.embed_query(
        "Engine warning light",
        "test-embedding",
        2,
    )

    assert vector == [
        0.1,
        0.2,
    ]
    assert token_count == 7
    assert client.embed_calls == [
        "Engine warning light",
    ]
    assert adapter.query_embedding_calls == 1
    assert adapter.generation_calls == 0


def test_adapter_generates_with_grounding_prompt() -> None:
    """Grounded generation should preserve safety settings."""

    client = FakeOllamaClient()
    adapter = ConfiguredOllamaRAGAdapter(
        client  # type: ignore[arg-type]
    )

    answer, input_tokens, output_tokens = (
        adapter.generate_grounded_answer(
            "Question and supplied context"
        )
    )

    assert answer == "A grounded answer [S1]."
    assert input_tokens == 30
    assert output_tokens == 8

    assert client.generate_calls == [
        {
            "prompt": "Question and supplied context",
            "system_prompt": GROUNDING_SYSTEM_PROMPT,
            "temperature": 0.0,
        }
    ]

    assert adapter.query_embedding_calls == 0
    assert adapter.generation_calls == 1


def test_adapter_rejects_index_embedding_model_mismatch() -> None:
    """The source index and configured model must agree."""

    client = FakeOllamaClient()
    adapter = ConfiguredOllamaRAGAdapter(
        client  # type: ignore[arg-type]
    )

    with pytest.raises(
        RuntimeError,
        match="does not match",
    ):
        adapter.embed_query(
            "Question",
            "different-embedding-model",
            2,
        )

    assert client.embed_calls == []
    assert adapter.query_embedding_calls == 0


def test_adapter_rejects_wrong_embedding_dimension() -> None:
    """A query vector must match the stored index dimension."""

    client = FakeOllamaClient()
    adapter = ConfiguredOllamaRAGAdapter(
        client  # type: ignore[arg-type]
    )

    with pytest.raises(
        RuntimeError,
        match="Expected 3, received 2",
    ):
        adapter.embed_query(
            "Question",
            "test-embedding",
            3,
        )

    assert client.embed_calls == [
        "Question",
    ]
    assert adapter.query_embedding_calls == 1


def test_adapter_rejects_calls_after_client_closes() -> None:
    """The adapter must not use a closed shared client."""

    client = FakeOllamaClient()
    client.close()

    adapter = ConfiguredOllamaRAGAdapter(
        client  # type: ignore[arg-type]
    )

    with pytest.raises(
        RuntimeError,
        match="Ollama client is closed",
    ):
        adapter.embed_query(
            "Question",
            "test-embedding",
            2,
        )

    with pytest.raises(
        RuntimeError,
        match="Ollama client is closed",
    ):
        adapter.generate_grounded_answer(
            "Question and context"
        )


def test_qdrant_retriever_uses_injected_embedding_provider() -> None:
    """Qdrant retrieval should use the injected provider."""

    provider_calls: list[
        tuple[
            str,
            str,
            int,
        ]
    ] = []

    def embedding_provider(
        query: str,
        embedding_model: str,
        expected_dimension: int,
    ) -> tuple[list[float], int | None]:
        provider_calls.append(
            (
                query,
                embedding_model,
                expected_dimension,
            )
        )

        return (
            [
                0.1,
                0.2,
            ],
            6,
        )

    qdrant_client = FakeQdrantClient()

    retriever = QdrantRetriever(
        qdrant_client,  # type: ignore[arg-type]
        collection_name="test_collection",
        top_k=1,
        query_embedding_provider=(
            embedding_provider
        ),
    )

    chunk = SimpleNamespace(
        chunk_id="chunk-001"
    )

    vector_index = SimpleNamespace(
        chunks=[
            chunk,
        ],
        embedding_model="test-embedding",
        embedding_dimension=2,
    )

    results, token_count = retriever.retrieve(
        "  Engine warning light  ",
        vector_index,  # type: ignore[arg-type]
    )

    assert provider_calls == [
        (
            "Engine warning light",
            "test-embedding",
            2,
        )
    ]

    assert token_count == 6
    assert len(results) == 1
    assert results[0].score == pytest.approx(
        0.91
    )
    assert results[0].chunk is chunk

    assert qdrant_client.query_arguments == {
        "collection_name": "test_collection",
        "query": [
            0.1,
            0.2,
        ],
        "limit": 1,
        "with_payload": True,
        "with_vectors": False,
    }


def test_service_passes_both_configured_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service should inject retrieval and generation."""

    vector_index = SimpleNamespace(
        chunks=[
            object(),
        ]
    )

    retriever = FakeServiceRetriever()

    service = service_module.GuardedRAGService(
        vector_index=(
            vector_index  # type: ignore[arg-type]
        ),
        retriever=(
            retriever  # type: ignore[arg-type]
        ),
    )

    def query_provider(
        query: str,
        embedding_model: str,
        expected_dimension: int,
    ) -> tuple[list[float], int | None]:
        del query
        del embedding_model
        del expected_dimension

        return (
            [
                0.1,
            ],
            1,
        )

    def generation_provider(
        user_message: str,
    ) -> tuple[str, int | None, int | None]:
        del user_message

        return (
            "Answer [S1].",
            2,
            3,
        )

    captured_arguments: dict[
        str,
        Any,
    ] = {}
    sentinel_result = object()

    def fake_run_guarded_rag(
        **arguments: Any,
    ) -> object:
        captured_arguments.update(
            arguments
        )

        return sentinel_result

    monkeypatch.setattr(
        service_module.guarded,
        "run_guarded_rag",
        fake_run_guarded_rag,
    )

    service.configure_model_providers(
        query_embedding_provider=query_provider,
        generation_provider=generation_provider,
    )

    result = service.answer(
        "  Engine warning  "
    )

    assert result is sentinel_result
    assert service.model_providers_configured is True
    assert retriever.embedding_provider is query_provider

    assert captured_arguments == {
        "question": "Engine warning",
        "vector_index": vector_index,
        "retriever": retriever,
        "generator": generation_provider,
    }


def test_runtime_configures_one_adapter_for_both_providers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runtime construction should bind one shared client."""

    FactoryRAGService.last_instance = None
    FactoryOllamaClient.last_instance = None

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

    service = FactoryRAGService.last_instance
    client = FactoryOllamaClient.last_instance
    adapter = runtime.rag_adapter

    assert service is not None
    assert client is not None
    assert adapter is not None

    assert service.model_providers_configured is True
    assert callable(
        service.query_embedding_provider
    )
    assert callable(
        service.generation_provider
    )

    vector, token_count = (
        service.query_embedding_provider(  # type: ignore[operator]
            "Question",
            "test-embedding",
            2,
        )
    )

    answer, input_tokens, output_tokens = (
        service.generation_provider(  # type: ignore[operator]
            "Question and context"
        )
    )

    assert vector == [
        0.1,
        0.2,
    ]
    assert token_count == 7
    assert answer == "A grounded answer [S1]."
    assert input_tokens == 30
    assert output_tokens == 8

    assert adapter.query_embedding_calls == 1
    assert adapter.generation_calls == 1
    assert client.embed_calls == [
        "Question",
    ]
    assert len(
        client.generate_calls
    ) == 1

    runtime.close()

    assert runtime.is_closed is True
    assert service.is_closed is True
    assert client.is_closed is True
    assert service.close_calls == 1
    assert client.close_calls == 1


def test_retrieval_abstention_skips_injected_generator() -> None:
    """Low similarity must stop before model generation."""

    generation_calls = 0

    def retriever(
        question: str,
        vector_index: object,
    ) -> tuple[list[object], int | None]:
        del question
        del vector_index

        return (
            [
                SimpleNamespace(
                    score=(
                        ABSTENTION_THRESHOLD
                        - 0.01
                    ),
                    chunk=object(),
                )
            ],
            4,
        )

    def generator(
        user_message: str,
    ) -> tuple[str, int | None, int | None]:
        del user_message

        nonlocal generation_calls
        generation_calls += 1

        raise AssertionError(
            "Generation should not have been called."
        )

    vector_index = SimpleNamespace(
        chunks=[
            object(),
        ]
    )

    result = run_guarded_rag(
        question="Unsupported question",
        vector_index=(
            vector_index  # type: ignore[arg-type]
        ),
        retriever=(
            retriever  # type: ignore[arg-type]
        ),
        generator=generator,
    )

    assert result.decision == "ABSTAIN"
    assert result.answer_strategy == (
        "retrieval abstention"
    )
    assert result.generation_called is False
    assert generation_calls == 0
