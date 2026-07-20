from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

from lessons.lesson_36_guarded_rag_service import (
    GuardedRAGService,
)
from lessons.lesson_42_application_settings import (
    AppSettings,
    get_settings,
)
from lessons.lesson_45_configured_ollama_client import (
    ConfiguredOllamaClient,
)


GROUNDING_SYSTEM_PROMPT = (
    "You are a grounded fleet-operations assistant. "
    "Use only the context supplied in the user message. "
    "Do not use outside knowledge or invent facts. "
    "Preserve source labels exactly in forms such as [S1]. "
    "Every factual sentence must cite at least one source "
    "that directly supports it. Answer concisely."
)


def model_names_equivalent(
    first_model: str,
    second_model: str,
) -> bool:
    """Compare Ollama model names while normalizing ':latest'."""

    def normalize(model_name: str) -> str:
        cleaned_name = model_name.strip()

        if cleaned_name.endswith(":latest"):
            return cleaned_name[
                : -len(":latest")
            ]

        return cleaned_name

    return normalize(first_model) == normalize(
        second_model
    )


class ConfiguredOllamaRAGAdapter:
    """
    Adapt ConfiguredOllamaClient to the guarded-RAG callables.

    One shared client now performs both query embedding and
    grounded answer generation.
    """

    def __init__(
        self,
        ollama_client: ConfiguredOllamaClient,
    ) -> None:
        """Initialize the callable adapter."""

        self._ollama_client = ollama_client
        self._query_embedding_calls = 0
        self._generation_calls = 0

    @property
    def query_embedding_calls(self) -> int:
        """Return the number of query-embedding requests."""

        return self._query_embedding_calls

    @property
    def generation_calls(self) -> int:
        """Return the number of grounded-generation requests."""

        return self._generation_calls

    def embed_query(
        self,
        query: str,
        embedding_model: str,
        expected_dimension: int,
    ) -> tuple[list[float], int | None]:
        """Embed one retrieval query through the shared client."""

        if self._ollama_client.is_closed:
            raise RuntimeError(
                "The configured Ollama client is closed."
            )

        if expected_dimension <= 0:
            raise ValueError(
                "The expected embedding dimension must be "
                "greater than zero."
            )

        if not model_names_equivalent(
            embedding_model,
            self._ollama_client.embedding_model,
        ):
            raise RuntimeError(
                "The vector index embedding model does not "
                "match the configured Ollama embedding model. "
                f"Index: {embedding_model!r}; configured: "
                f"{self._ollama_client.embedding_model!r}."
            )

        self._query_embedding_calls += 1

        result = self._ollama_client.embed(
            query
        )

        if not model_names_equivalent(
            result.model,
            self._ollama_client.embedding_model,
        ):
            raise RuntimeError(
                "Ollama returned an unexpected embedding "
                f"model: {result.model!r}."
            )

        if result.embedding_count != 1:
            raise RuntimeError(
                "A query-embedding request must return "
                "exactly one vector."
            )

        if result.embedding_dimension != expected_dimension:
            raise RuntimeError(
                "The query embedding dimension does not "
                "match the source vector index. "
                f"Expected {expected_dimension}, received "
                f"{result.embedding_dimension}."
            )

        return (
            result.embeddings[0],
            result.input_tokens,
        )

    def generate_grounded_answer(
        self,
        user_message: str,
    ) -> tuple[str, int | None, int | None]:
        """Generate a grounded answer through the shared client."""

        if self._ollama_client.is_closed:
            raise RuntimeError(
                "The configured Ollama client is closed."
            )

        self._generation_calls += 1

        result = self._ollama_client.generate(
            user_message,
            system_prompt=GROUNDING_SYSTEM_PROMPT,
            temperature=0.0,
        )

        if not model_names_equivalent(
            result.model,
            self._ollama_client.generation_model,
        ):
            raise RuntimeError(
                "Ollama returned an unexpected generation "
                f"model: {result.model!r}."
            )

        return (
            result.response,
            result.prompt_tokens,
            result.generated_tokens,
        )


@dataclass(frozen=True, slots=True)
class RuntimeValidationReport:
    """Validated status of the application dependencies."""

    stored_point_count: int
    source_chunk_count: int
    collection_name: str
    retrieval_top_k: int
    available_models: tuple[str, ...]
    generation_model_available: bool
    embedding_model_available: bool

    @property
    def all_required_models_available(self) -> bool:
        """Report whether both configured models are available."""

        return (
            self.generation_model_available
            and self.embedding_model_available
        )


class ApplicationRuntime:
    """
    Own the long-lived dependencies of the RAG application.

    The runtime combines:

    1. Validated application settings.
    2. The Qdrant-backed guarded-RAG service.
    3. The configured Ollama HTTP client.
    """

    def __init__(
        self,
        *,
        settings: AppSettings,
        rag_service: GuardedRAGService,
        ollama_client: ConfiguredOllamaClient,
        rag_adapter: ConfiguredOllamaRAGAdapter | None = None,
        owns_rag_service: bool = True,
        owns_ollama_client: bool = True,
    ) -> None:
        """Initialize a runtime from existing dependencies."""

        self._settings = settings
        self._rag_service = rag_service
        self._ollama_client = ollama_client
        self._rag_adapter = rag_adapter

        self._owns_rag_service = owns_rag_service
        self._owns_ollama_client = owns_ollama_client

        self._closed = False
        self._validation_report: (
            RuntimeValidationReport | None
        ) = None

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings,
    ) -> ApplicationRuntime:
        """
        Build the complete runtime from validated settings.

        If Ollama-client construction fails after the RAG
        service has been created, the RAG service is closed
        before the error is re-raised.
        """

        rag_service = (
            GuardedRAGService.from_local_qdrant(
                index_path=settings.vector_index_path,
                qdrant_path=settings.qdrant_path,
                collection_name=(
                    settings.qdrant_collection
                ),
                top_k=settings.retrieval_top_k,
            )
        )

        try:
            ollama_client = ConfiguredOllamaClient(
                settings
            )
        except Exception:
            rag_service.close()
            raise

        try:
            rag_adapter = ConfiguredOllamaRAGAdapter(
                ollama_client
            )

            configure_model_providers = getattr(
                rag_service,
                "configure_model_providers",
                None,
            )

            if configure_model_providers is not None:
                if not callable(configure_model_providers):
                    raise TypeError(
                        "The RAG service model-provider "
                        "configuration hook is not callable."
                    )

                configure_model_providers(
                    query_embedding_provider=(
                        rag_adapter.embed_query
                    ),
                    generation_provider=(
                        rag_adapter.generate_grounded_answer
                    ),
                )
            else:
                # Compatibility path for minimal test doubles
                # that predate the model-provider hook.
                rag_adapter = None
        except Exception:
            ollama_client.close()
            rag_service.close()
            raise

        return cls(
            settings=settings,
            rag_service=rag_service,
            ollama_client=ollama_client,
            rag_adapter=rag_adapter,
            owns_rag_service=True,
            owns_ollama_client=True,
        )

    @property
    def settings(self) -> AppSettings:
        """Return the validated runtime settings."""

        return self._settings

    @property
    def rag_service(self) -> GuardedRAGService:
        """Return the managed guarded-RAG service."""

        self._require_open()

        return self._rag_service

    @property
    def ollama_client(self) -> ConfiguredOllamaClient:
        """Return the managed Ollama client."""

        self._require_open()

        return self._ollama_client

    @property
    def rag_adapter(
        self,
    ) -> ConfiguredOllamaRAGAdapter | None:
        """Return the configured guarded-RAG adapter."""

        self._require_open()

        return self._rag_adapter

    @property
    def is_closed(self) -> bool:
        """Report whether the runtime has been closed."""

        return self._closed

    @property
    def validation_report(
        self,
    ) -> RuntimeValidationReport | None:
        """Return the most recent validation report."""

        return self._validation_report

    def validate(self) -> RuntimeValidationReport:
        """
        Validate Qdrant and the configured Ollama models.

        Validation confirms that:

        1. The Qdrant collection matches the source index.
        2. Ollama is reachable.
        3. The generation model is installed.
        4. The embedding model is installed.
        """

        self._require_open()

        stored_point_count = (
            self._rag_service.validate()
        )

        available_models = tuple(
            self._ollama_client.health_check()
        )

        generation_model_available = (
            model_is_available(
                configured_model=(
                    self._settings.generation_model
                ),
                available_models=available_models,
            )
        )

        embedding_model_available = (
            model_is_available(
                configured_model=(
                    self._settings.embedding_model
                ),
                available_models=available_models,
            )
        )

        report = RuntimeValidationReport(
            stored_point_count=stored_point_count,
            source_chunk_count=(
                self._rag_service.source_chunk_count
            ),
            collection_name=(
                self._rag_service.collection_name
            ),
            retrieval_top_k=(
                self._rag_service.top_k
            ),
            available_models=available_models,
            generation_model_available=(
                generation_model_available
            ),
            embedding_model_available=(
                embedding_model_available
            ),
        )

        if not report.all_required_models_available:
            missing_models: list[str] = []

            if not generation_model_available:
                missing_models.append(
                    self._settings.generation_model
                )

            if not embedding_model_available:
                missing_models.append(
                    self._settings.embedding_model
                )

            raise RuntimeError(
                "Required Ollama models are unavailable: "
                + ", ".join(missing_models)
            )

        self._validation_report = report

        return report

    def close(self) -> None:
        """Close all runtime-owned dependencies safely."""

        if self._closed:
            return

        close_error: Exception | None = None

        if self._owns_rag_service:
            try:
                self._rag_service.close()
            except Exception as error:
                close_error = error

        if self._owns_ollama_client:
            try:
                self._ollama_client.close()
            except Exception as error:
                if close_error is None:
                    close_error = error

        self._closed = True

        if close_error is not None:
            raise RuntimeError(
                "A runtime dependency could not be "
                "closed cleanly."
            ) from close_error

    def __enter__(
        self,
    ) -> ApplicationRuntime:
        """Enter a managed runtime context."""

        self._require_open()

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close runtime dependencies when leaving context."""

        del exception_type
        del exception_value
        del traceback

        self.close()

    def _require_open(self) -> None:
        """Reject runtime operations after closure."""

        if self._closed:
            raise RuntimeError(
                "The application runtime is closed."
            )


def model_is_available(
    *,
    configured_model: str,
    available_models: tuple[str, ...],
) -> bool:
    """
    Match configured Ollama model names safely.

    Ollama may report an untagged configured model using
    the explicit ':latest' tag.
    """

    normalized_configured = (
        configured_model.strip()
    )

    normalized_available = {
        model_name.strip()
        for model_name in available_models
        if model_name.strip()
    }

    if normalized_configured in normalized_available:
        return True

    if ":" not in normalized_configured:
        return (
            f"{normalized_configured}:latest"
            in normalized_available
        )

    if normalized_configured.endswith(
        ":latest"
    ):
        untagged_name = (
            normalized_configured[
                : -len(":latest")
            ]
        )

        return untagged_name in normalized_available

    return False


def print_model_status(
    *,
    model_name: str,
    available: bool,
) -> None:
    """Print one configured model status."""

    status_text = (
        "available"
        if available
        else "missing"
    )

    print(
        f"- {model_name}: {status_text}"
    )


def main() -> None:
    """Build and validate the application runtime."""

    settings = get_settings()

    print("APPLICATION RUNTIME")
    print("===================")
    print()
    print(
        "Creating configured runtime dependencies..."
    )
    print()

    with ApplicationRuntime.from_settings(
        settings
    ) as runtime:
        report = runtime.validate()

        print("QDRANT STATUS")
        print("-------------")
        print(
            "Collection:",
            report.collection_name,
        )
        print(
            "Source chunks:",
            report.source_chunk_count,
        )
        print(
            "Stored points:",
            report.stored_point_count,
        )
        print(
            "Retrieval top K:",
            report.retrieval_top_k,
        )
        print()

        print("OLLAMA STATUS")
        print("-------------")

        for model_name in report.available_models:
            print(
                "-",
                model_name,
            )

        print()
        print("REQUIRED MODELS")
        print("---------------")

        print_model_status(
            model_name=settings.generation_model,
            available=(
                report.generation_model_available
            ),
        )

        print_model_status(
            model_name=settings.embedding_model,
            available=(
                report.embedding_model_available
            ),
        )

        print()
        print("LIFECYCLE STATUS")
        print("----------------")
        print(
            "Runtime closed inside context:",
            runtime.is_closed,
        )
        print(
            "RAG service closed inside context:",
            runtime.rag_service.is_closed,
        )
        print(
            "Ollama client closed inside context:",
            runtime.ollama_client.is_closed,
        )
        print(
            "Shared model providers configured:",
            runtime.rag_service.model_providers_configured,
        )

    print(
        "Runtime closed after context:",
        runtime.is_closed,
    )
    print()

    print("STATUS")
    print("------")
    print(
        "Qdrant and Ollama dependencies were created, "
        "validated, and closed through one runtime."
    )


if __name__ == "__main__":
    main()
