from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import httpx

from lessons.lesson_42_application_settings import (
    AppSettings,
    get_settings,
)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Validated result from Ollama text generation."""

    model: str
    response: str
    prompt_tokens: int | None
    generated_tokens: int | None
    total_duration_nanoseconds: int | None


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Validated result from Ollama embedding generation."""

    model: str
    embeddings: list[list[float]]
    input_tokens: int | None
    total_duration_nanoseconds: int | None

    @property
    def embedding_count(self) -> int:
        """Return the number of generated vectors."""

        return len(self.embeddings)

    @property
    def embedding_dimension(self) -> int:
        """Return the shared vector dimension."""

        if not self.embeddings:
            return 0

        return len(self.embeddings[0])


class ConfiguredOllamaClient:
    """
    Reusable HTTP client for a configured Ollama server.

    The client obtains its base URL, models, and timeout
    from AppSettings.
    """

    def __init__(
        self,
        settings: AppSettings,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the configured Ollama client."""

        self._settings = settings
        self._closed = False

        if http_client is None:
            self._http_client = httpx.Client(
                base_url=settings.ollama_base_url,
                timeout=httpx.Timeout(
                    settings.ollama_timeout_seconds
                ),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

            self._owns_http_client = True
        else:
            self._http_client = http_client
            self._owns_http_client = False

    @property
    def base_url(self) -> str:
        """Return the configured Ollama base URL."""

        return self._settings.ollama_base_url

    @property
    def generation_model(self) -> str:
        """Return the configured generation model."""

        return self._settings.generation_model

    @property
    def embedding_model(self) -> str:
        """Return the configured embedding model."""

        return self._settings.embedding_model

    @property
    def timeout_seconds(self) -> float:
        """Return the configured request timeout."""

        return self._settings.ollama_timeout_seconds

    @property
    def is_closed(self) -> bool:
        """Report whether the component was closed."""

        return self._closed

    def health_check(self) -> list[str]:
        """
        Confirm that Ollama is reachable and list models.

        The Ollama tags endpoint returns information about
        models currently available to the local runtime.
        """

        self._require_open()

        payload = self._get_json(
            endpoint="/api/tags"
        )

        raw_models = payload.get(
            "models"
        )

        if not isinstance(raw_models, list):
            raise RuntimeError(
                "Ollama returned an invalid models response."
            )

        model_names: list[str] = []

        for raw_model in raw_models:
            if not isinstance(raw_model, dict):
                continue

            name = raw_model.get(
                "name"
            )

            if isinstance(name, str) and name.strip():
                model_names.append(
                    name.strip()
                )

        return model_names

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate one non-streaming text response."""

        self._require_open()

        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise ValueError(
                "The generation prompt cannot be empty."
            )

        if temperature < 0.0:
            raise ValueError(
                "Temperature cannot be negative."
            )

        request_payload: dict[str, Any] = {
            "model": self.generation_model,
            "prompt": cleaned_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system_prompt is not None:
            cleaned_system_prompt = (
                system_prompt.strip()
            )

            if not cleaned_system_prompt:
                raise ValueError(
                    "The system prompt cannot be blank."
                )

            request_payload["system"] = (
                cleaned_system_prompt
            )

        payload = self._post_json(
            endpoint="/api/generate",
            request_payload=request_payload,
        )

        model = self._require_string(
            payload=payload,
            field_name="model",
        )

        response_text = self._require_string(
            payload=payload,
            field_name="response",
        )

        prompt_tokens = self._optional_integer(
            payload=payload,
            field_name="prompt_eval_count",
        )

        generated_tokens = self._optional_integer(
            payload=payload,
            field_name="eval_count",
        )

        total_duration = self._optional_integer(
            payload=payload,
            field_name="total_duration",
        )

        return GenerationResult(
            model=model,
            response=response_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_duration_nanoseconds=(
                total_duration
            ),
        )

    def embed(
        self,
        inputs: str | Sequence[str],
    ) -> EmbeddingResult:
        """Generate embeddings for one or several texts."""

        self._require_open()

        normalized_inputs = self._normalize_inputs(
            inputs
        )

        request_input: str | list[str]

        if isinstance(inputs, str):
            request_input = normalized_inputs[0]
        else:
            request_input = normalized_inputs

        payload = self._post_json(
            endpoint="/api/embed",
            request_payload={
                "model": self.embedding_model,
                "input": request_input,
            },
        )

        model = self._require_string(
            payload=payload,
            field_name="model",
        )

        embeddings = self._require_embeddings(
            payload
        )

        if len(embeddings) != len(
            normalized_inputs
        ):
            raise RuntimeError(
                "Ollama returned a different number of "
                "embeddings than requested."
            )

        self._validate_shared_dimension(
            embeddings
        )

        input_tokens = self._optional_integer(
            payload=payload,
            field_name="prompt_eval_count",
        )

        total_duration = self._optional_integer(
            payload=payload,
            field_name="total_duration",
        )

        return EmbeddingResult(
            model=model,
            embeddings=embeddings,
            input_tokens=input_tokens,
            total_duration_nanoseconds=(
                total_duration
            ),
        )

    def close(self) -> None:
        """Close the owned HTTPX client safely."""

        if self._closed:
            return

        if self._owns_http_client:
            self._http_client.close()

        self._closed = True

    def __enter__(
        self,
    ) -> ConfiguredOllamaClient:
        """Enter a context-manager block."""

        self._require_open()

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the component after leaving the block."""

        del exception_type
        del exception_value
        del traceback

        self.close()

    def _get_json(
        self,
        *,
        endpoint: str,
    ) -> dict[str, Any]:
        """Send a GET request and return a JSON object."""

        try:
            response = self._http_client.get(
                endpoint
            )

            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise RuntimeError(
                "The Ollama request timed out."
            ) from error
        except httpx.RequestError as error:
            raise RuntimeError(
                "Could not connect to the configured "
                f"Ollama server at {self.base_url}."
            ) from error
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                "Ollama returned HTTP status "
                f"{error.response.status_code}."
            ) from error

        return self._decode_json_object(
            response
        )

    def _post_json(
        self,
        *,
        endpoint: str,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a JSON POST request and decode its response."""

        try:
            response = self._http_client.post(
                endpoint,
                json=request_payload,
            )

            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise RuntimeError(
                "The Ollama request timed out."
            ) from error
        except httpx.RequestError as error:
            raise RuntimeError(
                "Could not connect to the configured "
                f"Ollama server at {self.base_url}."
            ) from error
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                "Ollama returned HTTP status "
                f"{error.response.status_code}."
            ) from error

        return self._decode_json_object(
            response
        )

    @staticmethod
    def _decode_json_object(
        response: httpx.Response,
    ) -> dict[str, Any]:
        """Decode and validate a JSON object response."""

        try:
            payload = response.json()
        except ValueError as error:
            raise RuntimeError(
                "Ollama returned invalid JSON."
            ) from error

        if not isinstance(payload, dict):
            raise RuntimeError(
                "Ollama returned JSON that was not "
                "an object."
            )

        return payload

    @staticmethod
    def _normalize_inputs(
        inputs: str | Sequence[str],
    ) -> list[str]:
        """Validate and normalize embedding inputs."""

        if isinstance(inputs, str):
            candidates = [
                inputs
            ]
        else:
            candidates = list(
                inputs
            )

        if not candidates:
            raise ValueError(
                "At least one embedding input is required."
            )

        normalized: list[str] = []

        for candidate in candidates:
            if not isinstance(candidate, str):
                raise TypeError(
                    "Every embedding input must be a string."
                )

            cleaned_candidate = candidate.strip()

            if not cleaned_candidate:
                raise ValueError(
                    "Embedding inputs cannot be blank."
                )

            normalized.append(
                cleaned_candidate
            )

        return normalized

    @staticmethod
    def _require_string(
        *,
        payload: dict[str, Any],
        field_name: str,
    ) -> str:
        """Read and validate a required string field."""

        value = payload.get(
            field_name
        )

        if not isinstance(value, str):
            raise RuntimeError(
                "Ollama returned an invalid "
                f"{field_name!r} field."
            )

        cleaned_value = value.strip()

        if not cleaned_value:
            raise RuntimeError(
                "Ollama returned an empty "
                f"{field_name!r} field."
            )

        return cleaned_value

    @staticmethod
    def _optional_integer(
        *,
        payload: dict[str, Any],
        field_name: str,
    ) -> int | None:
        """Read an optional non-negative integer field."""

        value = payload.get(
            field_name
        )

        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise RuntimeError(
                "Ollama returned an invalid "
                f"{field_name!r} field."
            )

        return value

    @staticmethod
    def _require_embeddings(
        payload: dict[str, Any],
    ) -> list[list[float]]:
        """Read and validate Ollama embedding vectors."""

        raw_embeddings = payload.get(
            "embeddings"
        )

        if not isinstance(raw_embeddings, list):
            raise RuntimeError(
                "Ollama returned an invalid embeddings field."
            )

        embeddings: list[list[float]] = []

        for raw_vector in raw_embeddings:
            if (
                not isinstance(raw_vector, list)
                or not raw_vector
            ):
                raise RuntimeError(
                    "Ollama returned an invalid "
                    "embedding vector."
                )

            vector: list[float] = []

            for raw_value in raw_vector:
                if (
                    isinstance(raw_value, bool)
                    or not isinstance(
                        raw_value,
                        int | float,
                    )
                ):
                    raise RuntimeError(
                        "An embedding contains a "
                        "non-numeric value."
                    )

                vector.append(
                    float(raw_value)
                )

            embeddings.append(
                vector
            )

        if not embeddings:
            raise RuntimeError(
                "Ollama returned no embeddings."
            )

        return embeddings

    @staticmethod
    def _validate_shared_dimension(
        embeddings: list[list[float]],
    ) -> None:
        """Ensure every returned vector has equal length."""

        expected_dimension = len(
            embeddings[0]
        )

        if expected_dimension <= 0:
            raise RuntimeError(
                "The embedding dimension must be positive."
            )

        for vector in embeddings:
            if len(vector) != expected_dimension:
                raise RuntimeError(
                    "Ollama returned embeddings with "
                    "inconsistent dimensions."
                )

    def _require_open(self) -> None:
        """Reject work after the component has closed."""

        if self._closed:
            raise RuntimeError(
                "The configured Ollama client is closed."
            )


def print_models(
    model_names: list[str],
) -> None:
    """Print locally available Ollama models."""

    print("AVAILABLE OLLAMA MODELS")
    print("-----------------------")

    if not model_names:
        print(
            "No local Ollama models were reported."
        )
        print()
        return

    for model_name in model_names:
        print(
            "-",
            model_name,
        )

    print()


def main() -> None:
    """Demonstrate the settings-driven Ollama client."""

    settings = get_settings()

    print("CONFIGURED OLLAMA CLIENT")
    print("========================")
    print()
    print(
        "Base URL:",
        settings.ollama_base_url,
    )
    print(
        "Generation model:",
        settings.generation_model,
    )
    print(
        "Embedding model:",
        settings.embedding_model,
    )
    print(
        "Timeout seconds:",
        settings.ollama_timeout_seconds,
    )
    print()

    with ConfiguredOllamaClient(
        settings
    ) as client:
        print(
            "1. Checking the Ollama server..."
        )

        model_names = client.health_check()

        print(
            "   Ollama server responded."
        )
        print()

        print_models(
            model_names
        )

        required_models = {
            client.generation_model,
            client.embedding_model,
        }

        missing_models = sorted(
            model_name
            for model_name in required_models
            if model_name not in model_names
            and f"{model_name}:latest"
            not in model_names
        )

        if missing_models:
            raise RuntimeError(
                "Required Ollama models are missing: "
                + ", ".join(missing_models)
            )

        print(
            "2. Generating a configured response..."
        )

        generation_result = client.generate(
            (
                "Use only these supplied facts: Qdrant is a "
                "vector database; RAG means retrieval-augmented "
                "generation; Qdrant stores embeddings and "
                "retrieves semantically similar document chunks. "
                "Explain Qdrant's role in a RAG system in exactly "
                "one concise sentence."
            ),
            system_prompt=(
                "You are a precise AI engineering tutor. "
                "Preserve technical product names exactly and "
                "do not replace supplied definitions."
            ),
            temperature=0.0,
        )

        print(
            "   Generation completed."
        )
        print()

        print("GENERATION RESULT")
        print("-----------------")
        print(
            "Model:",
            generation_result.model,
        )
        print(
            "Prompt tokens:",
            generation_result.prompt_tokens,
        )
        print(
            "Generated tokens:",
            generation_result.generated_tokens,
        )
        print(
            "Response:"
        )
        print(
            generation_result.response
        )
        print()

        print(
            "3. Generating configured embeddings..."
        )

        embedding_result = client.embed(
            [
                (
                    "Qdrant stores and searches document "
                    "embeddings."
                ),
                (
                    "Guarded RAG abstains when retrieval "
                    "confidence is too low."
                ),
            ]
        )

        print(
            "   Embeddings generated."
        )
        print()

        print("EMBEDDING RESULT")
        print("----------------")
        print(
            "Model:",
            embedding_result.model,
        )
        print(
            "Embedding count:",
            embedding_result.embedding_count,
        )
        print(
            "Embedding dimension:",
            embedding_result.embedding_dimension,
        )
        print(
            "Input tokens:",
            embedding_result.input_tokens,
        )
        print()

        print(
            "Client closed inside context:",
            client.is_closed,
        )

    print(
        "Client closed after context:",
        client.is_closed,
    )
    print()

    print("STATUS")
    print("------")
    print(
        "Ollama generation and embedding configuration "
        "now comes from AppSettings."
    )


if __name__ == "__main__":
    main()