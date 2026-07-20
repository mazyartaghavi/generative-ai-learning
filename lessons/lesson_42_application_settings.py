from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import (
    Field,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class AppSettings(BaseSettings):
    """
    Validated configuration for the guarded-RAG application.

    Values can come from:

    1. Environment variables.
    2. A local .env file.
    3. The defaults declared below.

    Environment variables use the FLEET_ prefix.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FLEET_",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    api_title: str = Field(
        default="Fleet Guarded-RAG API",
        min_length=3,
        max_length=100,
    )

    api_version: str = Field(
        default="1.0.0",
        min_length=1,
        max_length=30,
    )

    api_host: str = Field(
        default="127.0.0.1",
        min_length=1,
        max_length=255,
    )

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    vector_index_path: Path = Path(
        "data/fleet_vector_index.json"
    )

    qdrant_path: Path = Path(
        "data/qdrant_local"
    )

    qdrant_collection: str = Field(
        default="fleet_manual_chunks",
        min_length=1,
        max_length=255,
    )

    retrieval_top_k: int = Field(
        default=3,
        ge=1,
        le=100,
    )

    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        min_length=8,
        max_length=500,
    )

    generation_model: str = Field(
        default="llama3.2:3b",
        min_length=1,
        max_length=200,
    )

    embedding_model: str = Field(
        default="embeddinggemma",
        min_length=1,
        max_length=200,
    )

    ollama_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        le=600.0,
    )

    @field_validator(
        "api_title",
        "api_version",
        "api_host",
        "qdrant_collection",
        "generation_model",
        "embedding_model",
    )
    @classmethod
    def reject_blank_strings(
        cls,
        value: str,
    ) -> str:
        """Strip surrounding whitespace and reject blanks."""

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "The configuration value cannot be blank."
            )

        return cleaned_value

    @field_validator(
        "ollama_base_url"
    )
    @classmethod
    def validate_ollama_url(
        cls,
        value: str,
    ) -> str:
        """Require an HTTP or HTTPS Ollama base URL."""

        cleaned_value = value.strip().rstrip(
            "/"
        )

        if not cleaned_value.startswith(
            (
                "http://",
                "https://",
            )
        ):
            raise ValueError(
                "The Ollama base URL must begin with "
                "'http://' or 'https://'."
            )

        return cleaned_value

    @field_validator(
        "vector_index_path",
        "qdrant_path",
    )
    @classmethod
    def reject_empty_paths(
        cls,
        value: Path,
    ) -> Path:
        """Reject empty filesystem path values."""

        if not str(value).strip():
            raise ValueError(
                "A configured filesystem path "
                "cannot be empty."
            )

        return value


@lru_cache
def get_settings() -> AppSettings:
    """
    Create and cache the application settings.

    Caching prevents the .env file and environment variables
    from being parsed again for every API request.
    """

    return AppSettings()


def print_setting(
    name: str,
    value: object,
) -> None:
    """Print one configuration value consistently."""

    print(
        f"{name}:",
        value,
    )


def main() -> None:
    """Load and display the validated settings."""

    settings = get_settings()

    print("APPLICATION SETTINGS")
    print("====================")
    print()

    print_setting(
        "API title",
        settings.api_title,
    )

    print_setting(
        "API version",
        settings.api_version,
    )

    print_setting(
        "API host",
        settings.api_host,
    )

    print_setting(
        "API port",
        settings.api_port,
    )

    print_setting(
        "Vector index path",
        settings.vector_index_path,
    )

    print_setting(
        "Qdrant path",
        settings.qdrant_path,
    )

    print_setting(
        "Qdrant collection",
        settings.qdrant_collection,
    )

    print_setting(
        "Retrieval top K",
        settings.retrieval_top_k,
    )

    print_setting(
        "Ollama base URL",
        settings.ollama_base_url,
    )

    print_setting(
        "Generation model",
        settings.generation_model,
    )

    print_setting(
        "Embedding model",
        settings.embedding_model,
    )

    print_setting(
        "Ollama timeout seconds",
        settings.ollama_timeout_seconds,
    )

    print()
    print("PATH STATUS")
    print("-----------")

    print_setting(
        "Vector index exists",
        settings.vector_index_path.exists(),
    )

    print_setting(
        "Qdrant path exists",
        settings.qdrant_path.exists(),
    )

    print()
    print("STATUS")
    print("------")
    print(
        "Application configuration was loaded and "
        "validated successfully."
    )


if __name__ == "__main__":
    main()