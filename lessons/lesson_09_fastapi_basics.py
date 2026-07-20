from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, status
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI(
    title="Generative AI Learning API",
    description=(
        "A small API for learning FastAPI, Pydantic, "
        "HTTP requests, and HTTP responses."
    ),
    version="0.1.0",
)


class WelcomeResponse(BaseModel):
    """Represent the response returned by the root endpoint."""

    message: str
    documentation_path: str


class HealthResponse(BaseModel):
    """Represent the current health of the API service."""

    status: Literal["ok"]
    service: str


class ApiMapping(BaseModel):
    """Represent one restaurant-to-software mapping."""

    model_config = ConfigDict(extra="forbid")

    restaurant_term: str = Field(
        min_length=1,
        max_length=50,
    )
    software_term: str = Field(
        min_length=1,
        max_length=100,
    )
    explanation: str = Field(
        min_length=10,
        max_length=300,
    )


class MappingReceipt(BaseModel):
    """Represent confirmation that a mapping was accepted."""

    accepted: bool
    mapping: ApiMapping
    message: str


@app.get(
    "/",
    response_model=WelcomeResponse,
)
def read_root() -> WelcomeResponse:
    """Return a welcome message."""

    return WelcomeResponse(
        message="The Generative AI Learning API is running.",
        documentation_path="/docs",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """Report whether the API service is running."""

    return HealthResponse(
        status="ok",
        service="generative-ai-learning-api",
    )


@app.post(
    "/mappings",
    response_model=MappingReceipt,
    status_code=status.HTTP_201_CREATED,
)
def create_mapping(mapping: ApiMapping) -> MappingReceipt:
    """Validate and accept one API analogy mapping."""

    return MappingReceipt(
        accepted=True,
        mapping=mapping,
        message="The mapping was accepted and validated.",
    )