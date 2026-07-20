from __future__ import annotations

import time
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"


app = FastAPI(
    title="Local LLM Chat API",
    description=(
        "A FastAPI service that validates chat requests "
        "and sends them to a local Ollama model."
    ),
    version="0.1.0",
)


class HealthResponse(BaseModel):
    """Represent the health of the FastAPI service."""

    status: Literal["ok"]
    service: str
    model: str


class ChatRequest(BaseModel):
    """Represent one user request sent to the LLM."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        min_length=1,
        max_length=2_000,
    )
    system_prompt: str = Field(
        default=(
            "You are a careful Generative AI tutor. "
            "Explain technical concepts clearly and accurately."
        ),
        min_length=10,
        max_length=1_000,
    )


class ChatResponse(BaseModel):
    """Represent the validated response returned to the client."""

    answer: str = Field(min_length=1)
    model: str
    elapsed_seconds: float = Field(ge=0)


async def ask_ollama(chat_request: ChatRequest) -> str:
    """Send an asynchronous chat request to Ollama."""

    request_body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": chat_request.system_prompt,
            },
            {
                "role": "user",
                "content": chat_request.prompt,
            },
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(
            timeout=120.0,
        ) as client:
            response = await client.post(
                OLLAMA_CHAT_URL,
                json=request_body,
            )

        response.raise_for_status()

    except httpx.ConnectError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Could not connect to Ollama. "
                "Confirm that the Ollama application is running."
            ),
        ) from error

    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "Ollama did not return an answer "
                "before the 120-second timeout."
            ),
        ) from error

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Ollama returned an unsuccessful HTTP status: "
                f"{error.response.status_code}."
            ),
        ) from error

    try:
        response_body = response.json()
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Ollama returned a response that was not valid JSON.",
        ) from error

    message = response_body.get("message")

    if not isinstance(message, dict):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Ollama returned an invalid message structure.",
        )

    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Ollama returned an empty or invalid answer.",
        )

    return content.strip()


@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    """Report whether the FastAPI service is running."""

    return HealthResponse(
        status="ok",
        service="local-llm-chat-api",
        model=MODEL_NAME,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def create_chat(
    chat_request: ChatRequest,
) -> ChatResponse:
    """Generate an answer using the local Ollama model."""

    start_time = time.perf_counter()

    answer = await ask_ollama(chat_request)

    elapsed_seconds = time.perf_counter() - start_time

    return ChatResponse(
        answer=answer,
        model=MODEL_NAME,
        elapsed_seconds=round(elapsed_seconds, 3),
    )