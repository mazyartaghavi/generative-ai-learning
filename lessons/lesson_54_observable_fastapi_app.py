from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Any, TextIO
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request, Response

from lessons.lesson_42_application_settings import (
    AppSettings,
    get_settings,
)
from lessons.lesson_52_runtime_fastapi_app import (
    RuntimeFactory,
    create_app,
)


REQUEST_ID_HEADER = "X-Request-ID"
LOGGER_NAME = "fleet_guarded_rag.api"


class JSONLogFormatter(logging.Formatter):
    """Format application log records as one JSON object."""

    _standard_attributes = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        """Serialize a log record using stable JSON fields."""

        payload: dict[str, Any] = {
            "timestamp": (
                datetime.fromtimestamp(
                    record.created,
                    tz=UTC,
                ).isoformat()
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if (
                key in self._standard_attributes
                or key.startswith("_")
            ):
                continue

            payload[key] = make_json_safe(
                value
            )

        if record.exc_info is not None:
            payload["exception"] = (
                self.formatException(
                    record.exc_info
                )
            )

        if record.stack_info:
            payload["stack"] = (
                self.formatStack(
                    record.stack_info
                )
            )

        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )


def make_json_safe(
    value: Any,
) -> Any:
    """Convert arbitrary logging values into JSON-safe data."""

    if value is None or isinstance(
        value,
        str | int | float | bool,
    ):
        return value

    if isinstance(
        value,
        PurePath,
    ):
        return value.as_posix()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): make_json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list | tuple | set | frozenset,
    ):
        return [
            make_json_safe(
                item
            )
            for item in value
        ]

    return str(
        value
    )


def create_json_logger(
    *,
    name: str = LOGGER_NAME,
    level: int = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Logger:
    """
    Create one non-propagating JSON application logger.

    Recreating the logger replaces its handlers so development
    reloads and repeated tests do not duplicate log messages.
    """

    logger = logging.getLogger(
        name
    )

    logger.setLevel(
        level
    )

    logger.propagate = False

    logger.handlers.clear()

    handler = logging.StreamHandler(
        stream
        if stream is not None
        else sys.stdout
    )

    handler.setLevel(
        level
    )

    handler.setFormatter(
        JSONLogFormatter()
    )

    logger.addHandler(
        handler
    )

    return logger


def normalize_request_id(
    raw_request_id: str | None,
) -> str:
    """
    Validate an incoming request ID or generate a new one.

    Request IDs are intentionally restricted to visible ASCII
    characters and 128 characters to keep logs and headers safe.
    """

    if raw_request_id is None:
        return uuid4().hex

    cleaned_request_id = (
        raw_request_id.strip()
    )

    if not cleaned_request_id:
        return uuid4().hex

    if len(
        cleaned_request_id
    ) > 128:
        return uuid4().hex

    if any(
        ord(character) < 33
        or ord(character) > 126
        for character in cleaned_request_id
    ):
        return uuid4().hex

    return cleaned_request_id


def request_log_fields(
    *,
    request: Request,
    request_id: str,
    status_code: int,
    duration_ms: float,
) -> dict[str, Any]:
    """Build stable structured fields for one request log."""

    client_host = (
        request.client.host
        if request.client is not None
        else None
    )

    return {
        "event": "http_request_completed",
        "request_id": request_id,
        "http_method": request.method,
        "http_path": request.url.path,
        "http_status_code": status_code,
        "duration_ms": round(
            duration_ms,
            3,
        ),
        "client_host": client_host,
    }


def add_observability_middleware(
    app: FastAPI,
    *,
    logger: logging.Logger,
) -> None:
    """Add request correlation and structured access logging."""

    @app.middleware(
        "http"
    )
    async def observe_request(
        request: Request,
        call_next: Callable[
            [Request],
            Awaitable[Response],
        ],
    ) -> Response:
        request_id = normalize_request_id(
            request.headers.get(
                REQUEST_ID_HEADER
            )
        )

        request.state.request_id = (
            request_id
        )

        started_at = (
            time.perf_counter()
        )

        status_code = 500

        try:
            response = await call_next(
                request
            )

            status_code = (
                response.status_code
            )

            response.headers[
                REQUEST_ID_HEADER
            ] = request_id

            return response
        except Exception:
            duration_ms = (
                time.perf_counter()
                - started_at
            ) * 1000.0

            logger.exception(
                "HTTP request failed.",
                extra=request_log_fields(
                    request=request,
                    request_id=request_id,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )
                | {
                    "event": (
                        "http_request_failed"
                    )
                },
            )

            raise
        finally:
            if status_code != 500:
                duration_ms = (
                    time.perf_counter()
                    - started_at
                ) * 1000.0

                logger.info(
                    "HTTP request completed.",
                    extra=request_log_fields(
                        request=request,
                        request_id=request_id,
                        status_code=status_code,
                        duration_ms=duration_ms,
                    ),
                )


def create_observable_app(
    settings: AppSettings,
    *,
    runtime_factory: RuntimeFactory | None = None,
    logger: logging.Logger | None = None,
) -> FastAPI:
    """
    Create the runtime FastAPI application with observability.

    The base application still owns dependency startup and
    shutdown. This function adds request IDs and JSON access
    logs without changing the guarded-RAG business logic.
    """

    active_logger = (
        create_json_logger()
        if logger is None
        else logger
    )

    app = create_app(
        settings,
        runtime_factory=runtime_factory,
    )

    app.state.api_logger = (
        active_logger
    )

    add_observability_middleware(
        app,
        logger=active_logger,
    )

    return app


settings = get_settings()

api_logger = create_json_logger()

app = create_observable_app(
    settings,
    logger=api_logger,
)


def main() -> None:
    """Run the observable runtime FastAPI application."""

    print("OBSERVABLE RUNTIME FASTAPI APPLICATION")
    print("======================================")
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
    print(
        "Structured logger:",
        api_logger.name,
    )
    print(
        "Request ID header:",
        REQUEST_ID_HEADER,
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
        access_log=False,
    )


if __name__ == "__main__":
    main()
