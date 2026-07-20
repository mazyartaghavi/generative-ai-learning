from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lessons.lesson_54_observable_fastapi_app import (
    LOGGER_NAME,
    REQUEST_ID_HEADER,
    JSONLogFormatter,
    add_observability_middleware,
    create_json_logger,
    make_json_safe,
    normalize_request_id,
)


def unique_logger_name() -> str:
    """Return an isolated logger name for one test."""

    return (
        f"{LOGGER_NAME}.tests."
        f"{uuid4().hex}"
    )


def read_json_log_lines(
    stream: io.StringIO,
) -> list[dict[str, object]]:
    """Decode every non-empty JSON log line."""

    return [
        json.loads(line)
        for line in stream.getvalue().splitlines()
        if line.strip()
    ]


def make_test_app(
    *,
    logger: logging.Logger,
) -> FastAPI:
    """Create a small API protected by observability middleware."""

    app = FastAPI()

    add_observability_middleware(
        app,
        logger=logger,
    )

    @app.get("/ok")
    def ok() -> dict[str, str]:
        """Return one successful response."""

        return {
            "status": "ok",
        }

    @app.get("/boom")
    def boom() -> dict[str, str]:
        """Raise one unexpected application exception."""

        raise RuntimeError(
            "Simulated application failure."
        )

    return app


def test_json_log_formatter_serializes_standard_and_extra_fields() -> None:
    """JSON logs should contain stable fields and safe extras."""

    stream = io.StringIO()

    logger = create_json_logger(
        name=unique_logger_name(),
        stream=stream,
    )

    logger.info(
        "Observed test event.",
        extra={
            "event": "unit_test",
            "count": 3,
            "path": Path("data/example.json"),
            "nested": {
                "values": (
                    1,
                    2,
                )
            },
        },
    )

    records = read_json_log_lines(
        stream
    )

    assert len(records) == 1

    record = records[0]

    assert record["level"] == "INFO"
    assert record["message"] == (
        "Observed test event."
    )
    assert record["event"] == "unit_test"
    assert record["count"] == 3
    assert record["path"] == (
        "data/example.json"
    )
    assert record["nested"] == {
        "values": [
            1,
            2,
        ]
    }

    timestamp = record["timestamp"]

    assert isinstance(
        timestamp,
        str,
    )
    assert timestamp.endswith(
        "+00:00"
    )


def test_make_json_safe_handles_nested_values() -> None:
    """Arbitrary nested values should become JSON-compatible."""

    value = {
        "path": Path("data/index.json"),
        "items": (
            1,
            True,
            None,
        ),
        "set": {
            "alpha",
            "beta",
        },
    }

    converted = make_json_safe(
        value
    )

    assert converted["path"] == (
        "data/index.json"
    )
    assert converted["items"] == [
        1,
        True,
        None,
    ]

    assert sorted(
        converted["set"]
    ) == [
        "alpha",
        "beta",
    ]

    json.dumps(
        converted
    )


def test_recreating_logger_replaces_handlers() -> None:
    """Repeated setup should not duplicate emitted records."""

    logger_name = unique_logger_name()

    first_stream = io.StringIO()
    second_stream = io.StringIO()

    create_json_logger(
        name=logger_name,
        stream=first_stream,
    )

    logger = create_json_logger(
        name=logger_name,
        stream=second_stream,
    )

    assert len(
        logger.handlers
    ) == 1

    logger.info(
        "Only the active handler should receive this."
    )

    assert first_stream.getvalue() == ""

    records = read_json_log_lines(
        second_stream
    )

    assert len(records) == 1


@pytest.mark.parametrize(
    (
        "raw_request_id",
        "should_preserve",
    ),
    [
        (
            None,
            False,
        ),
        (
            "",
            False,
        ),
        (
            "   ",
            False,
        ),
        (
            "lesson-55-request",
            True,
        ),
        (
            "x" * 128,
            True,
        ),
        (
            "x" * 129,
            False,
        ),
    ],
)
def test_normalize_request_id_validates_or_generates(
    raw_request_id: str | None,
    should_preserve: bool,
) -> None:
    """Request IDs should be reused only when safe."""

    normalized = normalize_request_id(
        raw_request_id
    )

    if should_preserve:
        assert normalized == (
            raw_request_id.strip()
        )
        return

    assert re.fullmatch(
        r"[0-9a-f]{32}",
        normalized,
    )


def test_supplied_request_id_is_echoed_and_logged() -> None:
    """A safe caller-provided request ID should be correlated."""

    stream = io.StringIO()

    logger = create_json_logger(
        name=unique_logger_name(),
        stream=stream,
    )

    app = make_test_app(
        logger=logger
    )

    with TestClient(app) as client:
        response = client.get(
            "/ok",
            headers={
                REQUEST_ID_HEADER: (
                    "lesson-55-supplied"
                )
            },
        )

    assert response.status_code == 200

    assert (
        response.headers[
            REQUEST_ID_HEADER
        ]
        == "lesson-55-supplied"
    )

    records = read_json_log_lines(
        stream
    )

    assert len(records) == 1

    record = records[0]

    assert record["event"] == (
        "http_request_completed"
    )
    assert record["request_id"] == (
        "lesson-55-supplied"
    )
    assert record["http_method"] == "GET"
    assert record["http_path"] == "/ok"
    assert record["http_status_code"] == 200

    duration_ms = record[
        "duration_ms"
    ]

    assert isinstance(
        duration_ms,
        int | float,
    )
    assert duration_ms >= 0.0


def test_missing_request_id_is_generated_and_correlated() -> None:
    """The middleware should create and return a request ID."""

    stream = io.StringIO()

    logger = create_json_logger(
        name=unique_logger_name(),
        stream=stream,
    )

    app = make_test_app(
        logger=logger
    )

    with TestClient(app) as client:
        response = client.get(
            "/ok"
        )

    generated_request_id = (
        response.headers[
            REQUEST_ID_HEADER
        ]
    )

    assert re.fullmatch(
        r"[0-9a-f]{32}",
        generated_request_id,
    )

    records = read_json_log_lines(
        stream
    )

    assert records[0]["request_id"] == (
        generated_request_id
    )


def test_unsafe_request_id_is_replaced() -> None:
    """An oversized request ID should not enter logs or responses."""

    stream = io.StringIO()

    logger = create_json_logger(
        name=unique_logger_name(),
        stream=stream,
    )

    app = make_test_app(
        logger=logger
    )

    unsafe_request_id = "x" * 129

    with TestClient(app) as client:
        response = client.get(
            "/ok",
            headers={
                REQUEST_ID_HEADER: (
                    unsafe_request_id
                )
            },
        )

    returned_request_id = (
        response.headers[
            REQUEST_ID_HEADER
        ]
    )

    assert returned_request_id != (
        unsafe_request_id
    )

    assert re.fullmatch(
        r"[0-9a-f]{32}",
        returned_request_id,
    )

    records = read_json_log_lines(
        stream
    )

    assert records[0]["request_id"] == (
        returned_request_id
    )


def test_unhandled_exception_is_logged_as_failure() -> None:
    """Unexpected exceptions should create a structured failure log."""

    stream = io.StringIO()

    logger = create_json_logger(
        name=unique_logger_name(),
        stream=stream,
    )

    app = make_test_app(
        logger=logger
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.get(
            "/boom",
            headers={
                REQUEST_ID_HEADER: (
                    "lesson-55-failure"
                )
            },
        )

    assert response.status_code == 500

    records = read_json_log_lines(
        stream
    )

    assert len(records) == 1

    record = records[0]

    assert record["event"] == (
        "http_request_failed"
    )
    assert record["request_id"] == (
        "lesson-55-failure"
    )
    assert record["http_status_code"] == 500
    assert record["http_path"] == "/boom"

    exception_text = record[
        "exception"
    ]

    assert isinstance(
        exception_text,
        str,
    )
    assert (
        "Simulated application failure."
        in exception_text
    )


def test_formatter_class_is_attached_to_json_logger() -> None:
    """The configured handler should use JSONLogFormatter."""

    logger = create_json_logger(
        name=unique_logger_name(),
        stream=io.StringIO(),
    )

    assert len(
        logger.handlers
    ) == 1

    formatter = (
        logger.handlers[0].formatter
    )

    assert isinstance(
        formatter,
        JSONLogFormatter,
    )
