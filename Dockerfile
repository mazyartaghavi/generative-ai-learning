# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.11.29-python3.13-trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install locked runtime dependencies in a cacheable layer.
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-install-project

# Copy the application only after dependencies are cached.
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-editable

# Run the API as an unprivileged user.
RUN groupadd --system fleetapp \
    && useradd \
        --system \
        --gid fleetapp \
        --home-dir /app \
        fleetapp \
    && chown -R fleetapp:fleetapp /app

USER fleetapp

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=90s \
    --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"]

CMD ["python", "-m", "lessons.lesson_54_observable_fastapi_app"]
