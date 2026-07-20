from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

WORKFLOW_PATH = Path(".github/workflows/ci.yml")


def require(condition: bool, message: str) -> None:
    """Raise a clear workflow-validation error."""
    if not condition:
        raise RuntimeError(message)


def require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Require a dictionary-like YAML value."""
    require(isinstance(value, dict), f"{name} must be a YAML mapping.")
    return value


def require_sequence(value: Any, *, name: str) -> list[Any]:
    """Require a list-like YAML value."""
    require(isinstance(value, list), f"{name} must be a YAML sequence.")
    return value


def step_names(job: dict[str, Any]) -> set[str]:
    """Return all explicit step names from one job."""
    steps = require_sequence(job.get("steps"), name="job steps")
    return {
        str(step.get("name"))
        for step in steps
        if isinstance(step, dict) and step.get("name") is not None
    }


def main() -> None:
    """Validate the local GitHub Actions workflow."""
    print("LESSON 57 — CI WORKFLOW VALIDATION")
    print("==================================")
    print()

    require(WORKFLOW_PATH.is_file(), f"The workflow file is missing: {WORKFLOW_PATH}")
    raw_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    parsed = yaml.load(raw_text, Loader=yaml.BaseLoader)
    workflow = require_mapping(parsed, name="workflow")

    require(workflow.get("name") == "Continuous Integration", "The workflow name is incorrect.")
    triggers = require_mapping(workflow.get("on"), name="workflow triggers")
    require(
        {"push", "pull_request", "workflow_dispatch"}.issubset(triggers),
        "The workflow must support pushes, pull requests, and manual runs.",
    )

    permissions = require_mapping(workflow.get("permissions"), name="workflow permissions")
    require(permissions.get("contents") == "read", "The workflow must use read-only contents permission.")

    jobs = require_mapping(workflow.get("jobs"), name="workflow jobs")
    require(set(jobs) == {"python-tests", "docker-build"}, "The workflow must contain exactly two required jobs.")

    python_job = require_mapping(jobs["python-tests"], name="python-tests job")
    docker_job = require_mapping(jobs["docker-build"], name="docker-build job")
    require(python_job.get("runs-on") == "ubuntu-latest", "The Python job must run on ubuntu-latest.")
    require(docker_job.get("runs-on") == "ubuntu-latest", "The Docker job must run on ubuntu-latest.")
    require(docker_job.get("needs") == "python-tests", "The Docker build must wait for Python tests.")

    required_python_steps = {
        "Check out repository", "Set up Python", "Install uv", "Verify lock file",
        "Install locked dependencies", "Compile Python sources", "Run test suite",
    }
    require(required_python_steps.issubset(step_names(python_job)), "The Python job is missing required steps.")
    require(
        {"Check out repository", "Build container image"}.issubset(step_names(docker_job)),
        "The Docker job is missing required steps.",
    )

    required_text = (
        "actions/checkout@v7",
        "actions/setup-python@v6",
        "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b",
        'version: "0.11.29"',
        "uv lock --check",
        "uv sync --locked --all-extras --dev",
        "uv run python -m compileall lessons tests scripts -q",
        "uv run python -m pytest -q",
        "docker build",
        "--tag fleet-guarded-rag:ci",
    )
    for expected_text in required_text:
        require(expected_text in raw_text, f"The workflow is missing required text: {expected_text}")

    print(f"Workflow: {WORKFLOW_PATH}")
    print("Triggers: push, pull_request, workflow_dispatch")
    print("Permissions: contents read")
    print("Jobs: python-tests, docker-build")
    print("Python: 3.13")
    print("uv: 0.11.29")
    print()
    print("STATUS")
    print("------")
    print("The GitHub Actions workflow is structurally valid and contains all required CI checks.")


if __name__ == "__main__":
    main()
