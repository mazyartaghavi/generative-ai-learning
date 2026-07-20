from __future__ import annotations

import inspect
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from lessons import lesson_28_guarded_rag
from lessons import lesson_35_qdrant_retriever_component
from lessons import lesson_36_guarded_rag_service
from lessons import lesson_45_configured_ollama_client
from lessons import lesson_47_application_runtime


REPORT_PATH = Path(
    "data/lesson_49_integration_interface_audit.txt"
)

MODULES: tuple[
    tuple[str, ModuleType],
    ...,
] = (
    (
        "Lesson 28 — Guarded RAG",
        lesson_28_guarded_rag,
    ),
    (
        "Lesson 35 — Qdrant Retriever",
        lesson_35_qdrant_retriever_component,
    ),
    (
        "Lesson 36 — Guarded-RAG Service",
        lesson_36_guarded_rag_service,
    ),
    (
        "Lesson 45 — Configured Ollama Client",
        lesson_45_configured_ollama_client,
    ),
    (
        "Lesson 47 — Application Runtime",
        lesson_47_application_runtime,
    ),
)


def heading(
    title: str,
    marker: str = "=",
) -> str:
    """Create a consistently formatted report heading."""

    return "\n".join(
        (
            title,
            marker * len(title),
        )
    )


def safe_signature(
    target: object,
) -> str:
    """Return a target signature without crashing the audit."""

    try:
        return str(
            inspect.signature(
                target
            )
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        return (
            "<signature unavailable: "
            f"{type(error).__name__}: {error}>"
        )


def safe_source(
    target: object,
) -> str:
    """Return Python source without crashing the audit."""

    try:
        return inspect.getsource(
            target
        )
    except (
        OSError,
        TypeError,
    ) as error:
        return (
            "<source unavailable: "
            f"{type(error).__name__}: {error}>"
        )


def public_members(
    target: object,
) -> list[
    tuple[
        str,
        object,
    ]
]:
    """Return public callable members in alphabetical order."""

    members: list[
        tuple[
            str,
            object,
        ]
    ] = []

    for name, member in inspect.getmembers(
        target
    ):
        if name.startswith("_"):
            continue

        if (
            inspect.isfunction(member)
            or inspect.ismethod(member)
            or inspect.isclass(member)
        ):
            members.append(
                (
                    name,
                    member,
                )
            )

    return members


def render_callable_summary(
    *,
    qualified_name: str,
    target: object,
) -> str:
    """Render a callable name and signature."""

    return "\n".join(
        (
            f"TARGET: {qualified_name}",
            f"TYPE: {type(target).__name__}",
            f"SIGNATURE: {safe_signature(target)}",
        )
    )


def render_class_summary(
    *,
    qualified_name: str,
    target_class: type[Any],
) -> str:
    """Render a class and its public callable members."""

    lines = [
        f"CLASS: {qualified_name}",
        (
            "CONSTRUCTOR SIGNATURE: "
            f"{safe_signature(target_class)}"
        ),
        "",
        "PUBLIC CALLABLE MEMBERS:",
    ]

    members = public_members(
        target_class
    )

    if not members:
        lines.append(
            "- No public callable members found."
        )
    else:
        for name, member in members:
            lines.append(
                f"- {name}{safe_signature(member)}"
            )

    return "\n".join(
        lines
    )


def find_named_target(
    module: ModuleType,
    name: str,
) -> object | None:
    """Read a named target from a module."""

    return getattr(
        module,
        name,
        None,
    )


def render_key_interfaces() -> str:
    """Render signatures and source for integration targets."""

    sections: list[str] = [
        heading(
            "KEY INTEGRATION INTERFACES"
        )
    ]

    target_specs: tuple[
        tuple[
            str,
            ModuleType,
            str,
        ],
        ...,
    ] = (
        (
            "lesson_28_guarded_rag.run_guarded_rag",
            lesson_28_guarded_rag,
            "run_guarded_rag",
        ),
        (
            (
                "lesson_35_qdrant_retriever_component."
                "QdrantRetriever"
            ),
            lesson_35_qdrant_retriever_component,
            "QdrantRetriever",
        ),
        (
            (
                "lesson_36_guarded_rag_service."
                "GuardedRAGService"
            ),
            lesson_36_guarded_rag_service,
            "GuardedRAGService",
        ),
        (
            (
                "lesson_45_configured_ollama_client."
                "ConfiguredOllamaClient"
            ),
            lesson_45_configured_ollama_client,
            "ConfiguredOllamaClient",
        ),
        (
            (
                "lesson_47_application_runtime."
                "ApplicationRuntime"
            ),
            lesson_47_application_runtime,
            "ApplicationRuntime",
        ),
    )

    for qualified_name, module, target_name in target_specs:
        target = find_named_target(
            module,
            target_name,
        )

        sections.append(
            ""
        )

        if target is None:
            sections.append(
                "\n".join(
                    (
                        f"TARGET: {qualified_name}",
                        "STATUS: NOT FOUND",
                    )
                )
            )
            continue

        if inspect.isclass(
            target
        ):
            sections.append(
                render_class_summary(
                    qualified_name=qualified_name,
                    target_class=target,
                )
            )
        else:
            sections.append(
                render_callable_summary(
                    qualified_name=qualified_name,
                    target=target,
                )
            )

        sections.append(
            ""
        )

        sections.append(
            "SOURCE:"
        )

        sections.append(
            safe_source(
                target
            )
        )

    return "\n".join(
        sections
    )


def render_generation_candidates() -> str:
    """Find likely generation and embedding integration targets."""

    sections: list[str] = [
        heading(
            "GENERATION AND EMBEDDING CANDIDATES"
        )
    ]

    keywords = (
        "generate",
        "generation",
        "embed",
        "embedding",
        "ollama",
        "answer",
        "retrieve",
        "search",
    )

    for module_title, module in MODULES:
        sections.extend(
            (
                "",
                heading(
                    module_title,
                    "-",
                ),
            )
        )

        matches: list[
            tuple[
                str,
                object,
            ]
        ] = []

        for name, member in inspect.getmembers(
            module
        ):
            lowered_name = name.lower()

            if not any(
                keyword in lowered_name
                for keyword in keywords
            ):
                continue

            if (
                inspect.isfunction(member)
                or inspect.isclass(member)
            ):
                matches.append(
                    (
                        name,
                        member,
                    )
                )

        if not matches:
            sections.append(
                "No matching callable targets found."
            )
            continue

        for name, member in matches:
            sections.append(
                (
                    f"{name}: "
                    f"{safe_signature(member)}"
                )
            )

    return "\n".join(
        sections
    )


def read_module_file(
    module: ModuleType,
) -> str:
    """Read one imported module's complete source file."""

    module_file = getattr(
        module,
        "__file__",
        None,
    )

    if module_file is None:
        return (
            "<module does not expose a source-file path>"
        )

    path = Path(
        module_file
    ).resolve()

    try:
        source = path.read_text(
            encoding="utf-8"
        )
    except OSError as error:
        return (
            f"<could not read {path}: "
            f"{type(error).__name__}: {error}>"
        )

    return "\n".join(
        (
            f"FILE: {path}",
            "",
            source,
        )
    )


def render_complete_module_sources() -> str:
    """Render complete source for all integration modules."""

    sections: list[str] = [
        heading(
            "COMPLETE MODULE SOURCES"
        )
    ]

    for module_title, module in MODULES:
        sections.extend(
            (
                "",
                heading(
                    module_title,
                    "-",
                ),
                read_module_file(
                    module
                ),
            )
        )

    return "\n".join(
        sections
    )


def build_report() -> str:
    """Build the complete integration audit report."""

    environment = "\n".join(
        (
            heading(
                "LESSON 49 — INTEGRATION INTERFACE AUDIT"
            ),
            "",
            heading(
                "ENVIRONMENT"
            ),
            f"Python executable: {sys.executable}",
            f"Python version: {platform.python_version()}",
            f"Platform: {platform.platform()}",
            f"Project root: {Path.cwd().resolve()}",
            "",
        )
    )

    return "\n\n".join(
        (
            environment,
            render_key_interfaces(),
            render_generation_candidates(),
            render_complete_module_sources(),
        )
    )


def main() -> None:
    """Create the local integration-interface report."""

    report = build_report()

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("LESSON 49 — INTEGRATION INTERFACE AUDIT")
    print("=======================================")
    print()
    print(
        "Report created successfully."
    )
    print(
        "Report path:",
        REPORT_PATH.resolve(),
    )
    print()
    print(
        "Modules inspected:",
        len(MODULES),
    )
    print()
    print("STATUS")
    print("------")
    print(
        "The current guarded-RAG, Qdrant, Ollama, "
        "and runtime interfaces were captured."
    )


if __name__ == "__main__":
    main()