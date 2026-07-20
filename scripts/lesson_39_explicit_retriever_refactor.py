from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GUARDED_RAG_PATH = (
    PROJECT_ROOT
    / "lessons"
    / "lesson_28_guarded_rag.py"
)

SERVICE_PATH = (
    PROJECT_ROOT
    / "lessons"
    / "lesson_36_guarded_rag_service.py"
)


def create_backup(
    source_path: Path,
    backup_directory: Path,
) -> Path:
    """Copy one source file into the external backup directory."""

    backup_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_path = (
        backup_directory
        / source_path.name
    )

    shutil.copy2(
        source_path,
        backup_path,
    )

    return backup_path


def patch_guarded_rag(
    source: str,
) -> str:
    """
    Add an optional explicit retriever to run_guarded_rag().

    Existing callers remain compatible because the default
    continues to use the module's retrieve_chunks function.
    """

    updated = source

    future_import = (
        "from __future__ import annotations\n"
    )

    if "import collections.abc\n" not in updated:
        if future_import not in updated:
            raise RuntimeError(
                "Lesson 28 does not contain the expected "
                "__future__ import."
            )

        updated = updated.replace(
            future_import,
            (
                future_import
                + "\n"
                + "import collections.abc\n"
            ),
            1,
        )

    retriever_alias = """RetrieverFunction = collections.abc.Callable[
    [str, LoadedVectorIndex],
    tuple[list[SearchResult], int],
]
"""

    if (
        "RetrieverFunction = "
        "collections.abc.Callable"
        not in updated
    ):
        function_marker = (
            "def run_guarded_rag("
        )

        function_position = updated.find(
            function_marker
        )

        if function_position < 0:
            raise RuntimeError(
                "Could not find run_guarded_rag() "
                "in Lesson 28."
            )

        updated = (
            updated[:function_position]
            + retriever_alias
            + "\n\n"
            + updated[function_position:]
        )

    signature_pattern = re.compile(
        r"^def run_guarded_rag\(\n"
        r"(?P<parameters>.*?)"
        r"^\) -> (?P<return_type>[^:\n]+):",
        flags=re.MULTILINE | re.DOTALL,
    )

    signature_match = signature_pattern.search(
        updated
    )

    if signature_match is None:
        raise RuntimeError(
            "Could not identify the run_guarded_rag() "
            "signature."
        )

    parameters = signature_match.group(
        "parameters"
    )

    return_type = signature_match.group(
        "return_type"
    )

    if "retriever:" not in parameters:
        cleaned_parameters = parameters.rstrip()

        if "\n    *," in cleaned_parameters:
            new_parameters = (
                cleaned_parameters
                + "\n"
                + "    retriever: "
                "RetrieverFunction | None = None,\n"
            )
        else:
            new_parameters = (
                cleaned_parameters
                + "\n"
                + "    *,\n"
                + "    retriever: "
                "RetrieverFunction | None = None,\n"
            )

        new_signature = (
            "def run_guarded_rag(\n"
            + new_parameters
            + ") -> "
            + return_type
            + ":"
        )

        updated = (
            updated[
                : signature_match.start()
            ]
            + new_signature
            + updated[
                signature_match.end() :
            ]
        )

    function_start = updated.find(
        "def run_guarded_rag("
    )

    if function_start < 0:
        raise RuntimeError(
            "Could not locate the updated guarded-RAG "
            "function."
        )

    next_definition_match = re.search(
        r"^(?:def |if __name__)",
        updated[
            function_start + 1 :
        ],
        flags=re.MULTILINE,
    )

    if next_definition_match is None:
        function_end = len(updated)
    else:
        function_end = (
            function_start
            + 1
            + next_definition_match.start()
        )

    function_source = updated[
        function_start:function_end
    ]

    if "active_retriever =" not in function_source:
        retrieval_call_pattern = re.compile(
            r"^(?P<assignment>"
            r"    \(\n"
            r"(?:        [^\n]*\n)+?"
            r"    \) = )"
            r"retrieve_chunks\(",
            flags=re.MULTILINE,
        )

        retrieval_matches = list(
            retrieval_call_pattern.finditer(
                function_source
            )
        )

        if len(retrieval_matches) != 1:
            raise RuntimeError(
                "Expected exactly one multiline call to "
                "retrieve_chunks() inside run_guarded_rag(), "
                f"but found {len(retrieval_matches)}."
            )

        retrieval_match = (
            retrieval_matches[0]
        )

        dependency_block = """    active_retriever = (
        retrieve_chunks
        if retriever is None
        else retriever
    )

    if not callable(active_retriever):
        raise TypeError(
            "The retriever must be callable."
        )

"""

        replacement = (
            dependency_block
            + retrieval_match.group(
                "assignment"
            )
            + "active_retriever("
        )

        function_source = (
            function_source[
                : retrieval_match.start()
            ]
            + replacement
            + function_source[
                retrieval_match.end() :
            ]
        )

        updated = (
            updated[:function_start]
            + function_source
            + updated[function_end:]
        )

    if updated == source:
        print(
            "Lesson 28 already contains the explicit "
            "retriever refactor."
        )

    return updated


def patch_service(
    source: str,
) -> str:
    """
    Update GuardedRAGService.answer() to pass its retriever.

    This removes the temporary reassignment of
    guarded.retrieve_chunks from the service.
    """

    new_answer_method = '''    def answer(
        self,
        question: str,
    ) -> guarded.GuardedRAGResult:
        """
        Answer one question through the guarded pipeline.

        The Qdrant retriever is supplied explicitly, so the
        service does not modify Lesson 28's module-level
        retrieval dependency.
        """

        self._require_open()

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "The question cannot be empty."
            )

        if self._answer_in_progress:
            raise RuntimeError(
                "A guarded-RAG answer is already in progress "
                "for this service."
            )

        self._answer_in_progress = True

        try:
            return guarded.run_guarded_rag(
                question=cleaned_question,
                vector_index=self._vector_index,
                retriever=self._retriever,
            )
        finally:
            self._answer_in_progress = False

'''

    answer_method_pattern = re.compile(
        r"^    def answer\(\n"
        r".*?"
        r"(?=^    def answer_many\()",
        flags=re.MULTILINE | re.DOTALL,
    )

    matches = list(
        answer_method_pattern.finditer(
            source
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one GuardedRAGService.answer() "
            f"method, but found {len(matches)}."
        )

    answer_method = matches[0].group(0)

    if (
        "retriever=self._retriever"
        in answer_method
        and "original_retriever"
        not in answer_method
    ):
        print(
            "Lesson 36 already uses explicit retriever "
            "injection."
        )

        return source

    return answer_method_pattern.sub(
        new_answer_method,
        source,
        count=1,
    )


def write_source(
    path: Path,
    source: str,
) -> None:
    """Write Python source using UTF-8 and LF line endings."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(
            source
        )


def main() -> None:
    """Apply the checked Lesson 39 refactor."""

    print("LESSON 39 — EXPLICIT RETRIEVER REFACTOR")
    print("========================================")
    print()

    for required_path in (
        GUARDED_RAG_PATH,
        SERVICE_PATH,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                "Required source file not found: "
                f"{required_path}"
            )

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    backup_directory = (
        Path.home()
        / "generative-ai-learning-backups"
        / f"lesson-39-{timestamp}"
    )

    print(
        "1. Creating source backups..."
    )

    guarded_backup = create_backup(
        GUARDED_RAG_PATH,
        backup_directory,
    )

    service_backup = create_backup(
        SERVICE_PATH,
        backup_directory,
    )

    print(
        "   Guarded-RAG backup:",
        guarded_backup,
    )
    print(
        "   Service backup:",
        service_backup,
    )
    print()

    print(
        "2. Refactoring Lesson 28..."
    )

    original_guarded_source = (
        GUARDED_RAG_PATH.read_text(
            encoding="utf-8"
        )
    )

    updated_guarded_source = patch_guarded_rag(
        original_guarded_source
    )

    write_source(
        GUARDED_RAG_PATH,
        updated_guarded_source,
    )

    print(
        "   Lesson 28 now accepts an explicit retriever."
    )
    print()

    print(
        "3. Refactoring Lesson 36..."
    )

    original_service_source = (
        SERVICE_PATH.read_text(
            encoding="utf-8"
        )
    )

    updated_service_source = patch_service(
        original_service_source
    )

    write_source(
        SERVICE_PATH,
        updated_service_source,
    )

    print(
        "   Lesson 36 now passes its retriever directly."
    )
    print()

    print("STATUS")
    print("------")
    print(
        "The refactor completed successfully."
    )
    print(
        "Backups were stored outside the Git repository."
    )
    print(
        "Run compilation and tests before deleting any "
        "backup files."
    )


if __name__ == "__main__":
    main()