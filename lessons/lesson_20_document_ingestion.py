from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


SOURCE_PATH = Path(
    "data/fleet_operations_manual.txt"
)

OUTPUT_PATH = Path(
    "data/processed_fleet_sections.json"
)


@dataclass(frozen=True)
class DocumentSection:
    """One structured section extracted from a document."""

    section_id: str
    document_title: str
    section_title: str
    source: str
    content: str
    character_count: int
    word_count: int


def load_document(path: Path) -> str:
    """Load a UTF-8 text document from disk."""

    if not path.exists():
        raise FileNotFoundError(
            f"Source document was not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Source path is not a file: {path}"
        )

    return path.read_text(
        encoding="utf-8-sig"
    )


def clean_text(text: str) -> str:
    """Normalize line endings and unnecessary whitespace."""

    normalized_text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    cleaned_lines = [
        re.sub(
            r"[ \t]+$",
            "",
            line,
        )
        for line in normalized_text.splitlines()
    ]

    cleaned_text = "\n".join(
        cleaned_lines
    )

    cleaned_text = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned_text,
    )

    return cleaned_text.strip()


def build_section(
    section_number: int,
    document_title: str,
    section_title: str,
    source: Path,
    lines: list[str],
) -> DocumentSection | None:
    """Convert collected lines into a structured section."""

    content = "\n".join(lines).strip()

    if not content:
        return None

    return DocumentSection(
        section_id=f"section-{section_number:03d}",
        document_title=document_title,
        section_title=section_title,
        source=str(source),
        content=content,
        character_count=len(content),
        word_count=len(content.split()),
    )


def parse_sections(
    text: str,
    source: Path,
) -> list[DocumentSection]:
    """Split a Markdown-style document into sections."""

    document_title = "Untitled Document"
    current_section_title = "Document Overview"
    current_lines: list[str] = []
    sections: list[DocumentSection] = []

    for line in text.splitlines():
        stripped_line = line.strip()

        if (
            stripped_line.startswith("# ")
            and not stripped_line.startswith("## ")
        ):
            document_title = stripped_line[2:].strip()
            continue

        if stripped_line.startswith("## "):
            previous_section = build_section(
                section_number=len(sections) + 1,
                document_title=document_title,
                section_title=current_section_title,
                source=source,
                lines=current_lines,
            )

            if previous_section is not None:
                sections.append(previous_section)

            current_section_title = (
                stripped_line[3:].strip()
            )
            current_lines = []
            continue

        current_lines.append(line)

    final_section = build_section(
        section_number=len(sections) + 1,
        document_title=document_title,
        section_title=current_section_title,
        source=source,
        lines=current_lines,
    )

    if final_section is not None:
        sections.append(final_section)

    return sections


def save_sections(
    sections: list[DocumentSection],
    output_path: Path,
) -> None:
    """Serialize structured sections as formatted JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "section_count": len(sections),
        "sections": [
            asdict(section)
            for section in sections
        ],
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def create_preview(
    content: str,
    maximum_length: int = 100,
) -> str:
    """Create a short one-line preview of section content."""

    one_line_content = " ".join(
        content.split()
    )

    if len(one_line_content) <= maximum_length:
        return one_line_content

    return (
        one_line_content[
            : maximum_length - 3
        ]
        + "..."
    )


def main() -> None:
    """Run the first stage of document ingestion."""

    raw_text = load_document(
        SOURCE_PATH
    )

    cleaned_text = clean_text(
        raw_text
    )

    sections = parse_sections(
        text=cleaned_text,
        source=SOURCE_PATH,
    )

    if not sections:
        raise RuntimeError(
            "No document sections were extracted."
        )

    save_sections(
        sections=sections,
        output_path=OUTPUT_PATH,
    )

    print("DOCUMENT INGESTION REPORT")
    print("=========================")
    print()
    print("Source:", SOURCE_PATH)
    print("Output:", OUTPUT_PATH)
    print(
        "Raw character count:",
        len(raw_text),
    )
    print(
        "Cleaned character count:",
        len(cleaned_text),
    )
    print(
        "Sections extracted:",
        len(sections),
    )
    print()

    print("EXTRACTED SECTIONS")
    print("------------------")

    for section in sections:
        print(
            f"ID: {section.section_id}"
        )
        print(
            f"Document: {section.document_title}"
        )
        print(
            f"Section: {section.section_title}"
        )
        print(
            f"Characters: {section.character_count}"
        )
        print(
            f"Words: {section.word_count}"
        )
        print(
            "Preview:",
            create_preview(section.content),
        )
        print()

    print("STATUS")
    print("------")
    print(
        "The source document was loaded, cleaned, "
        "structured, and saved successfully."
    )


if __name__ == "__main__":
    main()