from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


INPUT_PATH = Path(
    "data/processed_fleet_sections.json"
)

OUTPUT_PATH = Path(
    "data/fleet_chunks.json"
)

MAX_WORDS = 35
OVERLAP_WORDS = 8


@dataclass(frozen=True)
class SourceSection:
    """One section loaded from the ingestion output."""

    section_id: str
    document_title: str
    section_title: str
    source: str
    content: str


@dataclass(frozen=True)
class DocumentChunk:
    """One retrieval-sized chunk with source metadata."""

    chunk_id: str
    section_id: str
    document_title: str
    section_title: str
    source: str
    content: str
    chunk_number: int
    start_word: int
    end_word_exclusive: int
    word_count: int
    character_count: int


def require_string(
    record: dict[str, Any],
    field_name: str,
    record_number: int,
) -> str:
    """Read and validate a required string field."""

    value = record.get(field_name)

    if not isinstance(value, str):
        raise ValueError(
            f"Section record {record_number} has an invalid "
            f"{field_name!r} field."
        )

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(
            f"Section record {record_number} has an empty "
            f"{field_name!r} field."
        )

    return cleaned_value


def load_sections(
    input_path: Path,
) -> list[SourceSection]:
    """Load and validate structured sections from JSON."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {input_path}"
        )

    if not input_path.is_file():
        raise ValueError(
            f"Input path is not a file: {input_path}"
        )

    try:
        payload = json.loads(
            input_path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Input file contains invalid JSON: {input_path}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            "The top-level JSON value must be an object."
        )

    raw_sections = payload.get("sections")

    if not isinstance(raw_sections, list):
        raise ValueError(
            "The JSON object must contain a sections list."
        )

    sections: list[SourceSection] = []

    for record_number, raw_section in enumerate(
        raw_sections,
        start=1,
    ):
        if not isinstance(raw_section, dict):
            raise ValueError(
                f"Section record {record_number} "
                "must be a JSON object."
            )

        section = SourceSection(
            section_id=require_string(
                raw_section,
                "section_id",
                record_number,
            ),
            document_title=require_string(
                raw_section,
                "document_title",
                record_number,
            ),
            section_title=require_string(
                raw_section,
                "section_title",
                record_number,
            ),
            source=require_string(
                raw_section,
                "source",
                record_number,
            ),
            content=require_string(
                raw_section,
                "content",
                record_number,
            ),
        )

        sections.append(section)

    if not sections:
        raise ValueError(
            "The input file contains no document sections."
        )

    return sections


def chunk_section(
    section: SourceSection,
    maximum_words: int,
    overlap_words: int,
) -> list[DocumentChunk]:
    """Split one section into overlapping word windows."""

    if maximum_words <= 0:
        raise ValueError(
            "maximum_words must be greater than zero."
        )

    if overlap_words < 0:
        raise ValueError(
            "overlap_words cannot be negative."
        )

    if overlap_words >= maximum_words:
        raise ValueError(
            "overlap_words must be smaller than "
            "maximum_words."
        )

    words = section.content.split()

    if not words:
        return []

    chunks: list[DocumentChunk] = []

    start_word = 0
    chunk_number = 1

    while start_word < len(words):
        end_word = min(
            start_word + maximum_words,
            len(words),
        )

        chunk_words = words[
            start_word:end_word
        ]

        chunk_content = " ".join(
            chunk_words
        )

        chunk = DocumentChunk(
            chunk_id=(
                f"{section.section_id}"
                f"-chunk-{chunk_number:03d}"
            ),
            section_id=section.section_id,
            document_title=section.document_title,
            section_title=section.section_title,
            source=section.source,
            content=chunk_content,
            chunk_number=chunk_number,
            start_word=start_word,
            end_word_exclusive=end_word,
            word_count=len(chunk_words),
            character_count=len(chunk_content),
        )

        chunks.append(chunk)

        if end_word == len(words):
            break

        next_start_word = (
            end_word - overlap_words
        )

        if next_start_word <= start_word:
            raise RuntimeError(
                "Chunking failed to make forward progress."
            )

        start_word = next_start_word
        chunk_number += 1

    return chunks


def save_chunks(
    chunks: list[DocumentChunk],
    source_section_count: int,
    output_path: Path,
) -> None:
    """Save chunks and chunking configuration as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "source_section_count": source_section_count,
        "chunk_count": len(chunks),
        "chunking_configuration": {
            "strategy": "overlapping_word_windows",
            "maximum_words": MAX_WORDS,
            "overlap_words": OVERLAP_WORDS,
        },
        "chunks": [
            asdict(chunk)
            for chunk in chunks
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
    maximum_length: int = 90,
) -> str:
    """Create a shortened one-line chunk preview."""

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
    """Load sections, create chunks, and save them."""

    sections = load_sections(
        INPUT_PATH
    )

    all_chunks: list[DocumentChunk] = []
    chunks_per_section: dict[str, int] = {}

    for section in sections:
        section_chunks = chunk_section(
            section=section,
            maximum_words=MAX_WORDS,
            overlap_words=OVERLAP_WORDS,
        )

        all_chunks.extend(
            section_chunks
        )

        chunks_per_section[
            section.section_id
        ] = len(section_chunks)

    if not all_chunks:
        raise RuntimeError(
            "No document chunks were created."
        )

    save_chunks(
        chunks=all_chunks,
        source_section_count=len(sections),
        output_path=OUTPUT_PATH,
    )

    print("DOCUMENT CHUNKING REPORT")
    print("========================")
    print()
    print("Input:", INPUT_PATH)
    print("Output:", OUTPUT_PATH)
    print(
        "Sections loaded:",
        len(sections),
    )
    print(
        "Maximum words per chunk:",
        MAX_WORDS,
    )
    print(
        "Overlap words:",
        OVERLAP_WORDS,
    )
    print(
        "Chunks created:",
        len(all_chunks),
    )
    print()

    print("CHUNKS PER SECTION")
    print("------------------")

    for section in sections:
        print(
            f"{section.section_id} "
            f"({section.section_title}): "
            f"{chunks_per_section[section.section_id]}"
        )

    print()
    print("CREATED CHUNKS")
    print("--------------")

    for chunk in all_chunks:
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Section: {chunk.section_title}")
        print(
            "Word range:",
            f"{chunk.start_word}:"
            f"{chunk.end_word_exclusive}",
        )
        print(
            "Words:",
            chunk.word_count,
        )
        print(
            "Characters:",
            chunk.character_count,
        )
        print(
            "Preview:",
            create_preview(chunk.content),
        )
        print()

    print("STATUS")
    print("------")
    print(
        "The document sections were divided into "
        "overlapping retrieval chunks successfully."
    )


if __name__ == "__main__":
    main()