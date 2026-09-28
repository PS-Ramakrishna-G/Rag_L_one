"""Single-entry PDF analysis pipeline for HR policy manuals.

This script is designed for manual documents with a chapter structure and tables,
using parent-child chunking to keep the policy hierarchy intact while creating
search-friendly child chunks for embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any

import fitz
import tiktoken
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
ENCODING = tiktoken.get_encoding("cl100k_base")


def load_env_file(path: Path) -> None:
    """Load the repository .env file into environment variables."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(ENV_FILE)


CHAPTER_RE = re.compile(
    r"(?:^|\n)\s*CHAPTER\s*\d+\s*(?:[-:–—]?)\s*(.*?)(?:\n|$)",
    re.IGNORECASE,
)
SECTION_RE = re.compile(r"(?:^|\n)\s*SECTION\s*\d+\s*(?:[-:–—]?)\s*(.*?)(?:\n|$)", re.IGNORECASE)
HEADER_RE = re.compile(
    r"^\s*(?:\d+\s+)?IIMA HR Policy Manual 2023(?:\s+\d+)?\s*$",
    re.IGNORECASE,
)
PAGE_NUMBER_ONLY_RE = re.compile(r"^\s*(?:\d+|[ivxlcdm]+)\s*$", re.IGNORECASE)
TOC_HINT_RE = re.compile(r"\b(?:contents|table of contents|organization chart|document control)\b", re.IGNORECASE)


def clean_page_text(text: str) -> str:
    """Remove running headers, page numbers, cover pages, and TOC fragments."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if HEADER_RE.fullmatch(line):
            continue
        if PAGE_NUMBER_ONLY_RE.fullmatch(line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise_text(text: str) -> bool:
    """Discard weak, title-only, or TOC-style content that should not be embedded."""
    value = (text or "").strip()
    if not value:
        return True
    if len(value) < 40:
        return True
    if re.fullmatch(r"\d+", value):
        return True
    if re.fullmatch(r"IIMA HR Policy Manual 2023\s*\d*", value, re.IGNORECASE):
        return True
    if TOC_HINT_RE.search(value):
        return True
    return False


def should_skip_page(page_number: int, text: str) -> bool:
    """Skip cover pages, declaration pages, and TOC pages before the policy body."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if page_number <= 10:
        return True
    if len(cleaned) < 80:
        return True
    return False


def extract_chapter_title(text: str) -> str:
    """Return the most relevant chapter title from a text block."""
    cleaned = (text or "").replace("\r", "\n").strip()
    if not cleaned:
        return ""

    chapter_match = CHAPTER_RE.search(cleaned)
    if chapter_match:
        title = chapter_match.group(1).strip()
        if title:
            return re.sub(r"\s+", " ", title)

    section_match = SECTION_RE.search(cleaned)
    if section_match:
        title = section_match.group(1).strip()
        if title:
            return re.sub(r"\s+", " ", title)

    # Fallback: grab the first non-trivial line if it looks like a title.
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        first = lines[0]
        if len(first) <= 120 and not re.fullmatch(r"[0-9]+", first):
            return first
    return ""


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text or ""))


def decode_tokens(tokens: list[int]) -> str:
    return ENCODING.decode(tokens)


def split_text_by_tokens(
    text: str,
    target_tokens: int = 250,
    max_tokens: int = 320,
    overlap_tokens: int = 30,
) -> List[str]:
    """Split text into child chunks while enforcing token limits."""
    text = (text or "").strip()
    if not text:
        return []

    tokens = ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks: List[str] = []
    start = 0
    total = len(tokens)

    while start < total:
        end = min(start + target_tokens, total)
        chunk_tokens = tokens[start:end]
        chunk_text = decode_tokens(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= total:
            break
        start = max(end - overlap_tokens, start + 1)

    return chunks


def split_into_sections(page_text: str) -> List[str]:
    """Split page text into logical headings and paragraphs."""
    if not page_text:
        return []

    blocks = re.split(r"\n{2,}", page_text.strip())
    cleaned = []
    for block in blocks:
        text = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if text:
            cleaned.append(text)
    return cleaned


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return cleaned or "section"


def _infer_section_metadata(section: Dict[str, Any], page_number: int, idx: int) -> Dict[str, Any]:
    title = str(section.get("title") or "").strip()
    content = str(section.get("content") or "").strip()
    chapter = str(section.get("chapter") or extract_chapter_title(title) or "Unspecified Chapter").strip()
    section_name = str(section.get("section") or title or "Unspecified Section").strip()
    document_id = str(section.get("document_id") or os.getenv("DOCUMENT_ID") or "hr_policy_2023").strip()

    if section_name.lower().startswith("chapter"):
        section_name = chapter

    parent_id = f"{document_id}_{_slugify(chapter)}_{_slugify(section_name)}"
    if not section_name or section_name == chapter:
        section_name = chapter

    return {
        "document_id": document_id,
        "chapter": chapter,
        "section": section_name,
        "parent_id": parent_id,
        "parent_title": chapter,
        "section_title": title or section_name,
        "page_start": int(section.get("page_start") or page_number or idx + 1),
        "page_end": int(section.get("page_end") or page_number or idx + 1),
        "parent_text": content,
    }


def build_parent_child_chunks(
    sections: Iterable[Dict[str, Any]],
    child_token_target: int = 250,
    child_max_tokens: int = 320,
    overlap_tokens: int = 30,
    parent_token_target: int = 1200,
) -> List[Dict[str, Any]]:
    """Create parent-child chunk records while preserving section boundaries."""
    chunks: List[Dict[str, Any]] = []

    for idx, section in enumerate(sections):
        title = str(section.get("title") or "").strip()
        content = str(section.get("content") or "").strip()
        page_number = int(section.get("page_number") or idx + 1)

        if not content:
            continue

        metadata = _infer_section_metadata(section, page_number, idx)
        parent_title = metadata["parent_title"]
        logical_parts = split_into_sections(content)
        if not logical_parts:
            logical_parts = [content]

        child_number = 0

        for part in logical_parts:
            part = part.strip()
            if not part:
                continue

            child_texts = split_text_by_tokens(
                part,
                target_tokens=child_token_target,
                max_tokens=child_max_tokens,
                overlap_tokens=overlap_tokens,
            )

            for child_text in child_texts:
                child_number += 1
                child_id = f"{metadata['parent_id']}_child_{child_number:02d}"
                chunks.append(
                    {
                        "document_id": metadata["document_id"],
                        "chapter": metadata["chapter"],
                        "section": metadata["section"],
                        "parent_id": metadata["parent_id"],
                        "child_id": child_id,
                        "chunk_id": child_id,
                        "parent_title": parent_title,
                        "section_title": metadata["section_title"],
                        "page_number": page_number,
                        "page_start": metadata["page_start"],
                        "page_end": metadata["page_end"],
                        "parent_text": metadata["parent_text"],
                        "child_text": child_text,
                        "token_count": count_tokens(child_text),
                        "child_token_target": child_token_target,
                        "child_max_tokens": child_max_tokens,
                        "overlap_tokens": overlap_tokens,
                        "parent_token_target": parent_token_target,
                    }
                )

    return chunks


@dataclass
class ChunkConfig:
    child_token_target: int = 250
    parent_token_target: int = 1200
    overlap_ratio: float = 0.12
    include_tables: bool = True


def extract_page_text(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract logical content from each PDF page while skipping cover, TOC, and header noise."""
    document = fitz.open(pdf_path)
    sections: List[Dict[str, Any]] = []

    for page_number in range(document.page_count):
        page = document[page_number]
        raw_text = page.get_text("text") or ""
        cleaned_text = clean_page_text(raw_text)

        if should_skip_page(page_number + 1, cleaned_text):
            continue

        blocks = split_into_sections(cleaned_text)
        if not blocks:
            continue

        title = blocks[0]
        chapter_title = extract_chapter_title(title)
        section_text = "\n\n".join(blocks[1:]) if len(blocks) > 1 else ""

        if is_noise_text(title) and not section_text:
            continue

        sections.append(
            {
                "title": chapter_title or title,
                "content": section_text or title,
                "page_number": page_number + 1,
            }
        )

    document.close()
    return sections


def create_sqlite_index(db_path: str = "storage/hr_policy_chunks.db") -> sqlite3.Connection:
    """Create SQLite tables to persist document, parent, and child metadata."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    connection = sqlite3.connect(db_path)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            title TEXT,
            source_pdf TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS parents (
            parent_id TEXT PRIMARY KEY,
            document_id TEXT,
            chapter TEXT,
            section TEXT,
            page_start INTEGER,
            page_end INTEGER,
            parent_text TEXT,
            token_count INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS children (
            child_id TEXT PRIMARY KEY,
            parent_id TEXT,
            document_id TEXT,
            chapter TEXT,
            section TEXT,
            page_start INTEGER,
            page_end INTEGER,
            child_text TEXT,
            token_count INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_chunks (
            id TEXT PRIMARY KEY,
            parent_title TEXT,
            section_title TEXT,
            page_number INTEGER,
            document_id TEXT,
            chapter TEXT,
            section TEXT,
            parent_id TEXT,
            page_start INTEGER,
            page_end INTEGER,
            parent_text TEXT,
            child_text TEXT,
            token_count INTEGER,
            child_token_target INTEGER,
            child_max_tokens INTEGER,
            overlap_tokens INTEGER,
            parent_token_target INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    columns = [row[1] for row in connection.execute("PRAGMA table_info(policy_chunks)").fetchall()]
    for column_name in ["document_id", "chapter", "section", "parent_id", "page_start", "page_end", "parent_text"]:
        if column_name not in columns:
            connection.execute(f"ALTER TABLE policy_chunks ADD COLUMN {column_name} TEXT")

    connection.commit()
    return connection


def store_chunks_sqlite(chunks: List[Dict[str, Any]], db_path: str = "storage/hr_policy_chunks.db") -> None:
    connection = create_sqlite_index(db_path)
    document_id = chunks[0].get("document_id") if chunks else "hr_policy_2023"
    connection.execute(
        "INSERT OR REPLACE INTO documents (document_id, title, source_pdf) VALUES (?, ?, ?)",
        (document_id, "HR Policy Manual", "manual.pdf"),
    )

    seen_parents: set[str] = set()
    for chunk in chunks:
        parent_id = chunk.get("parent_id") or chunk.get("chunk_id")
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_text = str(chunk.get("parent_text") or "").strip()
            parent_row = (
                parent_id,
                chunk.get("document_id", document_id),
                chunk.get("chapter"),
                chunk.get("section"),
                chunk.get("page_start"),
                chunk.get("page_end"),
                parent_text,
                len(parent_text.split()) or 0,
            )
            connection.execute(
                "INSERT OR REPLACE INTO parents (parent_id, document_id, chapter, section, page_start, page_end, parent_text, token_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                parent_row,
            )

        connection.execute(
            "INSERT OR REPLACE INTO children (child_id, parent_id, document_id, chapter, section, page_start, page_end, child_text, token_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk.get("child_id") or chunk.get("chunk_id"),
                parent_id,
                chunk.get("document_id", document_id),
                chunk.get("chapter"),
                chunk.get("section"),
                chunk.get("page_start"),
                chunk.get("page_end"),
                chunk["child_text"],
                chunk.get("token_count", count_tokens(chunk["child_text"])),
            ),
        )

        connection.execute(
            """
            INSERT OR REPLACE INTO policy_chunks (
                id, parent_title, section_title, page_number, document_id, chapter, section, parent_id,
                page_start, page_end, parent_text, child_text, token_count, child_token_target,
                child_max_tokens, overlap_tokens, parent_token_target
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk["chunk_id"],
                chunk["parent_title"],
                chunk.get("section_title"),
                chunk.get("page_number"),
                chunk.get("document_id", document_id),
                chunk.get("chapter"),
                chunk.get("section"),
                parent_id,
                chunk.get("page_start"),
                chunk.get("page_end"),
                str(chunk.get("parent_text") or "").strip(),
                chunk["child_text"],
                chunk.get("token_count", count_tokens(chunk["child_text"])),
                chunk.get("child_token_target", 250),
                chunk.get("child_max_tokens", 320),
                chunk.get("overlap_tokens", 30),
                chunk.get("parent_token_target", 1200),
            ),
        )

    connection.commit()
    connection.close()


def save_chunks_json(chunks: List[Dict[str, Any]], output_path: str = "artifacts/hr_policy_chunks.json") -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def normalize_embedding(vector: List[float], target_dimension: int) -> List[float]:
    """Pad or truncate embeddings to the exact Pinecone index dimension."""
    vector = [float(v) for v in vector]
    if target_dimension <= 0:
        return vector
    if len(vector) == target_dimension:
        return vector
    if len(vector) < target_dimension:
        return vector + [0.0] * (target_dimension - len(vector))
    return vector[:target_dimension]


def build_pinecone_records(
    chunks: List[Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
    target_dimension: int | None = None,
) -> List[Dict[str, Any]]:
    """Convert chunk metadata into Pinecone-ready vector payloads."""
    model = SentenceTransformer(model_name)
    records: List[Dict[str, Any]] = []

    detected_dimension = target_dimension or int(os.getenv("PINECONE_VECTOR_DIMENSION") or os.getenv("pinecone_vector_dimension") or 0)
    for chunk in chunks:
        text = str(chunk.get("child_text") or "").strip()
        if not text:
            continue

        vector = model.encode(text)
        if detected_dimension > 0:
            vector = normalize_embedding(vector.tolist(), detected_dimension)
        else:
            vector = vector.tolist()

        records.append(
            {
                "id": str(chunk.get("chunk_id") or "chunk"),
                "values": vector,
                "metadata": {
                    "document_id": chunk.get("document_id"),
                    "parent_id": chunk.get("parent_id"),
                    "parent_title": chunk.get("parent_title"),
                    "chapter": chunk.get("chapter"),
                    "section": chunk.get("section"),
                    "section_title": chunk.get("section_title"),
                    "page_number": chunk.get("page_number"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "child_text": text,
                },
            }
        )

    return records


def upsert_to_pinecone(chunks: List[Dict[str, Any]], index_name: str = None, namespace: str = "hr-policy") -> List[Dict[str, Any]]:
    """Create Pinecone-ready records and optionally insert them if the SDK is configured."""
    target_dimension = None
    env_dim = os.getenv("PINECONE_VECTOR_DIMENSION") or os.getenv("pinecone_vector_dimension")
    if env_dim:
        target_dimension = int(env_dim)

    if target_dimension is None:
        try:
            from pinecone import Pinecone
        except ImportError:
            print("Pinecone SDK not installed; generated records only.")
            return build_pinecone_records(chunks)

        api_key = os.getenv("Pinecone_api_key") or os.getenv("PINECONE_API_KEY")
        host = os.getenv("pinecone_Host") or os.getenv("PINECONE_HOST")
        if not api_key or not host:
            print("Pinecone credentials missing; generated records only.")
            return build_pinecone_records(chunks)

        index_name = index_name or os.getenv("pinecone_index_name") or os.getenv("PINECONE_INDEX_NAME")
        client = Pinecone(api_key=api_key)
        try:
            description = client.describe_index(name=index_name) if index_name else None
            if description is not None:
                description_dict = description.to_dict() if hasattr(description, "to_dict") else description
                if isinstance(description_dict, dict):
                    target_dimension = int(description_dict.get("dimension") or 0)
        except Exception:
            target_dimension = None

        if not target_dimension:
            client = Pinecone(api_key=api_key)
            index = client.Index(host=host)
            meta = index.describe_index_stats()
            target_dimension = int(meta.get("dimension") if isinstance(meta, dict) else 0) or int(os.getenv("pinecone_dimension") or 0)

    records = build_pinecone_records(chunks, target_dimension=target_dimension or 384)
    if not index_name and not os.getenv("pinecone_Host") and not os.getenv("PINECONE_HOST"):
        return records

    try:
        from pinecone import Pinecone
    except ImportError:
        print("Pinecone SDK not installed; generated records only.")
        return records

    api_key = os.getenv("Pinecone_api_key") or os.getenv("PINECONE_API_KEY")
    host = os.getenv("pinecone_Host") or os.getenv("PINECONE_HOST")
    if not api_key or not host:
        print("Pinecone credentials missing; generated records only.")
        return records

    client = Pinecone(api_key=api_key)
    index = client.Index(host=host)

    batch_size = 150
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"Upserted batch {i // batch_size + 1}: {len(batch)} vectors")

    print(f"Completed upsert for {len(records)} vectors to Pinecone namespace '{namespace}'")
    return records


def analyze_policy_manual(
    pdf_path: str,
    db_path: str = "storage/hr_policy_chunks.db",
    pinecone_index_name: str | None = None,
    pinecone_namespace: str = "hr-policy",
) -> List[Dict[str, Any]]:
    """End-to-end analysis and preparation for a HR policy PDF."""
    sections = extract_page_text(pdf_path)
    chunks = build_parent_child_chunks(sections)
    store_chunks_sqlite(chunks, db_path)
    save_chunks_json(chunks)
    upsert_to_pinecone(chunks, index_name=pinecone_index_name, namespace=pinecone_namespace)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HR policy PDFs and generate chunk metadata.")
    parser.add_argument("--pdf", required=True, help="Path to the HR policy PDF file")
    parser.add_argument("--db", default="storage/hr_policy_chunks.db", help="SQLite database path")
    parser.add_argument("--json-output", default="artifacts/hr_policy_chunks.json", help="JSON output path")
    parser.add_argument("--pinecone-index", default=None, help="Optional Pinecone index name/host configuration")
    parser.add_argument("--pinecone-namespace", default="hr-policy", help="Namespace to upsert into")
    args = parser.parse_args()

    chunks = analyze_policy_manual(
        args.pdf,
        db_path=args.db,
        pinecone_index_name=args.pinecone_index,
        pinecone_namespace=args.pinecone_namespace,
    )
    print(f"Extracted {len(chunks)} chunk records from {args.pdf}")
    print(f"Saved output to {args.json_output}")


if __name__ == "__main__":
    main()
