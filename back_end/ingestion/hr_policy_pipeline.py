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
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


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


def build_parent_child_chunks(
    sections: Iterable[Dict[str, Any]],
    child_token_target: int = 250,
    parent_token_target: int = 1200,
) -> List[Dict[str, Any]]:
    """Create parent-child chunk records from logical sections.

    Each returned chunk keeps the parent title, page number, and a child_text that is
    small enough to embed and search efficiently.
    """
    chunks: List[Dict[str, Any]] = []

    for idx, section in enumerate(sections):
        title = str(section.get("title") or "").strip()
        content = str(section.get("content") or "").strip()
        page_number = section.get("page_number") or idx + 1

        if not title and not content:
            continue

        parent_title = extract_chapter_title(title) if title else "UNSPECIFIED"
        if not parent_title and title:
            parent_title = re.sub(r"\s+", " ", title)

        child_parts = split_into_sections(content)
        if not child_parts:
            if content:
                child_parts = [content]

        for part in child_parts:
            if not part:
                continue
            text = normalize_whitespace(part)
            if len(text) == 0:
                continue
            chunks.append(
                {
                    "chunk_id": f"chunk_{len(chunks) + 1}",
                    "parent_title": parent_title,
                    "page_number": page_number,
                    "child_text": text,
                    "section_title": title,
                    "child_token_target": child_token_target,
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
    """Extract logical content from each PDF page with page numbers."""
    document = fitz.open(pdf_path)
    sections: List[Dict[str, Any]] = []

    for page_number in range(document.page_count):
        page = document[page_number]
        text = page.get_text("text")

        blocks = split_into_sections(text)
        if not blocks:
            continue

        title = blocks[0] if blocks else ""
        chapter_title = extract_chapter_title(title)
        section_text = "\n\n".join(blocks[1:]) if len(blocks) > 1 else ""

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
    """Create a SQLite table to persist parent-child metadata."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_chunks (
            id TEXT PRIMARY KEY,
            parent_title TEXT,
            section_title TEXT,
            page_number INTEGER,
            child_text TEXT,
            child_token_target INTEGER,
            parent_token_target INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.commit()
    return connection


def store_chunks_sqlite(chunks: List[Dict[str, Any]], db_path: str = "storage/hr_policy_chunks.db") -> None:
    connection = create_sqlite_index(db_path)
    for chunk in chunks:
        connection.execute(
            """
            INSERT OR REPLACE INTO policy_chunks (
                id, parent_title, section_title, page_number, child_text,
                child_token_target, parent_token_target
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk["chunk_id"],
                chunk["parent_title"],
                chunk.get("section_title"),
                chunk.get("page_number"),
                chunk["child_text"],
                chunk.get("child_token_target", 250),
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
                    "parent_title": chunk.get("parent_title"),
                    "section_title": chunk.get("section_title"),
                    "page_number": chunk.get("page_number"),
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
    index.upsert(vectors=records, namespace=namespace)
    print(f"Upserted {len(records)} vectors to Pinecone namespace '{namespace}'")
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
