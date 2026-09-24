from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from llm_engine.ollama_client import ask_qwen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = PROJECT_ROOT / "storage" / "hr_policy_chunks.db"

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "hr-policy",
)

PINECONE_DIMENSION = int(
    os.getenv(
        "PINECONE_VECTOR_DIMENSION",
        "512",
    )
)

PINECONE_API_KEY = (
    os.getenv("PINECONE_API_KEY")
    or os.getenv("Pinecone_api_key")
)

PINECONE_HOST = (
    os.getenv("PINECONE_HOST")
    or os.getenv("pinecone_Host")
)


SYSTEM_PROMPT = """
You are the HR Policy Answering Assistant for the IIMA HR Policy Manual.

Rules:
1. Answer only from the provided HR policy evidence.
2. Do not use outside knowledge.
3. Do not invent rules, amounts, dates, eligibility, benefits, or exceptions.
4. If evidence is insufficient, say:
   "I could not find sufficient information in the provided policy context."
5. If the question is outside the HR Policy Manual, say:
   "This question is outside the HR Policy Manual and cannot be answered from the provided policy context."
6. Give the direct answer first.
7. Include page references like [Page X] when available.
8. Keep the answer concise and grounded.
9. Never mention Pinecone, embeddings, vector databases, chunks, or retrieval internals.
""".strip()


# ============================================================
# LOAD EXPENSIVE RESOURCES ONCE
# ============================================================

EMBEDDING_MODEL = SentenceTransformer(
    MODEL_NAME
)

PINECONE_INDEX = None

if PINECONE_API_KEY and PINECONE_HOST:
    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    PINECONE_INDEX = pc.Index(
        host=PINECONE_HOST
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def _normalize_embedding(
    vector: list[float],
    target_dimension: int,
) -> list[float]:

    vector = [
        float(v)
        for v in vector
    ]

    if len(vector) == target_dimension:
        return vector

    if len(vector) < target_dimension:
        return (
            vector
            + [0.0]
            * (
                target_dimension
                - len(vector)
            )
        )

    return vector[:target_dimension]


def _get_target_dimension() -> int:
    return PINECONE_DIMENSION


def _embed_text(
    text: str,
) -> list[float]:

    vector = EMBEDDING_MODEL.encode(
        text,
        show_progress_bar=False,
    )

    return _normalize_embedding(
        vector.tolist(),
        PINECONE_DIMENSION,
    )


# ============================================================
# SQLITE FALLBACK
# ============================================================

def _load_sqlite_matches(
    question: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:

    if not DB_PATH.exists():
        return []

    # Simple fallback only.
    # We can replace this with BM25/FTS later.
    terms = [
        term.strip()
        for term in question.split()
        if len(term.strip()) >= 4
    ]

    if not terms:
        return []

    where_parts = []
    params = []

    for term in terms:
        where_parts.append(
            "LOWER(child_text) LIKE ?"
        )

        params.append(
            f"%{term.lower()}%"
        )

    sql = f"""
        SELECT
            id,
            parent_title,
            section_title,
            page_number,
            child_text
        FROM policy_chunks
        WHERE {" OR ".join(where_parts)}
        LIMIT ?
    """

    params.append(top_k)

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            sql,
            params,
        ).fetchall()

    return [
        {
            "id": row[0],
            "score": 0.0,
            "metadata": {
                "parent_title": row[1],
                "section_title": row[2],
                "page_number": row[3],
                "child_text": row[4],
            },
        }
        for row in rows
    ]


# ============================================================
# PINECONE RETRIEVAL
# ============================================================

def query_policy_store(
    question: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:

    if PINECONE_INDEX is not None:

        try:

            query_vector = _embed_text(
                question
            )

            response = PINECONE_INDEX.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False,
                namespace=PINECONE_NAMESPACE,
            )

            raw_matches = getattr(
                response,
                "matches",
                [],
            )

            matches = []

            for match in raw_matches:

                metadata = (
                    getattr(
                        match,
                        "metadata",
                        {},
                    )
                    or {}
                )

                matches.append(
                    {
                        "id": getattr(
                            match,
                            "id",
                            None,
                        ),

                        "score": float(
                            getattr(
                                match,
                                "score",
                                0.0,
                            )
                        ),

                        "metadata": metadata,
                    }
                )

            if matches:
                return matches

        except Exception as exc:
            print(
                f"[PINECONE ERROR] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    return _load_sqlite_matches(
        question,
        top_k=top_k,
    )


# ============================================================
# PROMPT
# ============================================================

def build_user_prompt(
    question: str,
    matches: list[dict[str, Any]],
) -> str:

    if not matches:
        return f"""
Question:
{question}

No policy evidence was retrieved.

Reply exactly:
"I could not find sufficient information in the provided policy context."
""".strip()

    sources = []

    for i, match in enumerate(
        matches,
        start=1,
    ):

        metadata = (
            match.get("metadata", {})
        )

        sources.append(
            f"""
[SOURCE {i}]
Page: {metadata.get("page_number", "Unknown")}
Parent: {metadata.get("parent_title", "Unknown")}
Section: {metadata.get("section_title", "Unknown")}

{metadata.get("child_text", "")}
""".strip()
        )

    context = "\n\n---\n\n".join(
        sources
    )

    return f"""
Answer the question using only the HR Policy Manual evidence below.

QUESTION:
{question}

EVIDENCE:
{context}

Requirements:
- Direct answer first.
- Cite relevant pages like [Page X].
- Do not use outside knowledge.
- Do not invent facts.
""".strip()


# ============================================================
# SERIALIZE SOURCES
# ============================================================

def _serialize_source(
    match: dict[str, Any],
) -> dict[str, Any]:

    metadata = (
        match.get("metadata", {})
    )

    return {
        "id": match.get("id"),
        "score": match.get("score", 0.0),
        "page_number": metadata.get(
            "page_number"
        ),
        "parent_title": metadata.get(
            "parent_title"
        ),
        "section_title": metadata.get(
            "section_title"
        ),
        "child_text": metadata.get(
            "child_text",
            "",
        ),
    }


# ============================================================
# FINAL ANSWER
# ============================================================

def answer_question(
    question: str,
    top_k: int = 5,
) -> dict[str, Any]:

    matches = query_policy_store(
        question,
        top_k=top_k,
    )

    source_payload = [
        _serialize_source(match)
        for match in matches
    ]

    retrieval_debug = {
        "index": os.getenv(
            "PINECONE_INDEX_NAME",
            "ragl1",
        ),
        "namespace": PINECONE_NAMESPACE,
        "embedding_model": MODEL_NAME,
        "top_k": top_k,
        "vector_dimension":
            PINECONE_DIMENSION,
    }

    if not matches:

        return {
            "answer":
                "I could not find sufficient information in the provided policy context.",

            "sources": [],

            "db_proof": {
                "database": str(DB_PATH),
                "rows_found": 0,
            },

            "retrieval_debug":
                retrieval_debug,

            "logs": [
                "Embedded query",
                "No evidence returned",
            ],

            "grounded": False,
            "source_pages": [],
        }

    prompt = build_user_prompt(
        question,
        matches,
    )

    answer = ask_qwen(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.1,
    )

    pages = []

    for match in matches:

        page = (
            match.get(
                "metadata",
                {},
            ).get(
                "page_number"
            )
        )

        if (
            page is not None
            and page not in pages
        ):
            pages.append(page)

    return {
        "answer": answer,

        "sources": source_payload,

        "db_proof": {
            "database": str(DB_PATH),
            "rows_found":
                len(matches),
            "namespace":
                PINECONE_NAMESPACE,
        },

        "retrieval_debug":
            retrieval_debug,

        "logs": [
            "Embedded query with all-MiniLM-L6-v2",
            f"Searched namespace: {PINECONE_NAMESPACE}",
            f"Retrieved {len(matches)} chunks",
            "Context sent to Qwen",
        ],

        "grounded": True,
        "source_pages": pages,
    }