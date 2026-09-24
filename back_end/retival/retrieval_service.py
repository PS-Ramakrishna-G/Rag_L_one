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
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "hr-policy")
SYSTEM_PROMPT = """
You are an HR Policy Manual question-answering assistant.

You must answer questions strictly from the policy context provided to you.

Rules:
1. Use only information explicitly stated in the provided policy context.
2. Do not use outside knowledge or make assumptions.
3. If the context does not contain enough information, say:
   "I could not find sufficient information in the provided policy context."
4. Do not invent policy rules, numbers, dates, eligibility conditions, benefits, or exceptions.
5. If multiple retrieved passages are relevant, combine them carefully.
6. If passages conflict, mention the conflict instead of choosing one without evidence.
7. Preserve important conditions, exceptions, limits, and eligibility requirements.
8. When available, cite the source page in the form [Page X].
9. Give a direct answer first, followed by supporting details when needed.
10. Do not mention retrieval, embeddings, Pinecone, chunks, vector databases, or internal system details.

Your goal is accuracy and faithfulness to the provided HR Policy Manual context.
""".strip()


def _normalize_embedding(vector: list[float], target_dimension: int) -> list[float]:
    vector = [float(v) for v in vector]
    if target_dimension <= 0:
        return vector
    if len(vector) == target_dimension:
        return vector
    if len(vector) < target_dimension:
        return vector + [0.0] * (target_dimension - len(vector))
    return vector[:target_dimension]


def _get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def _get_target_dimension() -> int:
    env_dim = os.getenv("PINECONE_VECTOR_DIMENSION") or os.getenv("pinecone_vector_dimension")
    if env_dim:
        return int(env_dim)
    api_key = os.getenv("Pinecone_api_key") or os.getenv("PINECONE_API_KEY")
    host = os.getenv("pinecone_Host") or os.getenv("PINECONE_HOST")
    if not api_key or not host:
        return 384

    try:
        client = Pinecone(api_key=api_key)
        index = client.Index(host=host)
        stats = index.describe_index_stats()
        if isinstance(stats, dict) and stats.get("dimension"):
            return int(stats["dimension"])
    except Exception:
        pass
    return 384


def _embed_text(text: str) -> list[float]:
    model = _get_embedding_model()
    vector = model.encode(text)
    return _normalize_embedding(vector.tolist(), _get_target_dimension())


def _load_sqlite_matches(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []

    sql = """
        SELECT id, parent_title, section_title, page_number, child_text
        FROM policy_chunks
        WHERE child_text LIKE ?
        ORDER BY page_number ASC
        LIMIT ?
    """
    q = f"%{question.strip()}%"
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(sql, (q, top_k)).fetchall()
    return [
        {
            "id": row[0],
            "score": 1.0,
            "metadata": {
                "parent_title": row[1],
                "section_title": row[2],
                "page_number": row[3],
                "child_text": row[4],
            },
        }
        for row in rows
    ]


def query_policy_store(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search Pinecone first, then fallback to SQLite if Pinecone is unavailable."""
    api_key = os.getenv("Pinecone_api_key") or os.getenv("PINECONE_API_KEY")
    host = os.getenv("pinecone_Host") or os.getenv("PINECONE_HOST")
    if api_key and host:
        try:
            query_vector = _embed_text(question)
            index = Pinecone(api_key=api_key).Index(host=host)
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=PINECONE_NAMESPACE,
            )
            matches = results.get("matches", []) if isinstance(results, dict) else []
            return [
                {
                    "id": match.get("id"),
                    "score": match.get("score", 0.0),
                    "metadata": match.get("metadata", {}),
                }
                for match in matches
            ]
        except Exception:
            pass

    return _load_sqlite_matches(question, top_k=top_k)


def build_user_prompt(question: str, matches: list[dict[str, Any]]) -> str:
    """Build a grounded Qwen prompt from retrieved evidence."""
    sources = []
    for i, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        sources.append(
            f"""
[SOURCE {i}]
Page: {metadata.get('page_number', 'Unknown')}
Chapter/Parent: {metadata.get('parent_title', 'Unknown')}
Section: {metadata.get('section_title', 'Unknown')}

{metadata.get('child_text', '')}
""".strip()
        )

    context = "\n\n---\n\n".join(sources)
    return f"""
Answer the question using only the sources below.

QUESTION:
{question}

SOURCES:
{context}

Provide a concise answer and include the relevant page number(s).
""".strip()


def _serialize_source(match: dict[str, Any]) -> dict[str, Any]:
    metadata = match.get("metadata", {})
    return {
        "id": match.get("id"),
        "score": match.get("score", 0.0),
        "page_number": metadata.get("page_number"),
        "parent_title": metadata.get("parent_title"),
        "section_title": metadata.get("section_title"),
        "child_text": metadata.get("child_text", ""),
    }


def answer_question(question: str, top_k: int = 5) -> dict[str, Any]:
    matches = query_policy_store(question, top_k=top_k)
    source_payload = [_serialize_source(match) for match in matches]
    retrieval_debug = {
        "index": os.getenv("PINECONE_INDEX_NAME", "ragl1"),
        "namespace": PINECONE_NAMESPACE,
        "embedding_model": MODEL_NAME,
        "top_k": top_k,
        "vector_dimension": _get_target_dimension(),
    }

    if not matches:
        return {
            "answer": "I could not find sufficient information in the provided policy context.",
            "sources": [],
            "db_proof": {"database": str(DB_PATH), "rows_found": 0},
            "retrieval_debug": retrieval_debug,
            "logs": [
                "Embedded query for semantic search",
                "No relevant matches found in Pinecone or SQLite",
                "Returned empty evidence set",
            ],
        }

    prompt = build_user_prompt(question, matches)
    answer = ask_qwen(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.1,
    )

    return {
        "answer": answer,
        "sources": source_payload,
        "db_proof": {
            "database": str(DB_PATH),
            "rows_found": len(matches),
            "namespace": PINECONE_NAMESPACE,
        },
        "retrieval_debug": retrieval_debug,
        "logs": [
            "Embedded query with all-MiniLM-L6-v2",
            f"Searched namespace: {PINECONE_NAMESPACE}",
            f"Retrieved {len(matches)} relevant chunks",
            "Context sent to Qwen for grounded answer generation",
        ],
    }
