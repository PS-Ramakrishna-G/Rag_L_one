from __future__ import annotations

import logging
import os
import re
import sqlite3
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from llm_engine.ollama_client import ask_qwen

logger = logging.getLogger("ragl1.retrieval")

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

Core rule:
Answer only from the policy evidence provided in the prompt. Do not use outside knowledge, common HR assumptions, or generic policy language.

Required behavior:
1. Give the direct answer first in 1-2 sentences.
2. Use only exact names, roles, procedures, and page references that appear in the supplied evidence.
3. If the evidence is missing or unclear, say exactly:
   "I could not find sufficient information in the provided policy context."
4. If the policy evidence does not specify a person, approver, or process, say exactly:
   "The policy evidence does not specify this."
5. If the question is outside the HR Policy Manual, say:
   "This question is outside the HR Policy Manual and cannot be answered from the provided policy context."
6. Include page references like [Page X] when available.
7. Ignore irrelevant sections and do not summarize unrelated policy content.
8. Never invent rules, dates, eligibility, benefits, approvals, or exceptions.
9. Never mention Pinecone, embeddings, vector databases, chunks, or retrieval internals.
""".strip()


AUTHORITY_HINTS = (
    "who",
    "whom",
    "contact",
    "ask",
    "need to ask",
    "report to",
    "approver",
    "approve",
    "approval",
    "authority",
)
EXPLICIT_AUTHORITY_TOKENS = (
    "director",
    "head of department",
    "department head",
    "reporting officer",
    "approving authority",
    "manager",
    "supervisor",
    "hr",
    "human resources",
    "chairperson",
    "committee",
)
AUTHORITY_ACTION_WORDS = (
    "ask",
    "contact",
    "report",
    "submit",
    "notify",
    "inform",
    "consult",
    "approval",
    "approve",
    "follow up",
    "seek",
)
LEAVE_TYPES = (
    "casual leave",
    "earned leave",
    "maternity leave",
    "paternity leave",
    "annual leave",
    "medical leave",
    "sick leave",
)
LEAVE_ALIASES = {
    "casual leave": ["casual leave", "casual leaves", "casula leave", "casula leaves", "casual"],
    "earned leave": ["earned leave", "earned leaves", "earned"],
    "maternity leave": ["maternity leave", "maternity", "maternal leave", "maternity leaves"],
    "paternity leave": ["paternity leave", "paternity", "paternal leave", "paternity leaves"],
    "annual leave": ["annual leave", "annual leaves", "yearly leave"],
    "medical leave": ["medical leave", "medical leaves", "med leave"],
    "sick leave": ["sick leave", "sick leaves", "medical leave", "illness leave"],
}


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip())


def _resolve_leave_type(text: str) -> str | None:
    normalized = _normalize_question(text).lower()
    if "leave" not in normalized:
        return None

    for canonical, aliases in LEAVE_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                return canonical

    for canonical, aliases in LEAVE_ALIASES.items():
        for alias in aliases:
            score = SequenceMatcher(None, normalized, alias).ratio()
            if score >= 0.72:
                return canonical

    return None


def route_question(question: str) -> dict[str, Any]:
    """Classify the question before retrieval and ask for clarification if needed."""
    normalized = _normalize_question(question).lower()
    if not normalized:
        return {
            "is_hr_related": False,
            "needs_clarification": False,
            "intent": "empty",
            "entities": {},
            "clarification_question": "",
            "rewritten_query": "",
            "search_terms": [],
        }

    leave_match = _resolve_leave_type(normalized)
    if "leave" in normalized and not leave_match:
        return {
            "is_hr_related": True,
            "needs_clarification": True,
            "intent": "leave_entitlement",
            "entities": {},
            "clarification_question": "Which leave type would you like to know about — Casual Leave, Earned Leave, Maternity Leave, Paternity Leave, or another leave type?",
            "rewritten_query": "",
            "search_terms": [],
        }

    if leave_match:
        specific = leave_match
        return {
            "is_hr_related": True,
            "needs_clarification": False,
            "intent": "leave_entitlement",
            "entities": {"leave_type": specific.title()},
            "clarification_question": "",
            "rewritten_query": f"How many days of {specific.title()} is an employee entitled to?",
            "search_terms": [specific.title(), "leave entitlement", "days", "calendar year"],
        }

    return {
        "is_hr_related": True,
        "needs_clarification": False,
        "intent": "general_hr_policy",
        "entities": {},
        "clarification_question": "",
        "rewritten_query": normalized,
        "search_terms": [token for token in re.split(r"\s+", normalized) if len(token) >= 3],
    }


def _expand_search_terms(question: str) -> list[str]:
    lowered = question.lower()
    raw_terms = [
        term.strip()
        for term in lowered.replace("?", " ").split()
        if len(term.strip()) >= 3
    ]

    synonyms = {
        "sick": ["sick", "medical", "illness"],
        "leave": ["leave", "annual leave", "sick leave", "medical leave"],
        "ask": ["ask", "contact", "report", "approver", "approval"],
        "who": ["who", "whom", "authority", "reporting officer"],
        "director": ["director", "manager", "supervisor", "head of department"],
    }

    expanded = []
    seen: set[str] = set()

    for term in raw_terms:
        for key, values in synonyms.items():
            if term in key or term in key.replace(" ", ""):
                for value in values:
                    if value not in seen:
                        expanded.append(value)
                        seen.add(value)
        if term not in seen:
            expanded.append(term)
            seen.add(term)

    if any(token in lowered for token in ("who", "whom", "contact", "ask", "report to", "approval", "approver")):
        for token in ("director", "manager", "supervisor", "hr", "human resources", "approving authority"):
            if token not in seen:
                expanded.append(token)
                seen.add(token)

    return expanded


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
    terms = _expand_search_terms(question)

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

    logger.info("Starting retrieval for question: %s", question[:150])
    pinecone_matches: list[dict[str, Any]] = []

    if PINECONE_INDEX is not None:
        try:
            query_vector = _embed_text(question)
            logger.info("Embedding generated for Pinecone query. Namespace=%s, top_k=%s", PINECONE_NAMESPACE, top_k)

            response = PINECONE_INDEX.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False,
                namespace=PINECONE_NAMESPACE,
            )

            raw_matches = getattr(response, "matches", [])
            for match in raw_matches:
                metadata = getattr(match, "metadata", {}) or {}
                pinecone_matches.append(
                    {
                        "id": getattr(match, "id", None),
                        "score": float(getattr(match, "score", 0.0)),
                        "metadata": metadata,
                    }
                )

            logger.info("Pinecone returned %s matches", len(pinecone_matches))
        except Exception:
            logger.exception("Pinecone query failed for question: %s", question[:150])

    sqlite_matches = _load_sqlite_matches(question, top_k=top_k)
    logger.info("SQLite returned %s matches", len(sqlite_matches))

    if pinecone_matches:
        combined = pinecone_matches + sqlite_matches
        unique: dict[str, dict[str, Any]] = {}
        for match in combined:
            key = str(match.get("id") or match.get("metadata", {}).get("child_text") or match.get("metadata", {}).get("page_number"))
            if key not in unique:
                unique[key] = match
        ranked = sorted(unique.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)
        logger.info("Combined retrieval result count: %s", len(ranked))
        return ranked[:top_k]

    if sqlite_matches:
        logger.info("Using SQLite results because Pinecone had no hits")
        return sqlite_matches

    logger.warning("No Pinecone and no SQLite matches found for question: %s", question[:150])
    return []


# ============================================================
# SAFETY CHECKS
# ============================================================

def _match_authority_question(question: str) -> bool:
    lowered = question.lower().strip()
    if not lowered:
        return False
    return any(token in lowered for token in AUTHORITY_HINTS)


def _policy_has_explicit_authority(match: dict[str, Any]) -> bool:
    metadata = match.get("metadata", {}) or {}
    text_blob = " ".join(
        [
            metadata.get("parent_title", ""),
            metadata.get("section_title", ""),
            metadata.get("child_text", ""),
        ]
    ).lower()

    for token in EXPLICIT_AUTHORITY_TOKENS:
        idx = text_blob.find(token)
        if idx == -1:
            continue
        window = text_blob[max(0, idx - 80): idx + 160]
        if any(action in window for action in AUTHORITY_ACTION_WORDS):
            return True

    return any(phrase in text_blob for phrase in ("ask the director", "contact the director", "report to the director", "submit to the director", "approval by the director", "approving authority", "reporting officer"))


def _should_refuse_authority_question(question: str, matches: list[dict[str, Any]]) -> bool:
    if not _match_authority_question(question):
        return False
    if not matches:
        return True
    return not any(_policy_has_explicit_authority(match) for match in matches)


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

No relevant policy evidence was retrieved.

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

RELEVANT POLICY EVIDENCE:
{context}

Requirements:
- Answer directly and specifically.
- Use only the evidence above; do not use general HR knowledge or assumptions.
- If a person, approver, authority, process, or policy detail is not named in the evidence, say: "The policy evidence does not specify this."
- If the evidence is not enough to support the question, say: "I could not find sufficient information in the provided policy context."
- Include page references and section names when available.
- Do not summarize unrelated sections.
- Keep the response concise and grounded in the actual policy text.
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

    logger.info("Processing answer request for question: %s", question[:150])
    route = route_question(question)

    if route.get("needs_clarification"):
        logger.info("Clarification needed for HR question: %s", question[:150])
        return {
            "answer": route["clarification_question"],
            "sources": [],
            "db_proof": {"database": str(DB_PATH), "rows_found": 0},
            "retrieval_debug": {
                "index": os.getenv("PINECONE_INDEX_NAME", "ragl1"),
                "namespace": PINECONE_NAMESPACE,
                "embedding_model": MODEL_NAME,
                "top_k": top_k,
                "vector_dimension": PINECONE_DIMENSION,
                "query_route": route,
            },
            "logs": ["Ambiguous HR question requires clarification"],
            "grounded": False,
            "source_pages": [],
            "query_route": route,
        }

    try:
        matches = query_policy_store(
            route.get("rewritten_query") or question,
            top_k=top_k,
        )
    except Exception:
        logger.exception("Retrieval pipeline failed before generating answer")
        raise

    logger.info("Retrieved %s candidate chunks", len(matches))

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
        "query_route": route,
    }

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
        if page is not None and page not in pages:
            pages.append(page)

    if not matches:
        logger.warning("No policy evidence found for question: %s", question[:150])
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

    if _should_refuse_authority_question(question, matches):
        logger.warning("Refusing vague authority question without explicit policy evidence: %s", question[:150])
        return {
            "answer": "The policy evidence does not specify this.",
            "sources": source_payload,
            "db_proof": {
                "database": str(DB_PATH),
                "rows_found": len(matches),
                "namespace": PINECONE_NAMESPACE,
            },
            "retrieval_debug": retrieval_debug,
            "logs": [
                "Embedded query with all-MiniLM-L6-v2",
                f"Retrieved {len(matches)} chunks",
                "Authority question was rejected because the policy text does not specify the approver",
            ],
            "grounded": False,
            "source_pages": pages,
        }

    prompt = build_user_prompt(
        question,
        matches,
    )
    logger.info("Sending %s chunks to LLM for answer generation", len(matches))

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
        "query_route": route,
    }