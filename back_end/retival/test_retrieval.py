from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "all-MiniLM-L6-v2"
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "hr-policy")
TOP_K = 10

model = SentenceTransformer(MODEL_NAME)


def normalize_embedding(vector: list[float], target_dimension: int) -> list[float]:
    vector = [float(v) for v in vector]

    if len(vector) == target_dimension:
        return vector

    if len(vector) < target_dimension:
        return vector + [0.0] * (target_dimension - len(vector))

    return vector[:target_dimension]


def get_index():
    api_key = os.getenv("PINECONE_API_KEY") or os.getenv("Pinecone_api_key")
    host = os.getenv("PINECONE_HOST") or os.getenv("pinecone_Host")

    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY")

    if not host:
        raise ValueError("Missing PINECONE_HOST")

    pc = Pinecone(api_key=api_key)
    return pc.Index(host=host)


def query(question: str):
    index = get_index()

    dimension = int(os.getenv("PINECONE_VECTOR_DIMENSION", "512"))
    vector = model.encode(question, show_progress_bar=False).tolist()
    vector = normalize_embedding(vector, dimension)

    response = index.query(
        vector=vector,
        top_k=TOP_K,
        namespace=NAMESPACE,
        include_metadata=True,
        include_values=False,
    )

    matches = getattr(response, "matches", [])

    print()
    print("=" * 100)
    print("QUESTION:")
    print(question)
    print("=" * 100)

    for rank, match in enumerate(matches, start=1):
        metadata = getattr(match, "metadata", {}) or {}

        print()
        print("-" * 100)
        print(f"RANK: {rank}")
        print(f"ID: {getattr(match, 'id', None)}")
        print(f"SCORE: {float(getattr(match, 'score', 0.0)):.4f}")
        print(f"PAGE: {metadata.get('page_number')}")
        print(f"PARENT: {metadata.get('parent_title')}")
        print(f"SECTION: {metadata.get('section_title')}")
        print()
        print(metadata.get("child_text", "")[:1500])


if __name__ == "__main__":
    test_questions = [
        "What is the maternity leave policy?",
        "What are the normal office hours?",
        "How many days of paternity leave are allowed?",
        "What is the recruitment process?",
        "What is the grievance process?",
    ]

    for question in test_questions:
        query(question)
