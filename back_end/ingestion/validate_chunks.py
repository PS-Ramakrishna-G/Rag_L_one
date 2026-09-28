from __future__ import annotations

import json
import statistics
from pathlib import Path

import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = PROJECT_ROOT / "artifacts" / "hr_policy_chunks.json"

encoding = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(encoding.encode(text or ""))


def main() -> None:
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Chunk file not found: {JSON_PATH}")

    chunks = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    print("=" * 80)
    print("HR POLICY CHUNK VALIDATION")
    print("=" * 80)
    print(f"Total chunks: {len(chunks)}")

    if not chunks:
        return

    token_counts = []
    empty_chunks = 0
    tiny_chunks = 0
    large_chunks = 0

    for chunk in chunks:
        text = (chunk.get("child_text") or chunk.get("text") or "").strip()
        tokens = token_count(text)
        token_counts.append(tokens)

        stored_tokens = chunk.get("token_count")
        if stored_tokens is not None and stored_tokens != tokens:
            print(
                "TOKEN COUNT MISMATCH:",
                chunk.get("chunk_id") or chunk.get("id"),
                "stored=",
                stored_tokens,
                "actual=",
                tokens,
            )

        if not text:
            empty_chunks += 1

        if 0 < tokens < 50:
            tiny_chunks += 1

        if tokens > 350:
            large_chunks += 1

    print()
    print("TOKEN STATISTICS")
    print("-" * 80)
    print(f"Minimum: {min(token_counts)}")
    print(f"Maximum: {max(token_counts)}")
    print(f"Average: {statistics.mean(token_counts):.2f}")
    print(f"Median: {statistics.median(token_counts):.2f}")
    print()
    print(f"Empty chunks: {empty_chunks}")
    print(f"Tiny chunks (<50 tokens): {tiny_chunks}")
    print(f"Large chunks (>350 tokens): {large_chunks}")

    print()
    print("=" * 80)
    print("FIRST 10 CHUNKS")
    print("=" * 80)

    for index, chunk in enumerate(chunks[:10], start=1):
        text = (chunk.get("child_text") or chunk.get("text") or "").strip()
        print()
        print("-" * 80)
        print(f"CHUNK {index}")
        print("ID:", chunk.get("chunk_id") or chunk.get("id"))
        print("Parent:", chunk.get("parent_title"))
        print("Section:", chunk.get("section_title"))
        print("Page:", chunk.get("page_number") or chunk.get("page_start"))
        print("Tokens:", token_count(text))
        print()
        print(text[:1000])

    print()
    print("=" * 80)
    print("SEARCH TEST: MATERNITY")
    print("=" * 80)

    found = 0

    for chunk in chunks:
        text = chunk.get("child_text") or chunk.get("text") or ""
        if "maternity" in text.lower():
            found += 1
            print()
            print("-" * 80)
            print("ID:", chunk.get("chunk_id") or chunk.get("id"))
            print("Parent:", chunk.get("parent_title"))
            print("Section:", chunk.get("section_title"))
            print("Page:", chunk.get("page_number") or chunk.get("page_start"))
            print("Tokens:", token_count(text))
            print()
            print(text[:1500])

    print()
    print(f"Maternity chunks found: {found}")

    print()
    print("=" * 80)
    print("ADDITIONAL SEARCH TESTS")
    print("=" * 80)

    test_terms = [
        "maternity",
        "paternity",
        "office hours",
        "recruitment",
        "grievance",
    ]

    for term in test_terms:
        term_found = 0
        for chunk in chunks:
            text = chunk.get("child_text") or chunk.get("text") or ""
            if term.lower() in text.lower():
                term_found += 1

        print(f"{term}: {term_found} chunks found")


if __name__ == "__main__":
    main()
