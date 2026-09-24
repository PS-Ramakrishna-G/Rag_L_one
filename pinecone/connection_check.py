"""Read-only Pinecone connection and index-status check.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe pinecone\\connection_check.py

Required .env values (the checker accepts these existing names and uppercase
aliases):
    Pinecone_api_key=...
    pinecone_index_name=...
    pinecone_Host=...              # recommended; index host from Pinecone console

The script never prints the API key or retrieves vector values. It only reads
account index metadata and the selected index's aggregate namespace statistics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE settings without requiring python-dotenv."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Shell/CI variables take precedence over the local .env file.
        os.environ.setdefault(key, value)


def setting(*names: str) -> str | None:
    return next((os.getenv(name) for name in names if os.getenv(name)), None)


def as_dict(value: Any) -> Any:
    """Convert Pinecone SDK response objects into printable dictionaries."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return value


def main() -> int:
    load_env_file(ENV_FILE)

    api_key = setting("Pinecone_api_key", "PINECONE_API_KEY")
    index_name = setting("pinecone_index_name", "PINECONE_INDEX_NAME")
    host = setting("pinecone_Host", "PINECONE_HOST")
    environment = setting("pinecone_region", "PINECONE_REGION")
    data_type = setting("pineconne_dataType", "PINECONE_DATA_TYPE")

    missing = []
    if not api_key:
        missing.append("Pinecone_api_key (or PINECONE_API_KEY)")
    if not index_name:
        missing.append("pinecone_index_name (or PINECONE_INDEX_NAME)")
    if missing:
        print("Configuration error: missing " + ", ".join(missing), file=sys.stderr)
        return 2

    try:
        from pinecone import Pinecone
    except ImportError:
        print(
            "Missing dependency: install the modern Pinecone SDK in the active "
            "environment with: pip install pinecone",
            file=sys.stderr,
        )
        return 2

    try:
        client = Pinecone(api_key=api_key)
        description = client.describe_index(name=index_name)
        description_dict = as_dict(description)

        # A host explicitly in .env is useful for a locked-down deployment, but
        # the host returned by describe_index is authoritative when not supplied.
        resolved_host = host or (
            description_dict.get("host")
            if isinstance(description_dict, dict)
            else getattr(description, "host", None)
        )
        if not resolved_host:
            print("Connection error: Pinecone did not return an index host.", file=sys.stderr)
            return 1

        index = client.Index(host=resolved_host)
        stats = as_dict(index.describe_index_stats())

        print("Pinecone connection: successful")
        print(f"Index: {index_name}")
        print(f"Host: {resolved_host}")
        if environment:
            print(f"Configured region: {environment}")
        if data_type:
            print(f"Configured vector data type: {data_type}")

        if isinstance(description_dict, dict):
            print(f"Dimension: {description_dict.get('dimension', 'unknown')}")
            print(f"Metric: {description_dict.get('metric', 'unknown')}")
            print(f"Index status: {description_dict.get('status', 'unknown')}")

        if isinstance(stats, dict):
            print(f"Total vector count: {stats.get('total_vector_count', 'unknown')}")
            namespaces = stats.get("namespaces", {}) or {}
            print(f"Namespace count: {len(namespaces)}")
            for namespace, namespace_stats in namespaces.items():
                vector_count = (
                    namespace_stats.get("vector_count", "unknown")
                    if isinstance(namespace_stats, dict)
                    else getattr(namespace_stats, "vector_count", "unknown")
                )
                print(f"  - {namespace or '(default)'}: {vector_count} vectors")
        else:
            print(f"Index statistics: {stats}")
        return 0

    except Exception as error:  # SDK error types vary by installed version.
        print(f"Pinecone connection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
