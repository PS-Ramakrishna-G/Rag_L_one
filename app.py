from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from back_end.ingestion.hr_policy_pipeline import analyze_policy_manual
from back_end.retival.retrieval_service import answer_question

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "storage" / "hr_policy_chunks.db"
JSON_PATH = PROJECT_ROOT / "artifacts" / "hr_policy_chunks.json"
PDF_PATH = PROJECT_ROOT / "Dataset" / "HR Policy Manual 2023.pdf"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("ragl1.backend")

app = Flask(
    __name__,
    template_folder="front_end/templates",
    static_folder="front_end/static",
)


@app.before_request
def log_request_start():
    logger.info("Incoming request: %s %s", request.method, request.path)
    logger.info("Request query params: %s", dict(request.args))
    logger.info("Request body size: %s bytes", len(request.get_data(cache=True, as_text=False)))

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        logger.info("Request JSON payload keys: %s", sorted(payload.keys()))


@app.after_request
def log_request_end(response):
    logger.info("Completed request: %s %s -> %s", request.method, request.path, response.status_code)
    return response


def ensure_policy_data() -> None:
    """Run ingestion automatically when the policy database is missing or stale."""
    if DB_PATH.exists() and JSON_PATH.exists():
        return

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Policy PDF not found: {PDF_PATH}")

    logger.info("Starting policy ingestion for %s", PDF_PATH)
    analyze_policy_manual(
        str(PDF_PATH),
        db_path=str(DB_PATH),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "hr-policy"),
    )
    logger.info("Policy ingestion complete.")


ensure_policy_data()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    logger.info("Chat request received. Question length: %s", len(question))

    if not question:
        logger.warning("Chat request rejected: empty question")
        return jsonify({"answer": "Please enter a question about the policy manual.", "sources": [], "db_proof": {}}), 400

    try:
        result = answer_question(question, top_k=5)
        logger.info("Chat request processed successfully. Answer length: %s", len(result.get("answer", "")))
        return jsonify(result)
    except Exception:
        logger.exception("Backend failure while processing chat request")
        raise


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
