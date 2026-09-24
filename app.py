from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from back_end.retival.retrieval_service import answer_question

app = Flask(
    __name__,
    template_folder="front_end/templates",
    static_folder="front_end/static",
)


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
    if not question:
        return jsonify({"answer": "Please enter a question about the policy manual.", "sources": [], "db_proof": {}}), 400

    result = answer_question(question, top_k=5)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
