# HR Policy Manual RAG Project

This project builds a retrieval-augmented generation (RAG) workflow for the HR Policy Manual PDF in `Dataset/HR Policy Manual 2023.pdf`.

The system is designed to:
- extract policy text from the PDF
- clean cover pages,headers, and TOC noise
- split long sections into token-aware parent/child chunks
- store chunk metadata in SQLite
- optionally upsert vectors to Pinecone
- retrieve relevant policy passages for a user question
- answer using a grounded HR-only prompt
- show evidence and debug output in a frontend chat UI

---

## High-level approach

The app follows a local, explainable RAG flow:

1. PDF ingestion
2. text cleaning and section parsing
3. parent/child chunking with token limits
4. SQLite persistence and JSON artifact export
5. optional Pinecone vector upsert
6. question embedding and semantic retrieval
7. Qwen answer generation with HR-only guardrails
8. UI display with evidence and source proof

This is built for policy/manual documents where correctness and traceability matter more than raw generation.

---

## Repository structure

```text
Rag_L_one/
├── app.py
├── requirements.txt
├── README.md
├── .env
├── Dataset/
│   └── HR Policy Manual 2023.pdf
├── artifacts/
│   ├── hr_policy_chunks.json
│   └── rag_evaluation_results.json
├── back_end/
│   ├── ingestion/
│   │   ├── hr_policy_pipeline.py
│   │   └── validate_chunks.py
│   └── retival/
│       ├── retrieval_service.py
│       ├── rag_evaluation.py
│       └── test_retrieval.py
├── front_end/
│   ├── static/
│   │   ├── app.js
│   │   └── style.css
│   └── templates/
│       └── index.html
├── llm_engine/
│   └── ollama_client.py
├── logs/
├── pinecone/
│   └── connection_check.py
├── storage/
│   └── hr_policy_chunks.db
└── Docuent_intelegence_engine/
    ├── __init__.py
    ├── document_analyzer.py
    └── strategy_registry.py
```

---

## Core design decisions

### 1) Ingestion is PDF-first
The ingestion pipeline reads the HR manual PDF and extracts page text with PyMuPDF.
It removes noisy content such as:
- cover page text
- document headers/footers
- isolated page numbers
- TOC pages
- very short non-policy fragments

The goal is to keep only meaningful HR policy content before chunking.

### 2) Parent/child chunking is token-aware
The chunking logic preserves section boundaries and then splits larger sections into smaller child chunks using token limits.

Current target values:
- child target: 250 tokens
- child max: 320 tokens
- overlap: 30 tokens

This keeps each chunk compact enough for embedding while preserving its policy meaning.

### 3) SQLite is the local source of truth
Every chunk is stored with metadata like:
- `parent_title`
- `section_title`
- `page_number`
- `token_count`
- `child_text`

This makes debugging, proof generation, and fallback retrieval easy.

### 4) Pinecone is optional but supported
If Pinecone credentials are configured, vectors are upserted in batches to stay below the request-size limit.
If Pinecone is not configured, the app still works using the SQLite evidence path and local retrieval logic.

### 5) Qwen is the local answer model
The answer generation layer uses Ollama with the local Qwen model configured in the app.

The local model used here is:
- `qwen2.5-coder:1.5b`

### 6) Guardrails are strict
The answer prompt is written to ensure the model answers only from the policy context and does not invent policy facts.

The model should only answer if:
- the question is HR-policy related
- the retrieved context supports the answer
- the answer is grounded in the provided policy text

---

## Local workflow

### Install dependencies

```powershell
cd C:\projects\Rag_L_one
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Pull the Ollama model

```powershell
ollama pull qwen2.5-coder:1.5b
```

### Ingest the policy PDF

```powershell
cd C:\projects\Rag_L_one
.\.venv\Scripts\python.exe back_end\ingestion\hr_policy_pipeline.py --pdf ".\Dataset\HR Policy Manual 2023.pdf"
```

### Validate chunk quality

```powershell
cd C:\projects\Rag_L_one
.\.venv\Scripts\python.exe back_end\ingestion\validate_chunks.py
```

This script checks:
- total chunk count
- token range and averages
- empty and tiny chunks
- oversized chunks
- token mismatch warnings
- presence of key HR-policy terms like maternity, paternity, office hours, recruitment, grievance

---

## Retrieval and answer flow

### Retrieval service
The main logic lives in:
- `back_end/retival/retrieval_service.py`

It does the following:
- embeds the user question
- searches Pinecone if credentials are available
- falls back to SQLite if Pinecone is missing
- builds a grounded prompt
- returns answer + sources + retrieval debug output

### Retrieval smoke tests

```powershell
cd C:\projects\Rag_L_one
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); from back_end.retival.retrieval_service import answer_question; print(answer_question('How many days of maternity leave are allowed?', top_k=5))"
```

You can also test the vector lookup script:

```powershell
cd C:\projects\Rag_L_one
.\.venv\Scripts\python.exe back_end\retival\test_retrieval.py
```

---

## Run the app

The app auto-checks for the policy data and runs ingestion automatically if the chunk database or artifact is missing.

```powershell
cd C:\projects\Rag_L_one
.\.venv\Scripts\python.exe app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The UI shows:
- chat question and answer
- retrieved evidence cards
- retrieval metadata
- debug logs

---

## Evaluation

The evaluation script checks question-answer quality against a set of known policy facts.

Run:

```powershell
cd C:\projects\Rag_L_one\back_end\retival
python .\rag_evaluation.py
```

Output is written to:

```text
artifacts/rag_evaluation_results.json
```

---

## Important project principle

This workflow is built to do the following in the correct order:

1. fix PDF extraction quality
2. fix chunk quality
3. validate chunk token sizes and content
4. ensure retrieval returns real evidence
5. only then ask the model to answer

If the chunk or evidence quality is poor, the model cannot recover with a good prompt. The system is intentionally structured to keep retrieval grounded and explainable.

---

## Current status

The project is currently operating as a working HR policy RAG prototype with:
- local PDF ingestion
- token-aware chunking
- SQLite persistence
- optional Pinecone batch upsert
- retrieval service with fallback logic
- local Ollama Qwen integration
- Flask chat API and frontend UI

---

## Notes

- App startup runs ingestion automatically when the policy artifact is missing.
- Pinecone is optional, but the app is designed to use it when configured.
- The answer layer is constrained to stay within the HR policy knowledge base.
- Evidence pages and source chunks are surfaced in the UI to make the answer explainable.

---

## License

This project is intended for internal demo and research purposes unless otherwise specified by the owning organization.
