# HR Policy RAG Assistant

This project builds a retrieval-augmented generation (RAG) system for an HR policy manual. It ingests policy content, stores chunked text in SQLite and Pinecone, retrieves relevant passages, and answers policy questions using a local Qwen model via Ollama.

## Project overview

We use a document ingestion pipeline to extract HR policy text from PDFs and convert it into clean, structured chunks.
Chunking is done to break long policy text into smaller passages so retrieval stays precise and the model gets focused context.
We store each chunk with metadata like page number, section, and parent title so evidence can be shown in the UI.
For embeddings, we use sentence-transformers with all-MiniLM-L6-v2 to convert text into vectors for semantic search.
For vector storage, we use Pinecone because it supports fast similarity search over large document collections.
For local fallback and proof, we also save chunk metadata into SQLite so the system still works when Pinecone is unavailable.
Retrieval uses the user question embedding, finds the closest policy chunks, and passes only those chunks into the LLM prompt.
The LLM layer uses Ollama and the local Qwen model to answer grounded questions without external API dependence.
The system prompt enforces strict policy-only behavior, so the model must answer from the retrieved policy text and not invent facts.
This entire flow is kept in one RAG pattern because it is a clean, explainable pipeline: ingest → chunk → embed → retrieve → answer → show evidence.
Why we use one file for the main logic
We use one central retrieval file, retrieval_service.py, because it keeps the orchestration simple and easy to trace:

question embedding
retrieval
prompt construction
answer generation
source proof formatting
This makes debugging easier, reduces duplication, and keeps the app easier to demo and explain. The frontend then just calls one API endpoint, and the backend handles the whole RAG logic in one place.

The system has four main layers:

1. Document ingestion and chunking
2. Vector storage in Pinecone
3. Retrieval and answer generation
4. Frontend demo UI

## Repository structure

```text
Rag_L_one/
├── app.py
├── requirements.txt
├── README.md
├── artifacts/
│   └── rag_evaluation.json
├── back_end/
│   ├── ingestion/
│   └── retival/
├── front_end/
│   ├── static/
│   └── templates/
├── llm_engine/
│   └── ollama_client.py
├── storage/
├── pinecone/
├── logs/
├── Dataset/
├── Docuent_intelegence_engine/
└── .env
```

## Main components

### 1) Ingestion pipeline
Responsible for extracting document content from policy PDFs, splitting into chunks, and storing metadata.

Key files:
- `back_end/ingestion/`
- `Docuent_intelegence_engine/document_analyzer.py`
- `Docuent_intelegence_engine/strategy_registry.py`

This layer prepares the document so it can be searched semantically.

### 2) Retrieval service
Responsible for:
- embedding the user question
- querying Pinecone for semantic matches
- falling back to SQLite when needed
- building a grounded prompt for the LLM

Key file:
- `back_end/retival/retrieval_service.py`

### 3) LLM engine
Connects to a local Ollama instance and calls the Qwen model.

Key file:
- `llm_engine/ollama_client.py`

### 4) Frontend app
Provides a ChatGPT-style interface showing the answer and retrieval proof.

Key files:
- `app.py`
- `front_end/templates/index.html`
- `front_end/static/app.js`
- `front_end/static/style.css`

## Environment setup

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Make sure Ollama is running locally.
4. Set your environment variables in `.env`.

Example `.env` values:

```env
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index
PINECONE_NAMESPACE=hr-policy
PINECONE_HOST=your_host
DEFAULT_OLLAMA_HOST=http://localhost:11434
DEFAULT_QWEN_MODEL=qwen2.5-coder:1.5b
```

## Install dependencies

```powershell
cd C:\projects\Rag_L_one
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the app

From the project root:

```powershell
cd C:\projects\Rag_L_one
.\.venv\Scripts\python.exe app.py
```

Then open:

```text
http://localhost:5000
```

## Run the evaluation

The evaluation script tests the RAG system on a known set of policy questions.

Run from the project root:

```powershell
cd C:\projects\Rag_L_one
.\.venv\Scripts\python.exe -m back_end.retival.rag_evaluation
```

This writes results to:

```text
artifacts/rag_evaluation.json
```

## How the RAG pipeline works

1. A user asks a question in the frontend.
2. Flask receives the request in `app.py`.
3. `answer_question()` in `retrieval_service.py` is called.
4. The question is embedded using `all-MiniLM-L6-v2`.
5. Pinecone is searched for similar policy chunks.
6. If Pinecone is unavailable, SQLite is used as a fallback.
7. The retrieved chunks are inserted into a grounded prompt.
8. Qwen is asked to answer using only the provided policy evidence.
9. The answer and source metadata are returned to the frontend.

## Evaluation purpose

The evaluation script measures whether the system can answer known policy questions with evidence from the document. It is useful for checking:

- retrieval quality
- answer faithfulness
- grounding to the policy text
- system reliability before demos

## Notes

- The local LLM is served through Ollama.
- The vector database is Pinecone.
- SQLite acts as a fallback and a local evidence store.
- The answer generation is constrained by a system prompt to keep replies grounded to the HR policy content.

## Recommended next steps

- Improve the evaluation metrics with exact match / answer similarity scoring
- Add stronger citation formatting in responses
- Expand the policy dataset coverage
- Add UI improvements for proof visibility and source browsing

## License

This project is for internal demo and research use unless otherwise specified by the owner.
