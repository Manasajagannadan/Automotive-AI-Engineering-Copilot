# Automotive AI Engineering Copilot

**An end-to-end, document-grounded AI assistant for automotive engineering knowledge retrieval and question answering.**

Built with Retrieval-Augmented Generation (RAG), Sentence Transformers, FAISS, Groq/Gemini LLM integration, FastAPI, and Streamlit. The system retrieves relevant passages from technical documents and supplies that evidence as context to an LLM before generating an answer.

The application extracts text from automotive PDF, DOCX, and TXT documents, chunks the text, embeds chunks with Sentence Transformers, indexes them in FAISS, retrieves relevant passages for a question, and generates a context-grounded answer. It includes a FastAPI backend, Streamlit frontend, a lightweight evaluation harness, unit tests, and an optional Gemini VLM client.

> **Important API detail:** The current Streamlit UI calls `POST /ask` at `http://127.0.0.1:8000` by default. The matching FastAPI entry point is `src.api.main:app`. The repository also contains a separate `api.main:app` application with different routes; it is not the entry point used by the current Streamlit UI.

## 1. Project objectives

Automotive engineering teams work with technical information distributed across specifications, manuals, reports, and test-related documents. Finding the right passage can take time. This project demonstrates a retrieval-grounded workflow that makes a supplied document collection searchable through natural-language questions.

The implementation focuses on:
- Converting supported technical documents into searchable text.
- Creating semantic vector representations of document chunks.
- Retrieving relevant passages for a user query.
- Passing retrieved context to an LLM to generate a response.
- Returning source information alongside the answer.
- Providing API, UI, evaluation, and test entry points.

## 2. System architecture

```text
PDF / DOCX / TXT documents
          |
          v
src/ingestion/loader.py       Extract and clean text
          |
          v
src/chunking/chunker.py       Word-based overlapping chunks
          |
          v
src/embeddings/embedder.py    all-MiniLM-L6-v2 embeddings (CPU)
          |
          v
src/retrieval/faiss_index.py  FAISS vector index + metadata
          |
          v
User question -> Input guardrails -> FAISS retrieval
                                      |
                                      v
                             Retrieval confidence check
                                      |
                                      v
                              Prompt construction
                                      |
                                      v
                                Groq / Gemini LLM
                                      |
                                      v
                             Answer + source filenames
                                      |
                                      v
                             FastAPI / Streamlit UI

Optional: Image + question -> Gemini VLM -> Image analysis
```

## 3. Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 (Dockerfile base image) |
| Document parsing | PyPDF, python-docx |
| Embeddings | Sentence Transformers, `all-MiniLM-L6-v2` |
| Vector search | FAISS CPU (`IndexFlatIP`) |
| LLM | Groq by default; Gemini alternative |
| Optional VLM | Google Generative AI / Gemini |
| API | FastAPI, Uvicorn |
| UI | Streamlit |
| Configuration | python-dotenv |
| Evaluation/tests | Custom evaluation script, pytest |
| Container | Docker |

## 4. Prerequisites

Install:
- Anaconda or Miniconda
- Git
- Groq API key for default text generation
- Google AI Studio API key only if you want optional Gemini image analysis

Links: [Anaconda](https://www.anaconda.com/download) · [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) · [Git](https://git-scm.com/downloads) · [Groq API keys](https://console.groq.com/keys) · [Google AI Studio](https://aistudio.google.com/apikey)

The embedding and retrieval steps are configured to run on CPU. LLM generation uses a hosted provider; a local GPU LLM is not required.

## 5. Clone the repository

Open **Anaconda Prompt** or **PowerShell**:

```powershell
cd C:\Projects
git clone https://github.com/Manasajagannadan/Automotive-AI-Engineering-Copilot.git
cd Automotive-AI-Engineering-Copilot
dir
```

Use your preferred project directory instead of `C:\Projects`.

## 6. Create and activate the Conda environment

The Dockerfile uses Python 3.11, so use the same version locally:

```powershell
conda create -n automotive-copilot python=3.11 -y
conda activate automotive-copilot
python --version
where python
```

The prompt should show `(automotive-copilot)`. Activate this environment in every new terminal before running project commands.

## 7. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The first embedding run may download `all-MiniLM-L6-v2`; allow internet access and disk space for the model cache.

## 8. Configure API keys

**Repository observation:** The current GitHub tree does not contain `.env.example`, although the original README refers to it. Create `.env` manually in the repository root for now:

```powershell
New-Item -Path .env -ItemType File
notepad .env
```

Add:

```env
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-20b

# Optional: Gemini image analysis
GOOGLE_API_KEY=

# Optional Gemini text model setting
GEMINI_MODEL=gemini-1.5-flash
```

Save the file. The code defaults to Groq when `LLM_PROVIDER` is not set. Set `LLM_PROVIDER=gemini` only if you intend to use Gemini for text generation and have configured `GOOGLE_API_KEY`.

**Security:** Never commit `.env` or real API keys. The repository should include a `.gitignore` that excludes at least:

```gitignore
.env
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.DS_Store
```

The current repository tree also contains tracked `__pycache__` and `.pyc` files; remove these generated files from Git and add the ignore rules.

## 9. Data and generated artifacts

```text
data/
├── documents/       # Source PDF, DOCX, TXT files
├── processed/       # Extracted/cleaned text files
├── chunks.jsonl     # Chunk records with source metadata
└── index/
    ├── index.faiss  # FAISS vector index
    └── metadata.pkl # Chunk metadata aligned with index vectors
```

The current loader supports `.pdf`, `.docx`, and `.txt`. It scans files directly inside `data/documents/` and does not recursively traverse subfolders. Add only documents you have permission to use.

## 10. Run the RAG pipeline

Run commands from the repository root with the Conda environment active. Run each stage only after the previous stage succeeds.

### Step 1 — Ingest documents

```powershell
python -m src.ingestion.loader --input-dir data/documents --out data/processed
```

The loader extracts text, normalizes whitespace, and writes one `.txt` output per successfully processed input file.

Check:

```powershell
Get-ChildItem data/processed
```

If zero documents are processed, check the input directory and supported extensions.

### Step 2 — Generate overlapping chunks

```powershell
python -m src.chunking.chunker --input data/processed --out data/chunks.jsonl
```

Current defaults:
- `--chunk-size 800` words
- `--overlap 150` words

You can set them explicitly:

```powershell
python -m src.chunking.chunker --input data/processed --out data/chunks.jsonl --chunk-size 800 --overlap 150
```

Each JSONL record stores `id`, `source`, `chunk_index`, and `text`.

Check:

```powershell
Get-Item data/chunks.jsonl
```

### Step 3 — Build the FAISS index

```powershell
python -m src.retrieval.faiss_index build --chunks data/chunks.jsonl --index-dir data/index
```

This generates normalized MiniLM embeddings on CPU, builds a FAISS `IndexFlatIP` index, and writes:
- `data/index/index.faiss`
- `data/index/metadata.pkl`

Because embeddings are normalized, inner-product similarity is equivalent to cosine similarity.

Check:

```powershell
Get-ChildItem data/index
```

### Step 4 — Test retrieval without calling the LLM (optional)

```powershell
python -m src.retrieval.faiss_index query --index-dir data/index --q "engine coolant temperature" --top-k 5
```

This prints similarity scores, source filenames, and snippets from retrieved chunks.

### Step 5 — Ask a question using the CLI

Ensure `.env` contains a valid Groq API key:

```powershell
python -m src.rag.generator --question "What should I check if coolant temperature is high?" --index-dir data/index --top-k 5
```

The generator checks input guardrails, retrieves chunks, checks the maximum similarity against the threshold, builds a context prompt, calls the configured LLM, and prints the answer, sources, retrieval confidence, and latency.

## 11. Evaluate the pipeline

The evaluation dataset is `src/evaluation/eval_set.json`.

Retrieval-only evaluation (does not call the LLM):

```powershell
python -m src.evaluation.evaluate --eval-set src/evaluation/eval_set.json --index-dir data/index --no-llm
```

Evaluation including answer keyword coverage:

```powershell
python -m src.evaluation.evaluate --eval-set src/evaluation/eval_set.json --index-dir data/index
```

Set the retrieval-accuracy gate threshold:

```powershell
python -m src.evaluation.evaluate --eval-set src/evaluation/eval_set.json --index-dir data/index --min-score 0.7
```

**Interpret the current metrics accurately:**
- `retrieval_accuracy`: keyword-based check that a retrieved chunk matches expected source/content keywords.
- `answer_keyword_coverage`: checks whether enough expected answer keywords appear in the generated answer.
- `avg_retrieval_confidence`: average top retrieval similarity.
- `--min-score` gates on `retrieval_accuracy`, not average similarity.
- The current evaluation set is small; expand it with representative questions from the actual corpus before treating the result as a robust quality gate.

This is a lightweight automated proxy, not a full semantic faithfulness evaluator or human-graded benchmark.

## 12. Start the FastAPI backend

The current Streamlit UI calls `POST /ask` on port `8000`. Start the matching application.

**Terminal 1** — from the repository root:

```powershell
conda activate automotive-copilot
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Routes in this application:
- `GET /health`
- `POST /ask`

Open API docs: http://127.0.0.1:8000/docs  
Health endpoint: http://127.0.0.1:8000/health

Example JSON body for `POST /ask`:

```json
{
  "question": "What should I check if coolant temperature is high?"
}
```

### Why the API command differs from the old README

There are two FastAPI entry points:

| Entry point | Routes | Role |
|---|---|---|
| `src.api.main:app` | `/health`, `/ask` | Used by current Streamlit UI and Dockerfile |
| `api.main:app` | `/health`, `/chat`, `/search`, `/vlm-chat` | Separate API implementation |

The old README command `uvicorn api.main:app --reload --port 8001` does not match the current Streamlit UI's default URL and `/ask` route.

## 13. Start the Streamlit frontend

Keep Terminal 1 running. Open **Terminal 2** in the repository root:

```powershell
conda activate automotive-copilot
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

The UI has:
1. **RAG Copilot** — sends questions to the backend `/ask` endpoint and displays answer, source filenames, retrieval confidence, and latency.
2. **VLM Image Analysis** — optional Gemini image analysis, enabled when `GOOGLE_API_KEY` is configured.

The UI's default API URL is `http://127.0.0.1:8000`. If you change the backend port, set `API_URL` to the same host and port before starting Streamlit.

## 14. Run tests

From the repository root:

```powershell
python -m pytest -v
```

Current tests cover chunk splitting, overlap continuity, empty chunk input, input guardrails, prompt-injection phrase rejection, and retrieval-confidence threshold behavior. They do not validate the full live LLM/API/deployment workflow.

## 15. Docker deployment (optional)

The repository Dockerfile uses Python 3.11, installs `requirements.txt`, copies `src`, `app`, and the prebuilt `data/index`, then starts `src.api.main:app` on port `8000`.

Build:

```powershell
docker build -t automotive-ai-engineering-copilot .
```

Run:

```powershell
docker run --rm -p 8000:8000 --env-file .env automotive-ai-engineering-copilot
```

Open http://127.0.0.1:8000/docs.

The Dockerfile starts the API only; it does not start Streamlit. Run the frontend separately or add a dedicated frontend/container orchestration setup. Since the Dockerfile copies the checked-in index, ensure `data/index/index.faiss` and `data/index/metadata.pkl` exist before building.

## 16. Troubleshooting

| Problem | Check |
|---|---|
| `conda` not recognized | Open Anaconda Prompt or repair Conda/PATH |
| Wrong Python interpreter | Activate `automotive-copilot`; run `python --version` and `where python` |
| Missing Python module | Confirm environment is active and rerun `python -m pip install -r requirements.txt` |
| Missing `GROQ_API_KEY` | Check `.env` and `LLM_PROVIDER` |
| No documents processed | Put supported files directly inside `data/documents/` |
| `chunks.jsonl` missing | Run ingestion before chunking |
| FAISS index missing | Run chunking before the FAISS build command |
| Embedding model download fails | Check internet access and model cache |
| UI cannot reach API | Start `src.api.main:app` on port `8000` |
| `/chat` returns 404 from UI | UI expects `/ask`; use `src.api.main:app` |
| Port 8000 already in use | Stop the conflicting process or configure both API and `API_URL` to use the same alternate port |
| Docker cannot find index | Check that both files in `data/index/` exist |

## 17. Quick-start command sequence (Windows)

```powershell
# Clone
git clone https://github.com/Manasajagannadan/Automotive-AI-Engineering-Copilot.git
cd Automotive-AI-Engineering-Copilot

# Conda environment
conda create -n automotive-copilot python=3.11 -y
conda activate automotive-copilot

# Dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check

# Create .env manually and add GROQ_API_KEY
notepad .env

# Ingestion -> chunking -> vector index
python -m src.ingestion.loader --input-dir data/documents --out data/processed
python -m src.chunking.chunker --input data/processed --out data/chunks.jsonl
python -m src.retrieval.faiss_index build --chunks data/chunks.jsonl --index-dir data/index

# Optional retrieval check
python -m src.retrieval.faiss_index query --index-dir data/index --q "engine coolant temperature" --top-k 5

# CLI question answering
python -m src.rag.generator --question "What should I check if coolant temperature is high?" --index-dir data/index

# Retrieval-only evaluation and tests
python -m src.evaluation.evaluate --eval-set src/evaluation/eval_set.json --index-dir data/index --no-llm
python -m pytest -v

# Terminal 1: backend
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2: frontend
streamlit run app/streamlit_app.py
```

In a new terminal, activate the Conda environment again before starting the frontend.

## 18. Engineering assessment and recommended next steps

These are observations from the current source tree:
1. Two FastAPI applications exist; the Streamlit app and Dockerfile use `src.api.main:app`.
2. The root `api/main.py` has additional routes, but the current Streamlit app does not call them.
3. `.env.example` is referenced by the old README but is absent from the current repository tree.
4. Generated `__pycache__` directories and `.pyc` files are tracked; remove them and add `.gitignore` rules.
5. The evaluation logic is keyword-based and the evaluation set is small. Add a larger, human-reviewed evaluation set before making quality claims.
6. The checked-in FAISS index is a generated artifact. Explain whether users should use it or rebuild it from source documents.
7. The VLM path is optional and uses Gemini; it is separate from the text RAG retrieval path.
8. Current guardrails are lightweight checks, not a complete security or safety framework.
9. Hosted model API availability, rate limits, and pricing depend on the provider and account.

Suggested next engineering improvements:
- Add `.env.example` with placeholder values and `.gitignore`.
- Remove tracked Python bytecode.
- Consolidate or clearly document the two API entry points.
- Add API integration tests and end-to-end tests using a small test corpus.
- Expand the evaluation dataset and add human-reviewed answer-quality checks.
- Add structured logging and request IDs to the primary API entry point.
- Add ingestion metadata, duplicate detection, and clearer ingestion error reporting.
- Add CI to run tests and the retrieval-only evaluation gate.

### Current scope and known limitations

- **Retrieval is not model fine-tuning.** The workflow uses a pretrained embedding model to retrieve document chunks; it does not train or fine-tune an LLM.
- **The vector index is corpus-dependent.** Rebuild it whenever source documents or chunking settings change.
- **Similarity is not a probability.** A FAISS similarity score is a retrieval signal, not a calibrated probability that an answer is correct.
- **Evaluation is a baseline.** Keyword-oriented checks are useful as lightweight regression tests, but they do not comprehensively measure factual correctness, groundedness, or safety.
- **Guardrails are basic controls.** Input-pattern checks and a retrieval threshold cannot guarantee protection against every prompt injection, hallucination, or unsafe output.
- **Provider calls require network access.** Groq/Gemini generation depends on API credentials, service availability, rate limits, and provider pricing.
- **There are two API entry points.** Keep the documented Streamlit workflow aligned with `src.api.main:app` unless the applications are intentionally consolidated.

### Recommended engineering roadmap

1. Add `.env.example` and `.gitignore`; remove committed secrets, Python bytecode, caches, and unnecessary generated artifacts.
2. Consolidate the two FastAPI applications or document and test their separate responsibilities.
3. Add API contract tests for `/health` and `/ask`, including invalid input and missing-index scenarios.
4. Add end-to-end tests with a deterministic document corpus and mocked LLM responses.
5. Expand the evaluation set with domain-reviewed questions, expected source passages, and human-reviewed answer assessments.
6. Track metrics such as Recall@K and MRR when ground-truth relevance labels are available.
7. Add structured request logging, request IDs, latency breakdowns, and provider error handling.
8. Add CI to install dependencies, run tests, and execute retrieval-only evaluation.
9. Define document versioning, access control, retention, and data-handling rules before using proprietary automotive documents.

## 19. Responsible-use and limitations

This project is a portfolio and engineering-assistance demonstration, not a validated automotive diagnostic or production safety system. It helps users search a configured collection of technical documents; it does not replace official OEM service procedures, qualified engineering judgment, or safety validation. Verify safety-critical diagnostics, repair instructions, and vehicle specifications against authoritative documentation before acting.
