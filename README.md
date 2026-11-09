# Automotive AI Engineering Copilot — RAG + LLM + VLM

A retrieval-augmented assistant over real automotive technical documents (NHTSA
Technical Service Bulletins + public OEM owner's manuals) — with an evaluation
harness, guardrails, observability logging, and CI eval gates. Free-tier deployable.

Because a 4GB-VRAM laptop can't usefully run a local LLM/VLM, generation calls a
**free hosted API** (Groq by default; swap in Google AI Studio/Gemini if you prefer)
instead of self-hosting a model — retrieval, evaluation, guardrails and orchestration
are all still yours, running locally on CPU.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get an LLM API key (free — only one key required)

- **Groq — required, and this is genuinely free with no credit card:**
  https://console.groq.com/keys. This is the only key the core system needs.
- **Google AI Studio (Gemini) — optional, only for the image-analysis tab.**
  Skip this entirely if you want; the RAG chat works fully without it. If you do
  want it: https://aistudio.google.com/apikey — but since March 2026, some regions
  prompt new AI Studio accounts to set up prepaid billing before issuing a key.
  If you hit that, don't fight it — it's not worth it for an optional module.

```bash
cp .env.example .env
# then edit .env and paste GROQ_API_KEY (leave GOOGLE_API_KEY blank if you're skipping VLM)
```

## 3. Build your document corpus

Don't use a single toy PDF. Recommended real corpus:
1. Download 10-15 NHTSA Technical Service Bulletins (free, public):
   https://www.nhtsa.gov/nhtsa-datasets-and-apis (or search "site:nhtsa.gov TSB pdf")
2. Download 5-10 public owner's manuals from manufacturer sites (most publish these
   as free PDFs — search "<make> <model> owner's manual pdf").
3. Place everything in `data/documents/` (PDF, DOCX, or TXT).

## 4. Ingest, chunk, embed, index

```bash
python -m src.ingestion.loader --input-dir data/documents --out data/processed
python -m src.chunking.chunker --input data/processed --out data/chunks.jsonl
python -m src.retrieval.faiss_index build --chunks data/chunks.jsonl --index-dir data/index
```

## 5. Ask a question from the CLI

```bash
python -m src.rag.generator --question "What should I check if coolant temperature is high?" --index-dir data/index
```

## 6. Run the evaluation harness

```bash
python -m src.evaluation.evaluate --eval-set src/evaluation/eval_set.json --index-dir data/index
```
This scores retrieval accuracy + answer faithfulness against a curated Q&A set and
is the script CI runs to block a deploy if quality regresses — see `--min-score`.

## 7. Serve it

```bash
uvicorn api.main:app --reload --port 8001
```
Then:
```bash
streamlit run app/streamlit_app.py
```

## 8. Docker

```bash
docker build -t automotive-copilot .
docker run -p 8001:8001 --env-file .env automotive-copilot
```

## 9. Deploy for free

- **API:** Render free web service (Docker deploy), with `GROQ_API_KEY` set as an
  environment variable in Render's dashboard (never commit it to the repo).
- **UI:** Streamlit Community Cloud, `API_URL` pointing at your Render URL.

## Honest scope note

This is engineered as a genuine production RAG pipeline (retrieval, evaluation,
guardrails, observability, CI gates) — but it's rate-limited by whichever free-tier
LLM API you use, so it's scoped for low-QPS engineering-team use, not
high-throughput production traffic. Say that plainly rather than let someone assume
otherwise.
