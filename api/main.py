"""
FastAPI service for the Automotive AI Engineering Copilot.

Run locally:
    uvicorn api.main:app --reload --port 8001

Endpoints:
    GET  /health
    POST /chat     -- {"question": "..."} -> grounded answer + sources
    POST /search   -- {"question": "..."} -> raw retrieved chunks (no LLM call)
    POST /vlm-chat  -- multipart image + question -> VLM answer
"""
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.rag.generator import generate_answer  # noqa: E402
from src.retrieval.faiss_index import search  # noqa: E402
from src.vlm.vlm_client import analyze_image, vlm_available, VLMNotConfigured  # noqa: E402
from observability.logger import log_query  # noqa: E402

INDEX_DIR = Path(os.environ.get("INDEX_DIR", "data/index"))

app = FastAPI(title="Automotive AI Engineering Copilot", version="0.1.0")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health():
    index_exists = (INDEX_DIR / "index.faiss").exists()
    return {"status": "ok", "index_built": index_exists, "vlm_available": vlm_available()}


@app.post("/chat")
def chat(req: ChatRequest):
    result = generate_answer(req.question, INDEX_DIR, top_k=req.top_k)
    log_query(req.question, result)
    return result


@app.post("/search")
def search_endpoint(req: ChatRequest):
    results = search(INDEX_DIR, req.question, top_k=req.top_k)
    return {"results": results}


@app.post("/vlm-chat")
async def vlm_chat(question: str = Form(...), file: UploadFile = File(...)):
    if not vlm_available():
        return {
            "answer": None,
            "vlm_available": False,
            "message": "VLM mode isn't configured (optional module — the core RAG "
                       "chat works without it). See src/vlm/vlm_client.py for setup notes.",
        }
    image_bytes = await file.read()
    try:
        answer = analyze_image(image_bytes, question)
    except VLMNotConfigured as e:
        return {"answer": None, "vlm_available": False, "message": str(e)}
    return {"answer": answer, "vlm_available": True}
