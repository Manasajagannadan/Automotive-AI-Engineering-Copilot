from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.generator import generate_answer


app = FastAPI(
    title="Automotive AI Engineering Copilot",
    description="RAG + LLM API for automotive technical documents",
    version="1.0.0",
)


INDEX_DIR = Path("data/index")


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "automotive-rag-copilot",
    }


@app.post("/ask")
def ask(request: QuestionRequest):
    result = generate_answer(
        request.question,
        INDEX_DIR,
        top_k=2,
    )

    return result