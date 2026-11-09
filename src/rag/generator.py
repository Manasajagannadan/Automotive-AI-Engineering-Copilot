"""
Orchestrates: question -> FAISS retrieval -> prompt construction -> free-tier LLM
API call (Groq by default) -> grounded answer + cited sources.

Usage (CLI):
    python -m src.rag.generator --question "What causes engine overheating?" --index-dir data/index
"""
import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from src.retrieval.faiss_index import search
from src.rag.prompt import SYSTEM_PROMPT, build_prompt
from src.guardrails.guardrails import passes_input_guardrails, retrieval_confidence_ok

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")


def call_groq(system_prompt: str, user_prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low temperature — this is a grounded-answer task, not creative writing
    )
    return resp.choices[0].message.content


def call_gemini(system_prompt: str, user_prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
    resp = model.generate_content(user_prompt)
    return resp.text


def generate_answer(question: str, index_dir: Path, top_k: int = 2) -> dict:
    start = time.time()

    if not passes_input_guardrails(question):
        return {
            "answer": "This question could not be processed — please rephrase it as a "
                      "specific automotive engineering question.",
            "sources": [],
            "retrieval_confidence": 0.0,
            "latency_ms": round((time.time() - start) * 1000, 1),
            "blocked_by_guardrail": True,
        }

    retrieved = search(index_dir, question, top_k=top_k)
    max_score = max((r["score"] for r in retrieved), default=0.0)

    if not retrieval_confidence_ok(max_score):
        return {
            "answer": "Information not available in the provided documents "
                      "(no sufficiently relevant content was retrieved).",
            "sources": [],
            "retrieval_confidence": round(max_score, 3),
            "latency_ms": round((time.time() - start) * 1000, 1),
            "blocked_by_guardrail": False,
        }

    prompt = build_prompt(question, retrieved)

    if LLM_PROVIDER == "groq":
        answer = call_groq(SYSTEM_PROMPT, prompt)
    elif LLM_PROVIDER == "gemini":
        answer = call_gemini(SYSTEM_PROMPT, prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    sources = sorted({r["source"] for r in retrieved})

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_confidence": round(max_score, 3),
        "latency_ms": round((time.time() - start) * 1000, 1),
        "blocked_by_guardrail": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    result = generate_answer(args.question, Path(args.index_dir), args.top_k)
    print("\nAnswer:\n", result["answer"])
    print("\nSources:", result["sources"])
    print(f"Retrieval confidence: {result['retrieval_confidence']}")
    print(f"Latency: {result['latency_ms']} ms")


if __name__ == "__main__":
    main()
