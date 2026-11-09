"""
Evaluation harness for the RAG pipeline. Scores:
  - Retrieval accuracy: did retrieval surface a chunk whose source/content matches
    the expected topic keywords for that question?
  - Answer keyword coverage: does the generated answer mention the expected concepts?
  - Retrieval confidence: the raw similarity score of the top retrieved chunk.

This is a lightweight, dependency-free proxy for "did the system actually work,"
not a substitute for human-graded faithfulness scoring — but it's real, automatable,
and exactly the kind of check that should run in CI before a deploy is allowed.

IMPORTANT: src/evaluation/eval_set.json ships with only 3 template questions. Before
this is a meaningful gate, expand it to 40-50 real questions written against YOUR
actual document corpus — that's what makes the "90%+ query resolution" claim
honest instead of aspirational.

Usage:
    python -m src.evaluation.evaluate --eval-set src/evaluation/eval_set.json \
        --index-dir data/index --min-score 0.7
"""
import argparse
import json
import sys
from pathlib import Path

from src.retrieval.faiss_index import search
from src.rag.generator import generate_answer


def score_retrieval(question_item: dict, retrieved: list[dict]) -> bool:
    keywords = [k.lower() for k in question_item["expected_source_keywords"]]
    for chunk in retrieved:
        haystack = (chunk["source"] + " " + chunk["text"]).lower()
        if any(k in haystack for k in keywords):
            return True
    return False


def score_answer(question_item: dict, answer: str) -> bool:
    keywords = [k.lower() for k in question_item["expected_answer_keywords"]]
    lowered = answer.lower()
    hits = sum(1 for k in keywords if k in lowered)
    return hits >= max(1, len(keywords) // 2)  # at least half the expected keywords present


def run_evaluation(eval_set_path: Path, index_dir: Path, call_llm: bool = True) -> dict:
    eval_items = json.loads(eval_set_path.read_text())

    retrieval_hits = 0
    answer_hits = 0
    confidences = []
    per_question_results = []

    for item in eval_items:
        question = item["question"]
        retrieved = search(index_dir, question, top_k=2)
        top_score = max((r["score"] for r in retrieved), default=0.0)
        confidences.append(top_score)

        retrieval_ok = score_retrieval(item, retrieved)
        retrieval_hits += int(retrieval_ok)

        answer_ok = None
        if call_llm:
            result = generate_answer(question, index_dir, top_k=2)
            answer_ok = score_answer(item, result["answer"])
            answer_hits += int(bool(answer_ok))

        per_question_results.append({
            "question": question,
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
            "top_score": round(top_score, 3),
        })

    n = len(eval_items)
    summary = {
        "n_questions": n,
        "retrieval_accuracy": round(retrieval_hits / n, 3) if n else 0.0,
        "answer_keyword_coverage": round(answer_hits / n, 3) if (call_llm and n) else None,
        "avg_retrieval_confidence": round(sum(confidences) / n, 3) if n else 0.0,
        "per_question": per_question_results,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--min-score", type=float, default=0.7, help="Minimum retrieval accuracy to pass the gate")
    ap.add_argument("--no-llm", action="store_true", help="Skip LLM calls, score retrieval only (faster, no API key needed)")
    args = ap.parse_args()

    summary = run_evaluation(Path(args.eval_set), Path(args.index_dir), call_llm=not args.no_llm)

    print(json.dumps(summary, indent=2))
    print(f"\nRetrieval accuracy: {summary['retrieval_accuracy']} (gate threshold: {args.min_score})")

    if summary["retrieval_accuracy"] < args.min_score:
        print("EVAL GATE FAILED — deployment blocked.")
        sys.exit(1)

    print("Eval gate passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
