"""
Simple structured query logger: what was asked, what was retrieved, what was
answered, confidence, latency. Writes append-only JSON lines — enough to build a
"what is this system actually doing in production" story without needing a full
observability platform.
"""
import json
import time
from pathlib import Path

LOG_PATH = Path("observability/query_log.jsonl")


def log_query(question: str, result: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "question": question,
        "sources": result.get("sources", []),
        "retrieval_confidence": result.get("retrieval_confidence"),
        "latency_ms": result.get("latency_ms"),
        "blocked_by_guardrail": result.get("blocked_by_guardrail", False),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_recent_logs(n: int = 50) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-n:]]
