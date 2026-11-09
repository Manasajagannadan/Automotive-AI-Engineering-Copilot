"""
Lightweight guardrails for the RAG pipeline:
  1. Input guardrail — rejects empty/too-short/non-question input before it burns an
     LLM API call.
  2. Retrieval-confidence guardrail — refuses to generate an answer when the best
     retrieved chunk's similarity score is below threshold, instead of letting the
     LLM improvise from weak or irrelevant context (this is the main lever against
     hallucination in a RAG system — worth being able to explain in an interview).
"""

MIN_QUESTION_LENGTH = 6
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.25  # cosine similarity on normalized MiniLM embeddings

BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "you are now",
    "system prompt",
]


def passes_input_guardrails(question: str) -> bool:
    if not question or len(question.strip()) < MIN_QUESTION_LENGTH:
        return False
    lowered = question.lower()
    if any(pattern in lowered for pattern in BLOCKED_PATTERNS):
        return False
    return True


def retrieval_confidence_ok(max_score: float, threshold: float = RETRIEVAL_CONFIDENCE_THRESHOLD) -> bool:
    return max_score >= threshold
