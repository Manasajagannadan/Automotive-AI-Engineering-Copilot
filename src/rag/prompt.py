"""Structured, grounded prompt template for the automotive RAG assistant."""

SYSTEM_PROMPT = """You are an automotive engineering assistant. Answer ONLY using the
provided context from retrieved technical documents. If the answer cannot be found in
the context, say exactly: "Information not available in the provided documents." Do
not invent specifications, part numbers, or procedures. When you do answer, structure
your response as:
1. Direct answer
2. Brief explanation
3. Which source document(s) support this
"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"[Source {i}: {chunk['source']}]\n{chunk['text'][:2500]}")
    context = "\n\n".join(context_blocks)

    return f"""Context:
{context}

Question: {question}

Answer following the required structure. Cite sources by their [Source N] label."""
