"""
Wraps a small, CPU-friendly sentence-transformers model for embedding chunks and
queries. all-MiniLM-L6-v2 is ~80MB and fast on CPU — no GPU needed for this step,
which matters on a 4GB-VRAM machine since the GPU should be reserved for vision
model training in Project 1, not tied up embedding text.
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
