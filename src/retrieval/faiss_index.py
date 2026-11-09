"""
Builds and queries a FAISS index over chunk embeddings.

Usage:
    python -m src.retrieval.faiss_index build --chunks data/chunks.jsonl --index-dir data/index
    (searching is used programmatically by src/rag/generator.py — see load_index/search below)
"""
import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from src.embeddings.embedder import embed_texts, embed_query


def build_index(chunks_path: Path, index_dir: Path):
    chunks = [json.loads(line) for line in open(chunks_path)]
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path} — run the chunker first.")

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks (CPU, MiniLM)...")
    embeddings = embed_texts(texts).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Wrote FAISS index ({index.ntotal} vectors, dim={dim}) to {index_dir}")


def load_index(index_dir: Path):
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search(index_dir: Path, query: str, top_k: int = 5) -> list[dict]:
    index, metadata = load_index(index_dir)
    q_emb = embed_query(query).astype("float32").reshape(1, -1)
    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx]
        results.append({**chunk, "score": float(score)})
    return results


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--chunks", required=True)
    build.add_argument("--index-dir", required=True)

    query_cmd = sub.add_parser("query")
    query_cmd.add_argument("--index-dir", required=True)
    query_cmd.add_argument("--q", required=True)
    query_cmd.add_argument("--top-k", type=int, default=5)

    args = ap.parse_args()

    if args.command == "build":
        build_index(Path(args.chunks), Path(args.index_dir))
    elif args.command == "query":
        results = search(Path(args.index_dir), args.q, args.top_k)
        for r in results:
            print(f"[{r['score']:.3f}] {r['source']} :: {r['text'][:150]}...")


if __name__ == "__main__":
    main()
