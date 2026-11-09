"""
Splits cleaned document text into overlapping chunks, tagged with their source
document — this is what makes citations possible downstream.

Chunk size / overlap are deliberately exposed as CLI args, not hardcoded, because
"why this chunk size?" is one of the most common RAG interview questions — you
should be able to point at an experiment, not a guess. See src/evaluation/evaluate.py
for a script to compare chunk-size choices against retrieval accuracy.

Usage:
    python -m src.chunking.chunker --input data/processed --out data/chunks.jsonl \
        --chunk-size 800 --overlap 150
"""
import argparse
import json
from pathlib import Path


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Word-based sliding-window chunking (simple, explainable, no extra dependency)."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def process_directory(input_dir: Path, chunk_size: int, overlap: int) -> list[dict]:
    all_chunks = []
    for path in sorted(input_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{path.stem}::chunk{i}",
                "source": path.name,
                "chunk_index": i,
                "text": chunk,
            })
        print(f"[ok] {path.name} -> {len(chunks)} chunks")
    return all_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory of cleaned .txt files")
    ap.add_argument("--out", required=True, help="Output .jsonl path")
    ap.add_argument("--chunk-size", type=int, default=800, help="Words per chunk")
    ap.add_argument("--overlap", type=int, default=150, help="Word overlap between chunks")
    args = ap.parse_args()

    chunks = process_directory(Path(args.input), args.chunk_size, args.overlap)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"\nWrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
