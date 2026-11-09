"""
Loads PDF / DOCX / TXT documents from a directory, extracts text, and writes one
cleaned .txt file per source document (keeping source filenames for citation later).

Usage:
    python -m src.ingestion.loader --input-dir data/documents --out data/processed
"""
import argparse
import re
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument


def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_txt(path: Path) -> str:
    return path.read_text(errors="ignore")


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


EXTRACTORS = {".pdf": extract_pdf, ".docx": extract_docx, ".txt": extract_txt}


def process_directory(input_dir: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = []
    for path in sorted(input_dir.iterdir()):
        ext = path.suffix.lower()
        if ext not in EXTRACTORS:
            continue
        try:
            raw = EXTRACTORS[ext](path)
        except Exception as e:
            print(f"[skip] failed to extract {path.name}: {e}")
            continue

        cleaned = clean_text(raw)
        if not cleaned:
            print(f"[skip] no extractable text in {path.name}")
            continue

        out_path = out_dir / (path.stem + ".txt")
        out_path.write_text(cleaned, encoding="utf-8")
        processed.append(path.name)
        print(f"[ok] {path.name} -> {out_path.name} ({len(cleaned)} chars)")

    return processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    processed = process_directory(Path(args.input_dir), Path(args.out))
    print(f"\nProcessed {len(processed)} documents.")
    if not processed:
        print("No documents found — place PDFs/DOCX/TXT files in the input directory first.")


if __name__ == "__main__":
    main()
