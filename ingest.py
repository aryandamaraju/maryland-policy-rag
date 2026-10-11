import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader

DOCS = Path("data/docs")
INDEX = Path("index")
CHUNK_CHARS = 1000
OVERLAP = 200


def chunk_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_CHARS])
        start += CHUNK_CHARS - OVERLAP
    return chunks


def main():
    INDEX.mkdir(exist_ok=True)
    pdfs = sorted(DOCS.glob("*.pdf"))
    records = []
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        for page_num, page in enumerate(reader.pages, start=1):
            for chunk in chunk_text(page.extract_text() or ""):
                if len(chunk) > 50:
                    records.append({"file": pdf.name, "page": page_num, "text": chunk})
    if not records:
        raise SystemExit("No text found. Put PDFs in data/docs first.")
    print(f"{len(records)} chunks from {len(pdfs)} PDFs")

    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    vectors = np.array(list(model.embed([r["text"] for r in records])), dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    np.save(INDEX / "vectors.npy", vectors)
    (INDEX / "chunks.json").write_text(json.dumps(records), encoding="utf-8")
    print("Index saved.")


if __name__ == "__main__":
    main()