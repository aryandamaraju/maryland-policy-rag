import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()
INDEX = Path("index")
PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
REFUSAL = "I can't find that in the provided documents."

_embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
_vectors = np.load(INDEX / "vectors.npy")
CHUNKS = json.loads((INDEX / "chunks.json").read_text(encoding="utf-8"))


def retrieve(question, k=5):
    q = np.array(list(_embedder.embed([question]))[0], dtype="float32")
    q /= np.linalg.norm(q)
    scores = _vectors @ q
    top = np.argsort(-scores)[:k]
    return [{**CHUNKS[i], "score": float(scores[i])} for i in top]


def generate(system, user):
    if PROVIDER == "anthropic":
        import anthropic

        msg = anthropic.Anthropic().messages.create(
            model=MODEL,
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
    import ollama

    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options={"temperature": 0},
    )
    return resp["message"]["content"]


def answer(question, k=5):
    hits = retrieve(question, k)
    context = "\n\n".join(
        f"[{i + 1}] ({h['file']}, p.{h['page']})\n{h['text']}" for i, h in enumerate(hits)
    )
    system = (
        "Answer using ONLY the numbered excerpts. Cite sources like [1]. "
        f"If the excerpts do not contain the answer, reply exactly: {REFUSAL}"
    )
    user = f"Excerpts:\n{context}\n\nQuestion: {question}"
    return generate(system, user), hits