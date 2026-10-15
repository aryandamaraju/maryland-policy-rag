import json
import os
from pathlib import Path

# Serverless hosts (Vercel) only allow writes under /tmp, so keep every model cache there.
if os.getenv("VERCEL"):
    os.environ.setdefault("HF_HOME", "/tmp/hf")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/fastembed")

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()
_HERE = Path(__file__).resolve().parent
INDEX = _HERE / "index"
if "FASTEMBED_CACHE_PATH" not in os.environ and (_HERE / "models").exists():
    os.environ["FASTEMBED_CACHE_PATH"] = str(_HERE / "models")  # bundled model weights
PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
REFUSAL = "I can't find that in the provided documents."

_embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
_vectors = np.load(INDEX / "vectors.npy")
CHUNKS = json.loads((INDEX / "chunks.json").read_text(encoding="utf-8"))


import math
import re
from collections import Counter

_tok = lambda t: re.findall(r"[a-z0-9$%.]+", t.lower())
_docs = [_tok(c["text"]) for c in CHUNKS]
_df = Counter(w for d in _docs for w in set(d))
_avg = sum(len(d) for d in _docs) / len(_docs)
_tf = [Counter(d) for d in _docs]
_reranker = None


def _bm25(question, k1=1.5, b=0.75):
    q = _tok(question)
    n = len(_docs)
    out = np.zeros(n, dtype="float32")
    for w in set(q):
        if w not in _df:
            continue
        idf = math.log(1 + (n - _df[w] + 0.5) / (_df[w] + 0.5))
        for i, tf in enumerate(_tf):
            f = tf.get(w, 0)
            if f:
                out[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(_docs[i]) / _avg))
    return out


def _dense(question):
    q = np.array(list(_embedder.embed([question]))[0], dtype="float32")
    q /= np.linalg.norm(q)
    return _vectors @ q


def retrieve(question, k=5, mode=None):
    """mode: dense | hybrid | rerank (default from RETRIEVAL_MODE env, else dense)."""
    mode = mode or os.getenv("RETRIEVAL_MODE", "dense")
    dense = _dense(question)
    if mode == "dense":
        top = np.argsort(-dense)[:k]
        return [{**CHUNKS[i], "score": float(dense[i])} for i in top]
    # hybrid: reciprocal rank fusion of dense and BM25 rankings
    bm = _bm25(question)
    rrf = np.zeros(len(CHUNKS), dtype="float32")
    for scores in (dense, bm):
        for rank, i in enumerate(np.argsort(-scores)[:50]):
            rrf[i] += 1.0 / (60 + rank)
    if mode == "hybrid":
        top = np.argsort(-rrf)[:k]
        return [{**CHUNKS[i], "score": float(rrf[i])} for i in top]
    # rerank: cross-encoder rescoring of the top 20 hybrid candidates
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _reranker = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    cand = list(np.argsort(-rrf)[:20])
    rs = list(_reranker.rerank(question, [CHUNKS[i]["text"] for i in cand]))
    order = np.argsort(-np.array(rs))[:k]
    return [{**CHUNKS[cand[j]], "score": float(rs[j])} for j in order]


_pipe = None


def _gen_transformers(system, user):
    global _pipe
    if _pipe is None:
        from transformers import pipeline

        _pipe = pipeline("text-generation", model=MODEL)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out = _pipe(msgs, max_new_tokens=300, do_sample=False)
    return out[0]["generated_text"][-1]["content"]


def _payload(system, user):
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 1500,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if "gpt-oss" in MODEL:  # reasoning models: keep thinking short so the answer fits
        body["reasoning_effort"] = "low"
    return body


def _gen_openai_compat(system, user):
    """Any OpenAI-compatible chat API (Groq, etc.). Key comes from the LLM_API_KEY env var."""
    import json as _json
    import urllib.request

    req = urllib.request.Request(
        os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1") + "/chat/completions",
        data=_json.dumps(_payload(system, user)).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "snap-rag/1.0 (+https://github.com/aryandamaraju/maryland-policy-rag)",
            "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return _json.loads(r.read())["choices"][0]["message"]["content"]


def generate(system, user):
    if PROVIDER == "openai_compat":
        return _gen_openai_compat(system, user)
    if PROVIDER == "transformers":
        return _gen_transformers(system, user)
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