from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import rag

app = FastAPI(title="Regulatory Document Q&A")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(rag.CHUNKS), "llm": f"{rag.PROVIDER}:{rag.MODEL}"}


@app.post("/ask")
def ask(q: Question):
    question = q.question.strip()[:500]
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    try:
        text, hits = rag.answer(question, 5)
    except Exception as e:  # generation can fail; still show retrieved sources
        hits = rag.retrieve(question, 5)
        text = None
        err = f"{type(e).__name__}: {str(e)[:160]}"
        print("generation error:", err)
    return {
        "error": locals().get("err"),
        "answer": text,
        "refused": bool(text) and rag.REFUSAL.lower() in text.lower(),
        "sources": [
            {"n": i + 1, "file": h["file"], "page": h["page"], "score": round(h["score"], 3), "snippet": h["text"][:400]}
            for i, h in enumerate(hits)
        ],
    }


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")
