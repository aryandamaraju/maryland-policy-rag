---
title: Regulatory Document QA
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Maryland SNAP Policy RAG

A question-answering system over eight public chapters of the Maryland SNAP manual. It retrieves relevant passages, answers using only those passages with citations, and refuses when the documents don't contain the answer. It includes a small evaluation harness.

Not benefits or legal advice. Source: [Maryland DHS SNAP manual](https://dhs.maryland.gov/supplemental-nutrition-assistance-program/food-supplement-program-manual/).

## How it works

1. `ingest.py` extracts text from the PDFs (pypdf), splits each page into 1000-character chunks with 200 overlap, embeds them with BAAI/bge-small-en-v1.5 (fastembed), and saves the index with numpy.
2. `rag.py` embeds the question, takes the top 5 chunks by cosine similarity, and prompts an LLM to answer only from those excerpts with [n] citations, or reply with a fixed refusal sentence. The LLM runs locally through Ollama (qwen2.5:7b). Setting `LLM_PROVIDER=anthropic` switches to the Anthropic API.
3. `eval.py` scores retrieval and refusals on `eval/questions.jsonl`.

## Setup

Requires Python 3.12. Ollama is needed only for the local LLM.

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements-dev.txt
    ollama pull qwen2.5:7b

`requirements.txt` lists only what the web app needs at runtime (it is what Vercel installs). `requirements-dev.txt` adds ingestion (pypdf) and the Ollama and Anthropic clients.

Create a `.env` file:

    LLM_PROVIDER=ollama
    LLM_MODEL=qwen2.5:7b

To use an OpenAI-compatible API such as Groq instead, set `LLM_PROVIDER=openai_compat`, `LLM_MODEL` (for example `llama-3.1-8b-instant`) and `LLM_API_KEY`. `LLM_BASE_URL` defaults to Groq's endpoint.

Download the PDF versions of chapters 100, 115, 120, 130, 200, 211, 408 and 600 from the manual page into `data/docs` as `ch100_household.pdf`, `ch115_categorical.pdf`, `ch120_citizenship.pdf`, `ch130_work.pdf`, `ch200_resources.pdf`, `ch211_excluded_income.pdf`, `ch408_verification.pdf`, `ch600_standards.pdf`. Then:

    python ingest.py
    python ask.py "your question"
    python eval.py
    uvicorn app:app --port 8000

## Evaluation

48 hand-written questions covering all eight chapters: 40 answerable (each labeled with a source file and a phrase that must appear in a retrieved passage) and 8 unanswerable. Run with `python eval.py dense|hybrid|rerank`.

| Retrieval mode | Retrieval hit rate@5 | Correct refusals |
|---|---|---|
| Dense: bge-small cosine top-5 | 34/40 (85%) | 8/8 |
| Hybrid: dense + BM25, reciprocal rank fusion | 38/40 (95%) | 8/8 |
| Rerank: hybrid top-20, rescored by ms-marco-MiniLM-L-6-v2 cross-encoder | 40/40 (100%) | 8/8 |

All runs: 1000-char chunks with 200 overlap, qwen2.5:7b via Ollama for the refusal checks.

Caveats:
- Small set. Most questions were written while reading the passages, so their wording overlaps the source. Expect lower scores on harder questions.
- Hit rate checks retrieval only, not whether the final answer is correct.
- Refusals are judged by exact match on the refusal sentence.

## What failed and why

Dense retrieval missed 6 questions. One example: "Do resources matter when deciding eligibility for a categorically eligible household?" The passage that states the rule (ch200 p.1) was not in the top 5. Similarity scores were bunched between 0.80 and 0.83 across many passages about categorical eligibility, and the embedding did not separate the one that states the rule. Dense also missed exact-figure lookups such as the maximum allotment for a household of three.

Hybrid fixes most of these, because BM25 matches exact terms and numbers. It still misses two paraphrased questions where the question shares few words with the source ("deduct all of its housing costs" vs. "uncapped shelter deduction"; "buys food separately" for the separate-purchase income limit). Reranking the top 20 hybrid candidates with a cross-encoder recovers both.

## Next

LLM-graded answer correctness and hallucination rate, and a harder question set written without looking at the source passages.