# Maryland SNAP Policy RAG

A question-answering system over eight public chapters of the Maryland SNAP manual. It retrieves relevant passages, answers using only those passages with citations, and refuses when the documents don't contain the answer. It includes a small evaluation harness.

Not benefits or legal advice. Source: [Maryland DHS SNAP manual](https://dhs.maryland.gov/supplemental-nutrition-assistance-program/food-supplement-program-manual/).

## How it works

1. `ingest.py` extracts text from the PDFs (pypdf), splits each page into 1000-character chunks with 200 overlap, embeds them with BAAI/bge-small-en-v1.5 (fastembed), and saves the index with numpy.
2. `rag.py` embeds the question, takes the top 5 chunks by cosine similarity, and prompts an LLM to answer only from those excerpts with [n] citations, or reply with a fixed refusal sentence. The LLM runs locally through Ollama (qwen2.5:7b). Setting `LLM_PROVIDER=anthropic` switches to the Anthropic API.
3. `eval.py` scores retrieval and refusals on `eval/questions.jsonl`.

## Setup

Requires Python 3.12 and Ollama.

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ollama pull qwen2.5:7b

Create a `.env` file:

    LLM_PROVIDER=ollama
    LLM_MODEL=qwen2.5:7b

Download the PDF versions of chapters 100, 115, 120, 130, 200, 211, 408 and 600 from the manual page into `data/docs` as `ch100_household.pdf`, `ch115_categorical.pdf`, `ch120_citizenship.pdf`, `ch130_work.pdf`, `ch200_resources.pdf`, `ch211_excluded_income.pdf`, `ch408_verification.pdf`, `ch600_standards.pdf`. Then:

    python ingest.py
    python ask.py "your question"
    python eval.py

## Evaluation

15 hand-written questions: 12 answerable (each labeled with a source file and a phrase that must appear in a retrieved passage) and 3 unanswerable.

| Variant | Retrieval hit rate@5 | Correct refusals |
|---|---|---|
| Baseline: 1000-char chunks, bge-small, cosine top-5, qwen2.5:7b | 11/12 (92%) | 3/3 |

Caveats:
- Small set. Questions were written while reading the passages, so their wording overlaps the source. Expect lower scores on harder questions.
- Hit rate checks retrieval only, not whether the final answer is correct.
- Questions cover chapters 115, 120, 130, 200 and 408. Chapters 100, 211 and 600 have none yet.

## What failed and why

One miss: "Do resources matter when deciding eligibility for a categorically eligible household?" The passage that states the rule (ch200 p.1) was not in the top 5. The top 5 were ch200 p.8, ch115 p.6 (twice), ch408 p.12 and ch200 p.4, with similarity scores bunched between 0.80 and 0.83. Many passages discuss categorical eligibility, and the embedding does not separate the one that states the rule. The model still answered "No", which is correct, but its stated reason ("not verified") is not what the rule says.

## Next

Hybrid (BM25 plus embeddings) search, reranking, LLM-graded answer correctness and hallucination rate, a harder question set, a FastAPI service, Docker, a CI eval gate, and deployment.