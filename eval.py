import json
import os
import sys

from rag import CHUNKS, REFUSAL, answer, retrieve

K = 5
MODE = sys.argv[1] if len(sys.argv) > 1 else None
if MODE:  # answer() reads the mode from the environment, so the refusal checks use it too
    os.environ["RETRIEVAL_MODE"] = MODE
RETRIEVAL_ONLY = os.getenv("EVAL_RETRIEVAL_ONLY") == "1"
MIN_HIT_RATE = float(os.getenv("MIN_HIT_RATE", "0"))

with open("eval/questions.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
answerable = [r for r in rows if r.get("gold_file")]
unanswerable = [r for r in rows if not r.get("gold_file")]

valid = passed = 0
for r in answerable:
    phrase = r["gold_phrase"].lower()
    exists = any(c["file"] == r["gold_file"] and phrase in c["text"].lower() for c in CHUNKS)
    if not exists:
        print("BAD LABEL (phrase not found in that file):", r["question"])
        continue
    valid += 1
    hits = retrieve(r["question"], K, MODE)
    ok = any(h["file"] == r["gold_file"] and phrase in h["text"].lower() for h in hits)
    passed += ok
    print("PASS" if ok else "FAIL", "-", r["question"])

rate = passed / valid if valid else 0.0
if valid:
    print(f"\nRetrieval hit rate@{K}: {passed}/{valid} = {rate:.0%}")

if RETRIEVAL_ONLY:
    if rate < MIN_HIT_RATE:
        print(f"GATE FAILED: hit rate {rate:.0%} is below {MIN_HIT_RATE:.0%}")
        sys.exit(1)
    sys.exit(0)

refused = 0
for r in unanswerable:
    text, _ = answer(r["question"], K)
    ok = REFUSAL.lower() in text.lower()
    refused += ok
    print("PASS" if ok else "FAIL", "- [unanswerable]", r["question"])
if unanswerable:
    print(f"Correct refusals: {refused}/{len(unanswerable)}")
