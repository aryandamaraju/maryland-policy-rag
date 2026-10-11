import json
import random
import sys

with open("index/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for c in random.sample(chunks, n):
    print(f"--- {c['file']} p.{c['page']}")
    print(c["text"])
    print()