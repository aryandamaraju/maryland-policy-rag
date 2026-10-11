import sys

from rag import answer

text, hits = answer(" ".join(sys.argv[1:]))
print(text)
print("\nSources:")
for h in hits:
    print(f"- {h['file']} p.{h['page']} (score {h['score']:.2f})")