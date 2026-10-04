"""
Week 8: how often does injection.py flag a LEGITIMATE document?

Runs the detector over every chunk of the real data/ corpus (the 7 tickets and
the PDF) -- none of which is an attack -- and reports every hit. A detector
that is never measured against clean text is an unmeasured defence: the number
here is the false-positive rate, and it is the reason the posture is "degrade,
never refuse".

No LLM, no network, no embedding model.

Usage:
    python evals/injection_false_positives.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import injection
from ingestion import load_and_chunk_all


def main():
    chunks = load_and_chunk_all()
    flagged = []
    for chunk in chunks:
        detections = injection.scan(chunk["text"])
        if detections:
            flagged.append((chunk, detections))

    print(f"Scanned {len(chunks)} real chunks from {len({c['source'] for c in chunks})} documents.")
    print(f"Flagged: {len(flagged)} ({100 * len(flagged) / max(len(chunks), 1):.1f}%)\n")
    for chunk, detections in flagged:
        print(f"[{chunk['source']}] chunk {chunk['chunk_index']} page {chunk['page']}")
        for d in detections:
            print(f"    {d.describe()}")
    if not flagged:
        print("No false positives on the real corpus.")
    return len(flagged)


if __name__ == "__main__":
    main()
