"""
Week 6, Phase 4, step 1: generate real answers for every eval case and save
them to judge_sample.json with an empty "human_verdict" field per entry.

Run this once, then open evals/judge_sample.json and fill in
"human_verdict": "GOOD" or "POOR" for every entry (an optional
"human_reason" helps you remember why later, and shows up if you disagree
with the judge). Then run validate_judge.py.

This step has to be a human (that's the whole point of Track A's
deliverable) -- this script only prepares the sample, it does not grade it.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# See validate_judge.py's comment on this same line.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import DEFAULT_TOP_K
from eval_questions import EVAL_QUESTIONS, REGRESSION_CASES
from generate import answer_question
from ingestion import get_client, get_embedding_model
from retrieval import retrieve

SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "judge_sample.json")


def main():
    model = get_embedding_model()
    client = get_client()

    all_cases = EVAL_QUESTIONS + REGRESSION_CASES
    sample = []
    for case in all_cases:
        # Retrieved separately from answer_question() (same pattern
        # classify_failures.py already uses) so the judge can be given the
        # actual context the answer should be grounded in -- without it,
        # the judge can only guess whether a refusal was warranted or a
        # claim was accurate, which is exactly what caused it to give an
        # unstable, factually wrong verdict during validation.
        chunks = retrieve(case["question"], top_k=DEFAULT_TOP_K, model=model, client=client)
        context = "\n\n".join(c["text"] for c in chunks) if chunks else "(no context retrieved)"

        result = answer_question(case["question"], top_k=DEFAULT_TOP_K, model=model, client=client)
        sample.append({
            "id": case["id"],
            "question": case["question"],
            "answer": result["answer"],
            "context": context,
            "skipped_llm": result["skipped_llm"],
            "human_verdict": None,   # <-- fill in "GOOD" or "POOR" for each entry
            "human_reason": "",      # optional, helps you remember why later
        })
        print(f"[{case['id']}] done")

    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(sample)} answers to {SAMPLE_FILE}")
    print('Open that file, fill in "human_verdict": "GOOD" or "POOR" for each entry,')
    print("then run: python evals/validate_judge.py")


if __name__ == "__main__":
    main()
