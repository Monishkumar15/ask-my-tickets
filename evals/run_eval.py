"""
Week 6: the one-command test runner. Runs every eval case (Week 3/4's
original 15 + Week 6's regression cases from evals/eval_questions.py)
through the REAL persisted pipeline (deliberately not an in-memory copy --
this tests what a user actually gets, not a clean-room reproduction),
scores each with assertions.py's free rule-based checks, and prints one
scorecard broken down by problem_type.

Usage:
    python evals/run_eval.py                    # run + print a scorecard
    python evals/run_eval.py --save baseline     # also save a named snapshot
    python evals/run_eval.py --compare baseline  # diff this run against a snapshot
    python evals/run_eval.py --with-judge        # also run the LLM judge (1 extra call/case --
                                                  #  only meaningful once validate_judge.py has
                                                  #  confirmed it agrees with human grading)
    python evals/run_eval.py --with-ragas        # also run faithfulness/relevancy/precision/recall
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# See validate_judge.py's comment on this same line -- an answer containing
# a character Windows' cp1252 console can't display would otherwise crash
# this whole run partway through, after every LLM call already happened.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from assertions import run_assertions
from config import DEFAULT_TOP_K
from eval_questions import EVAL_QUESTIONS, REGRESSION_CASES
from generate import answer_question
from ingestion import get_client, get_embedding_model
from ragas_metrics import answer_relevancy, context_precision, context_recall, faithfulness
from retrieval import retrieve

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


def run_one_case(case, model, client, with_judge, with_ragas):
    chunks = retrieve(case["question"], top_k=DEFAULT_TOP_K, model=model, client=client)
    answer_result = answer_question(case["question"], top_k=DEFAULT_TOP_K, model=model, client=client)

    assertion_results = run_assertions(case, chunks, answer_result)
    all_passed = all(a["passed"] for a in assertion_results) if assertion_results else True

    record = {
        "id": case["id"],
        "question": case["question"],
        "problem_type": case.get("problem_type", "other"),
        "assertions": assertion_results,
        "assertions_passed": all_passed,
        "answer": answer_result.get("answer"),
        "skipped_llm": answer_result.get("skipped_llm"),
    }

    if with_ragas:
        record["ragas"] = {
            "faithfulness": faithfulness(case["question"], answer_result.get("answer"), chunks),
            "answer_relevancy": answer_relevancy(case["question"], answer_result.get("answer"), model),
            "context_precision": context_precision(case, chunks),
            "context_recall": context_recall(case, chunks),
        }

    if with_judge:
        from judge import judge_answer
        context = "\n\n".join(c["text"] for c in chunks) if chunks else "(no context retrieved)"
        if answer_result.get("answer"):
            record["judge"] = judge_answer(case["question"], answer_result["answer"], context)
        else:
            record["judge"] = {"verdict": "N/A", "reason": "no answer to judge (refused)"}

    return record


def scorecard(records):
    by_problem_type = {}
    for r in records:
        pt = r["problem_type"]
        bucket = by_problem_type.setdefault(pt, {"total": 0, "assertions_passed": 0, "judge_good": 0, "judge_total": 0})
        bucket["total"] += 1
        if r["assertions_passed"]:
            bucket["assertions_passed"] += 1
        if "judge" in r and r["judge"]["verdict"] in ("GOOD", "POOR"):
            bucket["judge_total"] += 1
            if r["judge"]["verdict"] == "GOOD":
                bucket["judge_good"] += 1

    print("\n=== Scorecard by problem type ===")
    for pt, bucket in sorted(by_problem_type.items()):
        line = f"{pt:<20} assertions: {bucket['assertions_passed']}/{bucket['total']}"
        if bucket["judge_total"]:
            line += f"   judge GOOD: {bucket['judge_good']}/{bucket['judge_total']}"
        print(line)

    overall_passed = sum(1 for r in records if r["assertions_passed"])
    print(f"\nOverall: {overall_passed}/{len(records)} cases passed all assertions")
    return by_problem_type


def load_snapshot(name):
    path = os.path.join(SNAPSHOT_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"No snapshot named '{name}' found at {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(name, records, by_problem_type):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "records": records,
            "by_problem_type": by_problem_type,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved snapshot '{name}' to {path}")


def print_comparison(old_snapshot, new_by_problem_type, new_records):
    print(f"\n=== Before/after vs snapshot saved at {old_snapshot.get('saved_at', '?')} ===")
    old_by_pt = old_snapshot["by_problem_type"]
    all_types = sorted(set(old_by_pt) | set(new_by_problem_type))
    for pt in all_types:
        old = old_by_pt.get(pt, {"total": 0, "assertions_passed": 0})
        new = new_by_problem_type.get(pt, {"total": 0, "assertions_passed": 0})
        old_rate = old["assertions_passed"] / old["total"] if old["total"] else 0
        new_rate = new["assertions_passed"] / new["total"] if new["total"] else 0
        arrow = "UP" if new_rate > old_rate else ("DOWN" if new_rate < old_rate else "same")
        print(f"  {pt:<20} {old['assertions_passed']}/{old['total']} ({old_rate:.0%}) -> "
              f"{new['assertions_passed']}/{new['total']} ({new_rate:.0%})  [{arrow}]")

    # Per-case flips are the clearest evidence that a specific fix worked --
    # not just an aggregate rate moving.
    old_by_id = {r["id"]: r for r in old_snapshot["records"]}
    print("\n  Per-case changes:")
    changed = False
    for r in new_records:
        old_r = old_by_id.get(r["id"])
        if old_r and old_r["assertions_passed"] != r["assertions_passed"]:
            changed = True
            print(f"    {r['id']}: {old_r['assertions_passed']} -> {r['assertions_passed']}")
    if not changed:
        print("    (no case flipped pass/fail status)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="NAME", help="save a named snapshot after this run")
    parser.add_argument("--compare", metavar="NAME", help="diff this run against a saved snapshot")
    parser.add_argument("--with-judge", action="store_true", help="also run the LLM judge (1 extra call/case)")
    parser.add_argument("--with-ragas", action="store_true", help="also run faithfulness/relevancy/precision/recall")
    args = parser.parse_args()

    model = get_embedding_model()
    client = get_client()

    all_cases = EVAL_QUESTIONS + REGRESSION_CASES
    records = []
    for case in all_cases:
        # A single case's LLM call failing (e.g. both Gemini and Groq
        # rate-limited at once, which genuinely happened running this
        # after a day of heavy eval traffic) must not crash the whole run
        # and discard every result already collected -- record it as a
        # failed case and keep going, same pattern classify_failures.py
        # already uses for exactly this.
        try:
            record = run_one_case(case, model, client, args.with_judge, args.with_ragas)
        except Exception as e:
            print(f"[ERROR] {case['id']} ({case.get('problem_type', 'other')}): {case['question'][:60]}")
            print(f"    {type(e).__name__}: {e}")
            records.append({
                "id": case["id"],
                "question": case["question"],
                "problem_type": case.get("problem_type", "other"),
                "assertions": [],
                "assertions_passed": False,
                "answer": None,
                "skipped_llm": None,
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        status = "PASS" if record["assertions_passed"] else "FAIL"
        print(f"[{status}] {case['id']} ({record['problem_type']}): {case['question'][:60]}")
        for a in record["assertions"]:
            if not a["passed"]:
                print(f"    FAILED {a['name']}: {a['reason']}")
        records.append(record)

    by_problem_type = scorecard(records)

    if args.compare:
        old_snapshot = load_snapshot(args.compare)
        if old_snapshot:
            print_comparison(old_snapshot, by_problem_type, records)

    if args.save:
        save_snapshot(args.save, records, by_problem_type)


if __name__ == "__main__":
    main()
