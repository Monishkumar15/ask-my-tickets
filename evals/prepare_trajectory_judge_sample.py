"""
Week 8, step 1 of validating trajectory_judge.py: build a sample to hand-grade.

Reads a saved trajectory_eval.py snapshot (default: week8_baseline), pulls
each case's full trace via tracing.get(trace_id), and writes
trajectory_judge_sample.json with an empty "human_verdict" field per entry --
same two-step process Week 6 used for judge.py (prepare_judge_sample.py,
then validate_judge.py), applied to trajectory quality instead of answer
quality.

Usage:
    python evals/prepare_trajectory_judge_sample.py [snapshot_name]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from trajectory_judge import render_trajectory
from tracing import get as get_trace

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "trajectory_judge_sample.json")


def main():
    snapshot_name = sys.argv[1] if len(sys.argv) > 1 else "week8_baseline"
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"{snapshot_name}.json")
    if not os.path.exists(snapshot_path):
        print(f"No snapshot named '{snapshot_name}' found at {snapshot_path} -- "
              f"run trajectory_eval.py --save {snapshot_name} first.")
        return

    with open(snapshot_path, encoding="utf-8") as f:
        snapshot = json.load(f)

    sample = []
    for record in snapshot["records"]:
        if not record.get("trace_id"):
            print(f"[SKIP] {record['id']}: no trace_id (this case errored during the eval run)")
            continue

        trace = get_trace(record["trace_id"])
        if trace is None:
            print(f"[SKIP] {record['id']}: trace {record['trace_id']} not found on disk")
            continue

        sample.append({
            "id": record["id"],
            "question": record["question"],
            "trace_id": record["trace_id"],
            "trajectory_text": render_trajectory(trace),
            "answer": record["answer"],
            "outcome_passed": record["outcome_passed"],
            "trajectory_rule_checks_passed": record["trajectory_passed"],
            "human_verdict": None,
            "human_reason": None,
            "judge_verdict": None,
            "judge_reason": None,
        })

    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(sample)} entries to {SAMPLE_FILE}.")
    print("Next: fill in \"human_verdict\": \"GOOD\" or \"POOR\" (and optionally "
          "\"human_reason\") for each entry by reading its trajectory_text, then "
          "run validate_trajectory_judge.py.")


if __name__ == "__main__":
    main()
