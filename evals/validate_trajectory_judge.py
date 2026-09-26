"""
Week 8, step 2 of validating trajectory_judge.py: check it against YOUR OWN
grading before trusting its verdicts anywhere else -- same discipline
Week 6 applied to judge.py (validate_judge.py), same threshold, same
disagreement-reporting shape. Reads trajectory_judge_sample.json (prepared
by prepare_trajectory_judge_sample.py, then hand-graded by filling in
"human_verdict" for each entry), runs trajectory_judge on the same
trajectories, and reports agreement.

Nothing downstream (trajectory_eval.py's --with-judge) should be treated as
meaningful until this agreement is high enough -- an unvalidated AI judge is
just a confident number nobody should trust.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from trajectory_judge import judge_trajectory
from tracing import get as get_trace

SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "trajectory_judge_sample.json")
AGREEMENT_THRESHOLD = 0.80  # below this, don't trust the judge yet


def main():
    if not os.path.exists(SAMPLE_FILE):
        print(f"{SAMPLE_FILE} doesn't exist yet -- run prepare_trajectory_judge_sample.py first.")
        return

    with open(SAMPLE_FILE, encoding="utf-8") as f:
        sample = json.load(f)

    ungraded = [s for s in sample if not s.get("human_verdict")]
    if ungraded:
        print(f"{len(ungraded)} entries still need a human_verdict filled in -- grade these first:")
        for s in ungraded:
            print(f"  [{s['id']}] {s['question']}")
        print(f"\nEdit {SAMPLE_FILE} and set \"human_verdict\": \"GOOD\" or \"POOR\" for each, then re-run.")
        return

    agreements = 0
    disagreements = []
    for entry in sample:
        trace = get_trace(entry["trace_id"])
        judge_result = judge_trajectory(entry["question"], trace, entry["answer"])
        entry["judge_verdict"] = judge_result["verdict"]
        entry["judge_reason"] = judge_result["reason"]

        human = entry["human_verdict"].strip().upper()
        agree = judge_result["verdict"] == human
        if agree:
            agreements += 1
        else:
            disagreements.append(entry)

        print(f"[{entry['id']}] human={human:<5} judge={judge_result['verdict']:<5} "
              f"{'AGREE' if agree else 'DISAGREE'}")

    n = len(sample)
    agreement_rate = agreements / n
    print(f"\n=== Agreement: {agreements}/{n} = {agreement_rate:.1%} ===")

    if disagreements:
        print("\n=== Disagreements (read both reasons to see where they diverge) ===")
        for entry in disagreements:
            print(f"\n[{entry['id']}] \"{entry['question']}\"")
            print(f"  human ({entry['human_verdict']}): {entry.get('human_reason') or '(no reason given)'}")
            print(f"  judge ({entry['judge_verdict']}): {entry['judge_reason']}")

    if agreement_rate >= AGREEMENT_THRESHOLD:
        print(f"\nJudge is VALIDATED (>= {AGREEMENT_THRESHOLD:.0%} agreement) -- "
              f"safe to trust with --with-judge in trajectory_eval.py.")
    else:
        print(f"\nJudge is NOT validated yet (< {AGREEMENT_THRESHOLD:.0%} agreement) -- "
              f"do not treat its scores as meaningful until this improves.")

    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
