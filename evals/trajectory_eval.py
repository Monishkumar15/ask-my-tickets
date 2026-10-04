"""
Week 8: the one-command trajectory test runner. Runs every eval case
(evals/eval_questions.py's EVAL_QUESTIONS + REGRESSION_CASES) through the
Week 7 agent (agent.py's run_agent(), not the fixed pipeline), grades each
run on TWO independent axes -- outcome (assertions.py, same rule-based
checks Week 6 already built) and trajectory (trajectory_checks.py, new
this week) -- and specifically reports any case where they disagree: a
right answer reached by a wrong path, the outcome-vs-trajectory gap this
week is about.

Usage:
    python evals/trajectory_eval.py                      # run + print both scorecards
    python evals/trajectory_eval.py --save NAME           # also save a named snapshot
    python evals/trajectory_eval.py --compare NAME        # diff this run against a snapshot
    python evals/trajectory_eval.py --with-judge          # also run the validated trajectory_judge
                                                           #  (1 extra call/case -- only meaningful
                                                           #  once validate_trajectory_judge.py has
                                                           #  confirmed it agrees with hand grading)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent import run_agent
from assertions import run_assertions
from config import DEFAULT_TOP_K
from eval_questions import EVAL_QUESTIONS, REGRESSION_CASES
from ingestion import get_client, get_embedding_model
from trajectory_checks import run_trajectory_checks
from trajectory_judge import judge_trajectory
from tracing import get as get_trace

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


def run_one_case(case, model, client, with_judge):
    t0 = time.perf_counter()
    result = run_agent(case["question"], top_k=DEFAULT_TOP_K, model=model, client=client)
    elapsed = time.perf_counter() - t0

    trace = get_trace(result["trace_id"]) or {}
    answer_result = {"answer": result["answer"], "skipped_llm": result["skipped_llm"]}

    outcome_checks = run_assertions(case, result["chunks_seen"], answer_result)
    outcome_passed = all(c["passed"] for c in outcome_checks) if outcome_checks else True

    trajectory_checks = run_trajectory_checks(case, trace)
    trajectory_rule_passed = all(c["passed"] for c in trajectory_checks) if trajectory_checks else True

    judge_result = None
    if with_judge:
        judge_result = judge_trajectory(case["question"], trace, result["answer"])

    # The judge catches semantic reasoning failures the rules can't see
    # (e.g. misreading truncated evidence); the rules catch structural
    # failures the judge's rubric doesn't ask it to penalize (e.g. wasted
    # steps) -- see validate_trajectory_judge.py's one documented
    # disagreement. Combined, not either alone.
    trajectory_passed = trajectory_rule_passed and (judge_result is None or judge_result["verdict"] != "POOR")

    record = {
        "id": case["id"], "question": case["question"],
        "problem_type": case.get("problem_type", "other"),
        "seconds": elapsed, "llm_calls": result["llm_calls"], "steps_taken": result["steps_taken"],
        "stopped_reason": result["stopped_reason"], "escalated": result["escalated"],
        "answer": result["answer"], "trace_id": result["trace_id"],
        "outcome_checks": outcome_checks, "outcome_passed": outcome_passed,
        "trajectory_checks": trajectory_checks, "trajectory_rule_passed": trajectory_rule_passed,
        "trajectory_passed": trajectory_passed,
        # The literal deliverable: right answer, wrong path.
        "outcome_trajectory_gap": outcome_passed and not trajectory_passed,
    }
    if judge_result is not None:
        record["judge_verdict"] = judge_result["verdict"]
        record["judge_reason"] = judge_result["reason"]
    return record


def _run_with_rate_limit_retry(case, model, client, with_judge, max_attempts=3):
    """
    Both providers can hit a free-tier rate limit under this eval's
    back-to-back request rate (an agent call plus a judge call, per case).
    A 429 here is throttling, not a real failure -- worth a short backoff
    and retry before letting it surface as an actual error.
    """
    for attempt in range(max_attempts):
        try:
            return run_one_case(case, model, client, with_judge)
        except requests.exceptions.HTTPError as e:
            is_rate_limited = e.response is not None and e.response.status_code == 429
            if not is_rate_limited or attempt == max_attempts - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"[RATE LIMITED] {case['id']} -- waiting {wait}s before retry {attempt + 2}/{max_attempts}")
            time.sleep(wait)


def _percentile(values, pct):
    """Simple nearest-rank percentile -- no numpy dependency, consistent
    with this project's minimal-dependency pattern elsewhere. At small n
    (this eval set has 20 cases), p99 is honestly close to the max --
    stated in the printed report rather than overclaiming statistical
    rigor from too few samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[rank]


def print_scorecard(records):
    print("\n=== Outcome vs Trajectory scorecard ===")
    print(f"{'ID':<5} {'OUTCOME':<9} {'TRAJECTORY':<12} {'STEPS':<6} {'CALLS':<6} {'SECONDS':<8}")
    print("-" * 55)
    for r in records:
        if r.get("errored"):
            # A case that never ran (quota, network) is not an agent failure.
            print(f"{r['id']:<5} {'ERROR':<9} {'ERROR':<12} {'-':<6} {'-':<6} {'-':<8}")
            continue
        print(f"{r['id']:<5} {'PASS' if r['outcome_passed'] else 'FAIL':<9} "
              f"{'PASS' if r['trajectory_passed'] else 'FAIL':<12} "
              f"{r['steps_taken']:<6} {r['llm_calls']:<6} {r['seconds']:<8.2f}")

    # Errored cases are counted separately and excluded from every rate below:
    # reporting them as FAIL made a quota outage look like an agent regression.
    errored = [r for r in records if r.get("errored")]
    records = [r for r in records if not r.get("errored")]
    n = len(records)
    outcome_passed_n = sum(1 for r in records if r["outcome_passed"])
    trajectory_passed_n = sum(1 for r in records if r["trajectory_passed"])
    print(f"\nOutcome passed:    {outcome_passed_n}/{n}")
    print(f"Trajectory passed: {trajectory_passed_n}/{n}")
    if errored:
        print(f"\n!! {len(errored)} case(s) ERRORED and are excluded from the counts above: "
              f"{', '.join(r['id'] for r in errored)}")
        print("!! This run is INCOMPLETE. Do not quote it, compare it, or save it as a snapshot.")
    if n == 0:
        return

    print("\n=== Outcome-vs-trajectory gap: right answer, wrong path ===")
    gaps = [r for r in records if r["outcome_trajectory_gap"]]
    if not gaps:
        print("  None found in this run.")
    else:
        for r in gaps:
            print(f"  {r['id']}: \"{r['question'][:60]}\" (trace {r['trace_id']})")
            for c in r["trajectory_checks"]:
                if not c["passed"]:
                    print(f"      FAILED {c['name']}: {c['reason']}")

    seconds = [r["seconds"] for r in records]
    calls = [r["llm_calls"] for r in records]
    print("\n=== Cost per task ===")
    print(f"  Seconds  -- mean: {sum(seconds)/n:.2f}s   p99: {_percentile(seconds, 99):.2f}s"
          f"   (n={n}, so p99 is close to the observed max, not a robust tail estimate)")
    print(f"  LLM calls -- mean: {sum(calls)/n:.2f}   p99: {_percentile(calls, 99):.2f}")


def load_snapshot(name):
    path = os.path.join(SNAPSHOT_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"No snapshot named '{name}' found at {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(name, records):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"saved_at": datetime.now(timezone.utc).isoformat(), "records": records}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved snapshot '{name}' to {path}")


def print_comparison(old_snapshot, new_records):
    print(f"\n=== Before/after vs snapshot saved at {old_snapshot.get('saved_at', '?')} ===")
    old_by_id = {r["id"]: r for r in old_snapshot["records"]}
    changed = False
    for r in new_records:
        old_r = old_by_id.get(r["id"])
        if not old_r or r.get("errored") or old_r.get("errored"):
            continue
        if old_r["outcome_passed"] != r["outcome_passed"] or old_r["trajectory_passed"] != r["trajectory_passed"]:
            changed = True
            print(f"  {r['id']}: outcome {old_r['outcome_passed']}->{r['outcome_passed']}, "
                  f"trajectory {old_r['trajectory_passed']}->{r['trajectory_passed']}")
    if not changed:
        print("  (no case flipped outcome or trajectory pass/fail status)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="NAME", help="save a named snapshot after this run")
    parser.add_argument("--compare", metavar="NAME", help="diff this run against a saved snapshot")
    parser.add_argument("--with-judge", action="store_true",
                         help="also run the validated trajectory_judge (1 extra call/case)")
    args = parser.parse_args()

    model = get_embedding_model()
    client = get_client()

    all_cases = EVAL_QUESTIONS + REGRESSION_CASES
    records = []
    for case in all_cases:
        try:
            record = _run_with_rate_limit_retry(case, model, client, args.with_judge)
        except Exception as e:
            # Same lesson as run_eval.py / race_agent_vs_workflow.py: one
            # case's LLM call failing must not discard every result
            # already collected.
            print(f"[ERROR] {case['id']}: {case['question'][:60]}")
            print(f"    {type(e).__name__}: {e}")
            records.append({
                "id": case["id"], "question": case["question"], "problem_type": case.get("problem_type", "other"),
                "seconds": 0.0, "llm_calls": 0, "steps_taken": 0, "stopped_reason": "error", "escalated": False,
                "answer": None, "trace_id": None, "outcome_checks": [], "outcome_passed": False,
                "trajectory_checks": [], "trajectory_rule_passed": False, "trajectory_passed": False,
                "outcome_trajectory_gap": False, "error": f"{type(e).__name__}: {e}",
                "errored": True,
            })
            continue
        records.append(record)

    print_scorecard(records)

    if args.compare:
        old_snapshot = load_snapshot(args.compare)
        if old_snapshot:
            print_comparison(old_snapshot, records)

    if args.save:
        if any(r.get("errored") for r in records):
            print(f"\nNot saving snapshot '{args.save}': some cases errored, so it would record "
                  f"a quota outage as a baseline. Re-run when every case completes.")
        else:
            save_snapshot(args.save, records)


if __name__ == "__main__":
    main()
