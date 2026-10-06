"""
Week 7, Phase 5: race the hand-built agent (agent.py) against the existing
fixed pipeline (generate.py's answer_question()) on the SAME questions,
scored with the EXACT SAME rule-based checks (assertions.py, from Week 6)
-- no separate judgment call for "did the agent get it right" vs "did the
fixed pipeline get it right".

Measures the three things the brief asks for, with real numbers:
  speed       -- wall-clock seconds per question
  cost        -- number of LLM calls per question (a real proxy: the agent
                 makes one call per step, the fixed pipeline always makes
                 exactly one)
  reliability -- assertions.py's pass/fail, unchanged from Week 6

Uses the real persisted index, same as run_eval.py -- this tests what a
user actually gets from each approach, not a clean-room reproduction.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import run_agent
from assertions import run_assertions
from config import DEFAULT_TOP_K
from eval_questions import EVAL_QUESTIONS, REGRESSION_CASES
from generate import answer_question
from ingestion import get_client, get_embedding_model
from retrieval import retrieve

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "race_results.json")


def _with_rate_limit_retry(fn, *args, max_attempts=3, **kwargs):
    """
    Found live: run_fixed() had NO exception handling at all, so a 429 on
    EITHER provider (even a momentary one that would have cleared itself
    within seconds) crashed the entire race, discarding every result
    already collected. Same lesson trajectory_eval.py's own
    _run_with_rate_limit_retry() already learned for that script -- ported
    here, since this script never got the same fix. A 429 here is
    throttling, not a real failure -- worth a short backoff and retry
    before letting it crash the whole run.
    """
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            is_rate_limited = e.response is not None and e.response.status_code == 429
            if not is_rate_limited or attempt == max_attempts - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"  [RATE LIMITED] waiting {wait}s before retry {attempt + 2}/{max_attempts}")
            time.sleep(wait)


def run_fixed(case, model, client):
    """The existing single-pass pipeline -- always exactly 1 LLM call."""
    t0 = time.perf_counter()
    chunks = retrieve(case["question"], top_k=DEFAULT_TOP_K, model=model, client=client)
    answer_result = answer_question(case["question"], top_k=DEFAULT_TOP_K, model=model, client=client)
    elapsed = time.perf_counter() - t0

    assertion_results = run_assertions(case, chunks, answer_result)
    passed = all(a["passed"] for a in assertion_results) if assertion_results else True

    return {
        "seconds": elapsed, "llm_calls": 1, "passed": passed,
        "assertions": assertion_results, "answer": answer_result.get("answer"),
    }


def run_agentic(case, model, client):
    """The Week 7 agent -- 1+ LLM calls, however many steps it actually took."""
    t0 = time.perf_counter()
    try:
        result = run_agent(case["question"], top_k=DEFAULT_TOP_K, model=model, client=client)
    except requests.exceptions.HTTPError as e:
        # A 429 specifically is let through uncaught (not converted to a
        # failure record here) so _with_rate_limit_retry() -- wrapping this
        # function's caller -- can actually retry it. Swallowing it here
        # the way every OTHER exception is below would silently record a
        # real pass as a permanent FAIL instead of giving it the same
        # retry chance run_fixed() now gets.
        if e.response is not None and e.response.status_code == 429:
            raise
        elapsed = time.perf_counter() - t0
        return {"seconds": elapsed, "llm_calls": None, "passed": False,
                "assertions": [], "answer": None, "error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        # A single case's LLM call failing must not crash the whole race --
        # same lesson learned (and fixed) in run_eval.py earlier this week.
        elapsed = time.perf_counter() - t0
        return {"seconds": elapsed, "llm_calls": None, "passed": False,
                "assertions": [], "answer": None, "error": f"{type(e).__name__}: {e}"}
    elapsed = time.perf_counter() - t0

    answer_result = {"answer": result["answer"], "skipped_llm": result["skipped_llm"]}
    assertion_results = run_assertions(case, result["chunks_seen"], answer_result)
    passed = all(a["passed"] for a in assertion_results) if assertion_results else True

    return {
        "seconds": elapsed, "llm_calls": result["llm_calls"], "passed": passed,
        "assertions": assertion_results, "answer": result["answer"],
        "steps_taken": result["steps_taken"], "stopped_reason": result["stopped_reason"],
        "escalated": result["escalated"],
    }


def main():
    model = get_embedding_model()
    client = get_client()

    all_cases = EVAL_QUESTIONS + REGRESSION_CASES
    records = []

    print(f"{'ID':<5} {'FIXED':<22} {'AGENT':<32}")
    print("-" * 65)
    for case in all_cases:
        try:
            fixed = _with_rate_limit_retry(run_fixed, case, model, client)
        except requests.exceptions.HTTPError as e:
            # Both providers were still rate-limited after every retry --
            # same non-crashing treatment run_agentic() already gives a
            # failure, so one case's exhausted quota doesn't discard every
            # result already collected.
            fixed = {"seconds": 0.0, "llm_calls": None, "passed": False,
                      "assertions": [], "answer": None, "error": f"{type(e).__name__}: {e}"}
        try:
            agent = _with_rate_limit_retry(run_agentic, case, model, client)
        except requests.exceptions.HTTPError as e:
            agent = {"seconds": 0.0, "llm_calls": None, "passed": False,
                      "assertions": [], "answer": None, "error": f"{type(e).__name__}: {e}"}

        fixed_label = f"{'PASS' if fixed['passed'] else 'FAIL'} {fixed['seconds']:.1f}s/1call"
        agent_label = (f"{'PASS' if agent['passed'] else 'FAIL'} {agent['seconds']:.1f}s/"
                        f"{agent.get('llm_calls', '?')}calls/{agent.get('steps_taken', '?')}steps")
        print(f"{case['id']:<5} {fixed_label:<22} {agent_label:<32}")

        records.append({
            "id": case["id"], "question": case["question"],
            "problem_type": case.get("problem_type", "other"),
            "fixed": fixed, "agent": agent,
        })

    # --- Summary ---
    n = len(records)
    fixed_total_time = sum(r["fixed"]["seconds"] for r in records)
    agent_total_time = sum(r["agent"]["seconds"] for r in records)
    fixed_total_calls = sum(r["fixed"].get("llm_calls") or 0 for r in records)
    agent_total_calls = sum(r["agent"].get("llm_calls") or 0 for r in records)
    fixed_passed = sum(1 for r in records if r["fixed"]["passed"])
    agent_passed = sum(1 for r in records if r["agent"]["passed"])

    print("\n=== Summary (n={}) ===".format(n))
    print(f"{'Metric':<28} {'Fixed pipeline':<20} {'Agent':<20}")
    print(f"{'Total time':<28} {fixed_total_time:<20.1f} {agent_total_time:<20.1f}")
    print(f"{'Avg time/question':<28} {fixed_total_time/n:<20.2f} {agent_total_time/n:<20.2f}")
    print(f"{'Total LLM calls':<28} {fixed_total_calls:<20} {agent_total_calls:<20}")
    print(f"{'Avg LLM calls/question':<28} {fixed_total_calls/n:<20.2f} {agent_total_calls/n:<20.2f}")
    print(f"{'Assertions passed':<28} {f'{fixed_passed}/{n}':<20} {f'{agent_passed}/{n}':<20}")

    # Where they actually disagree -- the useful evidence, not just the
    # aggregate numbers.
    print("\n=== Cases where fixed and agent disagree on pass/fail ===")
    disagreements = [r for r in records if r["fixed"]["passed"] != r["agent"]["passed"]]
    if disagreements:
        for r in disagreements:
            print(f"  {r['id']}: fixed={'PASS' if r['fixed']['passed'] else 'FAIL'} "
                  f"agent={'PASS' if r['agent']['passed'] else 'FAIL'}")
    else:
        print("  None -- both approaches passed/failed the exact same cases.")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "records": records,
            "summary": {
                "n": n,
                "fixed_total_time": fixed_total_time, "agent_total_time": agent_total_time,
                "fixed_total_calls": fixed_total_calls, "agent_total_calls": agent_total_calls,
                "fixed_passed": fixed_passed, "agent_passed": agent_passed,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
