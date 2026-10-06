"""
Week 10: race the manager+2-specialists team (multi_agent.py) against the
single agent (agent.py, Week 7-9) on the SAME 20 questions, scored with the
EXACT SAME rule-based checks (assertions.py) -- no separate judgment call
for "did the team get it right" vs "did the single agent get it right".

Directly extends evals/race_agent_vs_workflow.py's shape (same cases, same
assertions, same disagreement-reporting idea) with:
  - a third arm (the team) instead of the fixed pipeline
  - two new, REAL metrics neither existing race script captures:
    tokens (prompt_tokens + completion_tokens, summed across every LLM
    call in that arm) and cost (split input/output rates from config.py,
    verified against real provider pricing -- see config.py's own comment)

Reports exactly the brief's four numbers: Quality (Pass rate), Speed
(Execution latency), Tokens Used, Cost ($).

Usage:
    python evals/race_team_vs_agent.py
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
from config import (
    DEFAULT_TOP_K,
    GEMINI_INPUT_COST_PER_1K,
    GEMINI_OUTPUT_COST_PER_1K,
)
from eval_questions import EVAL_QUESTIONS, REGRESSION_CASES
from ingestion import get_client, get_embedding_model
from multi_agent import run_team

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "team_race_results.json")


def _with_rate_limit_retry(fn, *args, max_attempts=3, **kwargs):
    """Ported from race_agent_vs_workflow.py's own fix this same week --
    this script never got it, so a 429 on either arm would silently
    record a permanent FAIL instead of a short backoff + retry. See that
    script's _with_rate_limit_retry() for the original live-crash context."""
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

# Both providers answered with Gemini in every real run observed while
# building this (Groq never got exercised in testing) -- cost is computed
# with Gemini's verified rate as the default proxy. Documented as such in
# FINDINGS_week10.md rather than silently assumed to be exactly right for
# every single call.
INPUT_RATE = GEMINI_INPUT_COST_PER_1K
OUTPUT_RATE = GEMINI_OUTPUT_COST_PER_1K


def _cost(prompt_tokens, completion_tokens):
    return (prompt_tokens * INPUT_RATE + completion_tokens * OUTPUT_RATE) / 1000


def run_single(case, model, client):
    """The Week 7-9 single agent -- same call evals/race_agent_vs_workflow.py
    already uses, now also reading the real token counts Week 10 added to
    agent.py's return dict."""
    t0 = time.perf_counter()
    try:
        result = run_agent(case["question"], top_k=DEFAULT_TOP_K, model=model, client=client)
    except requests.exceptions.HTTPError as e:
        # A 429 specifically is let through uncaught so _with_rate_limit_retry()
        # wrapping this function's caller can actually retry it -- same
        # reasoning as race_agent_vs_workflow.py's run_agentic().
        if e.response is not None and e.response.status_code == 429:
            raise
        elapsed = time.perf_counter() - t0
        return {"seconds": elapsed, "llm_calls": None, "passed": False,
                "assertions": [], "answer": None, "prompt_tokens": 0, "completion_tokens": 0,
                "error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"seconds": elapsed, "llm_calls": None, "passed": False,
                "assertions": [], "answer": None, "prompt_tokens": 0, "completion_tokens": 0,
                "error": f"{type(e).__name__}: {e}"}
    elapsed = time.perf_counter() - t0

    answer_result = {"answer": result["answer"], "skipped_llm": result["skipped_llm"]}
    assertion_results = run_assertions(case, result["chunks_seen"], answer_result)
    passed = all(a["passed"] for a in assertion_results) if assertion_results else True

    return {
        "seconds": elapsed, "llm_calls": result["llm_calls"], "passed": passed,
        "assertions": assertion_results, "answer": result["answer"],
        "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
        "steps_taken": result["steps_taken"], "stopped_reason": result["stopped_reason"],
    }


def run_team_case(case, model, client):
    """The Week 10 manager+2-specialists team. model/client: loaded once
    in main() and threaded through to multi_agent.run_team() -- without
    this, each specialist's retrieve() call reloads its own embedding
    model from scratch (found live: ~5-6s, every single call, since
    get_embedding_model() has no caching), which would inflate the
    team's measured latency with pure reload overhead unrelated to real
    multi-agent cost and bias this race's own "Speed" numbers."""
    t0 = time.perf_counter()
    try:
        result = run_team(case["question"], model=model, client=client)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            raise
        elapsed = time.perf_counter() - t0
        return {"seconds": elapsed, "llm_calls": None, "passed": False,
                "assertions": [], "answer": None, "prompt_tokens": 0, "completion_tokens": 0,
                "error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"seconds": elapsed, "llm_calls": None, "passed": False,
                "assertions": [], "answer": None, "prompt_tokens": 0, "completion_tokens": 0,
                "error": f"{type(e).__name__}: {e}"}
    elapsed = time.perf_counter() - t0

    # run_assertions() needs "chunks" (for expected_source/forbidden_sources/
    # must_cite_page checks) -- the team's result doesn't expose raw chunks
    # the same shape agent.py's chunks_seen does, so reconstruct the minimal
    # shape those checks actually read (source/page) from what we do have:
    # the final declared sources. This means must_cite_page can't be
    # checked for the team (no page data survives the merge) -- documented
    # honestly in FINDINGS_week10.md rather than faked.
    pseudo_chunks = [{"source": s, "page": None} for s in result["sources"]]
    answer_result = {"answer": result["answer"], "skipped_llm": result["skipped_llm"]}
    assertion_results = run_assertions(case, pseudo_chunks, answer_result)
    passed = all(a["passed"] for a in assertion_results) if assertion_results else True

    return {
        "seconds": elapsed, "llm_calls": result["llm_calls"], "passed": passed,
        "assertions": assertion_results, "answer": result["answer"],
        "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
        "specialists_used": result["specialists_used"],
    }


def main():
    model = get_embedding_model()
    client = get_client()

    all_cases = EVAL_QUESTIONS + REGRESSION_CASES
    records = []

    print(f"{'ID':<5} {'SINGLE AGENT':<28} {'TEAM':<32}")
    print("-" * 70)
    for case in all_cases:
        try:
            single = _with_rate_limit_retry(run_single, case, model, client)
        except requests.exceptions.HTTPError as e:
            single = {"seconds": 0.0, "llm_calls": None, "passed": False,
                      "assertions": [], "answer": None, "prompt_tokens": 0, "completion_tokens": 0,
                      "error": f"{type(e).__name__}: {e}"}
        try:
            team = _with_rate_limit_retry(run_team_case, case, model, client)
        except requests.exceptions.HTTPError as e:
            team = {"seconds": 0.0, "llm_calls": None, "passed": False,
                    "assertions": [], "answer": None, "prompt_tokens": 0, "completion_tokens": 0,
                    "error": f"{type(e).__name__}: {e}"}

        single_label = f"{'PASS' if single['passed'] else 'FAIL'} {single['seconds']:.1f}s/{single.get('llm_calls', '?')}calls"
        team_label = f"{'PASS' if team['passed'] else 'FAIL'} {team['seconds']:.1f}s/{team.get('llm_calls', '?')}calls"
        print(f"{case['id']:<5} {single_label:<28} {team_label:<32}")

        records.append({
            "id": case["id"], "question": case["question"],
            "problem_type": case.get("problem_type", "other"),
            "single": single, "team": team,
        })

    # --- Summary: exactly the brief's four numbers ---
    n = len(records)

    def _agg(arm):
        total_s = sum(r[arm]["seconds"] for r in records)
        total_calls = sum(r[arm].get("llm_calls") or 0 for r in records)
        total_prompt = sum(r[arm]["prompt_tokens"] for r in records)
        total_completion = sum(r[arm]["completion_tokens"] for r in records)
        passed = sum(1 for r in records if r[arm]["passed"])
        total_cost = sum(_cost(r[arm]["prompt_tokens"], r[arm]["completion_tokens"]) for r in records)
        return {
            "avg_seconds": total_s / n, "total_calls": total_calls,
            "total_tokens": total_prompt + total_completion,
            "avg_tokens": (total_prompt + total_completion) / n,
            "total_cost": total_cost, "avg_cost": total_cost / n,
            "passed": passed,
        }

    single_agg = _agg("single")
    team_agg = _agg("team")

    single_quality = f"{single_agg['passed']}/{n} ({single_agg['passed']/n:.0%})"
    team_quality = f"{team_agg['passed']}/{n} ({team_agg['passed']/n:.0%})"
    single_speed = f"{single_agg['avg_seconds']:.2f}s avg"
    team_speed = f"{team_agg['avg_seconds']:.2f}s avg"
    single_tokens = f"{single_agg['avg_tokens']:.0f} avg/question"
    team_tokens = f"{team_agg['avg_tokens']:.0f} avg/question"
    single_cost = f"${single_agg['avg_cost']:.4f} avg/question"
    team_cost = f"${team_agg['avg_cost']:.4f} avg/question"

    print(f"\n=== Summary (n={n}) ===")
    print(f"{'Metric':<28} {'Single Agent':<24} {'Team (Manager+2)':<24}")
    print(f"{'Quality (Pass rate)':<28} {single_quality:<24} {team_quality:<24}")
    print(f"{'Speed (Execution latency)':<28} {single_speed:<24} {team_speed:<24}")
    print(f"{'Tokens Used':<28} {single_tokens:<24} {team_tokens:<24}")
    print(f"{'Cost ($)':<28} {single_cost:<24} {team_cost:<24}")

    print("\n=== Cases where single agent and team disagree on pass/fail ===")
    disagreements = [r for r in records if r["single"]["passed"] != r["team"]["passed"]]
    if disagreements:
        for r in disagreements:
            print(f"  {r['id']}: single={'PASS' if r['single']['passed'] else 'FAIL'} "
                  f"team={'PASS' if r['team']['passed'] else 'FAIL'}")
    else:
        print("  None -- both approaches passed/failed the exact same cases.")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "records": records,
            "summary": {"n": n, "single": single_agg, "team": team_agg},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
