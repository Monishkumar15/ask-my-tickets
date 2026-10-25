"""
Week 11: the actual measured cost improvement the brief asks for -- not an
estimate. Runs EVAL_QUESTIONS through the real fixed pipeline
(generate.py::answer_question()) twice: once against a deliberately
cleared (cold) cache, once again (warm) immediately after, and reports the
real token/cost difference plus the real cache hit rate -- using the exact
same usage_out/estimate_cost() machinery Week 11's Part A already built
for the trace data, not a separate calculation.

Usage:
    python evals/measure_cache_savings.py
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import semantic_cache
from config import DEFAULT_TOP_K
from eval_questions import EVAL_QUESTIONS, REGRESSION_CASES
from generate import answer_question
from ingestion import get_client, get_embedding_model

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "cache_savings_results.json")


def _run_pass(cases, model, client):
    records = []
    for case in cases:
        result = answer_question(case["question"], top_k=DEFAULT_TOP_K, model=model, client=client)
        records.append({
            "id": case["id"], "question": case["question"],
            "cache_hit": result.get("cache_hit", False),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_cost_usd": result.get("total_cost_usd", 0.0),
            "skipped_llm": result.get("skipped_llm", False),
        })
    return records


def _summarize(records):
    n = len(records)
    total_tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in records)
    total_cost = sum(r["total_cost_usd"] for r in records)
    hits = sum(1 for r in records if r["cache_hit"])
    return {"n": n, "total_tokens": total_tokens, "total_cost_usd": total_cost, "cache_hits": hits}


def main():
    model = get_embedding_model()
    client = get_client()
    all_cases = EVAL_QUESTIONS + REGRESSION_CASES

    # A genuinely cold cache, not just "probably empty" -- confirms the
    # "before" numbers aren't accidentally measuring a warm cache left
    # over from a previous run of this same script or of answer_question()
    # elsewhere (the UI, run_eval.py, etc.).
    semantic_cache.clear(client=client)

    print(f"Running {len(all_cases)} cases COLD (empty cache)...")
    cold_records = _run_pass(all_cases, model, client)
    cold = _summarize(cold_records)

    print(f"Running {len(all_cases)} cases WARM (same questions, cache now populated)...")
    warm_records = _run_pass(all_cases, model, client)
    warm = _summarize(warm_records)

    cold_hits_label = f"{cold['cache_hits']}/{cold['n']}"
    warm_hits_label = f"{warm['cache_hits']}/{warm['n']}"

    print(f"\n=== Cache savings (n={cold['n']}) ===")
    print(f"{'Metric':<22} {'Cold (empty cache)':<22} {'Warm (populated)':<22}")
    print(f"{'Total tokens':<22} {cold['total_tokens']:<22} {warm['total_tokens']:<22}")
    print(f"{'Total cost ($)':<22} {cold['total_cost_usd']:<22.6f} {warm['total_cost_usd']:<22.6f}")
    print(f"{'Cache hits':<22} {cold_hits_label:<22} {warm_hits_label:<22}")

    if cold["total_cost_usd"] > 0:
        pct_saved = 100 * (1 - warm["total_cost_usd"] / cold["total_cost_usd"])
        print(f"\nCost reduction on the warm pass: {pct_saved:.1f}%")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "cold_records": cold_records, "warm_records": warm_records,
            "cold_summary": cold, "warm_summary": warm,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
