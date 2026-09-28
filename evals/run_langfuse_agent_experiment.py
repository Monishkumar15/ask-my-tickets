"""
Week 8 equivalent of run_langfuse_experiment.py: runs the real agent
(agent.py's run_agent(), the same function evals/trajectory_eval.py already
uses) against the ask_my_tickets_agent_trajectory_set dataset.

Two scoring layers, same architecture as the fixed-pipeline experiment:
  1. Local, free checks -- BOTH evals/assertions.py's outcome checks AND
     evals/trajectory_checks.py's structural trajectory checks (repeated
     searches, wasted steps, silently-incomplete runs, terminal action,
     step efficiency) -- run in-process, no LLM call, posted as Scores.
  2. The hosted `agent-trajectory-judge` evaluator (langfuse_eval_setup.py)
     runs automatically server-side, reading the rendered thought/action/
     observation trail -- the semantic reasoning check a structural rule
     can't do (misread evidence, invented detail, unjustified escalate).

Usage: python evals/run_langfuse_agent_experiment.py [--limit N]
"""
import argparse
import io
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langfuse import Evaluation, Langfuse  # noqa: E402

import config  # noqa: E402
from agent import run_agent  # noqa: E402
from evals.assertions import run_assertions  # noqa: E402
from evals.langfuse_eval_setup import TRAJECTORY_DATASET_NAME  # noqa: E402
from evals.trajectory_checks import run_trajectory_checks  # noqa: E402
from evals.trajectory_judge import render_trajectory  # noqa: E402
from tracing import get as get_trace  # noqa: E402


def _run_one(question):
    """
    The real agent call -- run_agent(), the exact same proven, synchronous
    entry point evals/trajectory_eval.py already uses successfully for all
    20 cases. Deliberately called OUTSIDE run_experiment()'s own asyncio
    task scheduling (see main(), below) -- awaiting _run_agent_async()
    directly from inside an async task() function was tried first and
    failed intermittently ("unhandled errors in a TaskGroup") on ~half the
    items, reproducible even at max_concurrency=1 and after a full MCP
    server restart. Root cause: mcp_client.py's MCPToolSession relies on
    anyio task groups (via mcp's streamable_http_client) whose cancel
    scopes are tied to a specific asyncio Task identity -- run_experiment()
    schedules each item's task() as its own asyncio Task even at
    concurrency 1, which anyio's cancel-scope tracking doesn't tolerate.
    Running the agent in a plain top-level loop first (this function),
    then handing run_experiment() only an already-computed result to look
    up, sidesteps the incompatibility entirely rather than fighting it.
    """
    result = run_agent(question)
    trace = get_trace(result["trace_id"]) or {}
    trajectory_text = render_trajectory(trace)
    return {
        "answer": result["answer"],
        "trajectory_text": trajectory_text,
        "chunks_seen": result["chunks_seen"],
        "skipped_llm": result.get("skipped_llm", False),
        "steps_taken": result["steps_taken"],
        "llm_calls": result["llm_calls"],
        "stopped_reason": result["stopped_reason"],
        "trace": trace,
    }


def _run_one_with_retry(question, max_attempts=3):
    """Same rate-limit-retry discipline as evals/trajectory_eval.py's
    _run_with_rate_limit_retry() -- an agent call plus (previously) a judge
    call per case can exceed the free-tier rate under back-to-back
    requests; a 429 here is throttling, not a real failure. Also retries a
    plain network ReadTimeout (found live running this batch) -- a
    transient connectivity blip, not a real failure either."""
    for attempt in range(max_attempts):
        try:
            return _run_one(question)
        except requests.exceptions.HTTPError as e:
            is_rate_limited = e.response is not None and e.response.status_code == 429
            if not is_rate_limited or attempt == max_attempts - 1:
                raise
            wait = 15 * (attempt + 1)
            print(f"    [RATE LIMITED] waiting {wait}s before retry {attempt + 2}/{max_attempts}")
            time.sleep(wait)
        except requests.exceptions.Timeout:
            if attempt == max_attempts - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"    [TIMEOUT] waiting {wait}s before retry {attempt + 2}/{max_attempts}")
            time.sleep(wait)


def local_evaluator(*, input, output, expected_output, metadata, **kwargs):
    case = input
    answer_result = {"answer": output["answer"], "skipped_llm": output["skipped_llm"]}
    outcome_results = run_assertions(case, output["chunks_seen"], answer_result)
    trajectory_results = run_trajectory_checks(case, output["trace"])
    evaluations = [
        Evaluation(name=f"assertion.{r['name']}", value=1.0 if r["passed"] else 0.0, comment=r["reason"], data_type="NUMERIC")
        for r in outcome_results
    ]
    evaluations += [
        Evaluation(name=f"trajectory.{r['name']}", value=1.0 if r["passed"] else 0.0, comment=r["reason"], data_type="NUMERIC")
        for r in trajectory_results
    ]
    return evaluations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N dataset items (cost/time control)")
    args = parser.parse_args()

    lf = Langfuse()
    dataset = lf.get_dataset(TRAJECTORY_DATASET_NAME)
    items = list(dataset.items)
    if args.limit:
        items = items[: args.limit]
    print(f"Running {len(items)} agent call(s) sequentially (outside Langfuse's scheduler -- see _run_one's docstring)...")

    precomputed = {}
    for i, item in enumerate(items):
        question = item.input["question"]
        print(f"  [{i + 1}/{len(items)}] {question[:60]}")
        precomputed[item.id] = _run_one_with_retry(question)

    def task(*, item, **kwargs):
        return precomputed[item.id]

    print("Agent calls done -- now posting to Langfuse via run_experiment()...")
    result = lf.run_experiment(
        name="ask-my-tickets-agent-trajectory",
        description="agent.py's run_agent() against the ask_my_tickets_agent_trajectory_set dataset -- outcome + trajectory checks, plus the hosted trajectory judge.",
        data=items,
        task=task,
        evaluators=[local_evaluator],
    )
    print(result.format())
    lf.flush()


if __name__ == "__main__":
    main()
