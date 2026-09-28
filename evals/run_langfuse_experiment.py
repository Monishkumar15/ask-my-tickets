"""
Runs this project's real pipeline (retrieve() + answer_question(), the same
functions the live app calls) against the Langfuse dataset created by
langfuse_eval_setup.py, via the SDK's run_experiment().

Two scoring layers happen automatically, matching the reference project's
architecture but using OUR OWN logic:
  1. Local, free, rule-based checks (evals/assertions.py's run_assertions())
     run in-process as an `evaluators=[...]` function -- no LLM call, posts
     a Score per assertion straight to Langfuse (run_experiment() calls
     self.create_score() per Evaluation returned).
  2. The hosted `ticket-reply-judge` evaluator (langfuse_eval_setup.py) then
     runs automatically server-side, triggered by the evaluation rule that
     matches this dataset's experiment-root-spans -- no code here calls it
     directly.

Usage: python evals/run_langfuse_experiment.py [--limit N]
"""
import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langfuse import Evaluation, Langfuse  # noqa: E402

import config  # noqa: E402
from evals.assertions import run_assertions  # noqa: E402
from evals.langfuse_eval_setup import DATASET_NAME  # noqa: E402
from generate import answer_question  # noqa: E402
from retrieval import retrieve  # noqa: E402


def task(*, item, **kwargs):
    case = item.input
    question = case["question"]
    chunks = retrieve(question)
    answer_result = answer_question(question)
    context = "\n\n".join(c["text"] for c in chunks) if chunks else "(no context retrieved)"
    return {
        "answer": answer_result["answer"],
        "context": context,
        "sources": answer_result["sources"],
        "skipped_llm": answer_result["skipped_llm"],
        "chunks": chunks,
    }


def assertions_evaluator(*, input, output, expected_output, metadata, **kwargs):
    case = input
    answer_result = {"answer": output["answer"], "skipped_llm": output["skipped_llm"]}
    results = run_assertions(case, output["chunks"], answer_result)
    return [
        Evaluation(name=f"assertion.{r['name']}", value=1.0 if r["passed"] else 0.0, comment=r["reason"], data_type="NUMERIC")
        for r in results
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N dataset items (cost control)")
    args = parser.parse_args()

    lf = Langfuse()
    dataset = lf.get_dataset(DATASET_NAME)
    items = list(dataset.items)
    if args.limit:
        items = items[: args.limit]
    print(f"Running experiment on {len(items)} dataset item(s)...")

    result = lf.run_experiment(
        name="ask-my-tickets-fixed-pipeline",
        description="Fixed pipeline (retrieve() + answer_question()) against the ask_my_tickets_eval_set dataset.",
        data=items,
        task=task,
        evaluators=[assertions_evaluator],
    )
    print(result.format())
    lf.flush()


if __name__ == "__main__":
    main()
