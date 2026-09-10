"""
Quick utility: print the most recent trace's step-by-step thoughts/actions --
whichever ran most recently, agent or fixed pipeline. Run with:

    python evals/show_last_trace.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "traces")


def main():
    files = sorted(os.listdir(TRACE_DIR), key=lambda f: os.path.getmtime(os.path.join(TRACE_DIR, f)), reverse=True)
    if not files:
        print("No traces found yet -- run agent.py or ask a question first.")
        return

    with open(os.path.join(TRACE_DIR, files[0]), encoding="utf-8") as f:
        trace = json.load(f)

    print("Question:", trace["question"])
    print("Outcome:", trace.get("outcome"))
    print()
    for stage in trace["stages"]:
        d = stage["data"]
        print(f"--- {stage['name']} (at {stage['elapsed_ms']}ms) ---")
        if "thought" in d:
            print("  thought:", d.get("thought"))
            print("  action:", d.get("action"), d.get("action_input"))
        else:
            print(" ", d)
        print()


if __name__ == "__main__":
    main()
