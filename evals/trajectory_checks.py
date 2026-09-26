"""
Week 8: cheap, rule-based checks over an agent run's TRAJECTORY (the
step-by-step thought/action/observation history in its trace) -- not its
final answer. Same shape and convention as assertions.py: each check takes
one eval case plus the agent's full trace dict, and returns
(passed: bool | None, reason: str). None means "doesn't apply to this
case," filtered out by run_trajectory_checks() the same way
run_assertions() already does.

Why this file exists, separate from assertions.py: an agent can pass every
assertion in assertions.py (right final answer, right sources cited) while
still having taken a wrong, inefficient, or silently-incomplete path to get
there -- exactly the "outcome vs trajectory gap" this week is about. These
checks read the trace's stages, not the returned answer/chunks.
"""


def _agent_steps(trace):
    """Every 'agent_step' stage's data dict, in execution order."""
    return [s["data"] for s in trace.get("stages", []) if s["name"] == "agent_step"]


def check_terminal_action(case, trace):
    """
    Tool-choice accuracy: does the trace's actual terminal action
    (finish vs escalate) match what the case says it SHOULD be, grounded
    in the ticket data (see eval_questions.py's expected_terminal_action
    comment for exactly which ticket text justifies each case's value)?
    """
    expected = case.get("expected_terminal_action")
    if not expected:
        return None, "no expected_terminal_action on this case"

    outcome = trace.get("outcome")
    actual = {"agent_answer": "finish", "agent_escalated": "escalate"}.get(outcome)
    if actual is None:
        return False, f"agent never reached a terminal action (outcome={outcome})"
    if actual == expected:
        return True, f"correctly used '{actual}'"
    return False, f"expected '{expected}', agent used '{actual}'"


def check_no_repeated_search(trace):
    """
    Loop failure mode: two search_tickets steps with the same
    (normalized) query gain no new information -- the agent is circling,
    not progressing. Applies to every agent trace, no case field needed.
    """
    queries = [
        step["action_input"].get("query", "").strip().lower()
        for step in _agent_steps(trace)
        if step.get("action") == "search_tickets"
    ]
    if len(queries) < 2:
        return None, "fewer than 2 searches -- nothing to repeat"

    seen = set()
    for q in queries:
        if q and q in seen:
            return False, f"repeated an identical search query: \"{q}\""
        seen.add(q)
    return True, f"no repeated queries across {len(queries)} searches"


def check_no_wasted_steps(trace):
    """
    A distinct loop/stuck failure mode from check_no_repeated_search: a
    step where the model's action was unrecognized or its JSON malformed
    (agent.py's _parse_action() self-correction path) burns a step of the
    budget without making any progress. Found live: a case where 3
    consecutive unrecognized-action steps used 4 of 5 available steps
    before the agent finally recovered -- the run still finished with a
    correct answer, but the path came within one step of exhausting the
    budget on nothing. Structural checks that only look at search_tickets
    steps (check_no_repeated_search, check_step_efficiency) can't see this
    at all -- this is what actually catches it.
    """
    wasted = sum(
        1 for step in _agent_steps(trace)
        if step.get("error") in ("unrecognized_action",) or step.get("action") == "_parse_error"
    )
    if wasted == 0:
        return True, "no wasted (unrecognized/malformed) steps"
    return False, f"{wasted} step(s) burned on an unrecognized or malformed action"


def check_not_silently_incomplete(trace):
    """
    "Gives up quietly": the loop ran out of steps/time (agent_stopped)
    instead of reaching finish/escalate. The partial answer text may still
    look plausible -- this check flags it regardless of wording, since the
    agent never actually confirmed it was done.
    """
    outcome = trace.get("outcome")
    if outcome != "agent_budget_exhausted":
        return None, "did not exit via budget exhaustion"
    return False, "agent ran out of steps/time without reaching finish or escalate"


def check_step_efficiency(case, trace):
    """
    Lightweight expected-tool-sequence proxy: for a compound question with
    N known distinct topics (expected_sources_all), the agent shouldn't
    need dramatically more than N search steps (+1 slack for a genuine
    re-search) to cover them. Not an exact ordered-sequence match --  real
    agent paths legitimately vary -- just a bound on gross inefficiency.
    """
    expected_sources = case.get("expected_sources_all")
    if not expected_sources:
        return None, "no expected_sources_all on this case"

    search_steps = sum(1 for step in _agent_steps(trace) if step.get("action") == "search_tickets")
    budget = len(expected_sources) + 1
    if search_steps <= budget:
        return True, f"{search_steps} search step(s) for {len(expected_sources)} expected topic(s)"
    return False, (
        f"{search_steps} search step(s) for only {len(expected_sources)} expected "
        f"topic(s) -- expected at most {budget}"
    )


# Every check in one place -- same "one list the runner iterates" pattern
# as assertions.py's ALL_CHECKS, so adding a new trajectory check later is
# one line here, not a change to trajectory_eval.py.
ALL_CHECKS = [
    ("terminal_action", check_terminal_action, "case_and_trace"),
    ("no_repeated_search", check_no_repeated_search, "trace_only"),
    ("no_wasted_steps", check_no_wasted_steps, "trace_only"),
    ("not_silently_incomplete", check_not_silently_incomplete, "trace_only"),
    ("step_efficiency", check_step_efficiency, "case_and_trace"),
]


def run_trajectory_checks(case, trace):
    """
    Run every applicable trajectory check for one agent run. Returns a
    list of {name, passed, reason} -- only for checks whose relevant field
    or condition actually applies (a check returning None, ... is filtered
    out, same convention as assertions.run_assertions()).
    """
    results = []
    for name, check_fn, arg_kind in ALL_CHECKS:
        passed, reason = check_fn(case, trace) if arg_kind == "case_and_trace" else check_fn(trace)
        if passed is None:
            continue
        results.append({"name": name, "passed": passed, "reason": reason})
    return results
