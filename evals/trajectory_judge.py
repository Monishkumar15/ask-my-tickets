"""
Week 8: LLM-as-judge for an agent run's TRAJECTORY -- not its final answer
(that's judge.py's job), and not the structural checks in
trajectory_checks.py (query repeats, terminal action, step counts). This
judge reads the actual thought/action/observation sequence and asks
whether the REASONING that produced the answer was sound -- the thing a
structural rule can't check, and specifically the mechanism for catching a
right-looking answer reached by a wrong or lucky path (a genuine
outcome-vs-trajectory gap a rule-based check would miss entirely).

Same design choices as judge.py, for the same reasons (see that file's
docstring): binary GOOD/POOR (not 1-10, so it can actually be validated
against human agreement -- see validate_trajectory_judge.py), G-Eval
style (explicit rubric + required one-sentence reason, not a bare
verdict), and it reuses generate.py's call_llm() for the same
Gemini-primary/Groq-fallback-on-429 behavior as the rest of this project.

This judge is NOT trusted until validate_trajectory_judge.py confirms it
agrees with hand-grading -- same discipline judge.py went through in Week 6
before run_eval.py's --with-judge treated its scores as meaningful.
"""
import json

from generate import call_llm

RUBRIC = """You are grading whether a customer-support AGENT'S REASONING PROCESS was
sound -- NOT whether its final answer merely reads correctly. You are given
the full step-by-step thought/action/observation trail it produced, plus its
final answer. Grade the PATH, not just the destination.

Give a verdict of GOOD or POOR, using this rubric:

GOOD: every search targeted something the customer actually asked about (not
a tangent), each "thought" correctly reflects what its own observation
actually said (no misreading, no ignoring evidence that contradicts the
eventual answer, no inventing a detail no observation supports), and the
final action (finish or escalate) is one the gathered evidence genuinely
justifies. A GOOD trajectory can still take extra steps or phrase things
imperfectly -- that's not what you're grading.

POOR: the reasoning searched for the wrong thing yet still concluded as if it
had covered the question; a thought misreads or contradicts its own
observation; the final answer states something no observation actually
supports (even if it happens to sound plausible or matches a fact by luck);
a genuinely distinct part of a multi-part question was never searched for at
all; or escalate/finish was chosen for a reason the evidence doesn't
actually establish. This is exactly the case of "the final answer looks
right, but the path that produced it was wrong" -- grade THAT as POOR even
when the final answer text reads fine on its own.

Respond with ONLY a JSON object, no other text:
{"reason": "<one sentence pointing at the specific step, if any, where the reasoning broke down>", "verdict": "GOOD" or "POOR"}
"""


def _strip_code_fence(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


def render_trajectory(trace):
    """
    Turn a trace's agent_step/agent_stopped stages into the same
    readable thought/action/observation text show_last_trace.py already
    prints for a human -- reused here as the judge's input, so both a
    human reviewer and this judge are reading the identical rendering.
    """
    lines = []
    for stage in trace.get("stages", []):
        d = stage["data"]
        if stage["name"] == "agent_step":
            lines.append(f"Step {d.get('step')}: thought=\"{d.get('thought')}\" "
                          f"action={d.get('action')}({json.dumps(d.get('action_input'))})")
            if "observation" in d:
                # The actual retrieved text the agent saw -- required so the
                # judge can check whether "thought" reflects this correctly,
                # not just count how many results came back.
                lines.append(f"  -> observation:\n{d['observation']}")
        elif stage["name"] == "agent_stopped":
            lines.append(f"[STOPPED: {d.get('reason')}]")
    return "\n".join(lines) if lines else "(no steps recorded)"


def _call_judge(question, trajectory_text, answer_text):
    prompt = (
        f"{RUBRIC}\n\nCustomer question: {question}\n\n"
        f"Agent's step-by-step trail:\n{trajectory_text}\n\n"
        f"Agent's final answer: {answer_text}\n\nJSON:"
    )
    content, _provider_used = call_llm(prompt, temperature=0)  # judging should be as deterministic as possible
    parsed = json.loads(_strip_code_fence(content))
    return parsed["verdict"].strip().upper(), parsed["reason"].strip()


def judge_trajectory(question, trace, answer_text):
    """
    trace: the full trace dict (tracing.get(trace_id)) for one agent run.
    answer_text: the agent's final answer.

    Returns {"verdict": "GOOD"|"POOR"|"ERROR", "reason": str}. Never
    raises -- a judge failure is itself useful information, same
    convention as judge.judge_answer().
    """
    trajectory_text = render_trajectory(trace)
    try:
        verdict, reason = _call_judge(question, trajectory_text, answer_text)
        if verdict not in ("GOOD", "POOR"):
            return {"verdict": "ERROR", "reason": f"unexpected verdict from judge: {verdict!r}"}
        return {"verdict": verdict, "reason": reason}
    except Exception as e:
        return {"verdict": "ERROR", "reason": f"judge call failed: {e}"}
