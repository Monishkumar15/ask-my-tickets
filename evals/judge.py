"""
Week 6: LLM-as-judge for ticket-reply quality -- things a rule can't check
(tone, helpfulness, completeness), unlike assertions.py's rule-based checks
(source cited, refusal correct, etc.).

Design choices, explained rather than just implemented:

  Binary vs 1-10 scoring: this uses a BINARY verdict (GOOD/POOR), not a
  1-10 scale. A 1-10 scale gives more nuance, but it's much harder to
  validate against human agreement -- two graders (a person and an LLM,
  or two people) rarely land on the exact same number out of 10 for a
  subjective quality, even when they broadly agree the answer is "pretty
  good". Binary agreement is a far more reliable signal to validate
  against (validate_judge.py) -- which is the actual deliverable this
  project's Track A asks for: validate the judge before trusting its
  number, and a number you can't reliably validate isn't one worth having.

  G-Eval style, not a bare "rate this 1-10": the judge is given an
  explicit rubric and required to give a short reason alongside its
  verdict (closer to chain-of-thought than a silent number). A wrong
  verdict is debuggable -- you can read WHY it said POOR -- instead of a
  black box.

  The judge is given the retrieved context, not just the question and
  answer. The original version judged (question, answer_text) alone --
  and validate_judge.py caught it giving a confidently WRONG, unstable
  verdict on a refusal-correctness case (asserting a fact "can be
  answered from the guide" that a direct read of the full PDF text had
  already confirmed isn't there, flip-flopping between repeated calls on
  the exact same input). Without the actual source material, the judge
  can only assess surface quality (tone, completeness relative to the
  question's own wording) -- it structurally cannot check whether a
  refusal was warranted or a claim is accurate, because it has nothing to
  check them against. Passing context fixes that at the root, rather than
  patching the rubric wording around a blind spot.
"""
import json

from generate import call_llm

RUBRIC = """You are grading a customer support assistant's reply for quality,
using the CONTEXT below as the source of truth for what's actually true --
not your own general knowledge.

Give a verdict of GOOD or POOR, using this rubric:

GOOD: the reply correctly answers what the customer actually asked, using
ONLY facts the context actually supports (numbers, conditions, timeframes --
not vague generalities, and nothing invented or overclaimed beyond what the
context says), and reads like a real support agent wrote it (clear,
professional, not robotic or overly formal, and never leaking internal
agent-only instructions into a customer-facing reply). An HONEST refusal
("I don't know based on the available documents", or a message saying no
relevant information was found) is ALSO GOOD if the CONTEXT genuinely does
not contain the answer -- check the context yourself before accepting or
rejecting a refusal; refusing correctly is the right outcome, not a
failure, and must never be marked POOR just for not providing an answer.

POOR: the reply is missing a fact the context actually contains, states
something the context does NOT support (an overclaim or a misreading -- e.g.
turning a rhetorical or open question in the context into a stated fact),
contradicts the context, ignores part of a multi-part question, reads oddly
(too terse to be useful, or padded with irrelevant detail), leaks internal
instructions meant for an agent rather than the customer, OR refuses when
the context actually does contain the answer.

Respond with ONLY a JSON object, no other text:
{"reason": "<one sentence explaining your verdict, citing the context if relevant>", "verdict": "GOOD" or "POOR"}
"""


def _strip_code_fence(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


def _call_judge(question, answer_text, context):
    # Reuses generate.py's call_llm() -- same Gemini-primary/Groq-fallback-
    # on-429 logic the rest of the app already relies on. A raw
    # Gemini-only request here (the original implementation) meant a
    # transient Gemini rate-limit silently failed the judge instead of
    # falling back, exactly the failure this project already solved
    # everywhere else it calls an LLM.
    prompt = (
        f"{RUBRIC}\n\nContext (the retrieved source material):\n{context}\n\n"
        f"Customer question: {question}\n\nAssistant's reply: {answer_text}\n\nJSON:"
    )
    content, _provider_used = call_llm(prompt, temperature=0)  # judging should be as deterministic as possible
    parsed = json.loads(_strip_code_fence(content))
    return parsed["verdict"].strip().upper(), parsed["reason"].strip()


def judge_answer(question, answer_text, context):
    """
    context: the retrieved source text the answer was (or should have
    been) grounded in -- e.g. "\n\n".join(c["text"] for c in chunks).
    Pass "(no context retrieved)" for a question where nothing came back,
    so the judge can correctly treat a refusal as warranted.

    Returns {"verdict": "GOOD"|"POOR"|"ERROR", "reason": str}. Never
    raises -- a judge failure is itself useful information (shows up as
    ERROR in a scorecard rather than crashing the whole eval run).
    """
    try:
        verdict, reason = _call_judge(question, answer_text, context)
        if verdict not in ("GOOD", "POOR"):
            return {"verdict": "ERROR", "reason": f"unexpected verdict from judge: {verdict!r}"}
        return {"verdict": verdict, "reason": reason}
    except Exception as e:
        return {"verdict": "ERROR", "reason": f"judge call failed: {e}"}
