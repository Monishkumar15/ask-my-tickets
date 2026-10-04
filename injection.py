"""
Week 8: code-level defence for indirect prompt injection.

Before this module, the only protection was a prompt asking the model to
treat retrieved text as data. That is a request, not a guarantee: nothing in
code ever looked at retrieved text, and a chunk containing the closing
delimiter (<<<RETRIEVED_DATA_END>>>) could close the trust boundary early.

What this module does:
    scan(text)       find instruction-shaped spans (rule name, line, span)
    neutralize(text) replace each span with a visible "[neutralized: rule]"
                     marker. Never silent, never empty, never a refusal.
    strip_delimiters remove any copy of our own boundary markers from
                     untrusted text, so a document cannot forge the boundary.

Posture: DEGRADE, NEVER REFUSE. A false positive costs one sentence of
fidelity and leaves a visible scar in the trace. Blocking on detection would
make any legitimate document that quotes an attack (a security-awareness
note, a ticket transcript) unanswerable.

What a regex cannot do (see FINDINGS_week8.md, "Residual risk"):
    * instructions phrased as plain content ("the correct answer is X")
    * a forged label for a real document, once the surrounding prose is clean
    * paraphrase with no trigger words, non-English text, base64/ROT13
    * an imperative split across a chunk boundary

Leaf module: `re`, `dataclasses` and config only, so importing it never pulls
in the pipeline.
"""
import re
from dataclasses import dataclass

import config

BEGIN_DATA = "<<<RETRIEVED_DATA_START>>>"
END_DATA = "<<<RETRIEVED_DATA_END>>>"

# Runtime switch. Defaults from config (INJECTION_DEFENCES env var). The eval
# harness flips it to run the same code with the defence off, so a before/after
# comparison is measured on identical code instead of remembered across commits.
_enabled = config.INJECTION_DEFENCES


def enabled():
    return _enabled


def set_enabled(value):
    global _enabled
    _enabled = bool(value)


# (rule, pattern). Rules are deliberately narrow: they target the lexically
# marked imperative that published payloads overwhelmingly use.
_RULES = (
    (
        "imperative-override",
        re.compile(
            r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier|the\s+above)"
            r"\s+(?:instructions?|rules?|prompts?|directions?)"
            r"|ignore\s+(?:the\s+)?(?:customer'?s?\s+|user'?s?\s+)?(?:actual\s+|real\s+|original\s+)"
            r"(?:question|request)",
            re.I,
        ),
    ),
    (
        "role-reassign",
        re.compile(
            r"(?:you\s+are\s+now\b|new\s+system\s+(?:prompt|instructions?)\b"
            r"|\bsystem\s+(?:override|notice|update)\s*[:\-]|^\s*system\s*[:\-]"
            r"|assistant\s+configuration|\bany\s+(?:agent|assistant|ai)\s+(?:reading|processing)\s+this)",
            re.I | re.M,
        ),
    ),
    (
        "citation-subversion",
        re.compile(
            r"(?:do\s+not|don'?t|must\s+not|never)\s+(?:cite|list|include\s+(?:a\s+|the\s+|any\s+)?"
            r"(?:citations?|sources?))",
            re.I,
        ),
    ),
    (
        "refusal-injection",
        re.compile(
            r"(?:must|always|you\s+will)\s+(?:reply|respond|answer)\s+(?:to\s+\w+\s+)?with\s+"
            r"(?:exactly|only|just)\b",
            re.I,
        ),
    ),
    (
        "exfiltration",
        re.compile(
            r"(?:append|include|print|output|reveal|return)\b[^.\n]{0,60}"
            r"\b(?:contents?|full\s+text|body)\b[^.\n]{0,60}\b(?:every|all|each|other)\b"
            r"|(?:include|append|print|output|reveal|return)\s+(?:the\s+)?(?:full|entire)\s+"
            r"(?:text|contents?)\s+of\s+(?:that|this|the|those|these|any|every|each)\b",
            re.I,
        ),
    ),
    (
        "output-hijack",
        re.compile(r"\b(?:call\s+finish|finish|reply|respond|answer)\s+with\s+exactly\b", re.I),
    ),
    (
        "source-injection",
        re.compile(
            r"(?:always\s+)?(?:list|add|include|cite)\s+['\"]?[\w.\-]+['\"]?\s+in\s+your\s+sources",
            re.I,
        ),
    ),
    (
        "tool-command",
        re.compile(
            r"\b(?:call|use|run|invoke|search\s+for)\s+(?:the\s+)?"
            r"(?:search_tickets|list_ticket_sources)\b",
            re.I,
        ),
    ),
    (
        "prompt-leak",
        re.compile(
            r"(?:repeat|print|reveal|show|disclose)\s+(?:your|the)\s+"
            r"(?:system\s+)?(?:prompt|instructions?)",
            re.I,
        ),
    ),
    (
        "react-frame",
        re.compile(
            r"^\s*(?:Observation|Action\s+Input|Action|Thought|Step\s+\d+)\s*:",
            re.I | re.M,
        ),
    ),
    (
        "label-forgery",
        re.compile(r"^\s*(?:\[Source:[^\]\n]{1,80}\]|\[[A-Za-z0-9][A-Za-z0-9._\-]*\])\s*$", re.M),
    ),
)

_DELIMITER_RE = re.compile(re.escape(BEGIN_DATA) + "|" + re.escape(END_DATA))


@dataclass(frozen=True)
class Detection:
    rule: str
    line: int  # 1-indexed within the scanned text
    span: str  # the literal match, clipped

    def describe(self):
        return f"{self.rule} at line {self.line}: {self.span!r}"


def _line_of(text, index):
    return text.count("\n", 0, index) + 1


def scan(text):
    """Return a list of Detection for every instruction-shaped span in text."""
    text = text or ""
    found = []
    for rule, pattern in _RULES:
        for match in pattern.finditer(text):
            found.append(Detection(rule, _line_of(text, match.start()), match.group(0).strip()[:80]))
    for match in _DELIMITER_RE.finditer(text):
        found.append(Detection("delimiter-forgery", _line_of(text, match.start()), match.group(0)))
    found.sort(key=lambda d: (d.line, d.rule))
    return found


def strip_delimiters(text):
    """Remove our own boundary markers from untrusted text, visibly."""
    return _DELIMITER_RE.sub("[neutralized: delimiter-forgery]", text or "")


def neutralize(text):
    """Return (cleaned_text, detections).

    Every matched span becomes "[neutralized: <rule>]". The result is never
    empty: if every line matched, a loud placeholder is returned instead, so
    the model is never reasoning over a silent hole.

    A no-op (text unchanged, no detections) when the defence is switched off,
    so the same call site serves both arms of an A/B run.
    """
    text = text or ""
    if not _enabled:
        return text, []

    detections = scan(text)
    if not detections:
        return text, []

    cleaned = strip_delimiters(text)
    for rule, pattern in _RULES:
        cleaned = pattern.sub(f"[neutralized: {rule}]", cleaned)

    if not cleaned.strip():
        rules = ", ".join(dict.fromkeys(d.rule for d in detections))
        cleaned = f"[neutralized: this text was entirely instruction-like ({rules}); withheld]"
    return cleaned, detections


def neutralize_chunk_text(text):
    """Convenience for call sites that only want the cleaned string."""
    return neutralize(text)[0]


def report_document(text, max_examples=5):
    """Scan a whole document (e.g. at upload) and summarize what looks like an
    instruction. Detection only: nothing is changed or blocked, because the
    upload path cannot tell a poisoned file from a legitimate one that quotes an
    attack. Independent of the on/off switch, since it never alters text.
    """
    detections = scan(text)
    return {
        "flagged": bool(detections),
        "count": len(detections),
        "rules": list(dict.fromkeys(d.rule for d in detections)),
        "examples": [d.describe() for d in detections[:max_examples]],
    }


def summarize(detections):
    """Compact, JSON-friendly summary for a trace stage."""
    return {
        "count": len(detections),
        "rules": list(dict.fromkeys(d.rule for d in detections)),
        "spans": [d.describe() for d in detections[:10]],
    }
