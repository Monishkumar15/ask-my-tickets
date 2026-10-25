"""
Week 6: cheap, rule-based, no-LLM-call checks -- run these before anything
that costs an LLM call (the judge, RAGAS-style metrics). Each function
takes one eval case (from eval_questions.py) and the actual result from
running the app, and returns (passed: bool, reason: str) -- reason is
always filled in, on both pass and fail, so a scorecard reads clearly
without re-running anything to find out why.
"""
import unicodedata


def check_expected_source(case, chunks):
    """The single primary source (older-style cases) must appear somewhere
    in the retrieved chunks."""
    expected = case.get("expected_source")
    if not expected:
        return None, "no expected_source on this case"
    sources = {c["source"] for c in chunks}
    if expected in sources:
        return True, f"'{expected}' present"
    return False, f"'{expected}' missing -- got {sorted(sources)}"


def check_expected_sources_all(case, chunks):
    """EVERY listed source must appear -- for multi-topic questions where
    "one of these" (expected_source) isn't strict enough."""
    expected_all = case.get("expected_sources_all")
    if not expected_all:
        return None, "no expected_sources_all on this case"
    sources = {c["source"] for c in chunks}
    missing = [s for s in expected_all if s not in sources]
    if not missing:
        return True, f"all {len(expected_all)} expected sources present"
    return False, f"missing: {missing} -- got {sorted(sources)}"


def check_forbidden_sources(case, chunks):
    """None of the listed sources may appear -- catches a specific source
    contaminating a slot that should have gone to the real answer."""
    forbidden = case.get("forbidden_sources")
    if not forbidden:
        return None, "no forbidden_sources on this case"
    sources = {c["source"] for c in chunks}
    present = [s for s in forbidden if s in sources]
    if not present:
        return True, "no forbidden sources present"
    return False, f"forbidden source(s) present: {present}"


def check_must_refuse(case, answer_result):
    """
    The correct behavior for this question is "I don't know" -- for
    questions with either no relevant evidence at all, or evidence that
    genuinely doesn't contain the asked-for fact.

    A refusal can come from either of two places, and both count:
      1) the confidence gate blocked the LLM call entirely
         (skipped_llm=True) -- nothing relevant enough was even retrieved.
      2) the LLM WAS called (something plausible-looking was retrieved)
         but it honestly declined via its own answer text -- the doc build_prompt()
         asks it to use verbatim on a genuine "not covered" case. Checking
         skipped_llm alone misses this second, equally valid case (this
         assertion originally did exactly that -- a false FAIL on a
         question the app was actually answering correctly).
    """
    if not case.get("must_refuse"):
        return None, "not a must_refuse case"
    answer_text = (answer_result.get("answer") or "").strip().lower()
    refused = answer_result.get("skipped_llm") or answer_text == "i don't know based on the available documents."
    if refused:
        return True, "correctly refused"
    return False, f"answered instead of refusing: \"{answer_result.get('answer', '')[:80]}\""


def check_must_cite_page(case, chunks):
    """At least one retrieved chunk must carry a non-null page number --
    catches a regression in PDF page-tagging."""
    if not case.get("must_cite_page"):
        return None, "not a must_cite_page case"
    pages = [c.get("page") for c in chunks]
    if any(p is not None for p in pages):
        return True, f"page(s) present: {[p for p in pages if p is not None]}"
    return False, f"no chunk carried a page number -- got pages={pages}"


# Week 11: closes the exact gap FINDINGS_week10.md Section 9 identified and
# explicitly deferred -- Groq's gpt-oss-120b writes numbers/punctuation
# using non-ASCII lookalikes (a narrow no-break space before "%", a
# non-breaking hyphen in a range like "31-60") that read identically to a
# human but fail a literal substring match. Tested directly against the
# real characters found in this project's own traces before trusting this
# fix: unicodedata.normalize("NFKC", ...) alone only gets PART of the way
# -- it correctly folds " " (narrow no-break space) to a plain space,
# but "‑" (non-breaking hyphen) only maps to "‐" (a DIFFERENT
# non-ASCII hyphen, still not "-"), and the full-width "【"/"】"
# brackets Groq sometimes wraps a citation in aren't touched by NFKC at
# all. The explicit replacement table below covers what NFKC misses,
# rather than shipping a fix that only works for the one lucky case.
_UNICODE_LOOKALIKES = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",  # hyphen/dash variants
    "‘": "'", "’": "'", "“": '"', "”": '"',  # curly quotes
}


def _normalize_for_keyword_match(text):
    normalized = unicodedata.normalize("NFKC", text)
    for unicode_char, ascii_char in _UNICODE_LOOKALIKES.items():
        normalized = normalized.replace(unicode_char, ascii_char)
    # Strip ALL whitespace, not just normalize it -- tested directly
    # against the real drill trace and found that NFKC-folding the narrow
    # no-break space to a plain space is NOT enough on its own: Groq used
    # it as "50 %" (no space intended, keyword is "50%") in that
    # exact answer, but FINDINGS_week10.md Section 9 also documented it as
    # "15 minutes" (A space IS intended, keyword is "15 minutes") --
    # there is no single space-preserving replacement correct for both.
    # Removing all whitespace from BOTH sides before comparing sidesteps
    # the ambiguity entirely, at the acceptable cost of being slightly
    # more permissive (e.g. "5 0%" would now also match "50%").
    return "".join(normalized.split())


def check_expected_keywords(case, answer_text):
    """Case-insensitive substring match -- every expected keyword must
    appear somewhere in the generated answer. Simple and honest about its
    limits (see classify_failures.py's docstring for the same caveat).
    Both the answer and each keyword are normalized the same way before
    comparing, so a keyword written with a plain ASCII space/hyphen still
    matches an answer that used a Unicode lookalike, and vice versa."""
    keywords = case.get("expected_keywords")
    if not keywords:
        return None, "no expected_keywords on this case"
    if answer_text is None:
        return False, "no answer text to check (refused or errored)"
    text_lower = _normalize_for_keyword_match(answer_text).lower()
    missing = [kw for kw in keywords if _normalize_for_keyword_match(kw).lower() not in text_lower]
    if not missing:
        return True, f"all keywords present: {keywords}"
    return False, f"missing keyword(s): {missing}"


# Every check in one place -- run_eval.py iterates this list so adding a
# new assertion type later is one line here, not a change to the runner.
ALL_CHECKS = [
    ("expected_source", check_expected_source, "chunks"),
    ("expected_sources_all", check_expected_sources_all, "chunks"),
    ("forbidden_sources", check_forbidden_sources, "chunks"),
    ("must_refuse", check_must_refuse, "answer_result"),
    ("must_cite_page", check_must_cite_page, "chunks"),
    ("expected_keywords", check_expected_keywords, "answer_text"),
]


def run_assertions(case, chunks, answer_result):
    """
    Run every applicable assertion for one case. Returns a list of
    {name, passed, reason} -- only for checks whose relevant field is
    actually present on the case (a check returns None, ... when its field
    is absent, and that's filtered out here rather than counted as a skip
    or a pass).
    """
    answer_text = answer_result.get("answer") if answer_result else None
    results = []
    for name, check_fn, arg_kind in ALL_CHECKS:
        arg = {"chunks": chunks, "answer_result": answer_result, "answer_text": answer_text}[arg_kind]
        passed, reason = check_fn(case, arg)
        if passed is None:
            continue  # this check doesn't apply to this case
        results.append({"name": name, "passed": passed, "reason": reason})
    return results
