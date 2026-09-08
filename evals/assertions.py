"""
Week 6: cheap, rule-based, no-LLM-call checks -- run these before anything
that costs an LLM call (the judge, RAGAS-style metrics). Each function
takes one eval case (from eval_questions.py) and the actual result from
running the app, and returns (passed: bool, reason: str) -- reason is
always filled in, on both pass and fail, so a scorecard reads clearly
without re-running anything to find out why.
"""


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


def check_expected_keywords(case, answer_text):
    """Case-insensitive substring match -- every expected keyword must
    appear somewhere in the generated answer. Simple and honest about its
    limits (see classify_failures.py's docstring for the same caveat)."""
    keywords = case.get("expected_keywords")
    if not keywords:
        return None, "no expected_keywords on this case"
    if answer_text is None:
        return False, "no answer text to check (refused or errored)"
    text_lower = answer_text.lower()
    missing = [kw for kw in keywords if kw.lower() not in text_lower]
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
