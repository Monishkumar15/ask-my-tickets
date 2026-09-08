"""
Week 4: labeled eval set for debugging retrieval.
Week 6: extended with regression cases from real bugs found and fixed
during Week 3-5 pipeline work (see evals/assertions.py for how the extra
fields below get checked).

Each question has an expected_source (the ticket file that SHOULD be
retrieved) and expected_keywords (used to auto-grade whether a generated
answer actually contains the right fact, without manual judgment).

Bucket comments below are HYPOTHESES to explain why each question was
chosen -- not confirmed outcomes. Only running classify_failures.py tells
you which bucket a question actually lands in.

Week 6 regression cases (below) use extra optional fields that older
scripts (classify_failures.py, compare_before_after.py, inspect_retrieval.py)
simply ignore -- they only ever read "id"/"question"/"expected_source"/
"expected_keywords", so adding these cases here doesn't touch their behavior:

    problem_type          -- which Error Analysis bucket this regression
                             guards against (retrieval_failure,
                             generation_failure, refusal_misfire)
    expected_sources_all  -- every source that MUST appear (multi-topic
                             questions -- expected_source alone can't
                             express "both of these", only "one of these")
    forbidden_sources     -- sources that must NOT appear (catches a
                             specific source contaminating the answer)
    must_refuse           -- True means the CORRECT behavior is a refusal
                             (skipped_llm=True) -- these questions exist to
                             catch a false answer, not a missing one
    must_cite_page        -- True means at least one retrieved chunk must
                             carry a non-null "page" (PDF page citation)
"""

EVAL_QUESTIONS = [
    # --- Bucket A: expected to clearly pass (semantic search + generation both fine) ---
    {"id": "q1", "question": "why did my account get locked when I typed my password wrong a few times?",
     "expected_source": "ticket_001_password_lockout.txt", "expected_keywords": ["15 minutes"]},
    {"id": "q2", "question": "my order still hasn't shown up, it's way past when it should have",
     "expected_source": "ticket_003_shipping_delay.txt", "expected_keywords": ["7 business days"]},

    # --- Bucket B: hypothesized RETRIEVAL failures (odd phrasing, or genuinely
    #     confusable with another document that shares vocabulary) ---
    {"id": "q3", "question": "3 attempts and now I'm locked out, is that normal?",
     "expected_source": "ticket_001_password_lockout.txt", "expected_keywords": ["15 minutes"]},
    {"id": "q4", "question": "I want to go to Account Settings, Subscription, then Cancel Plan -- what happens after that?",
     "expected_source": "ticket_004_account_cancellation.txt", "expected_keywords": ["90 days"]},
    {"id": "q5", "question": "there are two $29.99 charges on my statement from the same day",
     "expected_source": "ticket_005_billing_dispute.txt", "expected_keywords": ["reversed"]},
    {"id": "q6", "question": "it's been 45 days since I bought it, can I still get money back?",
     "expected_source": "ticket_002_refund_window.txt", "expected_keywords": ["50%"]},
    {"id": "q7", "question": "carrier says lost in transit, what do we do for the customer?",
     "expected_source": "ticket_003_shipping_delay.txt", "expected_keywords": ["reshipment"]},
    {"id": "q11", "question": "the item I got was damaged, will you refund me",
     "expected_source": "ticket_006_defective_item_refund.txt", "expected_keywords": ["90 days"]},
    {"id": "q12", "question": "you guys charged me the wrong price",
     "expected_source": "ticket_007_incorrect_charge_amount.txt", "expected_keywords": ["2 business days"]},

    # --- Bucket C: hypothesized GENERATION failures (right doc should be
    #     easy to retrieve, but the answer needs an exact fact an LLM might
    #     paraphrase, round, or drop) ---
    {"id": "q8", "question": "exactly how many days after cancelling is my data deleted?",
     "expected_source": "ticket_004_account_cancellation.txt", "expected_keywords": ["90"]},
    {"id": "q9", "question": "what percentage refund do I get if I ask between 31 and 60 days after buying?",
     "expected_source": "ticket_002_refund_window.txt", "expected_keywords": ["50%"]},
    {"id": "q10", "question": "how many business days do you wait before filing a lost-in-transit claim?",
     "expected_source": "ticket_003_shipping_delay.txt", "expected_keywords": ["3 days"]},

    # --- Bucket D: short, telegraphic, code/number-style queries -- semantic
    #     embeddings are built for full sentences, so short queries built
    #     from bare exact numbers are a genuine stress test, especially
    #     where the SAME number appears in two different tickets (90 days:
    #     ticket_004 AND ticket_006; $29.99: ticket_005 AND ticket_007) ---
    {"id": "q13", "question": "90 days delete data",
     "expected_source": "ticket_004_account_cancellation.txt", "expected_keywords": ["90"]},
    {"id": "q14", "question": "29.99 x2 reversed",
     "expected_source": "ticket_005_billing_dispute.txt", "expected_keywords": ["reversed"]},
    {"id": "q15", "question": "3 attempts locked 15 min",
     "expected_source": "ticket_001_password_lockout.txt", "expected_keywords": ["15 minutes"]},
]

# --- Week 6 regression cases: each one guards against a real bug found and
# fixed this session. If any of these ever fail again, a fix elsewhere
# silently reintroduced a problem that was already solved once. ---
REGRESSION_CASES = [
    # _diversify_by_source's leftover-slot tie-break used raw distance
    # instead of rerank_score, letting an unrelated ticket_003 shipping
    # chunk win a slot over the actual missing fact (ticket_004 chunk 2,
    # "data retained for 90 days") by a 0.007 distance gap.
    {"id": "r1", "problem_type": "retrieval_failure",
     "question": "how does customer support ticketing work and what happens to my data if I cancel my subscription",
     "expected_sources_all": ["ticket_004_account_cancellation.txt", "customer support.pdf"],
     "forbidden_sources": ["ticket_003_shipping_delay.txt"],
     "expected_keywords": ["90 days"]},

    # A question spanning MORE distinct topics than top_k allowed used to
    # drop the excess topic entirely (ticket_003) rather than truncate it --
    # fixed by MAX_DIVERSIFY_SOURCES widening the slot budget.
    {"id": "r2", "problem_type": "retrieval_failure",
     "question": "why is my order delayed, how do I get a refund for a defective item, and why was I charged the wrong amount",
     "expected_sources_all": [
         "ticket_006_defective_item_refund.txt",
         "ticket_003_shipping_delay.txt",
         "ticket_007_incorrect_charge_amount.txt",
     ],
     "forbidden_sources": ["ticket_002_refund_window.txt", "ticket_005_billing_dispute.txt"],
     "expected_keywords": ["90 days"]},

    # The confidence gate must still refuse a genuinely out-of-scope
    # question -- this is the "must be correctly refused" counterpart to
    # every "must be correctly answered" case above.
    {"id": "r3", "problem_type": "refusal_misfire",
     "question": "what's the best pizza topping?",
     "must_refuse": True},

    # A fact that genuinely isn't in the documents (verified by reading the
    # full PDF text) must be refused honestly, not hallucinated.
    {"id": "r4", "problem_type": "refusal_misfire",
     "question": "What is the recommended management ratio for customer support teams according to the guide?",
     "must_refuse": True},

    # PDF chunks must carry a page number (the pypdf -> PyMuPDF /
    # page-aware-chunking fix) -- a .txt-only regression here would mean
    # every PDF citation silently lost its page again.
    {"id": "r5", "problem_type": "retrieval_failure",
     "question": "how do I contact customer support",
     "expected_source": "customer support.pdf",
     "must_cite_page": True},
]
