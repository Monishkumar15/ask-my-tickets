"""
Week 4: labeled eval set for debugging retrieval.

Each question has an expected_source (the ticket file that SHOULD be
retrieved) and expected_keywords (used to auto-grade whether a generated
answer actually contains the right fact, without manual judgment).

Bucket comments below are HYPOTHESES to explain why each question was
chosen -- not confirmed outcomes. Only running classify_failures.py tells
you which bucket a question actually lands in.
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
