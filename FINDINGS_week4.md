# Findings — Week 4: Debugging Retrieval

**Branch:** `week4-debugging-retrieval` (cut from `week3-review-changes`)
**Track:** A — Customer Support Tickets (continuing the same project as Week 3)

## 1. Setup

- Added 2 new tickets to `data/` specifically to create **honest** ambiguity
  for retrieval to fail on (our original 5 tickets are topically distinct
  enough that semantic search was already perfect against them):
  - `ticket_006_defective_item_refund.txt` — shares "refund"/"days"
    vocabulary with `ticket_002`, but the actual policy is different
    (always full refund if defective, regardless of purchase date).
  - `ticket_007_incorrect_charge_amount.txt` — shares "$29.99"/"charged"
    vocabulary with `ticket_005`, but the resolution is different
    (manual correction, not auto-reversal).
- Built a 15-question labeled eval set (`eval_questions.py`), split across
  4 buckets by *hypothesis* (only running the scripts confirms real
  outcomes): straightforward questions, confusable-topic questions, exact-fact
  questions, and 3 deliberately short/telegraphic "code-style" questions
  (`q13`-`q15`) meant to stress-test semantic search's known weak spot.
- **The one improvement made this week:** `hybrid.py` + `bm25_search.py` —
  BM25 keyword search fused with the existing semantic search via
  Reciprocal Rank Fusion (RRF). `retrieve.py`, `generate.py`, and
  `DISTANCE_THRESHOLD` (0.75) were **not touched** at all — the existing
  pure-semantic pipeline still exists unchanged, so it could be compared
  directly against the new method.

## 2. Check 1 — retrieval failure vs. generation failure, with evidence

Ran all 15 questions through `classify_failures.py` (baseline `retrieve.py`,
unmodified). Every question found its correct document somewhere in the
top-3 (0 outright "wrong document fetched" misses at k=3) — **but one
question's ranking was degraded**, which is the real evidence this section
is about:

| ID | Question | Expected | Top-1 retrieved | Rank of correct doc |
|---|---|---|---|---|
| q14 | "29.99 x2 reversed" | `ticket_005_billing_dispute.txt` | `ticket_007_incorrect_charge_amount.txt` | **2** (not 1) |
| q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q15 | (14 other questions) | — | matches expected | 1 |

**q14 is our diagnosed failure, and here's the evidence for which kind it is:**
it is **not** a retrieval failure by the strict "wrong document fetched"
definition (the correct document, `ticket_005`, does appear — just at rank
2 instead of rank 1). It's a **retrieval quality/ranking issue**: the
correct answer was found, but not ranked first, so a `top_k=1` app or a UI
that only shows the #1 result would present the wrong document to the user.

**Root cause, confirmed directly:**
```
Semantic only: ticket_007 (rank 1), ticket_005 (rank 2)
BM25 only:     ticket_007 (score 5.93), ticket_005 (score 4.30, split across 2 chunks)
Hybrid (RRF):  ticket_007 (rank 1), ticket_005 (rank 2) -- unchanged
```
`ticket_007`'s single chunk happens to contain **both** the dollar amount
*and* the word "reversed" (in "not auto-**reversed**"), while `ticket_005`'s
matching content is **split across two separate chunks** — chunk 0 has the
duplicate-charge description, chunk 1 has "automatically **reversed**".
Neither of `ticket_005`'s two chunks alone has as much exact-term overlap
with the query as `ticket_007`'s single, denser chunk does. This is a
**chunking-boundary problem** (a Week 3 concept), not a weakness unique to
either search method — which is why hybrid search didn't fix it either.

**Generation-failure classification (RIGHT_DOC_WRONG_ANSWER vs.
RIGHT_DOC_RIGHT_ANSWER): blocked today.** All 15 attempted `generate.py`
calls returned OpenRouter `429` (daily free-tier quota exhausted from
earlier testing this session). `classify_failures.py` handled this
gracefully (no crash, `SKIPPED_LLM_ERROR` recorded per question,
full results saved to `week4_classify_results.json`) — this is a real,
known limitation of this write-up, not a hidden gap: **we do not yet have
evidence on generation-side failures**, only retrieval-side. Re-running
`classify_failures.py` once the quota resets will fill this in without
needing any other changes.

## 3. Check 2 — exactly one change

Confirmed: the only files added this week are `bm25_search.py`, `hybrid.py`,
plus supporting eval infrastructure (`eval_questions.py`, `metrics.py`,
`classify_failures.py`, `compare_before_after.py`) and 2 new data files.
**`retrieve.py`, `generate.py`, `vectorstore.py`, `store.py`, and
`DISTANCE_THRESHOLD` were not modified.** The existing semantic-only
`retrieve()` still works exactly as it did at the end of Week 3 — hybrid
search is a new, separate, opt-in function (`hybrid_retrieve()`), not a
replacement.

## 4. Check 3 — the before/after number

Computed with `compare_before_after.py` (zero LLM calls — pure retrieval
comparison, using an in-memory Qdrant collection that never touches the
real persisted database):

| | hit-rate@3 | MRR |
|---|---|---|
| **BEFORE** (semantic only) | 100.00% (15/15) | 0.9667 |
| **AFTER** (hybrid RRF) | 100.00% (15/15) | 0.9667 |

**No change on either metric.** hit-rate@3 was already at ceiling (every
question's correct document was somewhere in the top-3 even before hybrid
search), and MRR — which is sensitive to *rank*, not just presence — stayed
identical at 0.9667 because the one ranking issue found (q14) was
unaffected by the change (see Check 1's root-cause analysis: BM25 agreed
with semantic search's ranking here, so fusing them changed nothing for
this specific case).

## 5. Check 4 — what did NOT improve, and why

**q14 ("29.99 x2 reversed") did not improve.** Both the semantic method and
BM25 independently rank `ticket_007` above `ticket_005` for this query, so
RRF fusion — which combines two rankings that already agree — has nothing
to correct. This is the most important honest finding of the week: **the
chosen improvement (hybrid search) targets a different failure mode than
the one we actually found.** Hybrid search helps when semantic search and
keyword search *disagree* (one method's blind spot is covered by the
other's strength); it cannot help when both methods are led astray by the
*same* underlying cause — here, a chunk boundary that splits related
information apart.

**What would likely have fixed q14 instead:** adjusting chunking so
`ticket_005`'s duplicate-charge description and its "automatically
reversed" resolution land in the same chunk (e.g. a larger `CHUNK_SIZE`, or
overlap tuned to keep that specific pair of sentences together) — a
different lever entirely from this week's chosen improvement, and worth
testing as a genuine next step.

## 6. Known limitations (honest, not fixed this week)

- **Generation-failure evidence is missing today**, entirely due to
  OpenRouter's free-tier daily quota (50 requests/day) being exhausted from
  earlier testing. `classify_failures.py` is ready to produce this data —
  it just needs to be re-run once the quota resets (or credits are added).
- **Only one genuine ranking issue was found** (q14), and it was a *rank
  degradation* (correct doc at #2, not #1), not a full "wrong document"
  miss within top-3. The 15-question eval set, even with 2 deliberately
  confusable tickets and 3 adversarial short queries, mostly confirmed the
  existing pipeline is already strong — a real, if less dramatic, result
  than assuming failures would be easy to find.
- **The chosen improvement (hybrid search) did not fix the one issue
  found**, for a specific, understood reason (Section 5) — this itself is
  useful evidence about *which* improvement would have been the right one
  to pick, and demonstrates the improvement was tested honestly rather than
  reported as a success regardless of outcome.
- `RRF_K=60` and `CANDIDATE_POOL=15` were used as standard/reasonable
  defaults, not tuned against this dataset — not expected to change the
  q14 outcome regardless, since both underlying rankers already agreed.

## 7. Summary against the mentor checklist

- ✅ Can show a specific failure (q14) and which kind it is (a ranking
  degradation, not a full miss), with retrieved-list evidence.
- ✅ Made exactly one change (`hybrid.py`/`bm25_search.py`; nothing else touched).
- ✅ Have a real before/after number (hit-rate@3, MRR), computed with zero
  LLM calls so it isn't blocked by the OpenRouter quota.
- ✅ Explicitly identified what did NOT improve (q14) and diagnosed why
  (both underlying methods agreed on the wrong ranking; the real fix is a
  chunking change, not a retrieval-fusion change).
- ⬜ Generation-failure classification (RIGHT_DOC_WRONG_ANSWER vs.
  RIGHT_DOC_RIGHT_ANSWER) — blocked by quota, to be completed once reset.
