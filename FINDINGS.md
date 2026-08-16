# Findings — Ask My Tickets (Mini RAG App)

**Track:** A — Customer Support Tickets
**Stack:** Python, `sentence-transformers` (BGE-small, local/free) for embeddings,
ChromaDB (local, persistent) for vector storage, OpenRouter
(`google/gemma-4-26b-a4b-it:free`) for generation
**Documents:** 5 sample support tickets (`data/*.txt`) — password lockout,
refund window, shipping delay, account cancellation, billing dispute

## 1. Setup

- **Embedding model:** `BAAI/bge-small-en-v1.5`, loaded via `sentence-transformers`.
  Queries and documents are embedded asymmetrically — documents plain, queries
  prefixed with `"Represent this sentence for searching relevant passages: "` —
  per BGE's own training convention.
- **Vector store:** ChromaDB, persisted locally to `chroma_db/`.
- **Generation model:** OpenRouter, `google/gemma-4-26b-a4b-it:free` (free tier,
  20 req/min, 50 req/day on a $0 account).
- **Chunking:** sentence-aware — split on paragraph breaks first, then on
  sentence-ending punctuation within each paragraph, so a chunk boundary
  never falls mid-word or mid-sentence.

## 2. Bug caught during build — not a model issue, a chunking bug

The first chunker version split only on `.`/`!`/`?`. Our ticket files start
with a `Subject: ...` heading line that has **no trailing punctuation**, so
the splitter treated the entire heading + first sentence as one oversized
"sentence." Combined with 1-sentence overlap, this caused chunk 1 of several
tickets to almost fully duplicate chunk 0 instead of sharing a small, useful
bit of context — e.g. `ticket_003`'s chunk 0 and chunk 1 shared ~90% of their
text.

**Fix:** split on paragraph breaks (`\n\n`) first, *then* split each
paragraph into sentences. This correctly isolates heading lines as their own
unit regardless of punctuation, and generalizes to any document that uses
blank lines to separate sections — not just ours.

**Takeaway:** a sentence-splitting regex that looks reasonable in isolation
can behave badly on real formatting quirks (headings, list items, etc.) —
worth eyeballing actual chunk output, not just chunk counts, before trusting it.

## 3. Chunk-size comparison (the required experiment)

Compared two configurations against the same 5 reworded test questions (no
exact keyword overlap with the source documents), scored on whether the
correct source file came back as the #1 retrieval match:

| Config | Total chunks | Top-1 accuracy |
|---|---|---|
| SMALL (`chunk_size=100`) | 29 | 5/5 |
| LARGE (`chunk_size=600`) | 8 | 5/5 |

Both configurations retrieved the correct source every time — but the
**distance scores** (lower = more confident) told a different story:

| Question | SMALL distance | LARGE distance |
|---|---|---|
| Password lockout | **0.4748** | 0.5833 |
| Refund window | 0.5544 | 0.5655 |
| Shipping delay | **0.5103** | 0.6177 |
| Cancellation | 0.6758 | 0.6906 |
| Billing dispute | **0.2940** | 0.3666 |

SMALL was more confident on every single question. With `chunk_size=600` on
short ~700-900 character tickets, each chunk holds almost an entire ticket —
subject line, description, and resolution blended into one vector — diluting
the embedding into an "average" of several ideas rather than a sharp
representation of one fact.

**Why accuracy didn't differ even though confidence did:** our 5 tickets are
topically far apart (login vs. refund vs. shipping vs. cancellation vs.
billing), so even a diluted large-chunk embedding was still obviously closer
to the right topic than any wrong one. On a larger, more densely-related
document set, that safety margin would shrink, and diluted embeddings would
more plausibly retrieve the *wrong* chunk, not just a less-confident right one.

**Conclusion:** smaller chunks traded a bit of surrounding context for
measurably sharper retrieval confidence; larger chunks traded precision for
fewer, more complete chunks. For this document set, smaller chunks are the
safer default — consistent with the semantic-dilution pattern the reference
HR-RAG project also observed.

## 4. Grounded generation & refusal behavior

- Every in-scope test question returned a specific, correct answer with the
  correct `Source: <file>` citation.
- Out-of-scope questions ("what's the best pizza topping?", "what's the
  capital of France?", "what's the weather today?") consistently triggered
  the refusal: *"I don't know based on the available documents."*
- The refusal is enforced by a **pre-generation distance check**
  (`DISTANCE_THRESHOLD = 0.75` in `generate.py`) — if the best retrieved
  chunk is farther than that, the LLM is never called at all. This makes the
  refusal deterministic and free (no API call), rather than depending on the
  model's own judgment call every time.
- This threshold was set from observed distances during testing: in-scope
  questions consistently scored well under 0.7 (see table above), giving
  headroom before the 0.75 cutoff.

## 5. Known limitations (honest, not fixed this week)

- **Only 5 short documents.** We haven't tested a real multi-page document
  or a set of many closely-related tickets, where chunk-size dilution would
  more plausibly cause a *wrong* retrieval, not just a less-confident right one.
- **No multi-hop questions tested.** Every test question maps to a single
  fact in a single document — we haven't checked whether the model can
  combine two separate facts (e.g. "if my order is late AND I want a refund,
  what happens?").
- **Single model, no capability comparison.** We used one free model
  (`gemma-4-26b-a4b-it:free`) throughout — we don't have evidence for how a
  smaller/larger model would behave differently on the same chunks.
- **No retry/backoff on API failures.** A `429` (rate limit) currently
  surfaces as a friendly message and the loop continues, but the failed
  question itself is not automatically retried — deliberately deferred to a
  future week (see comment in `generate.py` history).
- **The 0.75 distance threshold is empirically set, not proven optimal** —
  calibrated from this project's 5 documents and test questions; a
  different document set would likely need re-tuning.

## 6. Summary against the mentor checklist

- ✅ App answers correctly from the documents, every answer cited.
- ✅ Refuses ("I don't know") on out-of-scope questions, confirmed on
  multiple unrelated test questions.
- ✅ Tried two chunk sizes (100 vs. 600 chars) and observed a concrete,
  quantified difference (distance/confidence, not just pass/fail).
- Went slightly further than required: traced *why* the difference occurred
  (semantic dilution in larger chunks) and *why* it didn't yet affect
  accuracy on this small, topically-distinct document set — plus caught and
  fixed a real chunking bug during development (Section 2).
