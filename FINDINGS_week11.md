# Findings -- Week 11: Production (Observability, Cost, Failure-to-Test Loop)

**Track:** A -- Customer Support Tickets
**Deliverable:** per-request logging you can search, a completed support drill,
a measured cost-per-request improvement, and one real failure turned into a
permanent test
**Mentor checklist:** *"Can they find a specific past answer from the logs
quickly? Do the logs show time and cost per step, not just one total? Did
they measure cost per request and improve it? Did they turn a real failure
into a test that now guards against it?"*

## 1. What this week builds, and why

Every prior week was graded by running an eval script and reading printed
numbers -- that only works because the exact question and expected answer
are already known. Week 11's premise is different: in production nobody is
watching the app ask-by-ask, so the app itself has to make its own history
searchable, its own cost visible, and its own real failures permanent.

Three gaps were confirmed directly against this repo before writing any
code, not assumed:

1. `api.py`'s `/traces` endpoint only ever supported `limit`/`offset`
   pagination -- no search by keyword, source, or outcome. With 1,668 real
   trace files already accumulated from this project's own testing, finding
   one specific bad answer from a vague complaint meant paging through by
   hand.
2. `generate.py::answer_question()` -- the fixed pipeline behind the real
   "Ask" UI page and `POST /ask`, the pipeline an actual user hits -- never
   captured token or cost data at all. Only `agent.py`/`multi_agent.py`
   (Week 10, CLI-only) did, and only as a final total, never per step.
3. A real, already-logged failure existed in this project's own trace
   history: trace `0b7819c32d2d432bb32255b0ff5efeb9`, the real `q9` case
   from Week 10's team race. `FINDINGS_week10.md` Section 9 had already
   root-caused it (Groq's formatting breaks a literal keyword match) and
   explicitly deferred the fix. Nobody had the trace id for it either --
   `multi_agent.py`'s own CLI never prints one to the screen.

## 2. Closing the cost-visibility gap (Part A)

`generate.py` gained `estimate_cost(provider, usage)`, using Week 10's
already-verified split input/output cost rates. `answer_question()` now
passes `usage_out` to `call_llm()` (it never did before) and writes
`prompt_tokens`/`completion_tokens`/`cost_usd` onto its `generation` trace
stage. The same per-step attachment was added to every `agent.py` step and
every `multi_agent.py` stage (`route`/`specialist`/`merge`), which already
computed the right numbers internally but never recorded them on the trace
itself. `tracing.py::Trace.finish()` now auto-sums a `total_cost_usd` from
whatever stages attached one, so no caller has to remember to total it up.

Verified live against real Gemini calls on all three pipelines -- correct
per-step numbers, correct totals, matching between the trace file and the
function's own return value.

## 2a. A second observability layer's own bug: Langfuse input bloat

Caught by inspecting a real Langfuse trace directly, not assumed:
`generate.py::answer_question()`'s `@observe(name="fixed-pipeline-ask")`
decorator has no `capture_input=False`, so Langfuse automatically captured
*every* argument passed to the function as the trace's Input -- including
`client`, an already-created `QdrantClient` (passed in by callers that
load it once for efficiency, the same pattern used throughout this
project). Langfuse then recursively serialized the ENTIRE client object:
internal `httpx` connection-pool state, gRPC channel pool, and even
Qdrant's full internal JSON Schema used for query validation -- none of it
question/answer data, all of it sent to Langfuse on every single request.
`agent.py::_run_agent_async()` had the same latent risk via its
`mcp_session` parameter (normally `None` in real use, but a heavy
`mcp.ClientSession` object on the Week 9 test-seam path).

**The fix**: `capture_input=False` on both decorators, with a manual
`get_langfuse_client().update_current_span(input={...})` call setting
just the meaningful fields (question/top_k/provider/temperature). Caught
one more real mistake applying this: this project's installed Langfuse
SDK version has no `update_current_trace()` -- only `update_current_span()`
exists, since an `@observe`-decorated function's own call IS the trace's
root span in this SDK's model, not a separate trace object. Found via a
real `AttributeError` on the first test run, not guessed correctly the
first time; fixed and re-verified live on both functions afterward.

**The convention this establishes, for anything added after this week**:
any `@observe`-decorated function that accepts a heavy object as a
parameter -- a DB client, a loaded ML model, an MCP session, or anything
similar -- needs `capture_input=False` (and `capture_output=False` if its
return value isn't already a small, clean dict) plus a manual
`update_current_span()` call. This is NOT automatic and NOT enforced by
the decorator itself -- a new parameter added later to an *already*
-decorated function, or a brand-new `@observe` added to
`multi_agent.py::_run_team_async()` (which has no Langfuse instrumentation
at all today), would silently reintroduce this exact bloat unless this
pattern is applied there too. Documented here specifically so it isn't
rediscovered the hard way a second time.

## 3. Real trace search (Part B)

`tracing.py::search()` scans every trace file (newest first, same mtime-sort
`list_recent()` already used) and matches a `query` as a case-insensitive
substring against the WHOLE serialized record -- the question, the final
answer, every stage's thoughts/observations, any retrieved chunk text, in
one pass -- plus optional `source`/`outcome`/`since` filters. Exposed as
`GET /traces/search` in `api.py`, declared before `/traces/{trace_id}` on
purpose: FastAPI matches routes in declaration order, and `{trace_id}`
would otherwise greedily swallow the literal path segment `search`.
Confirmed this mattered by testing the ordering directly.

The React Traces page got a real search box that calls this endpoint,
separate from the existing outcome-tab filter (which only ever filtered
whatever page was already loaded in memory).

Named honestly as a known limitation, not hidden: this is a linear per-file
scan, fine at today's scale (a few KB per file, ~1,700 files), but a real
production system would use a proper search index (OpenSearch/
Elasticsearch, or a database with a text index) instead.

## 4. The support drill: finding the real failure (Part B + D)

**Starting point -- a vague complaint:** *"A customer says the refund
percentage we quoted them looked garbled."* No trace id, no exact question
text, no timestamp -- exactly the brief's scenario.

**The search:**
```
GET /traces/search?query=percentage
```
-- 65 matches among 1,668+ files. Narrowed with an outcome filter:
```
GET /traces/search?query=percentage&outcome=team_answer
```
-- 5 matches. Trace `0b7819c32d2d432bb32255b0ff5efeb9` is among them: question
*"what percentage refund do I get if I ask between 31 and 60 days after
buying?"*, answered by the team pipeline, citing `ticket_002_refund_window.txt`.

**The real cause**, already root-caused in `FINDINGS_week10.md` Section 9:
the answer text contains `"50<narrow-space>%"` -- correct content, wrong
bytes for a literal `"50%"` substring check. `assertions.py`'s
`check_expected_keywords()` scored a genuinely correct answer as a FAIL.

**The fix** (`evals/assertions.py`): tested directly against this exact
trace's real text before trusting it, and found `unicodedata.normalize
("NFKC", ...)` alone is NOT sufficient -- it correctly folds the narrow
no-break space to a plain space, but only maps a non-breaking hyphen to a
*different* non-ASCII hyphen (never reaching plain `-`), and does nothing
for the full-width brackets Groq sometimes wraps a citation in. Added an
explicit lookalike-character table on top of NFKC, plus a second insight
found only by testing against the real data: the narrow space sometimes
means "no space" (`"50%"`) and sometimes means "a space" (`"15 minutes"`,
per Section 9's own earlier example) -- no single space-preserving mapping
is correct for both, so both the answer and each keyword are stripped of
all whitespace before comparing, sidestepping the ambiguity entirely.

**The permanent test** (`eval_questions.py`): added `r6`, reusing the real
`q9` wording, pointing directly at this trace id and at Section 9 in its
own comment. Full suite re-run: **20/21 passing**, `r6` now correctly
passes. The one remaining miss (`r4`) is a plain "I don't know" -- a
pre-existing, already-documented retrieval flakiness from Week 8, unrelated
to this fix, confirmed by reading its actual answer text rather than
assumed.

## 5. The one measured cost improvement: semantic caching (Part C)

**Design:** a second Qdrant collection (`semantic_cache.py`), same
embedding model already loaded for retrieval. Wired into
`answer_question()` only (the real user-facing pipeline) -- a cache hit
skips retrieval's confidence gate having already run (cheap) but skips the
LLM call entirely (the expensive part). Never caches a refusal -- caching
"I don't know" would permanently block a better-phrased retry of the same
underlying question.

**A real bug found and fixed during verification, not assumed away.** The
threshold was first set to 0.75 from 5 hand-picked example pairs
(paraphrases measured 0.336-0.674, distinct questions measured 0.856+).
Re-running the full eval suite afterward surfaced a real cross-contamination:
*"what percentage refund...?"* was served a cached answer about
lost-in-transit shipping timing -- a completely different question --
at distance 0.684, inside that threshold.

Investigating properly (computing every pairwise distance across the real
eval set, not another small hand-picked sample) showed the problem is
structural, not a bad constant: **genuinely different questions sat as
close as 0.413** -- closer than several genuine same-question paraphrases.
Short, fact-dense support questions about overlapping business topics
(refunds, billing, shipping) do not separate cleanly in embedding space the
way whole-document retrieval does. No single threshold is safe.

**The fix, in two layers:**
- `CACHE_SIMILARITY_THRESHOLD` tightened to a deliberately conservative
  `0.35` -- only accepts the pairs confirmed to be genuine identical-question
  paraphrases in the real measurement (0.0-0.337).
- A second, independent gate: `semantic_cache.get_cached()` now also
  requires the cached answer's stored sources to overlap with what the
  CURRENT question's own retrieval just found. The same "no single signal
  is enough" principle `retrieval.py`'s own `_diversify_by_source()`
  already relies on (distance alone isn't enough there either -- it also
  needs rerank-score agreement).

Re-verified directly: re-ran the exact scenario that broke (asking the
shipping question, then the refund question) -- no longer cross-contaminates,
each gets its own correct, freshly-generated answer.

**Honest residual limitation**, stated plainly rather than hidden: source
overlap is a real signal but not a perfect one -- two genuinely different
sub-questions about the *same* source document (e.g. two different facts
from the same ticket) would still share a source. The conservative distance
threshold is the main defense against that; this is a known, accepted trade
-off (lower recall, safer), not a claimed-solved problem.

**Staleness, caught as a second, separate gap**: none of the above protects
against a cached answer becoming outdated because the underlying ticket
changed -- distance and source-overlap only check whether a cached answer
matches the QUESTION, never whether it's still TRUE. Closed with two
independent mechanisms, solving two different problems rather than one
standing in for the other:
- `CACHE_TTL_SECONDS` (`config.py`, default 24h): every cache entry is now
  stamped with `cached_at`; `get_cached()` rejects (treats as a miss, does
  not physically delete) any entry older than the TTL. A backstop for
  silent/unnoticed document edits.
- `ingestion.py::build_database()` now calls `semantic_cache.clear()`
  immediately after rebuilding the index -- an actual document change (a
  manual re-ingest, or a real `/upload`) invalidates every cached answer
  right away, rather than waiting for the TTL to lapse. Deliberately
  placed inside `build_database()` only, never at normal app startup
  (`uvicorn api:app` does not call it) -- a routine restart should not
  throw away still-valid cached answers for no reason, only a genuine
  knowledge-base change should.

Both verified live: stored an entry, manually backdated its `cached_at` to
simulate expiry, confirmed `get_cached()` correctly misses; separately,
stored an entry, called `build_database()`, confirmed the same entry is no
longer servable immediately afterward.

**Real measured numbers** (21 cases, `evals/measure_cache_savings.py`,
re-measured after the fix):

```
=== Cache savings (n=21) ===
Metric                 Cold (empty cache)     Warm (populated)
Total tokens           12773                  0
Total cost ($)         0.004393               0.000000
Cache hits             3/21                   20/21
```

**100% cost reduction on the warm pass.** The cold pass's 3 hits (`q13`,
`q15`, `r6`) are each a genuine, verified identical-question paraphrase
already present earlier in the same run (`q8`/`q13`, `q3`/`q15`, `q9`/`r6`)
-- confirming the conservative threshold accepts exactly the safe cases and
nothing else. The warm pass's one non-hit (`r3`) is the one genuine
refusal in the set, correctly never cached by design.

## 6. The 10x capacity plan (conceptual, like Week 10's A2A section)

No load-testing code was built this week -- this is a written analysis,
grounded in this project's own real incidents, not a generic guess.

**What actually breaks first: free-tier LLM provider rate limits, not
Qdrant or the embedding model.** Across this project's own multi-week
session history, Gemini's daily request quota was confirmed fully
exhausted (501/500 RPD, verified via the AI Studio dashboard) more than
once, and both Gemini and Groq were observed rate-limited simultaneously
during real eval runs. At 10x traffic, this is the first and most severe
failure mode, by a wide margin -- a free-tier daily cap doesn't scale with
load at all; it just runs out sooner.

**What already helps, built in earlier weeks:**
- `generate.py::call_llm()`'s automatic Gemini<->Groq fallback on a 429 or
  5xx -- already absorbs single-provider exhaustion, not double exhaustion.
- `race_agent_vs_workflow.py`/`race_team_vs_agent.py`'s retry-with-backoff
  (added mid-session after real repeated crashes) -- smooths over a
  transient spike, not a sustained 10x load.
- Week 10's embedding-model-reuse fix (`model`/`client` threaded through
  rather than reloaded per call) -- already removes a real, measured
  ~5-6s-per-call tax that would otherwise multiply badly under load.
- This week's semantic cache -- directly reduces LLM call volume for
  repeated/paraphrased questions, the one lever that actually reduces
  demand on the scarce resource instead of just handling its failure.

**What a real 10x plan still needs, honestly, that isn't built:**
- A paid tier or multiple rotating API keys -- free-tier quotas are a hard
  ceiling no amount of code can work around.
- A real cache hit-rate target measured against real traffic patterns, not
  this week's eval-set numbers (which include duplicate questions by
  design).
- Qdrant and the embedding model are NOT expected to be the bottleneck at
  10x -- both are local/self-hosted with no external rate limit, and
  retrieval latency (confirmed throughout this project) is a small fraction
  of total request time compared to the LLM call itself.
- A real search index for `tracing.py::search()` (Section 3) -- the linear
  scan is fine at ~1,700 files, not necessarily at 10x the trace volume
  10x traffic would also produce.

## Known limitations

- **Trace search is a linear scan**, not a real search index -- correct and
  fast enough at today's scale, explicitly not claimed to scale indefinitely.
- **Semantic caching has a known, accepted false-negative trade-off**: the
  conservative threshold will miss some genuine paraphrases that a looser
  threshold would have safely caught, in exchange for eliminating the
  cross-contamination risk found this week. Not revisited further this
  week -- a more reliable grounding mechanism (e.g. an LLM-based
  equivalence check before serving a cached answer) is the natural next
  step, not attempted here.
- **Source-overlap is not a perfect safety net** on its own (Section 5) --
  two different sub-questions about the same source document would still
  pass it. The threshold is the primary defense; this is a second layer,
  not a guarantee.
- **Cost rates remain a standard-list-price proxy** (Week 10's own
  limitation, unchanged this week) -- real spend on this project's
  free-tier keys is $0.
- **A cache hit "locks in" whichever answer happened to get cached
  first, wording and all** -- found live, not assumed: normally
  `evals/run_eval.py` generates a fresh answer every re-run (temperature
  is not 0, so wording varies call to call), giving a flaky keyword check
  a real chance to pass on any given run. Once an answer is cached,
  every matching question gets that SAME stored wording every time --
  if that one cached version happens to omit the exact expected phrase
  (e.g. a correct answer that never literally says "15 minutes"), it now
  fails *consistently* instead of only occasionally. Not a correctness
  bug in the cache itself (the cached answer is still genuinely correct
  content), but a real interaction worth knowing about when comparing
  eval runs taken before and after the cache was populated.

## Summary against the mentor checklist

- [x] **Can they find a specific past answer from the logs quickly?** Yes
  -- Section 4, a real drill against 1,668+ real traces, narrowed to 5
  candidates with two filters.
- [x] **Do the logs show time and cost per step, not just one total?**
  Yes -- Section 2; every trace stage across all 3 pipelines now carries
  its own tokens/cost, with a summed total at the top level.
- [x] **Did they measure cost per request and improve it?** Yes -- Section
  5, with real before/after numbers, and an honestly-reported bug found and
  fixed in the improvement itself before trusting the numbers.
- [x] **Did they turn a real failure into a test that now guards against
  it?** Yes -- Section 4, `r6`, guarding the exact real failure this week's
  own drill found, verified passing.
