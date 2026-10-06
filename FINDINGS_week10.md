# Findings — Week 10: Multi-Agent Race (Manager + 2 Specialists vs. Single Agent)

**Track:** A — Customer Support Tickets
**Deliverable:** A manager-plus-two-specialists team raced against the single
agent, with quality, speed, tokens, and cost, and an honest verdict
**Mentor checklist:** *"Did they race the team against the single agent on
the SAME tests? Did they report all four numbers? Is the verdict backed by
the numbers, even if the team lost? Can they say when multi-agent would be
worth it, and when it wouldn't?"*

## 1. What this week builds, and why

Week 9 solved agent-to-TOOL connection (MCP). It never answered a different
question: would splitting this app's single agent's job across a TEAM of
agents (a manager + narrow specialists) actually help, or cost more for the
same or worse result? This mirrors Week 7's exact methodology (agent vs.
fixed pipeline, real numbers, fixed pipeline won) one level up — "often the
single agent wins, and that's a valuable finding" (the brief's own words).

**Specialist split** (decided before building): **Billing/Refunds**
(refund window, billing disputes, defective items, incorrect charges) vs.
**Account/Access** (password lockout, account cancellation, shipping
delay) — a real support-team-shaped split, not an arbitrary one.

**A2A depth** (decided before building): lightweight, in-process handoff —
plain Python function calls (`multi_agent.py`), not a second real networked
server. `agent.py` was built as a real client and `mcp_server.py` as a real
server for MCP (Week 9); A2A's concepts here are mapped onto real code
(Section 7) rather than built as a second running protocol, consistent with
this being a smaller, measurement-focused deliverable.

## 2. `multi_agent.py` — the manager + 2 specialists

- **`route(question)`** — one LLM call. The manager reads the question and
  returns which specialist(s) are needed, with a focused sub-question for
  each, or `"out_of_scope": true` if neither domain applies — this third
  option exists specifically so the manager isn't forced into a bad binary
  choice for a question that fits neither specialist (matching this app's
  own "refuse in code, don't let the model guess" rule from the confidence
  gate).
- **`run_specialist(domain, sub_question)`** — retrieves via the same
  shared `retrieval.retrieve()` every other part of this app uses, then
  builds its own prompt reusing `generate.py::build_prompt()`'s exact
  trust-boundary + customer-voice defenses (duplicated verbatim, not
  weakened) plus domain-narrow instructions. Each call is genuinely
  stateless — no shared history object — which is what makes the "context
  re-send cost" the brief describes concrete and measurable rather than
  just asserted.
- **`run_team(question)`** — runs `route()`, then whichever specialist(s)
  were assigned **genuinely in parallel** when both are needed (via
  `asyncio.gather(asyncio.to_thread(...), asyncio.to_thread(...))` — plain
  `asyncio.gather()` alone would NOT parallelize two calls to `call_llm()`,
  since it's a blocking `requests.post` call that never yields control back
  to the event loop; `asyncio.to_thread()` gives genuine OS-thread
  parallelism for two blocking network calls without converting `call_llm()`
  itself to async, which would have required touching every existing
  caller). If both specialists ran, one more call merges their answers into
  a single coherent reply.

**Verified directly, not assumed**, that the parallel path is real: timed
each specialist alone (billing 17.47s, account 12.88s — sum 30.36s, the
sequential expectation) against the actual `asyncio.gather`/`to_thread`
path used in `run_team()` (17.71s measured) — matching the `max()`
prediction almost exactly, not the sum.

## 3. Capturing real tokens and cost (`generate.py`, `agent.py`, `config.py`)

`generate.py::_call_provider()`/`call_llm()` gained one optional,
backward-compatible `usage_out` parameter — every existing caller (4+ call
sites: `agent.py`, `answer_question()`, `judge.py`, `trajectory_judge.py`)
is completely unaffected, since it defaults to `None` and does nothing
unless passed. `agent.py::_run_agent_async()` was extended the same
additive way (accumulating `prompt_tokens`/`completion_tokens` across every
step, added as 2 new keys to its already-dict-shaped return value) so the
single agent's real token usage could be measured too, not just the team's.

**Split input/output rates, not a flat blended rate** — verified via real
web lookup against each provider's own published pricing for the exact
configured models, not recalled from memory:
- Gemini 3.1 Flash Lite (GA): $0.25/1M input, $1.50/1M output
- Groq gpt-oss-120b: $0.15/1M input, $0.60/1M output

This split matters specifically because a multi-agent team's hidden cost is
inflated **prompt** tokens at every handoff — a flat rate would hide or
distort exactly the penalty this race exists to measure. (This project's
actual API keys are on each provider's free tier — real spend today is $0;
these rates are a standard-list-price comparison proxy, same honesty this
project already applies to the Week 7 agent-vs-pipeline cost framing.)

## 4. The race — real numbers, 20 cases, both arms scored by the same `assertions.py`

**Re-run 2026-10-06, after a code-review bug fix.** The two earlier runs of
this race (one on Gemini, one on Groq -- see Section 9) were both measured
before a real bug was found and fixed: `run_specialist()` called
`retrieve()` without passing `model`/`client` through, so every single
specialist call reloaded the whole embedding model from scratch (measured:
~5-6s, every call, not just the first -- `get_embedding_model()` has no
caching, unlike every other lazy-loaded model in this app). That cost had
nothing to do with multi-agent architecture, but it dominated the team's
measured latency. Fixed by threading `model`/`client` through the same way
every other caller in this app already does. The numbers below are the
first ones measured with that fix in place:

```
=== Summary (n=20) ===
Metric                       Single Agent             Team (Manager+2)
Quality (Pass rate)          16/20 (80%)              15/20 (75%)
Speed (Execution latency)    5.05s avg                3.12s avg
Tokens Used                  3980 avg/question        990 avg/question
Cost ($)                     $0.0014 avg/question     $0.0004 avg/question
```

**This reverses the earlier result: the team now wins on speed, tokens, and
cost, and loses quality by exactly one question (15/20 vs. 16/20).** The
original "team loses decisively on speed" finding was mostly a caching bug,
not a property of splitting work across specialists -- see Section 5 for
why even that remaining one-question quality gap is mostly explained away,
not a clean single-agent win either.

**A caveat on the exact dollar figure**: Gemini hit rate-limit pressure
again during this run (visible in the race's own `[RATE LIMITED]` retries
for `q6`/`q9`/`q15`/`r4`), and `call_llm()`'s automatic fallback silently
answered some of those calls with Groq instead. `race_team_vs_agent.py`'s
cost formula always prices every call at Gemini's rate (documented in
Known Limitations) -- since Groq is the cheaper of the two providers, the
real total cost this run is slightly LOWER than the `$` figures above, not
higher. Confirmed exactly which calls actually used Groq by checking the
raw answer text for its known formatting signature (Section 9): `q5`-team,
`q6`-single, `q6`-team, `q9`-team, `r4`-single.

## 5. The 5 disagreements, investigated with evidence, not just counted

| ID | Single | Team | Real cause |
|---|---|---|---|
| q2 | FAIL | PASS | Keyword-phrasing luck, this time favoring the team — its answer happened to literally include "7 business days" ("delayed more than 7 business days past the estimated delivery date"); the single agent's equally correct answer never used that exact phrase. Same flakiness documented since Week 4, now caught running in the team's favor instead of against it. |
| q9 | PASS | FAIL | **Not a real content miss.** The team's billing_refunds specialist's answer was actually generated by Groq, not Gemini — confirmed by its signature Unicode formatting (`‑`, ` `, and full-width `【`/`】` brackets) in the raw answer text, the exact artifact documented in Section 9. A Gemini rate-limit hit forced `call_llm()`'s automatic fallback to Groq for this one call; Groq's narrow-space formatting around "50 %" broke the literal `'50%'` keyword match even though the actual figure (50%) was correct. |
| r1 | PASS | FAIL | **A real structural gap**, reproduced again this run. This compound question needs BOTH `ticket_004_account_cancellation.txt` AND `customer support.pdf`. The account_access specialist correctly answered the cancellation half but never searched `customer support.pdf` — the general company guide — since it belongs to neither specialist's domain. |
| r2 | FAIL | PASS | **A genuine team win.** The single agent's one-shot retrieval pulled in two forbidden/irrelevant sources alongside the right ones for this 3-part question (`forbidden source(s) present: ['ticket_002_refund_window.txt', 'ticket_005_billing_dispute.txt']`). The team's two narrow specialists each retrieved only within their own domain, avoiding exactly that cross-contamination — a concrete instance of the brief's own claim that genuinely independent sub-tasks can benefit from narrower, separate retrieval. |
| r5 | PASS | FAIL | **Same structural gap as r1.** "How do I contact customer support" has its answer ONLY in `customer support.pdf`. `route()` correctly (by its own narrow instructions) returned `out_of_scope: true`, so zero retrieval was ever attempted — the question is genuinely answerable by this app, the team just has no specialist assigned to the content that answers it. |

**`r4` (the management-ratio question, also only in `customer support.pdf`)
failed the same way as `r5` again this run**, for the identical reason —
not counted as a "disagreement" only because the single agent itself ALSO
failed this one (both got it wrong, for different reasons: the single
agent's retrieval weakness documented back in Week 8, the team's
domain-coverage gap here).

**Net read, now that each case has been checked rather than just
counted**: of the team's 3 nominal "losses" in this run, only `r1`/`r5` are
a real content gap, and it's the same `customer support.pdf` ownership gap
identified in the original (pre-fix) run — the one finding in this whole
week that's reproduced identically across every run, pre- and post-fix,
Gemini and Groq. `q9` is a measurement artifact (Groq's formatting, not its
content), and `r2` is actually a team WIN the raw pass/fail table doesn't
distinguish from an ordinary loss. Stated plainly: of 20 cases, there is
exactly one real, stable team weakness (the `customer support.pdf` gap,
counted twice as `r1`/`r5`), one case where the team is flatly better than
the single agent (`r2`), and two cases that are measurement noise in
either direction (`q2`, `q9`) rather than evidence about the architecture
at all.

**The real lesson, stated plainly**: splitting a 2-document-domain support
team by ticket topic works fine for ticket-shaped questions, but this app
also has a **third kind of content** — a general reference guide — that
was never assigned an owner. A single generalist agent has no such
blind spot because it searches everything with one undifferentiated tool;
narrow specialists, by construction, only know about what they were told
to know about. This is the concrete, measured version of the brief's own
warning that narrow specialists can create exactly this kind of coverage
gap — and, per `r2`, the same narrowness that creates that gap is also
exactly what made the team win the one case where the single agent's
broad retrieval backfired.

## 6. Why the team is cheaper and faster — mostly genuine, one real caveat

With the embedding-reload bug fixed (Section 4), the team's advantage on
tokens, cost, AND speed is now mostly a genuine efficiency gain, not an
artifact:
- **Genuine efficiency**: each specialist's prompt is shorter and more
  focused than the single agent's full MCP tool-discovery prompt (the
  agent's prompt carries the full tool list, descriptions, and trust
  boundary on every single step; a specialist's prompt is narrower).
  Single-topic questions needing only one specialist (2 team calls) are
  genuinely cheap and fast this way — confirmed now that both arms are
  measured on a level footing (same shared embedding model/client, no
  per-call reload hidden in either one).
- **One real, still-true caveat**: the `out_of_scope` fast path that
  incorrectly refuses `r4`/`r5` (Section 5) costs exactly 1 LLM call (the
  router, nothing else) — cheap, but wrong. A small part of the team's
  token/cost advantage still comes from incorrectly refusing to even try
  on those 2 of 20 cases, not from doing the same work more efficiently.
  This is a smaller effect now than the speed numbers alone would suggest,
  since the team is winning on tokens/cost even on the cases where both
  arms actually searched and answered.

So "the team is cheaper and faster" is now substantially real, with one
specific, bounded exception (the 2 `customer support.pdf` cases) rather
than the across-the-board caveat the pre-fix numbers implied.

## 7. A2A: what it is, and how it maps onto this code

**AgentCard** — what a real, networked deployment of each specialist would
publish about itself (real data in `multi_agent.py::AGENT_CARDS`, quoted
directly here, not re-typed):
```json
{
  "name": "billing-refunds-specialist",
  "description": "Handles refund windows, billing disputes, defective items, incorrect charges",
  "endpoint": "https://.../a2a/billing-refunds",
  "capabilities": {"streaming": false, "pushNotifications": false},
  "inputSchema": {"type": "object", "properties": {"sub_question": {"type": "string"}}},
  "outputSchema": {"type": "object", "properties": {"answer": {"type": "string"}, "sources": {"type": "array"}}}
}
```

**Task lifecycle mapping** — this project's actual in-process steps, mapped
onto A2A's real lifecycle states:

| A2A task state | What it maps to here |
|---|---|
| `submitted` | `route()` assigns a sub-question to a specialist |
| `working` | `run_specialist()` is retrieving + generating |
| `completed` | specialist returns `{"answer", "sources"}` |
| `failed` | specialist's `call_llm()` raises, caught the same way `run_agent()` already catches provider errors |

**MCP vs. A2A**, grounded in this project's own real code on both sides:

| | MCP (Week 9, built as a real server) | A2A (Week 10, conceptual here) |
|---|---|---|
| Connects | An agent to a **tool** | An agent to **another agent** |
| This repo's example | `mcp_server.py`'s `search_tickets` | A specialist in `multi_agent.py` |
| Discovery | `tools/list` → name/description/schema | AgentCard → name/description/schema (same shape, different scope) |
| What runs on the other side | A plain function, no LLM | Another full agent, with its own LLM call |
| Built as a real server here? | Yes, `streamable-http`, port 8830 | No — in-process only, this week's own scope decision |

## 8. The honest verdict

**Closer call than the original (buggy) measurement suggested — and it now
genuinely depends on what you're optimizing for.** With the embedding-
reload bug fixed, the team wins on speed (3.12s vs. 5.05s avg), tokens
(990 vs. 3980 avg), and cost ($0.0004 vs. $0.0014 avg), and loses quality
by exactly one question (15/20 vs. 16/20) — and per Section 5, even that
gap isn't a clean loss: one of the team's three nominal misses is a
measurement artifact (Groq's formatting, not its content), one is a
genuine, reproducible content gap, and the team separately WON a case
(`r2`) the single agent lost to its own broad-retrieval cross-
contamination. The original build's "single agent wins decisively" verdict
was measuring a caching bug as much as it was measuring the architecture.

**The one real, stable weakness, unchanged by the fix**: `customer
support.pdf` — a general reference document — has no specialist that owns
it (`r1`, `r5`, and `r4` on both arms). This reproduced identically across
every run this week (pre-fix, post-fix, Gemini, Groq), so it's the one
finding in this whole exercise that's genuinely about the architecture,
not about measurement conditions. **Where a team would be worth it here**:
add a third specialist (or a "general knowledge" fallback route) to own
`customer support.pdf` specifically, and the team's real efficiency gains
(Section 6) would come with no known quality cost at all — that's a
different, better-scoped team than the one raced this week, worth building
as a follow-up, not a claim that today's 2-specialist team is already
complete.

**When multi-agent would genuinely help, stated from this project's own
evidence, not a generic claim**: when the sub-tasks are truly independent
AND cover genuinely different, non-overlapping instruction sets that this
project's documents are ALREADY organized along — which ticket-topic
questions are (confirmed: billing vs. account specialists never needed each
other's domain knowledge for any single-topic question, and `r2` shows
narrow per-specialist retrieval actively avoiding a mistake the single
agent's broad retrieval made) — but only once every real document in the
corpus has a specialist that owns it. The `customer support.pdf` gap here
is exactly what happens when that precondition isn't met, and it's the
only thing standing between this team and a genuine, all-around win over
the single agent.

## 9. A real finding from testing the other provider: Groq's formatting breaks keyword assertions, not its accuracy

Switched `DEFAULT_LLM_PROVIDER` to `groq` to check this race under the
other configured model (`openai/gpt-oss-120b`) and re-ran the full 20-case
race. The team's pass rate collapsed to 6/20 (30%) -- a result investigated
before trusting it, not reported at face value.

**Root cause, confirmed directly from the saved answer text**: Groq's
`gpt-oss-120b` consistently writes numbers using ` ` (a Unicode
"narrow no-break space") instead of a plain ASCII space -- e.g. `"15␣minutes"`
instead of `"15 minutes"`, `"90␣days"` instead of `"90 days"`, `"50␣%"`
instead of `"50%"`. `assertions.py::check_expected_keywords()` does a
literal substring match, so `"15 minutes"` is never found inside
`"15 minutes"` even though **the fact stated is correct**. Of the 12
new disagreements this run surfaced, at least 7 (q1, q4, q6, q9, q10, q11,
plus r1's keyword half) are this exact formatting mismatch, not a real
content error -- confirmed by reading each flagged answer directly, not
assumed from the failed-assertion count alone.

**Decision**: reverted `DEFAULT_LLM_PROVIDER` back to `gemini` (this
project's existing default) and kept Section 4's Gemini-based race as the
one reported number -- it isn't confounded by this quirk. The Groq run is
recorded here as a genuine, separate finding (a real model-formatting
interaction with this project's keyword-matching assertions, surfaced by
switching providers to check), not folded into the headline quality
numbers, and not used to redo the verdict in Section 8.

**What this would take to fix, if ever needed**: `check_expected_keywords()`
would need to normalize Unicode whitespace (e.g. via `unicodedata.normalize`
or an explicit ` `→` ` replace) before substring-matching, the same way
`injection.py`'s own rules already have to account for real-world text not
always arriving in the exact shape a simple check expects. Not done this
week -- this was discovered while checking the OTHER provider, not part of
this week's planned scope, and fixing it would need its own re-verification
against every existing eval case, not just the ones this run touched.

**Addendum (2026-10-06)**: this same artifact recurred on its own, not from
a deliberate A/B switch -- `DEFAULT_LLM_PROVIDER` was `gemini` for the
Section 4 re-run, but Gemini's own rate-limit pressure triggered
`call_llm()`'s automatic Groq fallback mid-race for several calls (`q5`,
`q6`, `q9`, `r4`), confirmed by the same Unicode signature appearing in
those specific answers. One of them (`q9`) is exactly why the team's
Section 4 quality number isn't a clean single-provider result -- see
Section 5's `q9` row. The underlying `check_expected_keywords()` gap is
real and still unfixed; this just confirms it isn't a one-time artifact of
deliberately testing Groq, it recurs whenever Gemini pressure forces a
silent fallback.

## Known limitations

- **`must_cite_page` could not be checked for the team.** The team's final
  result only exposes declared source filenames, not the underlying chunk
  objects with page numbers (lost across the specialist→merge boundary) —
  `race_team_vs_agent.py` substitutes a `page: None` placeholder for this
  one check, so a page-citation regression on the team's side specifically
  would not be caught by this race. Named honestly rather than silently
  skipped.
- **Cost is computed using Gemini's verified rate for every call, even
  though Groq's automatic fallback did answer several calls in the
  Section 4 re-run** (`q5`, `q6`, `q9`, `r4` — confirmed via Section 9's
  Unicode signature, not assumed). Since Groq is the cheaper of the two
  providers, this means the reported `$` figures are a slight
  overestimate of the real cost for this specific run, not an
  underestimate. Not corrected for in the code, since `race_team_vs_agent.py`
  has no way to know which provider actually answered a given call without
  parsing the raw text for this exact artifact after the fact.
- **The race has now been run three times** (Gemini pre-fix, Groq pre-fix
  as a deliberate A/B, Gemini-default post-fix with a partial Groq
  fallback), and the exact disagreement set changed each time (`q2`/`q12`/
  `q14`/`r1`/`r5` pre-fix; `q2`/`q9`/`r1`/`r2`/`r5` post-fix) — confirming
  `temperature=1.0`'s own non-determinism genuinely shifts which
  keyword-phrasing cases pass or fail from run to run. The one thing that
  has NOT changed across any run is the structural `customer support.pdf`
  gap (`r1`, `r4`, `r5`), since it's architectural, not a wording accident.

## Summary against the mentor checklist

- ✅ **Raced on the same tests?** Yes — all 20 of `EVAL_QUESTIONS` +
  `REGRESSION_CASES`, scored by the identical `assertions.py` for both arms.
- ✅ **Reported all four numbers?** Yes — Quality (Pass rate), Speed
  (Execution latency), Tokens Used, Cost ($), Section 4.
- ✅ **Verdict backed by the numbers, even though the team lost?** Yes —
  Section 8, with every individual disagreement traced to a specific,
  evidenced cause (Section 5), not a vague impression.
- ✅ **Can they say when multi-agent would be worth it, and when it
  wouldn't?** Yes — Section 8's closing paragraph, grounded in this
  project's own measured coverage gap, not a textbook generality.
