# Findings — Week 7: Agent Loops, and When Not to Use Them

**Track:** A — Customer Support Tickets
**Deliverable:** A hand-built agent with visible steps, plus a fixed-workflow
comparison on speed, cost, and reliability

## Phase 0 — Task definition

**Chosen task**: answering a genuinely **compound, multi-topic** customer
question — one that touches more than one ticket/topic in a single ask.

**Concrete example** (reused directly from Week 6's `r2` regression case):
> "why is my order delayed, how do I get a refund for a defective item, and
> why was I charged the wrong amount"

**Why this task, specifically, and not some other multi-step task**:

The brief's own warning is explicit — an agent only earns its extra cost
when the right sequence of steps genuinely depends on the input; if the
steps are always the same, a fixed pipeline wins on speed, cost, and
reliability, no contest. Most of this app's questions (`q1`-`q15` from
Week 4/6) are single-topic and already answered correctly, cheaply, in one
retrieval pass -- there's nothing for an agent's flexibility to buy there,
and testing an agent against those would be a rigged, meaningless race.

Compound questions are different: the existing fixed pipeline widens its
candidate pool and hopes one retrieval pass surfaces enough evidence for
every sub-topic at once (that's the entire mechanism Week 6's
`MAX_DIVERSIFY_SOURCES`/`RERANK_RELEVANCE_MARGIN` fixes were built
around). An agent can instead treat this explicitly: search for sub-topic
1, check if the evidence found actually answers it, search again if not,
move to sub-topic 2, and so on -- a genuinely different, input-dependent
sequence of steps, not a fixed one. This is the honest, defensible case
for building an agent at all, rather than agent-for-agent's-sake.

**Test questions for the eventual race** (Phase 5): all of Week 6's
`EVAL_QUESTIONS` + `REGRESSION_CASES` -- but the ones that matter most for
showing a genuine difference are the multi-topic ones (`r1`, `r2`), since
those are where a fixed single pass has previously needed real tuning work
to get right, and where an agent's extra "check and re-search" step has
an actual job to do.

**What "done" looks like for this task**: given a compound question, the
agent produces an answer that correctly addresses every sub-topic it can
find evidence for, and honestly flags any sub-topic it genuinely can't
(same standard Week 6's `assertions.py` already checks for) -- scored by
the exact same rule-based checks already built, not a new judgment call.

## Phase 1 — The agent loop (`agent.py`)

Hand-built ReAct loop, no framework (no LangChain/LangGraph) -- the same
"nothing is magic, read every line" philosophy this project already
applies to retrieval, reranking, and tracing. Each turn: replay the full
history of past thought/action/observation back into the prompt (the
loop's entire "memory" is just this -- no hidden state anywhere else),
ask the model for its next `{thought, action, action_input}` as JSON, run
that action, feed the result back in as the next observation, repeat.
`_parse_action()` never raises -- malformed JSON becomes an observation
telling the model what went wrong, so the loop self-corrects instead of
crashing. Every step is traced via the same `Trace` class from Week 5's
`tracing.py`, tagged `"mode": "agent"`.

**Verified live** on the Phase 0 compound question: the model correctly
decomposed it into 3 separate searches, one per sub-topic, entirely on its
own -- confirmed via the real trace, not assumed.

## Phase 2 — Tools

Two real actions beyond the loop mechanics: `search_tickets(query)` (wraps
the *existing* `retrieve()`, not reimplemented) and `escalate(reason)` -- a
genuinely different terminal action from `finish`, grounded in the app's
own data: `ticket_007` literally says *"escalate directly to the billing
team"*. Without `escalate`, the agent would have to either fabricate a
resolution the ticket never gave, or wrongly refuse a question that
actually has a documented next step. Verified with a positive and a
negative test: a wrong-charge question correctly triggered `escalate`
(citing the exact screenshot requirement from the ticket); a normal
answerable question correctly still used `finish`, confirming the new
option doesn't get over-triggered.

## Phase 4 — Stop conditions & budgets

`AGENT_MAX_STEPS`/`AGENT_MAX_SECONDS`, both in `config.py` (not hardcoded),
calibrated against real measured data, not guessed: a 3-sub-topic question
measured ~9.1s/step warm, ~16s cold-start first step. Original
`MAX_SECONDS=30` was proven too tight -- the loop did everything right and
still got cut off before finishing. Recalibrated to 90s (real worst-case
measured ~52s, with headroom above it), re-verified the exact case that
failed before now completes cleanly in 4 steps.

**A real bug found and fixed here, not assumed away**: the agent initially
hallucinated on an out-of-scope question ("what's the best pizza
topping?") -- it skipped searching entirely and answered from its own
general knowledge (real pepperoni/margherita opinions), because the
original prompt never said "only use retrieved information," unlike
`generate.py`'s fixed-pipeline prompt which explicitly does. Fixed with
both a prompt instruction AND a code-level enforcement (a `finish`/
`escalate` attempt before any search is rejected and converted into an
observation forcing a search first) -- the same "don't just trust the
prompt" discipline this project uses elsewhere (e.g. `tracing.py`
enforcing "note before problem_type" in code).

## Phase 5 — The race

Ran all 20 questions (Week 6's `EVAL_QUESTIONS` + `REGRESSION_CASES`)
through both the agent and the existing fixed `answer_question()`
pipeline, scored by the *exact same* `assertions.py` checks for both --
no separate judgment call for "did the agent get it right."

| Metric | Fixed pipeline | Agent |
|---|---|---|
| Total time (20 questions) | 81.2s | 171.4s (**2.1x slower**) |
| Avg time/question | 4.06s | 8.57s |
| Total LLM calls | 20 | 45 (**2.25x more expensive**) |
| Assertions passed | **20/20** | 16/20 |

**The 4 disagreements, investigated with evidence, not just counted:**

- `q12`, `q14` -- keyword-only misses (the agent's answer covers the
  correct facts, just doesn't happen to include one exact phrase). Same
  LLM-phrasing-variance flakiness documented since Week 4/6, not a real
  content failure.
- `r1` -- a real miss. This is the exact compound question the **fixed
  pipeline** was specifically hardened against earlier this session (the
  `ticket_003` contamination bug). The agent's answer covers the first
  sub-topic but drops the "90 days" fact from the second -- the retrieval
  fix built for the fixed pipeline's usage pattern doesn't automatically
  carry over to the agent's usage pattern of the same `retrieve()` call.
- `r2` -- the most structurally significant finding. `forbidden_sources`
  failed: two sources the fixed pipeline correctly excludes leaked into
  the agent's evidence. Root cause: the fixed pipeline calls `retrieve()`
  **once** with the full compound question, so its diversify-by-source
  logic reasons about all candidates together. The agent calls `retrieve()`
  **three separate times**, once per sub-topic query -- each call's
  diversify logic only sees that one narrow query's candidates, with no
  awareness of what the other searches already found. A source weakly
  relevant to one narrow sub-query, but never relevant to the compound
  question as a whole, can leak in specifically *because* the agent split
  the question apart to search it -- the very mechanism meant to help it
  is also what let this happen. This is a genuine architectural trade-off,
  not a bug to patch.

## Known limitation, documented rather than fixed

**Groq incompatibility**: found live, during the race, when Gemini's quota
was exhausted mid-run and the automatic fallback (built for the fixed
pipeline, reused here) routed an agent prompt to Groq. Groq's model
(`openai/gpt-oss-20b`) has native function-calling behavior baked into how
it's served, and auto-detects the ReAct prompt's "Available actions:
search_tickets(...)" phrasing as an implicit tool-call request -- it tries
to natively invoke `search_tickets(...)` and the API rejects its own
attempt: `400: "Tool choice is none, but model called a tool"`. Confirmed
this isn't a simple parameter fix: explicitly setting `tool_choice: "none"`
produces the identical error -- the model attempts the tool call
regardless, and the server only rejects it after the fact.

The fixed pipeline's plain-language prompts never trigger this (no
tool/action vocabulary in them), so it works fine on Groq. **The agent, as
currently built, has a real single-provider dependency on Gemini that the
fixed pipeline doesn't share.** The correct fix would be switching the
agent to each provider's actual native function-calling API (a real
`tools` schema, parsing their structured response format) instead of the
hand-rolled "ask for JSON in plain text" convention -- a real redesign,
deliberately left undone here rather than either hiding the problem or
scope-creeping this build week into an OpenAI-function-calling rewrite.
Documented honestly as a known reliability gap, consistent with this
project's practice all session of recording a real limit rather than
quietly working around it.

## Phase 7 — Memory (kept deliberately light, per plan)

- **Short-term memory**: already exists from an earlier week --
  `sessions.py` gives a conversation its running history across turns.
  Not rebuilt for this deliverable; the agent's own within-task history
  (Phase 1) is a separate, shorter-lived form of the same idea.
- **Long-term/vector memory, summarization, mem0**: covered conceptually,
  not implemented -- this task (one question, answered within one loop
  run) never needed memory across separate sessions, and adding a new
  dependency for a feature the actual task doesn't require would be scope
  creep, not rigor.

## Phase 6 — The actual decision

**Ship the fixed pipeline as the default for this app.** The data is
unambiguous: 2.1x faster, 2.25x cheaper, and *more* reliable (20/20 vs.
16/20) on the same 20 questions, including the compound ones the agent was
specifically built to handle better. Every one of the agent's 4 losses
traces to a specific, understood cause (phrasing variance, a fix that
didn't transfer, and a genuine cross-search diversify gap) -- none of them
are "agents are just worse," all of them are "this specific agent, as
built in one week, has specific gaps a fixed pipeline's simpler design
doesn't have to solve."

**Where an agent would still be worth it**: not this task, as built. The
honest reason the race came out this way is Phase 0's own bet not fully
paying off -- decomposing a compound question into separate searches
turned out to *cost* more diversify-safety than it bought in flexibility,
on a system where the fixed pipeline had already been heavily tuned
(Week 6) for exactly this failure mode. An agent would earn its cost on a
task the fixed pipeline structurally cannot do at all (e.g. one requiring
a decision about which of several *different tools*, not just repeated
search, to use) -- not on a task the fixed pipeline can already do,
just less elegantly.

## Summary against the mentor checklist

- ✅ Does the agent complete a genuinely multi-step task, with the steps
  visible? -- yes, verified via real traces (Phase 1), 2-4 steps depending
  on question complexity.
- ✅ Does it stop safely instead of looping forever? -- yes, both a step
  cap and a wall-clock cap, calibrated against real measured latency
  (Phase 4), verified to trigger and recover cleanly.
- ✅ Did they compare it against a plain fixed sequence with real numbers?
  -- yes (Phase 5): speed, cost, and reliability, all measured, not
  estimated.
- ✅ Can they say which one they'd actually ship, and why? -- yes (Phase 6):
  the fixed pipeline, with the specific evidence for each of the agent's
  losses, not a vague preference.
