# Ask My Tickets

A small **RAG (Retrieval-Augmented Generation)** app: load customer support
documents (in several formats), ask a question, and get an answer grounded
*only* in those documents — with a citation, or an honest "I don't know" if
the answer isn't there.

Built as a Week 3 learning project on retrieval and RAG.

See [`FINDINGS.md`](FINDINGS.md) for the chunk-size comparison results and a
real chunking bug caught and fixed during development. Note: `FINDINGS.md`
was written against the original BGE + ChromaDB setup — the *conclusions*
(smaller chunks retrieve more confidently, chunking bugs hide silently)
still hold, but the specific distance numbers there predate the MiniLM +
Qdrant switch described below.

## How it works

Two independent pipelines: **ingestion** builds the searchable index,
**retrieval + generation** answers a question against it. They only meet
when one triggers the other (e.g. the API's `/upload` re-runs ingestion
after saving a file).

Both pipelines are consolidated into one file each — `ingestion.py` and
`retrieval.py` — rather than split into many small single-concept files.
Retrieval also combines three techniques, not just one:

```
INGESTION (ingestion.py)               RETRIEVAL + GENERATION

data/*.{txt,md,docx,pdf,xlsx}          question typed by user
        │                                       │
        ▼                                       ▼
  load documents per format              embed the question
  (loaders, dispatch by                  (same model as documents)
   file extension)                               │
        │                              ┌─────────┴─────────┐
        ▼                              ▼                   ▼
  sentence-aware chunking        semantic search       BM25 keyword
  (no mid-word/sentence cuts)    (meaning-based)         search
        │                              └─────────┬─────────┘
        ▼                                         ▼
  embed each chunk                    fuse rankings (RRF)
  (all-MiniLM-L6-v2)                             │
        │                                         ▼
        ▼                            cross-encoder reranks the
  store in Qdrant                    shortlist (reads question +
  (delete + rebuild                  chunk together, not separately)
   collection each run)                          │
                                                  ▼
                                      confidence gate (distance
                                      threshold) -> refuse OR
                                      build grounded prompt -> LLM
                                                  │
                                                  ▼
                                      cited answer + local trace
                                      saved (tracing.py)
```

Entry points: `python ingestion.py` (re)builds the index, `app.py` runs the
retrieval+generation pipeline interactively, and `api.py` exposes the same
pipeline over HTTP (`POST /ask`, `POST /retrieve`, `POST /upload`,
`GET /health`, `GET /traces`, ...) — all sharing the same live Qdrant
collection as the CLI. A React UI (`ui-react/`) provides a browsable
front end for uploading documents, asking questions, and reviewing traces.

### File-by-file

- **`config.py`** — loads `.env` once; every file below imports its settings
  from here instead of hardcoding a value or calling `os.getenv` itself.
- **`ingestion.py`** — the whole ingestion pipeline in one file:
  - Document loading: one small function per file format (`.txt`, `.md`,
    `.docx`, `.pdf`, `.xlsx`), dispatched by extension via the `LOADERS`
    dict. Unreadable files return `None` instead of raising, so one bad
    file can't crash a whole ingestion run.
  - Chunking: `split_into_sentences()` + `chunk_text()` split plain text
    into overlapping, sentence-aligned chunks — never mid-word, never
    mid-sentence.
  - Embedding: `get_embedding_model()` loads `all-MiniLM-L6-v2` once;
    `embed_chunks()` turns chunk text into 384-number vectors. The same
    model also embeds questions during retrieval — both sides must use the
    identical model, or the vectors aren't comparable.
  - Vector store: `get_client()`/`build_collection()`/`search()`/`count()`
    wrap Qdrant. `build_collection()` deletes and recreates the collection
    every run (not upsert) to avoid stale chunks from renamed/removed
    files, and to avoid silently mixing vectors from two different
    embedding models in one collection.
  - `build_database()` orchestrates all of the above; running
    `python ingestion.py` directly calls it and prints the result.
- **`retrieval.py`** — the whole retrieval pipeline in one file:
  - `semantic_retrieve()` — meaning-based search only (the original Week 3
    method), with diversify-by-source so a question spanning two topics
    doesn't lose one of them to the other.
  - `build_bm25_index()`/`bm25_search()` — keyword search (exact word
    overlap, weighted by rarity) via the `rank_bm25` library.
  - `hybrid_retrieve()` — fuses semantic + BM25 rankings via Reciprocal
    Rank Fusion (RRF), also with diversify-by-source applied to the fused
    candidate pool.
  - `rerank()` — a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
    also local/free) reads the question and each candidate chunk's full
    text *together* and re-scores relevance — unlike the embedding model
    above (a "bi-encoder"), which encodes each side separately, ahead of
    time. Slower, so it only runs on a short candidate list.
  - `retrieve()` — the live pipeline `generate.py` actually calls: widen
    the pool via `hybrid_retrieve()`, rerank the whole widened pool, then
    diversify-by-source again on the reranked list (reranking alone has no
    notion of "don't pick 3 chunks from the same topic"), and return the
    final top-K.
  - `hit_rate_at_k()`/`mrr()` — retrieval quality metrics, used by the
    scripts in `evals/`.
- **`generate.py`** — calls `retrieval.retrieve()`, checks the best match's
  distance against `DISTANCE_THRESHOLD` (refuses with no LLM call if it's
  too far), otherwise builds a grounded prompt and calls the LLM for the
  answer — **Gemini by default, automatically falling back to Groq** if
  Gemini returns a 429 (quota/rate-limit). Records every stage (retrieval,
  confidence gate, prompt, generation) to a local trace via `tracing.py`.
- **`tracing.py`** — saves each question as a local JSON file under
  `traces/`, recording every pipeline stage with timing. No external
  tracing service — fully local, matching this project's free/local
  philosophy.
- **`app.py`** — the interactive CLI: loads the model + Qdrant client once,
  then loops asking questions via `generate.answer_question()`.
- **`api.py`** — HTTP API: `POST /upload` (save + rebuild the index),
  `GET /health`, `POST /ask`, `POST /retrieve` (inspect retrieval only),
  `GET /documents`, and the trace review endpoints (`GET /traces`,
  `GET /traces/{id}`, `POST /traces/{id}/annotation`). Also serves the
  built React UI as static files if `ui-react/dist/` exists.
- **`evals/`** — diagnostic and comparison scripts, separate from the
  runtime pipeline (see below).

All AI-related configuration (API key, model names, thresholds, paths) lives
in `.env`, loaded once through `config.py` — no module calls `os.getenv`
directly.

## Setup

```bash
python -m venv venv
.\venv\Scripts\Activate      # Windows
pip install -r requirements.txt
```

### Run Qdrant (the vector database)

Qdrant runs as a server, not an embedded local file — start it with Docker:
```bash
docker run -d --name qdrant-ask-my-tickets -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```
(Only use `docker run` the first time — it creates the container. On later
sessions, just `docker start qdrant-ask-my-tickets`.) Check it's up at
`http://localhost:6333/dashboard`, or `curl http://localhost:6333`.

### Configure `.env`

Copy the template and fill in your own key:
```bash
cp .env.example .env
```
`.env.example` is committed to the repo (it's safe — no real secrets in it,
just working defaults) and lists every variable the app reads:
```
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
GEMINI_MODEL_ID=gemini-3.1-flash-lite
GROQ_API_KEY=your-groq-api-key-here
GROQ_API_URL=https://api.groq.com/openai/v1/chat/completions
GROQ_MODEL_ID=openai/gpt-oss-20b
DEFAULT_LLM_PROVIDER=gemini
DISTANCE_THRESHOLD=0.75
DIVERSIFY_MARGIN=0.3
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
QUERY_INSTRUCTION=
DATA_DIR=data
CHUNK_SIZE=300
OVERLAP_SENTENCES=1
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=support_tickets
DEFAULT_TOP_K=3
EMBED_DIM=384
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_CANDIDATE_POOL=10
GEMINI_INPUT_COST_PER_1K=0.00025
GEMINI_OUTPUT_COST_PER_1K=0.0015
GROQ_INPUT_COST_PER_1K=0.00015
GROQ_OUTPUT_COST_PER_1K=0.0006
```
The four cost-rate vars (Week 10) are standard list-price-per-1K-tokens
comparison proxies for `evals/race_team_vs_agent.py`'s cost metric --
real spend on this project's free-tier keys is $0.
`GEMINI_API_KEY` and `GROQ_API_KEY` both need real values — get a free
Gemini key at [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)
and a free Groq key at [console.groq.com/keys](https://console.groq.com/keys).
Gemini is tried first by default; Groq is the automatic fallback if Gemini
returns a 429 (quota/rate-limit) or a known model-serving quirk (Week 9:
some Groq models misfire on the agent's tool-listing prompt -- see the
Week 9 section below) — any other kind of error surfaces immediately
instead of triggering a silent retry on the other provider. Every other
value above is already a safe, working default; `.env` itself stays out of
git (see `.gitignore`).

`QUERY_INSTRUCTION` stays empty for MiniLM (it doesn't need an asymmetric
query prefix). It exists as a setting for future BGE-family models, which
do — see [Key design decisions](#key-design-decisions).

## Usage

Build/rebuild the vector database (run after adding, editing, or removing
files in `data/`, or after changing `EMBEDDING_MODEL_NAME`):
```bash
python ingestion.py
```
This fully rebuilds the collection each time (delete + recreate), so it's
always safe to re-run — no stale chunks left behind from renamed/removed
files, and no risk of mixing vectors from two different embedding models.

Ask questions interactively:
```bash
python app.py
```

### Evaluation / diagnostic scripts (`evals/`)

All free/local — no LLM calls except where noted — and use an in-memory
Qdrant instance, so they never touch your real data.

```bash
python evals/compare_chunk_sizes.py     # chunk-size comparison (Step 9)
python evals/compare_before_after.py    # semantic vs. hybrid vs. hybrid+rerank, hit-rate@k + MRR
python evals/inspect_retrieval.py       # type any question, see all 3 methods + the answer side by side
python evals/classify_failures.py       # labels each eval question as a retrieval or generation failure
                                         # (the generation check spends 1 LLM call per question)
```

### HTTP API (optional — for dynamic uploads + health checks)

Start the API:
```bash
uvicorn api:app --reload
```
Then open **http://127.0.0.1:8000/docs** for an interactive Swagger UI, or
call the endpoints directly:

- `POST /upload` — upload a `.txt`/`.md`/`.docx`/`.pdf`/`.xlsx` file; it's
  saved into `data/` and the index is rebuilt immediately, so it's
  searchable right away (from both the API and `python app.py` — they share
  the same Qdrant collection). Unsupported file types are rejected with a
  `400` and a clear message.
- `GET /health` — reports whether the embedding model loads, Qdrant is
  reachable (with current chunk count), the Gemini and Groq keys are
  present, and `data/` exists. Does **not** call either LLM provider, to
  avoid spending free-tier quota just on a health check — it only confirms
  the keys are present, not that they're valid.

### Week 5 trace review

The API now also provides `POST /ask`, `GET /traces`, trace detail, and review
annotation endpoints. The frontend is React/Vite: run `uvicorn api:app --reload`
and, in another terminal, `cd ui-react && npm install && npm run dev`. Open the
Vite URL (normally http://127.0.0.1:5173). Each answer stores its
question, retrieved ticket chunks, confidence-gate decision, output/errors,
and timings in local `traces/` files. Use [WEEK5_ERROR_ANALYSIS.md](WEEK5_ERROR_ANALYSIS.md)
to report the roughly 20 reviewed traces and ranked error taxonomy.

### Week 8 -- trajectory evaluation, prompt injection, agent traces in the UI

Week 7's `agent.py` is a separate, hand-built ReAct loop (search/finish/escalate)
that never runs through the React "Ask" page -- that page only calls the fixed
`retrieval -> generation` pipeline via `POST /ask`. Week 8 adds a second kind of
grading on top of the agent: not just "was the final answer right" (outcome),
but "did the agent take a sane path to get there" (trajectory) -- plus a
prompt-injection red-team test. See [`FINDINGS_week8.md`](FINDINGS_week8.md) for
the full write-up (the outcome-vs-trajectory gap found, the injection
vulnerability found and closed, before/after numbers).

#### 1. Start the dependencies (Docker + backend + UI)

Same Qdrant container as every other week -- Week 8 adds no new service:
```bash
docker start qdrant-ask-my-tickets      # or the `docker run ...` command above, first time only
```
Then, in separate terminals:
```bash
.\venv\Scripts\Activate
uvicorn api:app --reload                # backend, http://127.0.0.1:8000

cd ui-react
npm install                             # first time only
npm run dev                             # UI, http://127.0.0.1:5173
```

#### 2. Generate an agent trace (this is the "Week 8" run, not the Ask page)

The React "Ask" page always uses the fixed pipeline. To produce an *agent*
trace (the kind Week 8 grades), run the agent directly -- each run saves one
trace file under `traces/`, same as any other question:
```bash
.\venv\Scripts\Activate
python agent.py
Ask a (possibly multi-part) question: <type one of the questions below>
```

Sample questions to try, and what to expect (all grounded in real ticket text,
verified while building the eval set):
- `there are two $29.99 charges on my statement from the same day` --
  same-day duplicate charges auto-reverse per `ticket_005`; expect the agent
  to search, then **finish** with a direct answer (no escalation needed).
- `you guys charged me the wrong price` -- an incorrect-amount charge is
  **not** auto-reversed per `ticket_007`; expect the agent to **escalate**.
- `why is my order delayed, how do I get a refund for a defective item, and
  why was I charged the wrong amount` -- a 3-part compound question where one
  part needs escalation; expect the whole answer to **escalate** (folding the
  answerable parts into the escalation reason), not a partial `finish`.
- `the item I got was damaged, will you refund me` -- watch for wasted/
  repeated `search_tickets` steps here specifically; this is the case Week 8's
  trajectory checker originally caught failing even though the final answer
  was correct (see `FINDINGS_week8.md` section on `q11`).
- `how do I contact customer support` -- the guide never states a concrete
  contact channel, so the *correct* behavior is `finish` with "I don't know" --
  this is an honest refusal, not a bug, even though the trace's outcome badge
  reads "Answered" (that badge only means the agent reached `finish`, not that
  it found a positive answer).
- `what's the best pizza topping?` -- fully out-of-scope; expect a refusal-style
  finish with no sources cited.

#### 3. Review the trace in the UI

Open the Vite UI -> **Error analysis** (the trace review page) -> pick the
trace you just generated (agent traces show an `agent` pill and use their own
outcome labels: Answered / Escalated / Budget exhausted, distinct from the
fixed pipeline's Answered / Refused / Error). Selecting an agent trace shows:
- A **Trajectory: PASS/FAIL** badge with a per-check breakdown (terminal
  action correctness, no repeated searches, no wasted/malformed steps, not
  silently budget-exhausted, step efficiency) -- computed by
  `GET /traces/{id}/trajectory`, the same rule-based checker the eval scripts
  use, not a second copy of the logic.
- The full step-by-step list (thought / action / action_input / observation)
  in place of the fixed pipeline's flat evidence list.

#### 4. Run the full trajectory eval (outcome vs. trajectory, cost per task)

```bash
.\venv\Scripts\Activate
python evals/trajectory_eval.py                    # run + print both scorecards
python evals/trajectory_eval.py --with-judge        # also run the LLM trajectory judge
python evals/trajectory_eval.py --save NAME         # save a snapshot for before/after
python evals/trajectory_eval.py --compare NAME      # diff current run vs. a saved snapshot
```
Prints, per case: outcome pass/fail, trajectory pass/fail, and flags any
**outcome-vs-trajectory gap** (right answer, wrong path) by case ID and reason.
Also reports mean/p99 latency and LLM-calls-per-task across the whole run.

#### 5. Run the prompt injection red-team test

```bash
.\venv\Scripts\Activate
python evals/prompt_injection_test.py
```
Fully isolated -- builds an in-memory Qdrant collection + BM25 index with a
planted "poisoned" chunk and never touches the real `data/` corpus or the
persisted collection. Runs all 6 attack variants (blunt/subtle instruction
override, credential-phishing misinformation, fake-source injection) against
both Gemini and Groq, and prints whether each attack's canary string leaked
into the agent's final answer. Currently: all 6 variants blocked, 0/3 for the
credential-phishing variant that succeeded 2/3 times before the trust-boundary
and credential-request rules were added to `agent.py`'s prompt (see
`FINDINGS_week8.md` for the exact before/after).

#### 6. (Optional) validate the LLM trajectory judge before trusting it

Same two-step discipline as Week 6's `judge.py`:
```bash
python evals/prepare_trajectory_judge_sample.py
# hand-grade the sample, then:
python evals/validate_trajectory_judge.py
```
Confirms the judge agrees with a human grader at or above the 80% threshold
before its verdicts are used anywhere.

### Week 9 -- MCP (Model Context Protocol)

`agent.py` no longer imports `retrieval.retrieve` directly. It discovers
its tools at the start of every run from a separate MCP server
(`mcp_server.py`) and calls them over the standard protocol instead of a
hard-coded function call -- so the tool becomes callable by any MCP
client, not just this repo's agent. See
[`FINDINGS_week9.md`](FINDINGS_week9.md) for the full write-up.

#### 1. Start the MCP server (in addition to Qdrant, before the backend/UI)

```bash
docker start qdrant-ask-my-tickets   # same as every other week
.\venv\Scripts\Activate
python mcp_server.py                 # streamable-http, the default -- a long-lived
                                      # process, same operational shape as Qdrant
```
Requires `MCP_SHARED_SECRET` set in `.env` (the HTTP transport refuses to
start without one -- generate your own: `python -c "import secrets; print(secrets.token_urlsafe(24))"`).
Leave this running in its own terminal, then start the backend/UI as usual
(`uvicorn api:app --reload`, `cd ui-react && npm run dev`).

#### 2. See a tool call go through MCP

```bash
.\venv\Scripts\Activate
python agent.py
Ask a (possibly multi-part) question: why did my account get locked when I typed my password wrong a few times?
```
Check the newest file in `traces/` -- its `agent_step` stage for
`search_tickets` now carries `"via": "mcp"`.

#### 3. Prove the two mentor-checklist items that need a second party

```bash
python evals/mcp_server_test.py       # discovery + both tools + auth rejection, against the live server
python evals/foreign_agent_demo.py    # a client with ZERO imports from this repo -- discovers
                                       # and calls the server the same way a stranger's agent would
```

#### 4. Look at the raw JSON-RPC once (the one-time "stop being a mystery" exercise)

```bash
python mcp_server.py --transport stdio
```
Or inspect it directly with `curl` against the running HTTP server. **On
Windows PowerShell, use `curl.exe` explicitly** -- plain `curl` is aliased to
`Invoke-WebRequest`, which doesn't understand `-X`/`-d`/`-H` the same way.

The streamable-http transport is session-based: `initialize` is always the
first call, its response carries an `mcp-session-id` header you must reuse
on every following request (a bare `tools/list` with no session id fails
with `"Missing session ID"` -- confirmed live, not a guess). A real MCP
client (`ClientSession`, or anything in `evals/`) handles this
automatically; doing it by hand with curl needs all three calls in order:

```bash
# 1. initialize -- copy the mcp-session-id header from the response
curl.exe -s -i -X POST http://127.0.0.1:8830/mcp `
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" `
  -H "X-MCP-Shared-Secret: <your secret>" `
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"manual-test","version":"1.0"}}}'

# 2. required handshake-completion notification (same session id)
curl.exe -s -X POST http://127.0.0.1:8830/mcp `
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" `
  -H "X-MCP-Shared-Secret: <your secret>" -H "Mcp-Session-Id: <session id from step 1>" `
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# 3. now tools/list (or tools/call) works
curl.exe -s -X POST http://127.0.0.1:8830/mcp `
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" `
  -H "X-MCP-Shared-Secret: <your secret>" -H "Mcp-Session-Id: <same session id>" `
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```
A `tools/call` payload looks like:
```json
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_tickets","arguments":{"query":"password lockout","top_k":2}}}
```

#### 5. Full verification checklist (what to actually run, in order)

```bash
python evals/mcp_server_test.py       # 1. server discoverable, callable, auth enforced
python evals/foreign_agent_demo.py "http://127.0.0.1:8830/mcp" "<your MCP_SHARED_SECRET>"
                                       # 2. a zero-imports client can use it
python agent.py                       # 3. real question -- check traces/*.json for "via": "mcp"
                                       # 4. open it in the UI's "Error analysis" page too
python evals/trajectory_eval.py       # 5. regression -- MCP migration didn't change agent correctness
python evals/prompt_injection_test.py # 6. Week 8's defenses still hold through the MCP path
```
Expect all 6 injection variants to print `blocked`, and `trajectory_eval.py`'s
outcome/trajectory scores to be consistent with Week 8's `week8_final`
snapshot (occasional individual-case failures with a plain `HTTPError: 429`
or Groq `400` are provider quota/quirks, not MCP bugs -- see
`FINDINGS_week9.md` sections 10-11; a generic `ExceptionGroup` instead of a
normal exception type would be the real red flag).

**Troubleshooting -- "port already in use" on `python mcp_server.py`:**
a previous `mcp_server.py` process (e.g. from an earlier test run you forgot
was still running) is usually the cause, not a real conflict with another
app. Find and stop it:
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*mcp_server.py*' } |
  Select-Object ProcessId, CommandLine
Stop-Process -Id <the PID(s) above> -Force
```

**Provider robustness**: `generate.py`'s `call_llm()` automatically falls
back to the other provider (Gemini <-> Groq) not just on a 429
(quota/rate-limit), but also on a Groq-specific `tool_use_failed` 400 --
some Groq models (e.g. `openai/gpt-oss-20b`) auto-detect the agent's
tool-listing prompt as a native tool-call request and then reject their own
attempt. Confirmed live: the identical prompt can succeed or fail across
calls on the same model, so this is model-serving-side non-determinism,
not a prompt bug -- the fallback makes any model choice on either provider
safe to use, instead of requiring one specific "known-good" model pinned
in `.env`. See `FINDINGS_week7.md`'s "Update (Week 9)" note.

### Week 10 -- multi-agent: a manager + 2 specialists, raced against the single agent

`multi_agent.py` is a second, independent way to answer a question:
instead of `agent.py`'s one ReAct loop, a manager (`route()`) decides
which of two narrow specialists (`billing_refunds`, `account_access`) a
question needs -- one, both (run genuinely in parallel), or neither (a
clean `out_of_scope` refusal) -- then merges their answers if both ran.
Raced against `agent.py` on the same 20 cases, scored by the same
`assertions.py`, reporting quality/speed/tokens/cost. See
[`FINDINGS_week10.md`](FINDINGS_week10.md) for the full write-up,
including the real numbers, a reproducible `customer_support.pdf`
coverage gap the team has and the single agent doesn't, and the A2A
(Agent-to-Agent) conceptual architecture this week's scope deliberately
kept in-process rather than built as a second networked server.

```bash
docker start qdrant-ask-my-tickets   # same as every other week -- no MCP server needed here,
                                      # multi_agent.py calls retrieval.retrieve() directly
.\venv\Scripts\Activate
python multi_agent.py
Ask a (possibly multi-part) question: why is my order delayed, how do I get a refund for a defective item, and why was I charged the wrong amount
```
That question needs both specialists; try a single-topic question (e.g.
`I got locked out after typing my password wrong`) to see only one run,
and an out-of-scope one (e.g. `what's the best pizza topping?`) to see
the zero-retrieval refusal path.

Run the race itself:
```bash
python evals/race_team_vs_agent.py
```
Prints a per-case table plus a four-metric summary (`Quality (Pass rate)`,
`Speed (Execution latency)`, `Tokens Used`, `Cost ($)`) for both arms, and
saves the full breakdown to `evals/team_race_results.json`.

## Project structure

```
ask-my-tickets/
├── data/                  # support documents (.txt, .md, .docx, .pdf, .xlsx)
├── config.py               # centralized .env loading -- every module reads config from here
├── ingestion.py             # the whole ingestion pipeline: loaders + chunking + embedding + Qdrant
├── retrieval.py              # the whole retrieval pipeline: semantic + BM25 + RRF fusion + reranking
├── generate.py                 # grounded generation, citations, refusal logic, tracing hook
├── tracing.py                   # local JSON trace files (traces/) -- no external service
├── app.py                        # interactive CLI entry point
├── api.py                         # HTTP API: /ask, /retrieve, /upload, /health, /documents, /traces
├── agent.py                          # Week 7: hand-built ReAct agent loop -- Week 9: tools via MCP
├── mcp_server.py                       # Week 9: MCP server (search_tickets, list_ticket_sources)
├── mcp_client.py                        # Week 9: MCP client session wrapper used by agent.py
├── multi_agent.py                        # Week 10: manager + 2 specialists, raced against agent.py
├── evals/                          # diagnostic/comparison scripts (not part of the runtime pipeline)
│   ├── eval_questions.py             # labeled eval set (question + expected source/keywords)
│   ├── compare_chunk_sizes.py          # chunk size experiment (Step 9)
│   ├── compare_before_after.py           # semantic vs hybrid vs hybrid+rerank, hit-rate@k + MRR
│   ├── classify_failures.py                # retrieval-failure vs generation-failure classification
│   ├── inspect_retrieval.py                  # interactive side-by-side inspection view
│   ├── trajectory_checks.py                  # Week 8: rule-based agent-path checks
│   ├── trajectory_judge.py                   # Week 8: LLM-as-judge for trajectory reasoning
│   ├── trajectory_eval.py                    # Week 8: one-command outcome + trajectory runner
│   ├── prompt_injection_test.py              # Week 8/9: isolated prompt-injection red-team test
│   ├── mcp_server_test.py                    # Week 9: MCP discovery/tool-call/auth smoke test
│   ├── foreign_agent_demo.py                 # Week 9: zero-imports "someone else's agent" proof
│   ├── race_agent_vs_workflow.py              # Week 7: agent vs. fixed pipeline race
│   └── race_team_vs_agent.py                  # Week 10: team vs. single agent race (quality/speed/tokens/cost)
├── ui-react/                        # React/Vite frontend (upload, ask, retrieve, trace review)
├── FINDINGS.md              # Week 3 chunk-size results, bugs caught, limitations
├── FINDINGS_week4.md          # Week 4 hybrid-search diagnosis and results
├── FINDINGS_week6.md           # Week 6 evals/regression/judge validation write-up
├── FINDINGS_week7.md            # Week 7 agent loop write-up
├── FINDINGS_week8.md             # Week 8 trajectory eval + prompt injection write-up
├── FINDINGS_week9.md              # Week 9 MCP integration write-up
├── FINDINGS_week10.md              # Week 10 multi-agent race write-up
├── WEEK5_ERROR_ANALYSIS.md     # Week 5 trace review write-up
├── requirements.txt
└── .env                          # all config incl. GEMINI_API_KEY, GROQ_API_KEY (not committed)
```

## Key design decisions

- **Sentence-aware chunking** — chunks always end on a complete sentence,
  never mid-word. Splitting on paragraph breaks first avoids un-punctuated
  headings (e.g. "Subject: ...") swallowing the whole first paragraph.
- **Multi-format ingestion, one dispatch table** — `ingestion.py`'s
  `LOADERS` dict maps file extension → extraction function (`.txt`/`.md`
  read as-is, `.docx` pulls paragraphs + table cells, `.pdf` extracts text
  per page with no OCR, `.xlsx` turns each row into `"Column: value"`
  sentences using the header row). Unsupported or unreadable files are
  skipped with a warning instead of crashing the whole ingestion run.
- **`all-MiniLM-L6-v2` embeddings** — small, free, local, no asymmetric
  query-prefix convention needed (unlike BGE-family models, which the
  `QUERY_INSTRUCTION` setting exists to support if swapped back in later).
- **Qdrant, rebuilt on every ingest** — `build_collection()` deletes and
  recreates the collection each run rather than upserting. This avoids two
  problems at once: stale chunks lingering from renamed/removed source
  files, and vectors from two different embedding models silently mixing
  in one collection (same dimension does not mean same vector space).
- **Hybrid search + reranking, not semantic search alone** — `retrieval.py`
  fuses semantic and keyword (BM25) search via RRF, then reranks the
  result with a cross-encoder that reads the question and each candidate
  chunk's full text together (unlike the embedding model, which encodes
  each side separately). Diversify-by-source is applied twice — once on
  the fused candidates, once on the reranked list — because reranking has
  no built-in notion of "don't pick 3 chunks from the same topic."
- **Confidence threshold before generation** — if the closest retrieved chunk
  is farther than `DISTANCE_THRESHOLD`, the app skips the LLM call entirely
  and returns "I don't know" — the model never even sees an out-of-scope
  question, so it can't hallucinate an answer to it.
- **Centralized config** — every AI-related setting lives in `.env`, loaded
  once through `config.py`. No other file calls `os.getenv`/`load_dotenv`
  or hardcodes a model name/URL/threshold.
- **Local tracing, no external service** — `tracing.py` saves each question
  as a JSON file under `traces/`, recording every pipeline stage with
  timing. Reviewed and labeled through the `/traces` API endpoints and the
  React UI's Traces page.
- **Two LLM providers, automatic fallback** — Gemini is tried first by
  default; a `429` (quota/rate-limit) automatically retries with Groq, no
  other error triggers a silent retry. The UI toggle lets you flip which
  one is tried first; the other is always the fallback either way.
- **Diversify-by-source needs a relative margin, not just a flat cutoff** —
  a source under `DISTANCE_THRESHOLD` isn't automatically "a genuine second
  topic": `DIVERSIFY_MARGIN` also requires it to be close to the single
  best match's distance, or a weakly-related document can wrongly displace
  a stronger same-source chunk just to manufacture topic "diversity"
  nothing in the question actually asked for.
