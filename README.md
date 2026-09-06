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
```
`GEMINI_API_KEY` and `GROQ_API_KEY` both need real values — get a free
Gemini key at [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)
and a free Groq key at [console.groq.com/keys](https://console.groq.com/keys).
Gemini is tried first by default; Groq is the automatic fallback if Gemini
returns a 429 (quota/rate-limit) — no other kind of error triggers a
silent retry on the other provider. Every other value above is already a
safe, working default; `.env` itself stays out of git (see `.gitignore`).

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
├── evals/                          # diagnostic/comparison scripts (not part of the runtime pipeline)
│   ├── eval_questions.py             # labeled eval set (question + expected source/keywords)
│   ├── compare_chunk_sizes.py          # chunk size experiment (Step 9)
│   ├── compare_before_after.py           # semantic vs hybrid vs hybrid+rerank, hit-rate@k + MRR
│   ├── classify_failures.py                # retrieval-failure vs generation-failure classification
│   └── inspect_retrieval.py                  # interactive side-by-side inspection view
├── ui-react/                        # React/Vite frontend (upload, ask, retrieve, trace review)
├── FINDINGS.md              # Week 3 chunk-size results, bugs caught, limitations
├── FINDINGS_week4.md          # Week 4 hybrid-search diagnosis and results
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
