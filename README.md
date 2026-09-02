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

```
INGESTION                              RETRIEVAL + GENERATION
(build the index)                      (answer a question)

data/*.{txt,md,docx,pdf,xlsx}          question typed by user
        │                                       │
        ▼                                       ▼
   loaders.py                              embed.py
   (extract text                       (embed the question,
    per format)                         same model as docs)
        │                                       │
        ▼                                       ▼
    chunk.py                             vectorstore.py
  (split into                          (search Qdrant for
sentence-aware chunks)                  top-K closest chunks)
        │                                       │
        ▼                                       ▼
    embed.py                              generate.py
 (turn each chunk                    (check distance threshold →
   into a vector)                    refuse OR build prompt → call LLM)
        │                                       │
        ▼                                       ▼
  vectorstore.py                          printed answer
 (delete + rebuild                        + source citation
  Qdrant collection)
        │
        ▼
     Qdrant
  (persists on the
  Docker server)
```

Entry points: `ingest.py` runs the ingestion pipeline, `app.py` runs the
retrieval+generation pipeline interactively, and `api.py` exposes ingestion
(`POST /upload`) and a dependency check (`GET /health`) over HTTP — both
share the same live Qdrant collection as the CLI.

### File-by-file, in pipeline order

```
Documents (data/*.txt, *.md, *.docx, *.pdf, *.xlsx)
        ↓
   loaders.py         -- extracts plain text per format (dispatch by extension)
        ↓
    chunk.py           -- sentence-aware chunking, no mid-word/mid-sentence cuts
        ↓
    embed.py            -- all-MiniLM-L6-v2 (local, free embedding model)
        ↓
 vectorstore.py           -- thin wrapper around Qdrant (build/search/count)
        ↓
    store.py               -- rebuilds the Qdrant collection from scratch each run
        ↓
   retrieve.py               -- top-K semantic similarity search
        ↓
   generate.py                 -- confidence threshold + grounded prompt + OpenRouter LLM
        ↓
      app.py                     -- interactive CLI loop
      api.py                      -- HTTP API: POST /upload, GET /health
```

- **`config.py`** — loads `.env` once; every file below imports its settings
  from here instead of hardcoding a value or calling `os.getenv` itself.
- **`loaders.py`** — one small function per file format (`.txt`, `.md`,
  `.docx`, `.pdf`, `.xlsx`), each returning plain text. Dispatches by file
  extension via the `LOADERS` dict. Unreadable files return `None` instead
  of raising, so one bad file can't crash a whole ingestion run.
- **`chunk.py`** — `load_documents()` walks `data/`, uses `loaders.py` to
  extract text from whatever format each file is, and skips
  unsupported/unreadable files with a `WARNING:`. `split_into_sentences()` +
  `chunk_text()` then split that plain text into overlapping,
  sentence-aligned chunks — completely unaware of which file format the text
  originally came from.
- **`embed.py`** — loads `all-MiniLM-L6-v2` once (`get_embedding_model()`)
  and turns a list of chunk texts into 384-number vectors
  (`embed_chunks()`). Used for both documents (during ingestion) and
  questions (during retrieval) — the same model must embed both sides, or
  the vectors aren't comparable.
- **`vectorstore.py`** — the only file that talks to Qdrant directly.
  `build_collection()` deletes and recreates the collection, then uploads
  every chunk's vector + payload (text/source/chunk_index). `search()` runs
  a similarity query and converts Qdrant's score into a "distance" (lower =
  closer) so the rest of the app has one consistent convention.
- **`store.py`** — the ingestion orchestrator: chunk → embed → store, calling
  the three files above in order. Also owns `get_client()`, reused by
  everything that needs to talk to Qdrant.
- **`ingest.py`** — the actual command you run (`python ingest.py`); a thin
  wrapper that calls `store.build_database()` and prints the result.
- **`retrieve.py`** — embeds a question (with `QUERY_INSTRUCTION` prefixed
  if set — empty for MiniLM), then calls `vectorstore.search()` to get the
  top-K closest chunks.
- **`generate.py`** — calls `retrieve()`, checks the best match's distance
  against `DISTANCE_THRESHOLD` (refuses with no LLM call if it's too far),
  otherwise builds a grounded prompt and calls OpenRouter for the answer.
- **`app.py`** — the interactive CLI: loads the model + Qdrant client once,
  then loops asking questions via `generate.answer_question()`.
- **`api.py`** — `POST /upload` saves a file into `data/` and calls the same
  `store.build_database()` ingestion path as `ingest.py`; `GET /health`
  independently checks the embedding model, Qdrant, the API key, and
  `data/`.

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
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_API_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_MODEL_ID=google/gemma-4-26b-a4b-it:free
DISTANCE_THRESHOLD=0.75
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
QUERY_INSTRUCTION=
DATA_DIR=data
CHUNK_SIZE=300
OVERLAP_SENTENCES=1
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=support_tickets
DEFAULT_TOP_K=3
EMBED_DIM=384
```
Only `OPENROUTER_API_KEY` needs a real value — get a free one at
[openrouter.ai/keys](https://openrouter.ai/keys), no cost, no credit card
needed for `:free` models. Every other value above is already a safe,
working default; `.env` itself stays out of git (see `.gitignore`).

`QUERY_INSTRUCTION` stays empty for MiniLM (it doesn't need an asymmetric
query prefix). It exists as a setting for future BGE-family models, which
do — see [Key design decisions](#key-design-decisions).

## Usage

Build/rebuild the vector database (run after adding, editing, or removing
files in `data/`, or after changing `EMBEDDING_MODEL_NAME`):
```bash
python ingest.py
```
This fully rebuilds the collection each time (delete + recreate), so it's
always safe to re-run — no stale chunks left behind from renamed/removed
files, and no risk of mixing vectors from two different embedding models.

Ask questions interactively:
```bash
python app.py
```

Compare retrieval quality across chunk sizes (no API calls, free/local only
— uses an in-memory Qdrant instance, doesn't touch your real data):
```bash
python compare_chunk_sizes.py
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
  reachable (with current chunk count), the OpenRouter key is present, and
  `data/` exists. Does **not** call OpenRouter itself, to avoid spending
  free-tier quota just on a health check — it only confirms the key is
  present, not that it's valid.

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
├── loaders.py               # per-format text extraction (dispatch by file extension)
├── chunk.py                  # document loading (via loaders.py) + sentence-aware chunking
├── embed.py                   # embedding model (all-MiniLM-L6-v2)
├── vectorstore.py               # thin wrapper around Qdrant (build/search/count)
├── store.py                      # ingestion pipeline: chunk -> embed -> store in Qdrant
├── ingest.py                      # the command you actually run to (re)build the index
├── retrieve.py                     # top-K semantic retrieval
├── generate.py                      # grounded generation, citations, refusal logic
├── app.py                            # interactive CLI entry point
├── api.py                             # HTTP API: POST /upload, GET /health
├── compare_chunk_sizes.py              # chunk size experiment (Step 9)
├── FINDINGS.md                          # chunk-size results, bugs caught, limitations
├── requirements.txt
└── .env                                  # all config incl. OPENROUTER_API_KEY (not committed)
```

## Key design decisions

- **Sentence-aware chunking** — chunks always end on a complete sentence,
  never mid-word. Splitting on paragraph breaks first avoids un-punctuated
  headings (e.g. "Subject: ...") swallowing the whole first paragraph.
- **Multi-format ingestion, one dispatch table** — `loaders.py` maps file
  extension → extraction function (`.txt`/`.md` read as-is, `.docx` pulls
  paragraphs + table cells, `.pdf` extracts text per page with no OCR,
  `.xlsx` turns each row into `"Column: value"` sentences using the header
  row). Unsupported or unreadable files are skipped with a warning instead
  of crashing the whole ingestion run.
- **`all-MiniLM-L6-v2` embeddings** — small, free, local, no asymmetric
  query-prefix convention needed (unlike BGE-family models, which the
  `QUERY_INSTRUCTION` setting exists to support if swapped back in later).
- **Qdrant, rebuilt on every ingest** — `store.py` deletes and recreates the
  collection each run rather than upserting. This avoids two problems at
  once: stale chunks lingering from renamed/removed source files, and
  vectors from two different embedding models silently mixing in one
  collection (same dimension does not mean same vector space).
- **Confidence threshold before generation** — if the closest retrieved chunk
  is farther than `DISTANCE_THRESHOLD`, the app skips the LLM call entirely
  and returns "I don't know" — the model never even sees an out-of-scope
  question, so it can't hallucinate an answer to it.
- **Centralized config** — every AI-related setting lives in `.env`, loaded
  once through `config.py`. No other file calls `os.getenv`/`load_dotenv`
  or hardcodes a model name/URL/threshold.
- **Free-tier aware** — no retry/backoff logic is included yet (deliberately
  deferred); the OpenRouter free-tier rate limit (20/min, 50/day on $0
  accounts) is a real constraint to be aware of when testing.
