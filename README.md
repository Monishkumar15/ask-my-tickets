# Ask My Tickets

A small **RAG (Retrieval-Augmented Generation)** app: load customer support
ticket documents, ask a question, and get an answer grounded *only* in those
documents — with a citation, or an honest "I don't know" if the answer isn't
there.

Built as a Week 3 learning project on retrieval and RAG.

See [`FINDINGS.md`](FINDINGS.md) for the chunk-size comparison results, a
real chunking bug caught and fixed during development, and honest known
limitations.

## How it works

```
Documents (data/*.txt)
        ↓
    chunk.py          -- sentence-aware chunking, no mid-word/mid-sentence cuts
        ↓
    embed.py           -- BAAI/bge-small-en-v1.5 (local, free embedding model)
        ↓
    store.py            -- ChromaDB (local persistent vector database)
        ↓
   retrieve.py           -- top-K semantic similarity search
        ↓
   generate.py            -- confidence threshold + grounded prompt + OpenRouter LLM
        ↓
      app.py                -- interactive CLI loop
```

## Setup

```bash
python -m venv venv
.\venv\Scripts\Activate      # Windows
pip install sentence-transformers chromadb requests python-dotenv
```

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your-openrouter-key-here
```
Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys) — no cost,
no credit card needed for `:free` models.

## Usage

Build the vector database (run once, or after changing `data/`):
```bash
python store.py
```

Ask questions interactively:
```bash
python app.py
```

Compare retrieval quality across chunk sizes (no API calls, free/local only):
```bash
python compare_chunk_sizes.py
```

## Project structure

```
ask-my-tickets/
├── data/                  # sample support ticket documents
├── chunk.py               # document loading + sentence-aware chunking
├── embed.py                # embedding model + query/document encoding
├── store.py                 # ChromaDB vector storage
├── retrieve.py                # top-K semantic retrieval
├── generate.py                 # grounded generation, citations, refusal logic
├── app.py                       # interactive CLI entry point
├── compare_chunk_sizes.py        # chunk size experiment (Step 9)
├── FINDINGS.md                    # chunk-size results, bugs caught, limitations
├── requirements.txt
└── .env                          # OPENROUTER_API_KEY (not committed)
```

## Key design decisions

- **Sentence-aware chunking** — chunks always end on a complete sentence,
  never mid-word. Splitting on paragraph breaks first avoids un-punctuated
  headings (e.g. "Subject: ...") swallowing the whole first paragraph.
- **Asymmetric BGE encoding** — documents are embedded as-is; queries get an
  instruction prefix (`"Represent this sentence for searching relevant
  passages: "`), matching how the BGE model family was trained.
- **Confidence threshold before generation** — if the closest retrieved chunk
  is farther than `DISTANCE_THRESHOLD`, the app skips the LLM call entirely
  and returns "I don't know" — the model never even sees an out-of-scope
  question, so it can't hallucinate an answer to it.
- **Free-tier aware** — no retry/backoff logic is included yet (deliberately
  deferred); the OpenRouter free-tier rate limit (20/min, 50/day on $0
  accounts) is a real constraint to be aware of when testing.
