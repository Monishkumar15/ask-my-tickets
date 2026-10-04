"""
Central place for all AI-related configuration. Loaded once from .env (with
sensible defaults for anything not set), so no other file calls load_dotenv()
or os.getenv() directly.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM providers: Gemini (primary) + Groq (automatic fallback) ---
# Both expose an OpenAI-compatible chat-completions endpoint, so the same
# request shape (Bearer token + {model, messages}) works for either --
# see generate.py's _call_provider().
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.1-flash-lite")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-20b")

# Which provider is tried FIRST by default ("gemini" or "groq"). The UI
# toggle overrides this per-request; falling back to the other provider
# only happens on a 429 (quota/rate-limit) from whichever is tried first.
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")

# How random/deterministic the LLM's wording is, independent of which
# provider answers -- 0 is closest to deterministic (always picks the most
# likely next word), higher values allow more varied phrasing. 1.0 matches
# the OpenAI-compatible API default that was already silently in effect
# (generate.py used to send no temperature field at all). The UI slider
# overrides this per-request.
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "1.0"))

DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.75"))

# How much worse than the single best match's distance another source is
# still allowed to be and count as "a genuine second topic" for multi-topic
# diversification (retrieval.py's _diversify_by_source()). A source under
# DISTANCE_THRESHOLD but far from the best match is more likely a weak,
# tangentially-related document than a real second topic in the question.
DIVERSIFY_MARGIN = float(os.getenv("DIVERSIFY_MARGIN", "0.3"))

# When MORE distinct topics turn out relevant than top_k allows, don't
# silently drop the excess ones (retrieval.py's _diversify_by_source()
# used to cut to the closest top_k sources and lose the rest entirely) --
# widen the guaranteed-a-slot budget up to how many relevant topics were
# actually found, capped here so one extreme multi-topic question can't
# blow the prompt up without bound.
MAX_DIVERSIFY_SOURCES = int(os.getenv("MAX_DIVERSIFY_SOURCES", "6"))

# A source can slip under DISTANCE_THRESHOLD/DIVERSIFY_MARGIN purely by raw
# embedding distance while being nothing the question is actually about --
# distance is a much coarser signal than the cross-encoder, which actually
# reads the question and chunk together. When a reranked rerank_score is
# available, a source's best chunk must be within this margin of the BEST
# rerank_score found (not just above some flat floor) to still count as
# "genuinely relevant" for diversification.
#
# This has to be a relative margin, not an absolute floor -- an absolute
# floor (e.g. "score must be >= 0") works for a 2-topic question (the real
# second topic scored barely positive) but wrongly excludes EVERY topic in a
# genuinely 3+-topic question: the cross-encoder reads one long compound
# question at once, which dilutes its confidence for each individual
# sub-topic, so even a real, asked-about topic can score negative there.
# Empirically validated at 6.0 against both a real 2-topic case (excludes a
# genuinely unrelated source scored -10 below the best match) and a real
# 3-topic case (keeps two real topics scored -2 and -3.8 below the best
# match, excludes two unrelated ones scored -6.1 and -7.3 below it).
RERANK_RELEVANCE_MARGIN = float(os.getenv("RERANK_RELEVANCE_MARGIN", "6.0"))

# --- Embeddings ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
# Only BGE-family models need an asymmetric query prefix. Empty = no prefix,
# which is correct for MiniLM. Kept settable via env so a future BGE-style
# model swap doesn't require touching any code.
QUERY_INSTRUCTION = os.getenv("QUERY_INSTRUCTION", "")

# --- Ingestion / data source ---
DATA_DIR = os.getenv("DATA_DIR", "data")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
OVERLAP_SENTENCES = int(os.getenv("OVERLAP_SENTENCES", "1"))

# --- Vector store (Qdrant) ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "support_tickets")

# --- Retrieval ---
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))

# How many extra candidates semantic_retrieve() (the standalone Week 3
# baseline) pulls before narrowing to top_k, so a multi-topic question has a
# chance to surface more than one relevant source before diversify runs.
CANDIDATE_POOL_MULTIPLIER = int(os.getenv("CANDIDATE_POOL_MULTIPLIER", "4"))

# Reciprocal Rank Fusion's damping constant -- the standard value from
# Cormack et al., "Reciprocal Rank Fusion". Flattens the curve so rank #1
# vs #2 doesn't dominate rank #10 vs #11 disproportionately when merging
# semantic and BM25 rankings.
RRF_K = int(os.getenv("RRF_K", "60"))

# How many candidates semantic search and BM25 each contribute to
# hybrid_retrieve() before RRF fusion.
HYBRID_CANDIDATE_POOL = int(os.getenv("HYBRID_CANDIDATE_POOL", "15"))

# --- Reranking (Week 4) ---
# A cross-encoder: unlike the embedding model above (which encodes the
# question and each chunk separately, ahead of time), this one reads the
# question and a chunk TOGETHER and outputs one relevance score. Slower, so
# it only runs on a short candidate list, not the whole document set.
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# How many hybrid-search results to feed into the reranker before cutting
# down to DEFAULT_TOP_K -- wider than top_k so reranking has real candidates
# to choose between, not just re-confirm the fusion's own top 3.
RERANK_CANDIDATE_POOL = int(os.getenv("RERANK_CANDIDATE_POOL", "10"))

# --- Vector dimensions ---
# Must match whatever EMBEDDING_MODEL_NAME actually outputs -- 384 is correct
# for all-MiniLM-L6-v2. This is a derived value, not an independently free
# choice: changing it without also changing the model (or vice versa) will
# break ingestion.
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

# --- Local storage (Week 5) ---
# Where trace files (tracing.py) and saved chat sessions (sessions.py) are
# written -- both were previously reading these env vars directly, which
# broke this file's own "no other file calls os.getenv() directly" rule.
TRACE_DIR = os.getenv("TRACE_DIR", "traces")
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "sessions")

# --- API server ---
# Origins the FastAPI backend accepts browser requests from (api.py's CORS
# middleware). Comma-separated. Only matters when the frontend and backend
# are on genuinely different origins with no proxy between them -- the Vite
# dev server's own proxy (ui-react/vite.config.js) sidesteps this entirely
# during local development, and the built dist/ is served from the same
# origin as the API in production, so this mainly matters for a frontend
# hosted separately from this backend.
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]

# --- Agent loop (Week 7) ---
# Stop conditions & budgets for agent.py's ReAct loop -- a loop where the
# model picks its own next step can otherwise run forever or cost
# unboundedly, so both a step count AND a wall-clock ceiling are enforced,
# not just one.
#
# MAX_SECONDS was empirically calibrated, not guessed: a real 3-sub-topic
# compound question measured ~9.1s/step in a warm process (embedding +
# reranker models already loaded) plus one-time ~16s model-load overhead
# on a cold process's first step. 5 steps cold-started measured ~52s in
# the worst case actually observed -- 90s leaves real headroom above that,
# rather than a number picked before ever measuring the real cost.
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))
AGENT_MAX_SECONDS = int(os.getenv("AGENT_MAX_SECONDS", "90"))

# --- Prompt-injection defence (Week 8) ---
# "off" disables injection.neutralize() everywhere (agent observations and
# the fixed pipeline's context). Exists so evals/prompt_injection_test.py can
# run the SAME code with the defence on and off for a clean A/B -- leave it on
# for real use.
INJECTION_DEFENCES = os.getenv("INJECTION_DEFENCES", "on").strip().lower() != "off"

# --- MCP server (Week 9) ---
# "http" is the default for real use (agent.py, evals/): mcp_server.py runs
# as its own long-lived process, same operational shape as Qdrant, so the
# embedding model / BM25 index it loads stay warm across calls instead of
# reloading per call. "stdio" is kept as an alternate transport -- launched
# as a local subprocess -- for inspecting the raw JSON-RPC handshake once
# and for the simplest possible zero-infrastructure discovery demo.
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "http")
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8830"))
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/mcp")

# Checked on every HTTP tool call (mcp_server.py's auth middleware) so an
# arbitrary caller can't invoke a tool with no check at all -- the brief's
# own "keeping it safe: access control" topic. Not present/enforced on the
# stdio transport: stdio has no network exposure to begin with (the caller
# already had to be able to launch a local subprocess), so there is no
# remote party to gate out there.
MCP_SHARED_SECRET = os.getenv("MCP_SHARED_SECRET", "")

# --- Langfuse (observability) ---
# Additive, hosted tracing layered on top of tracing.py's local JSON traces
# (not a replacement -- traces/*.json keeps working unchanged). The Langfuse
# SDK's own client also auto-reads these exact env-var names directly, but
# they're exposed here too so this project's "every env var goes through
# config.py" rule stays intact and /health can report present/missing like
# it already does for GEMINI_API_KEY/GROQ_API_KEY.
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
