"""
Central place for all AI-related configuration. Loaded once from .env (with
sensible defaults for anything not set), so no other file calls load_dotenv()
or os.getenv() directly.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- OpenRouter / LLM ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # no default: required for generate.py
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL_ID = os.getenv("OPENROUTER_MODEL_ID", "google/gemma-4-26b-a4b-it:free")
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.75"))

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

# --- Vector dimensions ---
# Must match whatever EMBEDDING_MODEL_NAME actually outputs -- 384 is correct
# for all-MiniLM-L6-v2. This is a derived value, not an independently free
# choice: changing it without also changing the model (or vice versa) will
# break ingestion.
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
