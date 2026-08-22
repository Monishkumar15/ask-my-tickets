"""
Step 4: Turn text chunks into embeddings (number-vectors) using a free,
local embedding model.
"""

from sentence_transformers import SentenceTransformer

from chunk import load_and_chunk_all
from config import EMBEDDING_MODEL_NAME

# all-MiniLM-L6-v2: a small, free, local embedding model that handles
# general English text well. Downloads once (~80MB) the first time you run
# this, then it's cached locally for future runs. Configurable via
# EMBEDDING_MODEL_NAME in .env if you want to swap models later.
MODEL_NAME = EMBEDDING_MODEL_NAME


def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks, model=None):
    """
    Given a list of chunk dicts (from chunk.py), add an 'embedding' key to
    each one containing its vector. Returns the same list, enriched.
    """
    if model is None:
        model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts)  # one vector per input text

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_all()
    print(f"Embedding {len(chunks)} chunks using '{MODEL_NAME}'...\n")

    chunks = embed_chunks(chunks)

    first = chunks[0]
    print(f"Example chunk from: {first['source']}")
    print(f"Text: {first['text'][:80]}...")
    print(f"Embedding length: {len(first['embedding'])} numbers")
    print(f"First 8 numbers: {first['embedding'][:8]}")
