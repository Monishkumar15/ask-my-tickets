"""
Step 4: Turn text chunks into embeddings (number-vectors) using a free,
local embedding model.
"""

from sentence_transformers import SentenceTransformer
from chunk import load_and_chunk_all

# BGE-small: a small, free, local embedding model from the BGE family named
# in the course material. Downloads once (~130MB) the first time you run
# this, then it's cached locally for future runs.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models are trained "asymmetrically": documents are embedded as-is, but
# queries are embedded with this instruction prefix in front, which measurably
# improves retrieval accuracy for this model family. Documents never get this
# prefix — only questions do (see retrieve.py).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


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
