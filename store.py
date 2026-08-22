"""
Step 5: Store embedded chunks in Qdrant (the vector database).
"""

import vectorstore
from chunk import load_and_chunk_all
from config import QDRANT_URL
from embed import embed_chunks, get_embedding_model


def get_client():
    """Connect to the Qdrant server configured in .env."""
    return vectorstore.get_client(url=QDRANT_URL)


def build_database():
    """
    Chunk the documents, embed them, and store everything in Qdrant.
    Rebuilds the collection from scratch every run -- see vectorstore.py's
    build_collection() docstring for why.
    """
    chunks = load_and_chunk_all()
    model = get_embedding_model()
    chunks = embed_chunks(chunks, model=model)

    client = get_client()
    embeddings = [c["embedding"].tolist() for c in chunks]
    vectorstore.build_collection(client, chunks, embeddings)

    return client


if __name__ == "__main__":
    client = build_database()
    print(f"Stored {vectorstore.count(client)} chunks in Qdrant at "
          f"'{QDRANT_URL}' (collection: '{vectorstore.COLLECTION_NAME}').")
