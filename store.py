"""
Step 5: Store embedded chunks in a local ChromaDB vector database.
"""

import chromadb

from chunk import load_and_chunk_all
from embed import embed_chunks, get_embedding_model

DB_PATH = "chroma_db"
COLLECTION_NAME = "support_tickets"


def get_collection():
    """Connect to (or create) a persistent local ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def build_database():
    """Chunk the documents, embed them, and store everything in ChromaDB."""
    chunks = load_and_chunk_all()
    model = get_embedding_model()
    chunks = embed_chunks(chunks, model=model)

    collection = get_collection()

    # ChromaDB wants parallel lists: one id, one embedding, one text (document),
    # and one metadata dict per item — all matched up by list position.
    ids = [f"{c['source']}_{c['chunk_index']}" for c in chunks]
    embeddings = [c["embedding"].tolist() for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    # upsert = insert new, or update if the same id already exists — safe to
    # re-run this script any time your documents change.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return collection


if __name__ == "__main__":
    collection = build_database()
    print(f"Stored {collection.count()} chunks in ChromaDB at '{DB_PATH}/' "
          f"(collection: '{COLLECTION_NAME}').")
