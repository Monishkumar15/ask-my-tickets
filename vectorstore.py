"""
Thin wrapper around Qdrant -- the vector database. Centralizes how we
connect, build a collection, search it, and count it, so store.py/retrieve.py
don't need to know Qdrant's specific API shape directly.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import COLLECTION_NAME, EMBED_DIM, QDRANT_URL


def get_client(url=QDRANT_URL):
    """Connect to a running Qdrant server (or an in-memory instance for tests)."""
    return QdrantClient(url=url) if url else QdrantClient(":memory:")


def build_collection(client, chunks, embeddings, collection_name=COLLECTION_NAME):
    """
    Delete-and-recreate the collection, then store every chunk + its vector.
    Rebuilding from scratch (rather than upserting) avoids two problems:
      1) stale chunks lingering after a source file is renamed/removed;
      2) leftover vectors from a previous embedding model silently mixing
         with new ones (same vector size doesn't mean same vector space).
    """
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=collection_name, points=points)


def search(client, query_vector, top_k=3, collection_name=COLLECTION_NAME):
    """
    Find the top_k closest chunks to query_vector. Returns a list of dicts:
    text, source, chunk_index, distance (lower = more relevant, matching the
    convention the rest of this project already uses).
    """
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )

    matches = []
    for point in result.points:
        # Qdrant's "score" for COSINE is a similarity (higher = closer,
        # roughly 0 to 1 for normalized embeddings). We convert it to a
        # "distance" (lower = closer) so generate.py's threshold check
        # keeps working the same way it always has.
        matches.append({
            "text": point.payload["text"],
            "source": point.payload["source"],
            "chunk_index": point.payload["chunk_index"],
            "distance": 1 - point.score,
        })
    return matches


def count(client, collection_name=COLLECTION_NAME):
    """How many chunks are currently stored."""
    return client.count(collection_name=collection_name).count


def all_chunks(client, collection_name=COLLECTION_NAME):
    """Return chunk payloads for the React document browser."""
    records, _ = client.scroll(collection_name=collection_name, limit=1000, with_payload=True, with_vectors=False)
    return [{"id": point.id, **point.payload} for point in records]
