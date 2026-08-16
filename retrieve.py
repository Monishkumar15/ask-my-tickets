"""
Step 6: Given a question, find the most relevant stored chunks.
"""

from embed import get_embedding_model, QUERY_INSTRUCTION
from store import get_collection
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def retrieve(question, top_k=3, model=None, collection=None):
    """
    Embed the question, then ask ChromaDB for the top_k closest chunks.
    Returns a list of dicts: text, source, chunk_index, distance.
    """
    if model is None:
        model = get_embedding_model()
    if collection is None:
        collection = get_collection()

    # BGE's asymmetric convention: queries get the instruction prefix,
    # documents (already embedded in embed.py) do not.
    prefixed_question = QUERY_INSTRUCTION + question
    question_embedding = model.encode([prefixed_question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
    )

    # ChromaDB returns everything as parallel lists nested one level for
    # batching (we only sent 1 question, so we index [0] to unwrap that).
    matches = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "text": text,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance,
        })

    return matches


if __name__ == "__main__":
    question = input("Ask a question: ")
    matches = retrieve(question)

    print(f"\nTop {len(matches)} matching chunks:\n")
    for i, match in enumerate(matches, start=1):
        print(f"{i}. [{match['source']} chunk {match['chunk_index']}] "
              f"(distance: {match['distance']:.4f})")
        print(f"   {match['text']}\n")
