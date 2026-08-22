"""
Step 6: Given a question, find the most relevant stored chunks.
"""

import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import vectorstore
from config import DEFAULT_TOP_K, DISTANCE_THRESHOLD, QUERY_INSTRUCTION
from embed import get_embedding_model
from store import get_client

# How many extra candidates to pull before narrowing down to top_k, so a
# multi-topic question has a chance to surface more than one relevant source.
CANDIDATE_POOL_MULTIPLIER = 4


def _diversify_by_source(candidates, top_k):
    """
    Given candidates sorted by distance (closest first), make sure at least
    one chunk from every genuinely relevant source (distance <= threshold)
    is represented -- not just whichever single source happens to dominate
    the closest matches. For a single-topic question, this changes nothing:
    if only one source qualifies as relevant, we fall back to plain top-K.
    """
    best_per_source = {}
    for c in candidates:
        if c["source"] not in best_per_source:
            best_per_source[c["source"]] = c

    relevant_sources = [c for c in best_per_source.values() if c["distance"] <= DISTANCE_THRESHOLD]
    relevant_sources.sort(key=lambda c: c["distance"])

    if len(relevant_sources) <= 1:
        # Single relevant topic (or none at all) -- no diversification needed,
        # plain top-K already gives the best, most detailed context.
        return candidates[:top_k]

    # Multi-topic: guarantee one chunk from each relevant source first, so no
    # topic gets silently dropped just because another topic's chunks happened
    # to be a bit closer. Then fill any remaining slots with the next-best
    # chunks overall, for extra depth/context.
    selected = list(relevant_sources[:top_k])
    selected_keys = {(c["source"], c["chunk_index"]) for c in selected}
    for c in candidates:
        if len(selected) >= top_k:
            break
        key = (c["source"], c["chunk_index"])
        if key not in selected_keys:
            selected.append(c)
            selected_keys.add(key)

    selected.sort(key=lambda c: c["distance"])
    return selected[:top_k]


def retrieve(question, top_k=DEFAULT_TOP_K, model=None, client=None):
    """
    Embed the question, then ask Qdrant for the top_k closest chunks -- but
    make sure a question spanning more than one relevant document doesn't
    lose a topic just because another topic's chunks were slightly closer.
    Returns a list of dicts: text, source, chunk_index, distance.
    """
    if model is None:
        model = get_embedding_model()
    if client is None:
        client = get_client()

    # Only BGE-family models need the asymmetric query prefix. QUERY_INSTRUCTION
    # is empty for MiniLM, so this is a no-op unless a future model needs it.
    prefixed_question = QUERY_INSTRUCTION + question if QUERY_INSTRUCTION else question
    # Qdrant wants a single flat vector, not a batch -- [0] unwraps the one
    # embedding we asked for out of model.encode()'s batch-shaped output.
    question_embedding = model.encode([prefixed_question]).tolist()[0]

    candidates = vectorstore.search(
        client, question_embedding, top_k=top_k * CANDIDATE_POOL_MULTIPLIER
    )
    return _diversify_by_source(candidates, top_k)


if __name__ == "__main__":
    question = input("Ask a question: ")
    matches = retrieve(question)

    print(f"\nTop {len(matches)} matching chunks:\n")
    for i, match in enumerate(matches, start=1):
        print(f"{i}. [{match['source']} chunk {match['chunk_index']}] "
              f"(distance: {match['distance']:.4f})")
        print(f"   {match['text']}\n")
