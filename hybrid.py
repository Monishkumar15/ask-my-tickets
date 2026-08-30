"""
Week 4: hybrid search -- the ONE improvement this week.

Combines semantic search (meaning) and BM25 (exact words) using Reciprocal
Rank Fusion (RRF), so a question can be helped by whichever method
actually understands it, instead of relying on semantic search alone.

Deliberately kept separate from retrieve.py -- the existing pure-semantic
retrieve() is left completely untouched, so both methods can be compared
side by side on the same eval set.
"""

from bm25_search import bm25_search
from config import QUERY_INSTRUCTION
from embed import get_embedding_model
from vectorstore import search as semantic_search

RRF_K = 60           # standard damping constant (Cormack et al., "Reciprocal Rank Fusion")
CANDIDATE_POOL = 15  # how many candidates each method contributes before fusion


def hybrid_retrieve(question, chunks, bm25_index, model=None, client=None, top_k=3):
    """
    Runs semantic search and BM25 search independently (CANDIDATE_POOL
    candidates each), then fuses their rankings with RRF:

        rrf_score(chunk) = sum over each ranker of  1 / (RRF_K + rank_in_that_ranker)

    where rank is the chunk's 1-based position in that ranker's own list.
    A chunk missing from one ranker's list simply contributes 0 for that
    term -- it isn't penalized, just not boosted by that method.

    Chunks are matched across the two rankers by (source, chunk_index).
    Returns the top_k chunks as {text, source, chunk_index, rrf_score}
    (higher rrf_score = better -- the OPPOSITE direction of retrieve()'s
    "distance" convention. Eval code should compare by whether
    expected_source shows up / at what rank, not by comparing raw
    distance vs. rrf_score numbers directly.)
    """
    if model is None:
        model = get_embedding_model()

    prefixed_question = QUERY_INSTRUCTION + question if QUERY_INSTRUCTION else question
    question_embedding = model.encode([prefixed_question]).tolist()[0]

    semantic_results = semantic_search(client, question_embedding, top_k=CANDIDATE_POOL)
    bm25_results = bm25_search(question, bm25_index, chunks, top_k=CANDIDATE_POOL)

    # Build rank lookups: (source, chunk_index) -> 1-based rank in that list
    semantic_ranks = {
        (r["source"], r["chunk_index"]): rank
        for rank, r in enumerate(semantic_results, start=1)
    }
    bm25_ranks = {
        (r["source"], r["chunk_index"]): rank
        for rank, r in enumerate(bm25_results, start=1)
    }

    # Union of every chunk that appeared in EITHER list
    all_keys = set(semantic_ranks) | set(bm25_ranks)

    # Keep the actual chunk text/metadata handy for the final output
    chunk_lookup = {(r["source"], r["chunk_index"]): r for r in semantic_results}
    for r in bm25_results:
        chunk_lookup.setdefault((r["source"], r["chunk_index"]), r)

    fused = []
    for key in all_keys:
        score = 0.0
        if key in semantic_ranks:
            score += 1 / (RRF_K + semantic_ranks[key])
        if key in bm25_ranks:
            score += 1 / (RRF_K + bm25_ranks[key])

        chunk = chunk_lookup[key]
        fused.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "rrf_score": score,
        })

    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused[:top_k]


if __name__ == "__main__":
    from chunk import load_and_chunk_all
    from embed import embed_chunks
    from bm25_search import build_bm25_index
    import vectorstore

    chunks = load_and_chunk_all()
    model = get_embedding_model()

    embedded_chunks = embed_chunks(list(chunks), model=model)
    embeddings = [c["embedding"].tolist() for c in embedded_chunks]

    # in-memory client, default collection name -- doesn't touch the real DB,
    # and matches what hybrid_retrieve()'s semantic_search() call expects
    client = vectorstore.get_client(url=None)
    vectorstore.build_collection(client, chunks, embeddings)

    bm25_index, chunks = build_bm25_index(chunks)

    question = input("Ask a question (hybrid search): ")
    results = hybrid_retrieve(question, chunks, bm25_index, model=model, client=client, top_k=3)

    print(f"\nTop {len(results)} hybrid matches:\n")
    for i, r in enumerate(results, start=1):
        print(f"{i}. [{r['source']} chunk {r['chunk_index']}] (rrf_score: {r['rrf_score']:.4f})")
        print(f"   {r['text']}\n")
