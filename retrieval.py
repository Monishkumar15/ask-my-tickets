"""
Retrieval pipeline: question -> relevant chunks, using three techniques
combined -- semantic search, keyword search, and reranking.

Consolidated from retrieve.py, bm25_search.py, hybrid.py, and metrics.py
(Week 3-4), plus a new reranking stage (Week 4's missing concept).

    semantic_retrieve()  -- meaning-based search only (the Week 3 baseline)
    hybrid_retrieve()    -- semantic + BM25 keyword search, fused via RRF
    rerank()             -- cross-encoder re-scores a shortlist (the new part)
    retrieve()           -- the live pipeline generate.py actually uses:
                            hybrid_retrieve() widens the field, rerank()
                            picks the true best top_k

    hit_rate_at_k(), mrr()  -- retrieval quality metrics
"""

import os
import re

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    CANDIDATE_POOL_MULTIPLIER,
    DEFAULT_TOP_K,
    DISTANCE_THRESHOLD,
    DIVERSIFY_MARGIN,
    HYBRID_CANDIDATE_POOL,
    MAX_DIVERSIFY_SOURCES,
    QUERY_INSTRUCTION,
    RERANK_CANDIDATE_POOL,
    RERANK_RELEVANCE_MARGIN,
    RERANKER_MODEL_NAME,
    RRF_K,
)
from ingestion import get_client, get_embedding_model, load_and_chunk_all
from ingestion import search as semantic_search

# ===========================================================================
# 1. Keyword search (BM25) -- the "exact words" half of hybrid search
# ===========================================================================


def _tokenize(text):
    """Lowercase, keep only letters/digits as separate words."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(chunks):
    """
    chunks: list of {text, source, chunk_index, page} from load_and_chunk_all().
    Returns (bm25_index, chunks) -- chunks is returned alongside so callers
    can match a score back to its (source, chunk_index) by list position.
    """
    corpus = [_tokenize(c["text"]) for c in chunks]
    bm25_index = BM25Okapi(corpus)
    return bm25_index, chunks


def bm25_search(query, bm25_index, chunks, top_k):
    """
    Score every chunk against the query, return the top_k as a list of
    dicts: text, source, chunk_index, page, bm25_score (higher = more relevant).
    """
    scores = bm25_index.get_scores(_tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)

    return [
        {
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "page": chunk.get("page"),
            "bm25_score": score,
        }
        for chunk, score in ranked[:top_k]
    ]


# ===========================================================================
# 2. Semantic retrieval -- the Week 3 baseline, meaning-based search only
# ===========================================================================


def _diversify_by_source(candidates, top_k):
    """
    Given candidates sorted by distance (closest first), make sure at least
    one chunk from every genuinely relevant source is represented -- not
    just whichever single source happens to dominate the closest matches.
    For a single-topic question, this changes nothing: if only one source
    qualifies as relevant, we fall back to plain top-K.

    "Genuinely relevant" requires BOTH:
      1) distance <= DISTANCE_THRESHOLD (the existing refusal-worthy cutoff)
      2) distance <= best_distance + DIVERSIFY_MARGIN (close to the single
         best match, not just anywhere under the flat threshold)
    Condition 2 matters: a source can sit under DISTANCE_THRESHOLD purely by
    being weakly/tangentially related, not because the question is
    genuinely about two topics. Without it, a real second topic (small gap
    from the best match) and a weak, unrelated document (large gap) get
    treated identically -- and the weak one can displace a stronger
    same-source chunk that actually answers the question, just to
    manufacture "diversity" nothing asked for.

    A third condition applies when candidates have already been reranked
    (rerank_score present): a source's best chunk must be within
    RERANK_RELEVANCE_MARGIN of the best rerank_score found. Distance alone
    isn't enough -- a source can slip under DISTANCE_THRESHOLD/
    DIVERSIFY_MARGIN by raw embedding distance while the cross-encoder
    (which actually reads the question and chunk together, a more
    discerning signal) considers it clearly off-topic. Without this, that
    source still gets a guaranteed slot and can displace a same-source
    chunk that actually completes the real answer -- exactly the failure
    distance-only diversification is supposed to prevent, just triggered by
    a different source this time. A RELATIVE margin, not an absolute floor,
    on purpose: a flat floor works for a 2-topic question but wrongly
    excludes every topic in a genuinely 3+-topic one, since the
    cross-encoder's confidence per sub-topic gets diluted reading one long
    compound question at once.

    Slot budget: normally exactly top_k chunks come back. But if MORE
    distinct topics turn out relevant than top_k allows, the budget widens
    up to MAX_DIVERSIFY_SOURCES rather than silently dropping the excess
    topics entirely -- a question asking about 4 tickets shouldn't lose the
    4th one just because it was 0.04 farther away than the 3 that fit. This
    can make the returned list longer than top_k; callers should size for
    "however many chunks come back," not assume an exact count.
    """
    best_per_source = {}
    for c in candidates:
        if c["source"] not in best_per_source:
            best_per_source[c["source"]] = c

    if not best_per_source:
        return candidates[:top_k]

    best_distance = min(c["distance"] for c in best_per_source.values())
    relevant_sources = [
        c for c in best_per_source.values()
        if c["distance"] <= DISTANCE_THRESHOLD and c["distance"] <= best_distance + DIVERSIFY_MARGIN
    ]

    # Only applies post-rerank -- the pre-rerank widening call in
    # hybrid_retrieve() has no rerank_score yet, so this is a no-op there.
    if relevant_sources and "rerank_score" in relevant_sources[0]:
        best_rerank = max(c["rerank_score"] for c in relevant_sources)
        relevant_sources = [c for c in relevant_sources if c["rerank_score"] >= best_rerank - RERANK_RELEVANCE_MARGIN]

    relevant_sources.sort(key=lambda c: c["distance"])

    if len(relevant_sources) <= 1:
        return candidates[:top_k]

    effective_top_k = max(top_k, min(len(relevant_sources), MAX_DIVERSIFY_SOURCES))

    selected = list(relevant_sources[:effective_top_k])
    selected_keys = {(c["source"], c["chunk_index"]) for c in selected}
    for c in candidates:
        if len(selected) >= effective_top_k:
            break
        key = (c["source"], c["chunk_index"])
        if key not in selected_keys:
            selected.append(c)
            selected_keys.add(key)

    selected.sort(key=lambda c: c["distance"])
    return selected[:effective_top_k]


def semantic_retrieve(question, top_k=DEFAULT_TOP_K, model=None, client=None):
    """
    Embed the question, then ask Qdrant for the top_k closest chunks --
    meaning-based search only, no keyword matching or reranking. This is
    the Week 3 method, kept unchanged as the baseline for comparisons.
    """
    if model is None:
        model = get_embedding_model()
    if client is None:
        client = get_client()

    prefixed_question = QUERY_INSTRUCTION + question if QUERY_INSTRUCTION else question
    question_embedding = model.encode([prefixed_question]).tolist()[0]

    candidates = semantic_search(client, question_embedding, top_k=top_k * CANDIDATE_POOL_MULTIPLIER)
    return _diversify_by_source(candidates, top_k)


# ===========================================================================
# 3. Hybrid fusion -- semantic + BM25, combined via Reciprocal Rank Fusion
# ===========================================================================


def hybrid_retrieve(question, chunks, bm25_index, model=None, client=None, top_k=DEFAULT_TOP_K):
    """
    Runs semantic search and BM25 search independently (HYBRID_CANDIDATE_POOL
    candidates each), then fuses their rankings with RRF:

        rrf_score(chunk) = sum over each ranker of  1 / (RRF_K + rank_in_that_ranker)

    A chunk missing from one ranker's list simply contributes 0 for that
    term. Chunks are matched across rankers by (source, chunk_index).

    Every returned chunk also carries its semantic "distance" (defaulting to
    1.0 -- "not confident" -- for a chunk that only BM25 found), so this
    output can still be fed through generate.py's confidence gate and
    through rerank() below without losing that information.
    """
    if model is None:
        model = get_embedding_model()

    prefixed_question = QUERY_INSTRUCTION + question if QUERY_INSTRUCTION else question
    question_embedding = model.encode([prefixed_question]).tolist()[0]

    semantic_results = semantic_search(client, question_embedding, top_k=HYBRID_CANDIDATE_POOL)
    bm25_results = bm25_search(question, bm25_index, chunks, top_k=HYBRID_CANDIDATE_POOL)

    semantic_ranks = {(r["source"], r["chunk_index"]): rank for rank, r in enumerate(semantic_results, start=1)}
    bm25_ranks = {(r["source"], r["chunk_index"]): rank for rank, r in enumerate(bm25_results, start=1)}
    all_keys = set(semantic_ranks) | set(bm25_ranks)

    # Keep the actual chunk text/metadata handy for the final output. Prefer
    # the semantic result (it carries a real "distance"); fall back to the
    # BM25 result with a placeholder distance if semantic search never saw it.
    chunk_lookup = {(r["source"], r["chunk_index"]): r for r in semantic_results}
    for r in bm25_results:
        key = (r["source"], r["chunk_index"])
        if key not in chunk_lookup:
            chunk_lookup[key] = {**r, "distance": 1.0}

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
            "page": chunk.get("page"),
            "distance": chunk.get("distance", 1.0),
            "rrf_score": score,
        })

    # Guarantee multi-topic coverage the same way semantic_retrieve() does --
    # RRF fusion alone has no such safeguard, and without it a question
    # spanning two topics can silently lose one of them just because the
    # other topic's chunks fused to a slightly higher score. Sort by
    # distance first (what _diversify_by_source expects), then restore
    # hybrid's own rrf_score ordering for anyone consuming this directly.
    fused.sort(key=lambda r: r["distance"])
    diversified = _diversify_by_source(fused, top_k)
    diversified.sort(key=lambda r: r["rrf_score"], reverse=True)
    return diversified


# ===========================================================================
# 4. Reranking -- a cross-encoder re-scores a shortlist (the new Week 4 piece)
# ===========================================================================

_reranker_cache = None


def get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        _reranker_cache = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_cache


def rerank(question, candidates, top_k=DEFAULT_TOP_K, reranker=None):
    """
    Re-score each candidate by feeding the question and its full chunk text
    TOGETHER into a cross-encoder model, then reorder by that fresh score.

    This is fundamentally different from the embedding model used for
    semantic search: an embedding model (a "bi-encoder") encodes the
    question and each chunk SEPARATELY, ahead of time -- fast, but it never
    actually reads them side by side. A cross-encoder reads both together,
    every time -- slower, so it only runs here on a short candidate list
    instead of the whole document set.

    Preserves each candidate's "distance" field, so the result can still
    be fed through generate.py's confidence gate afterward.
    """
    if not candidates:
        return []
    if reranker is None:
        reranker = get_reranker()

    pairs = [[question, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    reranked = [{**c, "rerank_score": float(score)} for c, score in zip(candidates, scores)]
    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


# ===========================================================================
# 5. The live pipeline -- what generate.py actually calls by default
# ===========================================================================

_bm25_cache = None  # (bm25_index, chunks), built once per process and reused


def _get_bm25():
    global _bm25_cache
    if _bm25_cache is None:
        chunks = load_and_chunk_all()
        _bm25_cache = build_bm25_index(chunks)
    return _bm25_cache


def retrieve(question, top_k=DEFAULT_TOP_K, model=None, client=None):
    """
    The full, current-best pipeline: hybrid search widens the candidate
    pool (semantic + BM25 via RRF), then a cross-encoder reranks those
    candidates for the final top_k. This is what generate.py uses.

    semantic_retrieve() and hybrid_retrieve() (above) remain available
    separately for comparisons -- e.g. "did reranking actually fix what
    hybrid search alone couldn't."
    """
    if model is None:
        model = get_embedding_model()
    if client is None:
        client = get_client()

    bm25_index, chunks = _get_bm25()
    candidates = hybrid_retrieve(question, chunks, bm25_index, model=model, client=client, top_k=RERANK_CANDIDATE_POOL)

    # Rerank the WHOLE widened pool first (not just the final top_k) --
    # reranking scores each chunk independently, with no notion of "don't
    # pick 3 chunks from the same topic," so a multi-topic question can
    # still lose a topic here even though hybrid_retrieve() already
    # guaranteed it was in the candidate pool. Diversify again on the
    # reranked list to guarantee coverage, same as hybrid_retrieve() does.
    #
    # Order by rerank_score (not distance) going in: _diversify_by_source()
    # always uses "distance" for its threshold/margin checks (that's what
    # DISTANCE_THRESHOLD and DIVERSIFY_MARGIN are calibrated for), but its
    # fill-loop just walks the incoming list in order to fill any leftover
    # slots -- and rerank_score is the more discerning signal for that tie-
    # break, since it's the one metric here that actually read the question
    # and each chunk's text TOGETHER. Sorting by raw distance instead let a
    # near-duplicate/uninformative chunk (e.g. a ticket's subject line) win
    # a leftover slot over the chunk containing the actual missing fact,
    # by a gap as small as 0.007 -- even though the cross-encoder had
    # already scored the useful chunk higher.
    reranked = rerank(question, candidates, top_k=RERANK_CANDIDATE_POOL)
    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    diversified = _diversify_by_source(reranked, top_k)
    diversified.sort(key=lambda c: c["rerank_score"], reverse=True)
    return diversified


# ===========================================================================
# 6. Retrieval quality metrics
# ===========================================================================


def hit_rate_at_k(results_list, expected_sources, k):
    """
    Returns the fraction of questions where the expected source appears
    ANYWHERE in that question's top-k results.
    """
    hits = 0
    for results, expected in zip(results_list, expected_sources):
        top_k_sources = {r["source"] for r in results[:k]}
        if expected in top_k_sources:
            hits += 1
    return hits / len(expected_sources)


def mrr(results_list, expected_sources):
    """
    Mean Reciprocal Rank: for each question, 1/rank of the first correct
    result (rank 1 -> 1.0, rank 2 -> 0.5, not found -> 0), averaged.
    """
    reciprocal_ranks = []
    for results, expected in zip(results_list, expected_sources):
        rank_score = 0.0
        for rank, r in enumerate(results, start=1):
            if r["source"] == expected:
                rank_score = 1 / rank
                break
        reciprocal_ranks.append(rank_score)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


if __name__ == "__main__":
    question = input("Ask a question: ")
    matches = retrieve(question)

    print(f"\nTop {len(matches)} matching chunks (hybrid + reranked):\n")
    for i, match in enumerate(matches, start=1):
        page_note = f", page {match['page']}" if match.get("page") else ""
        print(f"{i}. [{match['source']} chunk {match['chunk_index']}{page_note}] "
              f"(rerank_score: {match['rerank_score']:.4f}, distance: {match['distance']:.4f})")
        print(f"   {match['text']}\n")
