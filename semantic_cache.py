"""
Week 11: semantic caching -- reuse a past answer when a NEW question means
roughly the same thing as one already asked, not just an exact text match.
Modeled directly on retrieval.py's own pattern: a second small Qdrant
collection ("answer_cache"), the SAME embedding model already loaded for
retrieval, no new dependency.

Only wired into generate.py::answer_question() (the real user-facing fixed
pipeline, confirmed in this week's own cost-visibility audit to be the one
any real user actually hits) -- not agent.py/multi_agent.py, matching this
week's scope decision to make the real pipeline measurable and improved
first, not every pipeline at once.

CACHE_SIMILARITY_THRESHOLD (config.py) was measured empirically against
real paraphrase pairs before being set -- see that constant's own comment
for the actual numbers.

Staleness is handled by two SEPARATE mechanisms, solving two different
problems (not one covering for the other):
  - CACHE_TTL_SECONDS (config.py) limits how long ANY cached answer stays
    eligible for reuse at all, regardless of whether anything changed --
    a backstop for silent/unnoticed document edits.
  - clear() is called from ingestion.py::build_database() specifically
    WHEN the knowledge base is rebuilt (a real edit, or a new /upload) --
    an immediate invalidation, not waiting for the TTL to expire. It is
    NOT called on normal app startup -- only an actual re-ingestion should
    throw away cached answers; a routine restart shouldn't lose useful,
    still-valid ones.
"""
import time
import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams

from config import (
    CACHE_COLLECTION_NAME,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_TTL_SECONDS,
    EMBED_DIM,
    QUERY_INSTRUCTION,
)
from ingestion import get_client, get_embedding_model

# A fixed namespace so the SAME question text always hashes to the SAME
# point id across separate processes/runs (Python's own hash() is
# per-process-randomized for strings, which would silently stop the
# "ask the identical question twice" case from ever overwriting its own
# cache entry -- uuid5 is deterministic by design, exactly what upserting
# the same cached question later needs).
_NAMESPACE = uuid.UUID("c6f6a95e-6b7d-4c7e-8f0a-5f0c6b8f9f10")


def _embed(model, question):
    # Symmetric on purpose, unlike retrieval.py's query-vs-document
    # comparison: a cache lookup compares one QUESTION against other
    # PAST QUESTIONS, not a question against a document chunk, so the
    # same QUERY_INSTRUCTION prefix (empty for MiniLM today) is applied
    # on both the store and lookup side for a fair comparison.
    prefixed = QUERY_INSTRUCTION + question if QUERY_INSTRUCTION else question
    return model.encode([prefixed]).tolist()[0]


def get_cached(question, current_sources=None, model=None, client=None):
    """
    Returns a cached result dict (same shape answer_question() itself
    returns: answer/sources/provider) if a sufficiently similar past
    question was already answered, else None. Never raises on a cold or
    missing collection -- a cache miss is the correct default on a fresh
    install, not an error condition.

    current_sources: the CURRENT question's own freshly-retrieved source
    filenames (already computed by the caller before this check, from the
    same retrieve() call the confidence gate just used). Required as a
    second, independent safety gate -- found live, via the real eval set,
    that embedding distance ALONE cannot be trusted here: a genuinely
    different question ("what percentage refund...") matched a cached,
    completely unrelated answer ("...lost-in-transit claim timing") at
    distance 0.684, and a broader pairwise scan of the real eval set found
    genuinely different questions sitting CLOSER than some genuine
    paraphrase pairs. A cached answer is only served if its stored sources
    overlap with what this question's OWN retrieval just found -- the one
    signal here that's actually grounded in real document content, not
    just surface phrasing similarity. Not a perfect guarantee either (two
    different sub-questions about the same ticket still share a source --
    see FINDINGS_week11.md for the honest limitation), but a second,
    independent check layered on top of a deliberately conservative
    CACHE_SIMILARITY_THRESHOLD, the same "no single signal is enough"
    lesson retrieval.py's own _diversify_by_source already learned.
    """
    if model is None:
        model = get_embedding_model()
    if client is None:
        client = get_client()

    # A cache-layer failure (Qdrant unreachable, a transient query error,
    # etc.) must never crash an otherwise-healthy request -- retrieval has
    # already succeeded by the time this runs, so a real answer can still
    # be generated normally without the cache's help. Caught broadly and
    # deliberately: ANY failure here degrades to "treat it as a miss," the
    # same "one subsystem's failure shouldn't take down the whole request"
    # reasoning ingestion.py's load_document_text() already applies to a
    # single unreadable file.
    try:
        if not client.collection_exists(CACHE_COLLECTION_NAME):
            return None
        vector = _embed(model, question)
        result = client.query_points(collection_name=CACHE_COLLECTION_NAME, query=vector, limit=1)
    except Exception as exc:
        print(f"WARNING: semantic cache lookup failed ({type(exc).__name__}: {exc}) -- answering without cache.")
        return None

    if not result.points:
        return None

    point = result.points[0]
    distance = 1 - point.score  # same similarity -> distance convention as ingestion.py::search()
    if distance > CACHE_SIMILARITY_THRESHOLD:
        return None

    payload = point.payload
    cached_sources = set(payload.get("sources") or [])
    if current_sources is not None and not (cached_sources & set(current_sources)):
        return None

    # TTL: a DIFFERENT problem than the distance/source checks above --
    # those guard against matching the wrong past question; this guards
    # against matching the RIGHT one whose answer may no longer be true.
    # Nothing else here would ever notice if the underlying ticket's
    # policy changed after this entry was cached. Rejected here (a miss,
    # falls through to a fresh answer) rather than physically deleted --
    # cheap, and the entry simply gets overwritten next time this exact
    # question is stored again anyway.
    cached_at = payload.get("cached_at")
    if cached_at is not None and (time.time() - cached_at) > CACHE_TTL_SECONDS:
        return None

    return {
        "answer": payload["answer"],
        "sources": payload["sources"],
        "provider": payload.get("provider"),
        "cached_from_question": payload["question"],
        "cache_distance": distance,
    }


def store(question, answer_result, model=None, client=None):
    """
    Stores a real, successfully-generated answer for future reuse. Only
    ever called for a genuine answer -- NEVER for a refusal or an error --
    caching "I don't know" would permanently block a later, better-phrased
    version of the same underlying question from ever trying retrieval
    again, turning one unlucky phrasing into a standing false negative.
    """
    if model is None:
        model = get_embedding_model()
    if client is None:
        client = get_client()

    # Same reasoning as get_cached() -- a failed cache WRITE must not crash
    # a request that already produced a perfectly good, already-returned-
    # to-the-caller answer. Worst case on failure: this answer simply won't
    # be reusable next time, which is strictly no worse than caching never
    # having existed at all.
    try:
        if not client.collection_exists(CACHE_COLLECTION_NAME):
            client.create_collection(
                collection_name=CACHE_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

        vector = _embed(model, question)
        point_id = str(uuid.uuid5(_NAMESPACE, question))
        client.upsert(collection_name=CACHE_COLLECTION_NAME, points=[
            PointStruct(id=point_id, vector=vector, payload={
                "question": question,
                "answer": answer_result["answer"],
                "sources": answer_result["sources"],
                "provider": answer_result.get("provider"),
                "cached_at": time.time(),
            }),
        ])
    except Exception as exc:
        print(f"WARNING: semantic cache store failed ({type(exc).__name__}: {exc}) -- answer still returned normally.")


def clear(client=None):
    """Drops the cache collection entirely. Two callers, two different
    reasons: evals/measure_cache_savings.py uses it to guarantee a
    genuinely cold run before measuring "before" numbers; ingestion.py's
    build_database() uses it to invalidate every cached answer the moment
    the knowledge base actually changes, rather than waiting for
    CACHE_TTL_SECONDS to expire them one at a time."""
    if client is None:
        client = get_client()
    if client.collection_exists(CACHE_COLLECTION_NAME):
        client.delete_collection(CACHE_COLLECTION_NAME)
