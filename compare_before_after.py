"""
Week 4: the required before/after number.

Compares the OLD retrieval method (retrieve.py -- pure semantic search,
completely unmodified) against the NEW method (hybrid.py -- semantic +
BM25 fused via RRF), on the same 12-question eval set, using hit-rate@3
and MRR. Zero OpenRouter calls -- this is a pure retrieval comparison,
same as compare_chunk_sizes.py.

Uses an in-memory Qdrant collection; the real persisted collection is
never touched.
"""

import vectorstore
from bm25_search import build_bm25_index
from chunk import load_and_chunk_all
from config import DEFAULT_TOP_K
from embed import embed_chunks, get_embedding_model
from eval_questions import EVAL_QUESTIONS
from hybrid import hybrid_retrieve
from metrics import hit_rate_at_k, mrr
from retrieve import retrieve


def build_eval_setup():
    """Chunk once, embed once, build both the in-memory Qdrant collection
    and the BM25 index from the SAME chunk list, so (source, chunk_index)
    keys line up between the two retrieval methods."""
    chunks = load_and_chunk_all()
    model = get_embedding_model()
    embedded_chunks = embed_chunks(list(chunks), model=model)
    embeddings = [c["embedding"].tolist() for c in embedded_chunks]

    client = vectorstore.get_client(url=None)  # in-memory, real DB untouched
    vectorstore.build_collection(client, chunks, embeddings)

    bm25_index, chunks = build_bm25_index(chunks)

    return model, client, chunks, bm25_index


def run_comparison():
    model, client, chunks, bm25_index = build_eval_setup()

    expected_sources = [q["expected_source"] for q in EVAL_QUESTIONS]
    old_results_all = []
    new_results_all = []
    flips = []  # questions that changed status between old and new

    print(f"{'ID':<5} {'OLD (semantic)':<20} {'NEW (hybrid)':<20}")
    print("-" * 50)

    for q in EVAL_QUESTIONS:
        old_results = retrieve(q["question"], top_k=DEFAULT_TOP_K, model=model, client=client)
        new_results = hybrid_retrieve(q["question"], chunks, bm25_index, model=model, client=client, top_k=DEFAULT_TOP_K)

        old_results_all.append(old_results)
        new_results_all.append(new_results)

        old_hit = q["expected_source"] in {r["source"] for r in old_results}
        new_hit = q["expected_source"] in {r["source"] for r in new_results}

        old_label = "CORRECT" if old_hit else "WRONG"
        new_label = "CORRECT" if new_hit else "WRONG"
        print(f"{q['id']:<5} {old_label:<20} {new_label:<20}")

        if old_hit != new_hit:
            flips.append((q["id"], q["question"], old_label, new_label))

    old_hit_rate = hit_rate_at_k(old_results_all, expected_sources, k=DEFAULT_TOP_K)
    new_hit_rate = hit_rate_at_k(new_results_all, expected_sources, k=DEFAULT_TOP_K)
    old_mrr = mrr(old_results_all, expected_sources)
    new_mrr = mrr(new_results_all, expected_sources)

    n = len(EVAL_QUESTIONS)
    print("\n=== BEFORE vs AFTER ===")
    print(f"BEFORE (semantic only): hit-rate@{DEFAULT_TOP_K} = {old_hit_rate:.2%} ({round(old_hit_rate*n)}/{n}), MRR = {old_mrr:.4f}")
    print(f"AFTER  (hybrid RRF):    hit-rate@{DEFAULT_TOP_K} = {new_hit_rate:.2%} ({round(new_hit_rate*n)}/{n}), MRR = {new_mrr:.4f}")

    print("\n=== Questions where the result CHANGED (old -> new) ===")
    if flips:
        for id_, question, old_label, new_label in flips:
            print(f"  {id_}: \"{question}\" -- {old_label} -> {new_label}")
    else:
        print("  None -- no question's hit/miss status changed between the two methods.")

    # Mentor check #4: explicitly call out what did NOT change, not just what did.
    unchanged_still_correct = [
        q["id"] for q, old_r, new_r in zip(EVAL_QUESTIONS, old_results_all, new_results_all)
        if (q["expected_source"] in {r["source"] for r in old_r})
        and (q["expected_source"] in {r["source"] for r in new_r})
    ]
    print(f"\nQuestions correct under BOTH methods (hybrid made no difference here): {unchanged_still_correct}")


if __name__ == "__main__":
    run_comparison()
