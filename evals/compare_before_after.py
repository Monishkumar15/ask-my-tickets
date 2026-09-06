"""
Week 4: the required before/after number -- extended to three-way:
semantic only, hybrid (RRF), and hybrid + reranking (the new Week 4 piece).

Zero OpenRouter calls -- this is a pure retrieval comparison, same style as
compare_chunk_sizes.py. Uses an in-memory Qdrant collection; the real
persisted collection is never touched.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ingestion
from config import DEFAULT_TOP_K, RERANK_CANDIDATE_POOL
from eval_questions import EVAL_QUESTIONS
from retrieval import (
    build_bm25_index,
    hit_rate_at_k,
    hybrid_retrieve,
    mrr,
    rerank,
    semantic_retrieve,
)


def build_eval_setup():
    """Chunk once, embed once, build both the in-memory Qdrant collection
    and the BM25 index from the SAME chunk list, so (source, chunk_index)
    keys line up across all three retrieval methods."""
    chunks = ingestion.load_and_chunk_all()
    model = ingestion.get_embedding_model()
    embedded_chunks = ingestion.embed_chunks(list(chunks), model=model)
    embeddings = [c["embedding"].tolist() for c in embedded_chunks]

    client = ingestion.get_client(url=None)  # in-memory, real DB untouched
    ingestion.build_collection(client, chunks, embeddings)

    bm25_index, chunks = build_bm25_index(chunks)
    return model, client, chunks, bm25_index


def label(results, expected_source):
    return "CORRECT" if expected_source in {r["source"] for r in results} else "WRONG"


def run_comparison():
    model, client, chunks, bm25_index = build_eval_setup()
    expected_sources = [q["expected_source"] for q in EVAL_QUESTIONS]

    semantic_all, hybrid_all, reranked_all = [], [], []

    print(f"{'ID':<5} {'SEMANTIC':<10} {'HYBRID':<10} {'HYBRID+RERANK':<14}")
    print("-" * 45)

    for q in EVAL_QUESTIONS:
        semantic_results = semantic_retrieve(q["question"], top_k=DEFAULT_TOP_K, model=model, client=client)

        # Widen the hybrid pool before reranking -- same RERANK_CANDIDATE_POOL
        # the live app uses in retrieval.py's retrieve(), so this eval
        # reflects exactly what generate.py actually does.
        hybrid_candidates = hybrid_retrieve(
            q["question"], chunks, bm25_index, model=model, client=client, top_k=RERANK_CANDIDATE_POOL
        )
        hybrid_results = hybrid_candidates[:DEFAULT_TOP_K]
        reranked_results = rerank(q["question"], hybrid_candidates, top_k=DEFAULT_TOP_K)

        semantic_all.append(semantic_results)
        hybrid_all.append(hybrid_results)
        reranked_all.append(reranked_results)

        print(f"{q['id']:<5} {label(semantic_results, q['expected_source']):<10} "
              f"{label(hybrid_results, q['expected_source']):<10} "
              f"{label(reranked_results, q['expected_source']):<14}")

    n = len(EVAL_QUESTIONS)
    print("\n=== Aggregate metrics ===")
    for name, results_all in [
        ("SEMANTIC ONLY", semantic_all),
        ("HYBRID (RRF)", hybrid_all),
        ("HYBRID + RERANK", reranked_all),
    ]:
        hr = hit_rate_at_k(results_all, expected_sources, k=DEFAULT_TOP_K)
        m = mrr(results_all, expected_sources)
        print(f"{name:<16} hit-rate@{DEFAULT_TOP_K} = {hr:.2%} ({round(hr*n)}/{n}), MRR = {m:.4f}")

    # The interesting comparison: did reranking fix anything hybrid alone
    # couldn't? And did it break anything hybrid alone got right?
    flips = []
    for q, hybrid_r, reranked_r in zip(EVAL_QUESTIONS, hybrid_all, reranked_all):
        hybrid_hit = q["expected_source"] in {r["source"] for r in hybrid_r}
        reranked_hit = q["expected_source"] in {r["source"] for r in reranked_r}
        if hybrid_hit != reranked_hit:
            flips.append((q["id"], q["question"], hybrid_hit, reranked_hit))

    print("\n=== Questions where reranking changed the outcome vs. hybrid alone ===")
    if flips:
        for id_, question, hybrid_hit, reranked_hit in flips:
            old_label = "CORRECT" if hybrid_hit else "WRONG"
            new_label = "CORRECT" if reranked_hit else "WRONG"
            print(f"  {id_}: \"{question}\" -- hybrid {old_label} -> reranked {new_label}")
    else:
        print("  None.")


if __name__ == "__main__":
    run_comparison()
