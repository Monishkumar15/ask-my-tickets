"""
The inspection view -- type any question yourself (not limited to the
fixed eval set) and see, side by side:
  1. what SEMANTIC search alone found
  2. what HYBRID search (semantic + BM25 via RRF) found
  3. what HYBRID + RERANKING found (the live pipeline generate.py uses)
  4. the actual generated answer

Uses an in-memory Qdrant collection built from the real data/ folder --
never touches the real persisted collection.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

import ingestion
from config import DEFAULT_TOP_K, RERANK_CANDIDATE_POOL
from generate import answer_question
from retrieval import build_bm25_index, hybrid_retrieve, rerank, semantic_retrieve


def build_inspection_setup():
    """Chunk + embed + store into a fresh in-memory Qdrant client, and build
    a matching BM25 index -- both built from the exact same chunk list, so
    (source, chunk_index) lines up across all methods."""
    chunks = ingestion.load_and_chunk_all()
    model = ingestion.get_embedding_model()
    embedded_chunks = ingestion.embed_chunks(list(chunks), model=model)
    embeddings = [c["embedding"].tolist() for c in embedded_chunks]

    client = ingestion.get_client(url=None)  # in-memory, real DB untouched
    ingestion.build_collection(client, chunks, embeddings)

    bm25_index, chunks = build_bm25_index(chunks)

    return model, client, bm25_index, chunks


def print_results(label, results, score_key):
    print(f"  {label}:")
    for i, r in enumerate(results, start=1):
        print(f"    {i}. [{r['source']} chunk {r['chunk_index']}] ({score_key}: {r[score_key]:.4f})")


def inspect_one_question(question, model, client, bm25_index, chunks):
    semantic_results = semantic_retrieve(question, top_k=DEFAULT_TOP_K, model=model, client=client)

    hybrid_candidates = hybrid_retrieve(
        question, chunks, bm25_index, model=model, client=client, top_k=RERANK_CANDIDATE_POOL
    )
    hybrid_results = hybrid_candidates[:DEFAULT_TOP_K]
    reranked_results = rerank(question, hybrid_candidates, top_k=DEFAULT_TOP_K)

    print(f"\nQuestion: \"{question}\"\n")
    print_results("SEMANTIC only", semantic_results, "distance")
    print_results("HYBRID (semantic + BM25)", hybrid_results, "rrf_score")
    print_results("HYBRID + RERANKED (live pipeline)", reranked_results, "rerank_score")

    try:
        result = answer_question(question, top_k=DEFAULT_TOP_K, model=model, client=client)
        print(f"\n  Answer: {result['answer']}")
        if result["sources"]:
            label = "Source" if len(result["sources"]) == 1 else "Sources"
            print(f"  {label}: {', '.join(result['sources'])}")
    except requests.exceptions.HTTPError as e:
        print(f"\n  (Could not get a generated answer: {e})")
        print("  If this says '429', the free-tier daily quota is exhausted -- "
              "the retrieval comparison above is still valid, it doesn't need the LLM.")

    print()


def main():
    print("=== Retrieval inspection view ===")
    print("Type any question -- not limited to the fixed eval set.")
    print("Type 'exit' to quit.\n")

    model, client, bm25_index, chunks = build_inspection_setup()

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        inspect_one_question(question, model, client, bm25_index, chunks)


if __name__ == "__main__":
    main()
