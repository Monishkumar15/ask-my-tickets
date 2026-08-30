"""
Week 4: the inspection view.

For every eval question, prints side by side: the question, what the OLD
method (plain semantic search) retrieved, what the NEW method (hybrid RRF)
retrieved, whether each found the expected source, and (reused from
classify_failures.py's saved results, so no extra OpenRouter calls are
spent) the generated answer if one was already produced.
"""

import json
import os

import vectorstore
from bm25_search import build_bm25_index
from chunk import load_and_chunk_all
from config import DEFAULT_TOP_K
from embed import embed_chunks, get_embedding_model
from eval_questions import EVAL_QUESTIONS
from hybrid import hybrid_retrieve
from retrieve import retrieve

CLASSIFY_RESULTS_FILE = "week4_classify_results.json"


def load_cached_answers():
    """Reuse classify_failures.py's saved results if present, keyed by
    question id -- avoids spending any additional LLM calls just to display
    an answer we already generated once."""
    if not os.path.exists(CLASSIFY_RESULTS_FILE):
        return {}
    with open(CLASSIFY_RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {r["id"]: r for r in results}


def build_eval_setup():
    chunks = load_and_chunk_all()
    model = get_embedding_model()
    embedded_chunks = embed_chunks(list(chunks), model=model)
    embeddings = [c["embedding"].tolist() for c in embedded_chunks]

    client = vectorstore.get_client(url=None)  # in-memory, real DB untouched
    vectorstore.build_collection(client, chunks, embeddings)

    bm25_index, chunks = build_bm25_index(chunks)
    return model, client, chunks, bm25_index


def format_results(results, score_key, expected_source):
    """One line per retrieved chunk: source, chunk_index, score, hit marker."""
    lines = []
    for r in results:
        marker = " <-- expected" if r["source"] == expected_source else ""
        lines.append(f"      [{r['source']} chunk {r['chunk_index']}] {score_key}={r[score_key]:.4f}{marker}")
    return "\n".join(lines) if lines else "      (no results)"


def main():
    model, client, chunks, bm25_index = build_eval_setup()
    cached_answers = load_cached_answers()

    for q in EVAL_QUESTIONS:
        old_results = retrieve(q["question"], top_k=DEFAULT_TOP_K, model=model, client=client)
        new_results = hybrid_retrieve(q["question"], chunks, bm25_index, model=model, client=client, top_k=DEFAULT_TOP_K)

        old_hit = q["expected_source"] in {r["source"] for r in old_results}
        new_hit = q["expected_source"] in {r["source"] for r in new_results}

        print("=" * 90)
        print(f"[{q['id']}] \"{q['question']}\"")
        print(f"    expected source: {q['expected_source']}")
        print()
        print(f"  OLD (semantic only) -- hit: {old_hit}")
        print(format_results(old_results, "distance", q["expected_source"]))
        print()
        print(f"  NEW (hybrid RRF) -- hit: {new_hit}")
        print(format_results(new_results, "rrf_score", q["expected_source"]))
        print()

        cached = cached_answers.get(q["id"])
        if cached and cached.get("answer"):
            print(f"  Generated answer (cached from classify_failures.py):")
            print(f"      {cached['answer'][:300]}")
            if cached.get("keyword_grade") is not None:
                print(f"      keyword grade: {'PASS' if cached['keyword_grade'] else 'FAIL'}")
        else:
            print("  Generated answer: (not available -- run classify_failures.py first, "
                  "or check OpenRouter quota)")
        print()


if __name__ == "__main__":
    main()
