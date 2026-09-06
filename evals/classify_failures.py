"""
Week 4: classify each eval question as WRONG_DOCUMENT (retrieval failure),
RIGHT_DOC_WRONG_ANSWER (generation failure), or RIGHT_DOC_RIGHT_ANSWER.

Uses an in-memory Qdrant collection (never touches the real persisted
data) and semantic_retrieve() (plain semantic search, unmodified) -- this
script diagnoses the Week 3 baseline, before hybrid search/reranking.

Minimizes OpenRouter calls: retrieval-only check costs 0 LLM calls; the
generation check (1 LLM call) only runs for questions where the correct
document WAS retrieved -- there's nothing to grade otherwise.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

import ingestion
from config import DEFAULT_TOP_K
from eval_questions import EVAL_QUESTIONS
from generate import answer_question
from retrieval import semantic_retrieve

RESULTS_FILE = "week4_classify_results.json"


def answer_contains_keywords(answer_text, expected_keywords):
    """
    Case-insensitive substring match: True only if EVERY expected keyword
    appears somewhere in the answer text. Simple and honest about its
    limits -- an LLM writing "fifty percent" instead of "50%" would
    register as a miss. Worth spot-checking, not silently trusted.
    """
    text_lower = answer_text.lower()
    return all(kw.lower() in text_lower for kw in expected_keywords)


def build_eval_collection():
    """Chunk + embed + store everything into a fresh in-memory Qdrant client."""
    chunks = ingestion.load_and_chunk_all()
    model = ingestion.get_embedding_model()
    chunks = ingestion.embed_chunks(chunks, model=model)

    client = ingestion.get_client(url=None)  # in-memory, real DB untouched
    embeddings = [c["embedding"].tolist() for c in chunks]
    ingestion.build_collection(client, chunks, embeddings)

    return model, client


def classify_question(question_entry, model, client):
    """Returns a dict describing what happened for this one question."""
    question = question_entry["question"]
    expected_source = question_entry["expected_source"]

    retrieved = semantic_retrieve(question, top_k=DEFAULT_TOP_K, model=model, client=client)
    retrieved_sources = [r["source"] for r in retrieved]
    doc_hit = expected_source in retrieved_sources

    result = {
        "id": question_entry["id"],
        "question": question,
        "expected_source": expected_source,
        "retrieved_sources": retrieved_sources,
        "top1_distance": retrieved[0]["distance"] if retrieved else None,
        "doc_hit": doc_hit,
        "classification": None,
        "answer": None,
        "keyword_grade": None,
    }

    if not doc_hit:
        result["classification"] = "WRONG_DOCUMENT"
        return result

    # Doc was retrieved -- now check if generation actually got it right.
    # This is the only place this script spends an LLM call, and only when
    # there's actually something to grade.
    try:
        gen_result = answer_question(question, top_k=DEFAULT_TOP_K, model=model, client=client)
    except requests.exceptions.HTTPError as e:
        result["classification"] = "SKIPPED_LLM_ERROR"
        result["answer"] = f"(LLM call failed: {e})"
        return result

    result["answer"] = gen_result["answer"]

    if gen_result["skipped_llm"]:
        # The doc showed up in semantic_retrieve()'s top-k, but generate.py's
        # OWN (independent) distance check refused anyway -- still a
        # generation-side failure to deliver a usable answer despite
        # having the right document available.
        result["classification"] = "RIGHT_DOC_WRONG_ANSWER"
        result["keyword_grade"] = False
        return result

    keyword_grade = answer_contains_keywords(gen_result["answer"], question_entry["expected_keywords"])
    result["keyword_grade"] = keyword_grade
    result["classification"] = "RIGHT_DOC_RIGHT_ANSWER" if keyword_grade else "RIGHT_DOC_WRONG_ANSWER"
    return result


def main():
    model, client = build_eval_collection()

    results = []
    for question_entry in EVAL_QUESTIONS:
        result = classify_question(question_entry, model, client)
        results.append(result)

        print(f"[{result['classification']}] {result['id']}: \"{result['question']}\"")
        print(f"   expected: {result['expected_source']}")
        print(f"   retrieved: {result['retrieved_sources']} (top-1 distance: {result['top1_distance']})")
        if result["answer"]:
            print(f"   answer: {result['answer'][:150]}")
        print()

    tally = {}
    for r in results:
        tally[r["classification"]] = tally.get(r["classification"], 0) + 1

    print("=== Summary ===")
    for label, count in tally.items():
        print(f"{label}: {count}/{len(results)}")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
