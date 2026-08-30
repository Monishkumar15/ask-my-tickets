"""
Week 4: BM25 keyword search -- the "exact words" half of hybrid search.

Unlike embeddings (which compare meaning), BM25 scores documents by how
many of the question's exact words they contain, weighted by how rare
(and therefore informative) each word is across the whole collection.
"""

import re

from rank_bm25 import BM25Okapi


def _tokenize(text):
    """Lowercase, keep only letters/digits as separate words."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(chunks):
    """
    chunks: list of {text, source, chunk_index} from chunk.load_and_chunk_all().
    Returns (bm25_index, chunks) -- chunks is returned alongside so callers
    can match a score back to its (source, chunk_index) by list position.
    """
    corpus = [_tokenize(c["text"]) for c in chunks]
    bm25_index = BM25Okapi(corpus)
    return bm25_index, chunks


def bm25_search(query, bm25_index, chunks, top_k):
    """
    Score every chunk against the query, return the top_k as a list of
    dicts: text, source, chunk_index, bm25_score (higher = more relevant).
    """
    scores = bm25_index.get_scores(_tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)

    return [
        {
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "bm25_score": score,
        }
        for chunk, score in ranked[:top_k]
    ]


if __name__ == "__main__":
    from chunk import load_and_chunk_all

    chunks = load_and_chunk_all()
    bm25_index, chunks = build_bm25_index(chunks)

    question = input("Ask a question (BM25 keyword search only): ")
    results = bm25_search(question, bm25_index, chunks, top_k=3)

    print(f"\nTop {len(results)} keyword matches:\n")
    for i, r in enumerate(results, start=1):
        print(f"{i}. [{r['source']} chunk {r['chunk_index']}] (bm25_score: {r['bm25_score']:.4f})")
        print(f"   {r['text']}\n")
