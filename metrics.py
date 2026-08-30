"""
Week 4: retrieval quality metrics -- turning "did the right document show
up" into an actual number, instead of a feeling.
"""


def hit_rate_at_k(results_list, expected_sources, k):
    """
    results_list: a list of per-question retrieval results (each a list of
    dicts with a "source" key, already sorted best-first).
    expected_sources: a parallel list -- expected_sources[i] is the correct
    source for results_list[i].

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
    Mean Reciprocal Rank: for each question, find the 1-based rank of the
    first result whose source matches the expected one, and score it as
    1/rank (rank 1 -> 1.0, rank 2 -> 0.5, rank 3 -> 0.33, not found -> 0).
    Then average across all questions.

    Rewards ranking the correct answer HIGH, not just "somewhere in top-k"
    -- catches improvement that hit_rate_at_k alone might miss (e.g. moving
    the correct source from rank 3 to rank 1 doesn't change hit-rate@3, but
    does change MRR).
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
    # A tiny worked example so you can see the formulas in action.
    fake_results = [
        [{"source": "A"}, {"source": "B"}, {"source": "C"}],  # expected "A", rank 1
        [{"source": "B"}, {"source": "A"}, {"source": "C"}],  # expected "A", rank 2
        [{"source": "C"}, {"source": "B"}, {"source": "D"}],  # expected "A", not found
    ]
    expected = ["A", "A", "A"]

    print(f"hit_rate_at_k (k=3): {hit_rate_at_k(fake_results, expected, k=3):.2f}")  # 2/3 = 0.67
    print(f"mrr: {mrr(fake_results, expected):.4f}")  # (1.0 + 0.5 + 0) / 3 = 0.5
