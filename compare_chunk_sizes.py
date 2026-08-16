"""
Step 9: Compare retrieval quality across two different chunk sizes.

Uses a temporary in-memory ChromaDB (not your real chroma_db/ folder) so
your actual app's data is left untouched. No LLM/OpenRouter calls here --
this is a pure retrieval comparison, so it costs nothing and has no rate limit.
"""

import chromadb

from chunk import load_and_chunk_all
from embed import get_embedding_model, QUERY_INSTRUCTION

# One test question per ticket, deliberately reworded so it does NOT share
# exact words with the source document -- a real test of semantic matching.
TEST_QUESTIONS = [
    {"question": "why can't I log in after too many wrong passwords?",
     "expected_source": "ticket_001_password_lockout.txt"},
    {"question": "can I get money back if I bought something 6 weeks ago?",
     "expected_source": "ticket_002_refund_window.txt"},
    {"question": "my package is really late, what can be done?",
     "expected_source": "ticket_003_shipping_delay.txt"},
    {"question": "what happens to my info if I stop my plan?",
     "expected_source": "ticket_004_account_cancellation.txt"},
    {"question": "I was charged twice for the same thing",
     "expected_source": "ticket_005_billing_dispute.txt"},
]

CHUNK_CONFIGS = [
    {"label": "SMALL (chunk_size=100)", "chunk_size": 100, "overlap_sentences": 1},
    {"label": "LARGE (chunk_size=600)", "chunk_size": 600, "overlap_sentences": 1},
]


def build_temp_collection(chunk_size, overlap_sentences, model):
    """Chunk + embed + store into a fresh in-memory (non-persistent) collection."""
    chunks = load_and_chunk_all(chunk_size=chunk_size, overlap_sentences=overlap_sentences)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()

    client = chromadb.Client()  # in-memory only, nothing written to disk
    collection = client.create_collection(name=f"test_{chunk_size}")
    collection.add(
        ids=[f"{c['source']}_{c['chunk_index']}" for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return collection, len(chunks)


def run_comparison():
    model = get_embedding_model()

    for config in CHUNK_CONFIGS:
        collection, chunk_count = build_temp_collection(
            config["chunk_size"], config["overlap_sentences"], model
        )

        print(f"\n=== {config['label']} -> {chunk_count} total chunks ===\n")

        correct = 0
        for test in TEST_QUESTIONS:
            query_text = QUERY_INSTRUCTION + test["question"]
            query_embedding = model.encode([query_text]).tolist()

            result = collection.query(query_embeddings=query_embedding, n_results=1)
            top_source = result["metadatas"][0][0]["source"]
            top_distance = result["distances"][0][0]

            is_correct = top_source == test["expected_source"]
            correct += is_correct
            status = "CORRECT" if is_correct else "WRONG"

            print(f"[{status}] \"{test['question']}\"")
            print(f"   -> top match: {top_source} (distance: {top_distance:.4f})")
            if not is_correct:
                print(f"   -> expected: {test['expected_source']}")
            print()

        print(f"Score: {correct}/{len(TEST_QUESTIONS)} correct top-1 matches "
              f"for {config['label']}")


if __name__ == "__main__":
    run_comparison()
