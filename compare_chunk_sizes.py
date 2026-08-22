"""
Step 9: Compare retrieval quality across two different chunk sizes.

Uses a temporary in-memory Qdrant instance (not your real Qdrant server) so
your actual app's data is left untouched. No LLM/OpenRouter calls here --
this is a pure retrieval comparison, so it costs nothing and has no rate limit.
"""

import vectorstore
from chunk import load_and_chunk_all
from config import QUERY_INSTRUCTION
from embed import get_embedding_model

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


def build_temp_collection(chunk_size, overlap_sentences, model, collection_name):
    """Chunk + embed + store into a fresh in-memory (non-persistent) Qdrant collection."""
    chunks = load_and_chunk_all(chunk_size=chunk_size, overlap_sentences=overlap_sentences)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()

    client = vectorstore.get_client(url=None)  # in-memory only, nothing written to disk
    vectorstore.build_collection(client, chunks, embeddings, collection_name=collection_name)
    return client, len(chunks)


def run_comparison():
    model = get_embedding_model()

    for config in CHUNK_CONFIGS:
        collection_name = f"test_{config['chunk_size']}"
        client, chunk_count = build_temp_collection(
            config["chunk_size"], config["overlap_sentences"], model, collection_name
        )

        print(f"\n=== {config['label']} -> {chunk_count} total chunks ===\n")

        correct = 0
        for test in TEST_QUESTIONS:
            query_text = QUERY_INSTRUCTION + test["question"] if QUERY_INSTRUCTION else test["question"]
            # Qdrant wants a single flat vector, not a batch -- [0] unwraps it.
            query_embedding = model.encode([query_text]).tolist()[0]

            results = vectorstore.search(client, query_embedding, top_k=1, collection_name=collection_name)
            top_source = results[0]["source"]
            top_distance = results[0]["distance"]

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
