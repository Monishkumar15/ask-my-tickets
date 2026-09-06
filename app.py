"""
Step 8: Interactive CLI app -- "Ask My Tickets"

Loads the embedding model and vector database once, then lets you ask
as many questions as you want in one session.
"""

import requests

from generate import answer_question
from ingestion import get_client, get_embedding_model


def main():
    print("=== Ask My Tickets ===")
    print("Type 'exit' to quit.\n")

    # Load once, reuse for every question -- avoids reloading the model /
    # reconnecting to Qdrant on every single loop iteration.
    model = get_embedding_model()
    client = get_client()

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue  # user just pressed Enter, ask again

        # A single failed request shouldn't crash the whole session -- show
        # a message and let the user try again, instead of the program
        # exiting on one bad call. Gemini already automatically falls back
        # to Groq on a 429, so seeing this means BOTH providers were
        # rate-limited/unavailable.
        try:
            result = answer_question(question, model=model, client=client)
        except requests.exceptions.HTTPError as e:
            print(f"\nSomething went wrong calling the AI model ({e}).")
            print("If this says '429', both Gemini and Groq are rate-limited"
                  " right now -- wait a few seconds and try again.\n")
            continue

        print(f"\nAnswer: {result['answer']}")
        if result["sources"]:
            label = "Source" if len(result["sources"]) == 1 else "Sources"
            print(f"{label}: {', '.join(result['sources'])}")
        if result["provider"]:
            print(f"(answered by: {result['provider']})")
        print()  # blank line before the next prompt


if __name__ == "__main__":
    main()
