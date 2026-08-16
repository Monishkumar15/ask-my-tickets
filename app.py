"""
Step 8: Interactive CLI app -- "Ask My Tickets"

Loads the embedding model and vector database once, then lets you ask
as many questions as you want in one session.
"""

import requests

from embed import get_embedding_model
from generate import answer_question
from store import get_collection


def main():
    print("=== Ask My Tickets ===")
    print("Type 'exit' to quit.\n")

    # Load once, reuse for every question -- avoids reloading the model /
    # reconnecting to the database on every single loop iteration.
    model = get_embedding_model()
    collection = get_collection()

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue  # user just pressed Enter, ask again

        # A single failed request (e.g. hitting the free model's rate limit)
        # shouldn't crash the whole session -- show a message and let the
        # user try again, instead of the program exiting on one bad call.
        try:
            result = answer_question(question, model=model, collection=collection)
        except requests.exceptions.HTTPError as e:
            print(f"\nSomething went wrong calling the AI model ({e}).")
            print("If this says '429', you've hit the free model's rate limit"
                  " -- wait a few seconds and try again.\n")
            continue

        print(f"\nAnswer: {result['answer']}")
        if result["source"]:
            print(f"Source: {result['source']}")
        print()  # blank line before the next prompt


if __name__ == "__main__":
    main()
