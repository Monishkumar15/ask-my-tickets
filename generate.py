"""
Step 7: Turn retrieved chunks into a grounded, cited answer -- or an honest
"I don't know" if nothing relevant was found.
"""

import os

import requests
from dotenv import load_dotenv

from retrieve import retrieve

load_dotenv()  # reads OPENROUTER_API_KEY from your .env file

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "google/gemma-4-26b-a4b-it:free"

# BGE-small distances for a *real* match landed around 0.55-0.65 in our own
# testing (Step 6). Anything much higher than that means the best chunk we
# found isn't actually relevant -- so we refuse instead of guessing.
DISTANCE_THRESHOLD = 0.75


def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""You are a support assistant. Answer the question using ONLY the
context below. Do not use any outside knowledge.

If the answer is not contained in the context, respond with exactly:
"I don't know based on the available documents."

Context:
{context}

Question: {question}

Answer:"""


def call_llm(prompt):
    # NOTE: no retry/backoff here on purpose for this week's assignment scope.
    # If you hit a 429 (rate limited), just wait a few seconds and rerun.
    # Retry logic is a good upcoming-week improvement -- see the comment
    # at the bottom of this file for what that would look like.
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    response.raise_for_status()  # raises an error if the request failed
    data = response.json()
    return data["choices"][0]["message"]["content"]


def answer_question(question, top_k=3, model=None, collection=None):
    chunks = retrieve(question, top_k=top_k, model=model, collection=collection)

    best_distance = chunks[0]["distance"] if chunks else float("inf")
    if best_distance > DISTANCE_THRESHOLD:
        return {
            "answer": "I don't know based on the available documents.",
            "source": None,
            "skipped_llm": True,
        }

    prompt = build_prompt(question, chunks)
    answer_text = call_llm(prompt)

    return {
        "answer": answer_text,
        "source": chunks[0]["source"],
        "skipped_llm": False,
    }


if __name__ == "__main__":
    question = input("Ask a question: ")
    result = answer_question(question)

    print(f"\nAnswer: {result['answer']}")
    if result["source"]:
        print(f"Source: {result['source']}")
