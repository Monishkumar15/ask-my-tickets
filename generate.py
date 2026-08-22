"""
Step 7: Turn retrieved chunks into a grounded, cited answer -- or an honest
"I don't know" if nothing relevant was found.
"""

import requests

from config import (
    DEFAULT_TOP_K,
    DISTANCE_THRESHOLD,
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    OPENROUTER_MODEL_ID,
)
from retrieve import retrieve

API_KEY = OPENROUTER_API_KEY
API_URL = OPENROUTER_API_URL
MODEL_ID = OPENROUTER_MODEL_ID
# DISTANCE_THRESHOLD now comes from config.py / .env. NOTE: this value was
# empirically tuned for the old BGE-small + ChromaDB setup -- after the
# MiniLM + Qdrant switch, re-measure real distances (see retrieve.py) and
# adjust DISTANCE_THRESHOLD in .env if refusal behavior looks off.


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


def answer_question(question, top_k=DEFAULT_TOP_K, model=None, client=None):
    chunks = retrieve(question, top_k=top_k, model=model, client=client)

    best_distance = chunks[0]["distance"] if chunks else float("inf")
    if best_distance > DISTANCE_THRESHOLD:
        return {
            "answer": f'I couldn\'t find relevant information to answer "{question}". '
                       f"Try asking something related to the topics covered in the "
                       f"documents.",
            "sources": [],
            "skipped_llm": True,
        }

    prompt = build_prompt(question, chunks)
    answer_text = call_llm(prompt)

    # List every distinct source that contributed context, closest match
    # first -- a question spanning multiple topics (see retrieve.py's
    # diversification) may pull chunks from more than one document, and the
    # citation should reflect all of them, not just the single best match.
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {
        "answer": answer_text,
        "sources": sources,
        "skipped_llm": False,
    }


if __name__ == "__main__":
    question = input("Ask a question: ")
    result = answer_question(question)

    print(f"\nAnswer: {result['answer']}")
    if result["sources"]:
        label = "Source" if len(result["sources"]) == 1 else "Sources"
        print(f"{label}: {', '.join(result['sources'])}")
