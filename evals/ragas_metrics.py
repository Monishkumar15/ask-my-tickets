"""
Week 6, Phase 5: lightweight custom equivalents of RAGAS's four standard
RAG metrics -- not the `ragas` package itself (deliberate: fewer new
dependencies, full control over what each number actually means, matching
this project's "local, minimal-dependency" pattern elsewhere -- local
tracing instead of Langfuse, local embeddings/reranker instead of a hosted
service). Each function's docstring says exactly how it approximates the
real RAGAS metric and where that approximation is weaker, rather than
quietly pretending to be the real thing.

    faithfulness       -- does the answer only claim things the retrieved
                          context actually supports? (catches hallucination)
    answer_relevancy   -- does the answer actually address the question?
    context_precision  -- of the chunks retrieved, what fraction were from
                          an actually-expected source?
    context_recall     -- of the expected source(s), what fraction did
                          retrieval actually find?

context_precision/context_recall need ground truth (expected_source or
expected_sources_all on the eval case) -- same "returns None without it"
pattern as assertions.py's checks, rather than silently scoring 0.
"""
import json

import numpy as np

from generate import call_llm

FAITHFULNESS_PROMPT = """You are checking whether an assistant's reply only
states things actually supported by the context below -- NOT whether the
reply is a good reply, only whether it is truthful relative to this context.

Context:
{context}

Assistant's reply: {answer}

List each distinct factual claim in the reply, and whether the context
supports it. Respond with ONLY a JSON object, no other text:
{{"total_claims": <int>, "supported_claims": <int>, "unsupported": ["<claim text>", ...]}}
"""


def _strip_code_fence(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


def faithfulness(question, answer_text, chunks):
    """
    score = supported_claims / total_claims -- 1.0 means every claim in
    the answer was traceable to the retrieved context, 0.0 means none
    were. Costs one LLM call.
    """
    if not answer_text or not answer_text.strip():
        return {"score": None, "error": "no answer text (refused or errored)"}

    context = "\n\n".join(c["text"] for c in chunks) if chunks else "(no context retrieved)"
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer_text)
    try:
        # Reuses generate.py's call_llm() for the same Gemini/Groq-fallback
        # reason as judge.py -- see its comment on the equivalent line.
        content, _provider_used = call_llm(prompt, temperature=0)
        parsed = json.loads(_strip_code_fence(content))
        total = parsed["total_claims"]
        supported = parsed["supported_claims"]
        score = supported / total if total else 1.0  # no claims made -> vacuously faithful
        return {"score": score, "unsupported": parsed.get("unsupported", [])}
    except Exception as e:
        return {"score": None, "error": f"faithfulness check failed: {e}"}


def answer_relevancy(question, answer_text, embedding_model):
    """
    Lightweight approximation of RAGAS's answer relevancy: cosine
    similarity between the question's embedding and the answer's
    embedding, using the SAME local embedding model already loaded for
    retrieval -- no extra LLM call, no extra dependency.

    Weaker than real RAGAS (which generates several synthetic questions
    FROM the answer and compares those back to the real question, catching
    a fluent-but-off-topic answer more reliably): a direct question-vs-
    answer similarity can be fooled by an answer that reuses the
    question's own words without actually answering it. Cheap and free,
    not a perfect substitute -- documented limitation, not a hidden one.
    """
    if not answer_text or not answer_text.strip():
        return {"score": None, "error": "no answer text (refused or errored)"}

    vectors = embedding_model.encode([question, answer_text])
    q_vec, a_vec = vectors[0], vectors[1]
    cosine_sim = float(np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec)))
    return {"score": cosine_sim}


def _ground_truth_sources(case):
    return set(case.get("expected_sources_all") or ([case["expected_source"]] if case.get("expected_source") else []))


def context_precision(case, chunks):
    """Of the retrieved chunks, what fraction came from an actually-
    expected source? No ground truth on this case -> None, not 0."""
    expected = _ground_truth_sources(case)
    if not expected:
        return {"score": None, "error": "no ground truth (expected_source/expected_sources_all) on this case"}
    if not chunks:
        return {"score": 0.0}
    relevant_count = sum(1 for c in chunks if c["source"] in expected)
    return {"score": relevant_count / len(chunks)}


def context_recall(case, chunks):
    """Of the expected source(s), what fraction did retrieval actually
    find? No ground truth on this case -> None, not 0."""
    expected = _ground_truth_sources(case)
    if not expected:
        return {"score": None, "error": "no ground truth (expected_source/expected_sources_all) on this case"}
    retrieved_sources = {c["source"] for c in chunks}
    found = expected & retrieved_sources
    return {"score": len(found) / len(expected)}
