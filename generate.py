"""
Step 7: Turn retrieved chunks into a grounded, cited answer -- or an honest
"I don't know" if nothing relevant was found.

Generation uses two providers: Gemini (tried first by default) and Groq
(automatic fallback). Both expose an OpenAI-compatible chat-completions
endpoint, so the same request shape works for either.
"""

import requests
from langfuse import observe
from langfuse import get_client as get_langfuse_client

from config import (
    DEFAULT_LLM_PROVIDER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DISTANCE_THRESHOLD,
    GEMINI_API_KEY,
    GEMINI_API_URL,
    GEMINI_MODEL_ID,
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL_ID,
)
from retrieval import retrieve
from tracing import Trace

# NOTE: DISTANCE_THRESHOLD was empirically tuned for the old BGE-small +
# ChromaDB setup -- after the MiniLM + Qdrant switch, re-measure real
# distances (see retrieval.py) and adjust DISTANCE_THRESHOLD in .env if
# refusal behavior looks off.

PROVIDERS = {
    "gemini": {"api_url": GEMINI_API_URL, "api_key": GEMINI_API_KEY, "model_id": GEMINI_MODEL_ID},
    "groq": {"api_url": GROQ_API_URL, "api_key": GROQ_API_KEY, "model_id": GROQ_MODEL_ID},
}


def build_prompt(question, chunks):
    def _label(c):
        page = c.get("page")
        return f"{c['source']}, page {page}" if page else c["source"]

    context = "\n\n".join(
        f"[Source: {_label(c)}]\n{c['text']}" for c in chunks
    )
    return f"""You are a support assistant. Answer the question using ONLY the
context below. Do not use any outside knowledge.

If the question has multiple parts, answer each part separately using
whatever context covers it. Only say you don't know about the SPECIFIC part
that has no supporting context -- do not refuse the entire answer just
because one part is missing evidence. If NONE of the question is covered by
the context, respond with exactly:
"I don't know based on the available documents."

Give a complete answer: include the relevant conditions, numbers, or
exceptions mentioned in the context, not just a bare one-line restatement.
Do not add any detail that isn't actually in the context below.

You are speaking directly to the customer. The context is often written as
internal instructions for a support AGENT (e.g. "agents should...",
"customer service representatives must inform the customer..." ) -- that is
backstage language, not something the customer should ever see. Rewrite any
such instruction in your own words as something YOU are doing for the
customer (first person: "we will...", "I'll..."), and never mention
"agents", "staff", "representatives", or that you were instructed to do
something.

Context:
{context}

Question: {question}

Answer:"""


@observe(as_type="generation", name="llm-call")
def _call_provider(provider_name, prompt, temperature=DEFAULT_TEMPERATURE):
    """POST to one provider's OpenAI-compatible chat-completions endpoint."""
    provider = PROVIDERS[provider_name]
    response = requests.post(
        provider["api_url"],
        headers={"Authorization": f"Bearer {provider['api_key']}"},
        json={
            "model": provider["model_id"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
        # Found live during Week 8 eval runs: with no timeout, a stalled
        # connection blocks forever -- no retry, no fallback to the other
        # provider, nothing recoverable. 60s is generous above any real
        # observed latency (worst case measured ~17s for a multi-step
        # agent's single call) but still bounded.
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    get_langfuse_client().update_current_generation(
        model=provider["model_id"],
        input=prompt,
        output=answer,
        usage_details=data.get("usage"),
    )
    return answer


def _is_tool_use_failed(response):
    """
    Some Groq models (e.g. openai/gpt-oss-20b) have native function-calling
    behavior baked into how they're served, and can auto-detect a ReAct-style
    prompt's "Available actions: <tool>(...)" phrasing as an implicit
    tool-call request -- then reject their OWN attempt: a 400 with
    {"error": {"code": "tool_use_failed", "message": "Tool choice is none,
    but model called a tool"}}. Confirmed live (Week 7, reconfirmed Week 9):
    non-deterministic -- the identical prompt can succeed or fail across
    calls, so this is a model-serving quirk, not a prompt bug to fix once.
    Never triggers on the fixed pipeline's plain-language prompts (no
    tool/action vocabulary in them) -- only agent.py's tool-listing prompt
    can hit this, on whichever model happens to be configured.
    """
    if response is None or response.status_code != 400:
        return False
    try:
        return response.json().get("error", {}).get("code") == "tool_use_failed"
    except ValueError:
        return False


def call_llm(prompt, preferred_provider=DEFAULT_LLM_PROVIDER, temperature=DEFAULT_TEMPERATURE):
    """
    Try preferred_provider first. Falls back to the other provider only for
    known-recoverable failures: a 429 (quota/rate-limit), or a model
    serving-side tool-call misfire (_is_tool_use_failed) -- both are
    properties of the specific model/provider handling this one request,
    not the request itself, so the other provider is expected to succeed
    where this one didn't. Any other kind of error (bad request, network
    issue, etc.) surfaces immediately instead of silently retrying
    elsewhere -- this project treats "retry automatically" as something
    each failure mode has to individually earn, not a default.

    Returns (answer_text, provider_that_actually_answered).
    """
    other_provider = "groq" if preferred_provider == "gemini" else "gemini"

    try:
        return _call_provider(preferred_provider, prompt, temperature), preferred_provider
    except requests.exceptions.HTTPError as e:
        is_quota_error = e.response is not None and e.response.status_code == 429
        if not (is_quota_error or _is_tool_use_failed(e.response)):
            raise
        return _call_provider(other_provider, prompt, temperature), other_provider


@observe(name="fixed-pipeline-ask")
def answer_question(question, top_k=DEFAULT_TOP_K, model=None, client=None,
                     provider=DEFAULT_LLM_PROVIDER, temperature=DEFAULT_TEMPERATURE,
                     return_trace_id=False):
    trace = Trace(question, {
        "top_k": top_k,
        "distance_threshold": DISTANCE_THRESHOLD,
        "preferred_provider": provider,
        "temperature": temperature,
    })
    try:
        chunks = retrieve(question, top_k=top_k, model=model, client=client)
        trace.stage("retrieval", chunks=chunks, returned_count=len(chunks))

        best_distance = chunks[0]["distance"] if chunks else float("inf")
        trace.stage("confidence_gate", best_distance=best_distance, threshold=DISTANCE_THRESHOLD, passed=best_distance <= DISTANCE_THRESHOLD)
        if best_distance > DISTANCE_THRESHOLD:
            get_langfuse_client().update_current_span(metadata={"refused": True, "best_distance": best_distance})
            result = {
                "answer": f'I couldn\'t find relevant information to answer "{question}". '
                           f"Try asking something related to the topics covered in the "
                           f"documents.",
                "sources": [],
                "skipped_llm": True,
                "provider": None,
            }
            result["trace_id"] = trace.finish(answer=result["answer"], outcome="refusal")
            return result

        prompt = build_prompt(question, chunks)
        trace.stage("prompt", source_count=len({chunk["source"] for chunk in chunks}))
        answer_text, provider_used = call_llm(prompt, preferred_provider=provider, temperature=temperature)
        trace.stage("generation", requested_provider=provider, provider_used=provider_used, completed=True)

        # List every distinct source that contributed context, closest match
        # first -- a question spanning multiple topics may pull chunks from
        # more than one document, and the citation should reflect all of them,
        # not just the single best match.
        sources = list(dict.fromkeys(c["source"] for c in chunks))

        result = {"answer": answer_text, "sources": sources, "skipped_llm": False, "provider": provider_used}
        result["trace_id"] = trace.finish(answer=answer_text, outcome="answer")
        return result
    except Exception as exc:
        trace.stage("error", error_type=type(exc).__name__, message=str(exc))
        trace.finish(outcome="error", error=str(exc))
        raise


if __name__ == "__main__":
    question = input("Ask a question: ")
    result = answer_question(question)

    print(f"\nAnswer: {result['answer']}")
    if result["sources"]:
        label = "Source" if len(result["sources"]) == 1 else "Sources"
        print(f"{label}: {', '.join(result['sources'])}")
    if result["provider"]:
        print(f"(answered by: {result['provider']})")
