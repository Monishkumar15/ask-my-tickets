"""
Small HTTP API: upload a document (any supported format) and get it
ingested immediately, plus a health check for the app's dependencies.

Run with: uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

import os
import shutil
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    CORS_ALLOWED_ORIGINS,
    DATA_DIR,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_TEMPERATURE,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)
from generate import answer_question
from ingestion import LOADERS, build_database, get_client, get_embedding_model, grouped_sources
from ingestion import count as count_chunks
from retrieval import retrieve
from sessions import add_message, create_session, delete_session, get_session, list_sessions
from tracing import annotate, get as get_trace, list_recent

# evals/ is normally a one-way dependency (eval scripts import from the app
# root, never the reverse) -- this is the one exception: the Week 8
# trajectory checker belongs in evals/ alongside assertions.py, but the
# React trace viewer needs the same check the eval scripts use, not a
# second copy of the logic.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals"))
from trajectory_checks import run_trajectory_checks

app = FastAPI(title="Ask My Tickets API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    # DELETE is required for /sessions/{id}; this list previously only had
    # GET/POST, which happened to not matter locally because the Vite dev
    # server's proxy makes every request same-origin (no browser CORS
    # preflight ever occurs) -- but it would silently break session
    # deletion for any frontend hosted on its own origin, without a proxy.
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=20)
    # "gemini" or "groq" -- which provider to try first for this request.
    # The other is the automatic fallback if this one hits a 429.
    provider: str | None = Field(default=None, pattern="^(gemini|groq)$")
    # How random/deterministic the answer's wording is -- 0 is closest to
    # deterministic, higher values allow more varied phrasing. Bounds match
    # what OpenAI-compatible chat-completions endpoints (Gemini's and
    # Groq's here) accept.
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    # Which saved chat session this question belongs to. When set, both the
    # question and the answer are appended to that session's history so the
    # chatbot UI can show the whole conversation, not just the last turn.
    session_id: str | None = Field(default=None, min_length=1, max_length=64)

class TraceAnnotation(BaseModel):
    reviewed: bool | None = None
    failure_note: str | None = Field(default=None, max_length=2000)
    problem_type: str | None = Field(default=None, max_length=120)
    severity: int | None = Field(default=None, ge=1, le=5)

class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=20)

# Loaded once, reused across requests -- same "load once" pattern as app.py.
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = get_embedding_model()
    return _model


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (.txt, .md, .docx, .pdf, .xlsx) into data/ and
    immediately rebuild the vector database so it's searchable right away.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in LOADERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {list(LOADERS.keys())}",
        )

    # Only keep the base filename -- never trust a path from the client
    # (prevents writing outside data/, e.g. "../../something").
    safe_name = os.path.basename(file.filename)
    dest_path = os.path.join(DATA_DIR, safe_name)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Rebuild the whole index so the new file is searchable immediately.
    # Fine at this project's scale; a much larger corpus would need a
    # smarter incremental-ingest approach instead of a full rebuild per upload.
    client = build_database()

    return {
        "message": f"Uploaded and ingested '{safe_name}'.",
        "total_chunks_in_index": count_chunks(client),
    }


@app.get("/health")
def health_check():
    """
    Checks each dependency the app needs and reports ok/error per check.
    Does NOT call Gemini/Groq (would burn free-tier quota) -- only confirms
    each API key is present, not that it's valid.
    """
    checks = {}

    try:
        _get_model().encode(["health check"])
        checks["embedding_model"] = "ok"
    except Exception as e:
        checks["embedding_model"] = f"error: {e}"

    try:
        client = get_client()
        chunk_count = count_chunks(client)
        checks["vector_store"] = f"ok ({chunk_count} chunks stored)"
    except Exception as e:
        checks["vector_store"] = f"error: {e}"

    checks["gemini_key"] = "present" if GEMINI_API_KEY else "missing"
    checks["groq_key"] = "present" if GROQ_API_KEY else "missing"
    checks["langfuse"] = "present" if (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY) else "missing"
    checks["data_dir"] = "ok" if os.path.isdir(DATA_DIR) else f"error: '{DATA_DIR}' not found"

    overall = "ok" if all(
        v == "present" or v.startswith("ok") for v in checks.values()
    ) else "degraded"

    return {"status": overall, "checks": checks}

@app.post("/ask")
def ask_question(request: AskRequest):
    if request.session_id and get_session(request.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        result = answer_question(
            request.question,
            top_k=request.top_k,
            model=_get_model(),
            client=get_client(),
            provider=request.provider or DEFAULT_LLM_PROVIDER,
            temperature=request.temperature,
            return_trace_id=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if request.session_id:
        add_message(request.session_id, "user", request.question)
        add_message(
            request.session_id,
            "assistant",
            result.get("answer", ""),
            extra={
                "sources": result.get("sources"),
                "provider": result.get("provider"),
                "trace_id": result.get("trace_id"),
            },
        )

    return result


@app.post("/sessions")
def new_session():
    return create_session()


@app.get("/sessions")
def sessions():
    return list_sessions()


@app.get("/sessions/{session_id}")
def session_detail(session_id: str):
    try:
        session = get_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    try:
        deleted = delete_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}

@app.post("/retrieve")
def inspect_retrieval(request: RetrieveRequest):
    try:
        return retrieve(request.question, top_k=request.top_k, model=_get_model(), client=get_client())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@app.get("/documents")
def documents():
    try:
        return grouped_sources(get_client())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@app.get("/traces")
def traces(limit: int = 50, offset: int = 0):
    if not 1 <= limit <= 200: raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    if offset < 0: raise HTTPException(status_code=422, detail="offset must be >= 0")
    records, total = list_recent(limit, offset)
    return {"traces": records, "total": total, "limit": limit, "offset": offset}

@app.get("/traces/{trace_id}")
def trace_detail(trace_id: str):
    record = get_trace(trace_id)
    if record is None: raise HTTPException(status_code=404, detail="Trace not found")
    return record

@app.get("/traces/{trace_id}/trajectory")
def trace_trajectory(trace_id: str):
    """
    Week 8: rule-based trajectory verdict for one agent trace, for the
    React trace viewer's "Error analysis" page. An arbitrary live trace has
    no eval "case" (no expected_terminal_action/expected_sources_all to
    check against) -- passing an empty case runs only the checks that
    don't need one (no_repeated_search, no_wasted_steps,
    not_silently_incomplete), which is exactly the right scope for
    inspecting a trace nobody labeled ahead of time.
    """
    record = get_trace(trace_id)
    if record is None: raise HTTPException(status_code=404, detail="Trace not found")
    if record.get("config", {}).get("mode") != "agent":
        raise HTTPException(status_code=400, detail="Not an agent trace")
    checks = run_trajectory_checks({}, record)
    passed = all(c["passed"] for c in checks) if checks else True
    return {"passed": passed, "checks": checks}

@app.post("/traces/{trace_id}/annotation")
def annotate_trace(trace_id: str, annotation: TraceAnnotation):
    values = annotation.model_dump(exclude_none=True) if hasattr(annotation, "model_dump") else annotation.dict(exclude_none=True)
    try: record = annotate(trace_id, values)
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None: raise HTTPException(status_code=404, detail="Trace not found")
    return record


# The lightweight browser review surface is served by the same app, avoiding a
# second dev server while the Week 5 sample is being collected.
frontend_dist = Path(__file__).parent / "ui-react" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="ui")
