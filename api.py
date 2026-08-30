"""
Small HTTP API: upload a document (any supported format) and get it
ingested immediately, plus a health check for the app's dependencies.

Run with: uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from config import DATA_DIR, OPENROUTER_API_KEY
from embed import get_embedding_model
from loaders import LOADERS
from store import build_database, get_client
from vectorstore import count as count_chunks
from generate import answer_question
from tracing import annotate, get as get_trace, list_recent

app = FastAPI(title="Ask My Tickets API")

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=20)

class TraceAnnotation(BaseModel):
    reviewed: bool | None = None
    failure_note: str | None = Field(default=None, max_length=2000)
    problem_type: str | None = Field(default=None, max_length=120)
    severity: int | None = Field(default=None, ge=1, le=5)

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
    Does NOT call OpenRouter (would burn free-tier quota) -- only confirms
    the API key is present, not that it's valid.
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

    checks["openrouter_key"] = "present" if OPENROUTER_API_KEY else "missing"
    checks["data_dir"] = "ok" if os.path.isdir(DATA_DIR) else f"error: '{DATA_DIR}' not found"

    overall = "ok" if all(
        v == "present" or v.startswith("ok") for v in checks.values()
    ) else "degraded"

    return {"status": overall, "checks": checks}

@app.post("/ask")
def ask_question(request: AskRequest):
    try:
        return answer_question(request.question, top_k=request.top_k, model=_get_model(), client=get_client(), return_trace_id=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@app.get("/traces")
def traces(limit: int = 50):
    if not 1 <= limit <= 200: raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    return list_recent(limit)

@app.get("/traces/{trace_id}")
def trace_detail(trace_id: str):
    record = get_trace(trace_id)
    if record is None: raise HTTPException(status_code=404, detail="Trace not found")
    return record

@app.post("/traces/{trace_id}/annotation")
def annotate_trace(trace_id: str, annotation: TraceAnnotation):
    values = annotation.model_dump(exclude_none=True) if hasattr(annotation, "model_dump") else annotation.dict(exclude_none=True)
    try: record = annotate(trace_id, values)
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None: raise HTTPException(status_code=404, detail="Trace not found")
    return record
