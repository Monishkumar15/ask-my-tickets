"""
Small HTTP API: upload a document (any supported format) and get it
ingested immediately, plus a health check for the app's dependencies.

Run with: uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile

from config import DATA_DIR, OPENROUTER_API_KEY
from embed import get_embedding_model
from loaders import LOADERS
from store import build_database, get_client
from vectorstore import count as count_chunks

app = FastAPI(title="Ask My Tickets API")

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
