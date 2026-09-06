"""
Local chat session storage for the Ask page's chatbot UI.

Separate from tracing.py on purpose: tracing.py records diagnostic detail
for the Week 4/5 error-analysis workflow (one file per question, read by
developers/mentors reviewing failures); this file stores actual
conversation history for the chat UI (one file per session, containing
every message in that conversation, read by the person using the chatbot).
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import SESSIONS_DIR as _SESSIONS_DIR

SESSIONS_DIR = Path(_SESSIONS_DIR)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _path(session_id):
    if not session_id or any(c not in "0123456789abcdef" for c in session_id):
        raise ValueError("Invalid session id")
    return SESSIONS_DIR / f"{session_id}.json"


def create_session():
    """Start a new, empty session. Returns the full session dict."""
    session = {
        "id": uuid.uuid4().hex,
        "title": "New chat",
        "created_at": _now(),
        "updated_at": _now(),
        "messages": [],
    }
    _save(session)
    return session


def _save(session):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    target = _path(session["id"])
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(target)


def get_session(session_id):
    path = _path(session_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions():
    """Summaries only (id, title, timestamps, message count) -- not the
    full message history, so the sidebar list loads fast even with many
    sessions."""
    if not SESSIONS_DIR.exists():
        return []

    summaries = []
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summaries.append({
            "id": session["id"],
            "title": session["title"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "message_count": len(session["messages"]),
        })

    return sorted(summaries, key=lambda s: s["updated_at"], reverse=True)


def add_message(session_id, role, content, extra=None):
    """
    Append one message to a session. role is "user" or "assistant".
    extra carries assistant-only fields (sources, provider, trace_id,
    skipped_llm) -- kept separate from user messages, which only ever have
    a question.

    The FIRST user message in a session also becomes that session's title
    (truncated), so "New chat" only shows until something's actually asked.
    """
    session = get_session(session_id)
    if session is None:
        return None

    message = {"role": role, "content": content, "created_at": _now()}
    if extra:
        message.update(extra)
    session["messages"].append(message)

    if role == "user" and session["title"] == "New chat":
        session["title"] = content[:60] + ("..." if len(content) > 60 else "")

    session["updated_at"] = _now()
    _save(session)
    return session


def delete_session(session_id):
    path = _path(session_id)
    if not path.exists():
        return False
    path.unlink()
    return True
