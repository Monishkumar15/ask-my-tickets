"""Local replayable traces for the Week 5 error-analysis workflow."""
from __future__ import annotations
import json, os, time, uuid
from datetime import datetime, timezone
from pathlib import Path

TRACE_DIR = Path(os.getenv("TRACE_DIR", "traces"))

def _now(): return datetime.now(timezone.utc).isoformat()
def _path(trace_id):
    if not trace_id or any(c not in "0123456789abcdef" for c in trace_id): raise ValueError("Invalid trace id")
    return TRACE_DIR / f"{trace_id}.json"

class Trace:
    def __init__(self, question, config):
        self.started = time.perf_counter()
        self.data = {"trace_id": uuid.uuid4().hex, "created_at": _now(), "question": question, "config": config, "stages": [], "annotation": {"reviewed": False, "failure_note": None, "problem_type": None, "severity": None}}
    @property
    def id(self): return self.data["trace_id"]
    def stage(self, name, **data): self.data["stages"].append({"name": name, "elapsed_ms": round((time.perf_counter()-self.started)*1000), "data": data})
    def finish(self, answer=None, outcome="answer", error=None):
        self.data.update({"completed_at": _now(), "duration_ms": round((time.perf_counter()-self.started)*1000), "answer": answer, "outcome": outcome, "error": error})
        save(self.data); return self.id

def save(record):
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    target = _path(record["trace_id"]); temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"); temporary.replace(target)
def get(trace_id):
    path = _path(trace_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
def list_recent(limit=50):
    if not TRACE_DIR.exists(): return []
    records=[]
    for path in TRACE_DIR.glob("*.json"):
        try: records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError): pass
    return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)[:limit]
def annotate(trace_id, values):
    record=get(trace_id)
    if record is None: return None
    allowed={"reviewed","failure_note","problem_type","severity"}
    record.setdefault("annotation",{}).update({key:value for key,value in values.items() if key in allowed})
    note=record["annotation"].get("failure_note") or ""
    if record["annotation"].get("problem_type") and not note.strip(): raise ValueError("Write an open-coded failure note before assigning a problem type.")
    save(record); return record
