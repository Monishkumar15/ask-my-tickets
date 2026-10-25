"""Local replayable traces for the Week 5 error-analysis workflow."""
from __future__ import annotations
import json, time, uuid
from datetime import datetime, timezone
from pathlib import Path

from config import TRACE_DIR as _TRACE_DIR

TRACE_DIR = Path(_TRACE_DIR)

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
        # Week 11: summed from whatever stages actually attached a cost_usd
        # (generate.py's "generation" stage, agent.py's "agent_step" stages,
        # multi_agent.py's "route"/"specialist"/"merge" stages) -- not
        # passed in by the caller, so no call site has to remember to total
        # it up itself, and a stage that never captured cost (e.g. a
        # refusal with no LLM call) just contributes 0.
        total_cost_usd = sum(stage["data"].get("cost_usd", 0) or 0 for stage in self.data["stages"])
        self.data.update({"completed_at": _now(), "duration_ms": round((time.perf_counter()-self.started)*1000), "answer": answer, "outcome": outcome, "error": error, "total_cost_usd": total_cost_usd})
        save(self.data); return self.id

def save(record):
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    target = _path(record["trace_id"]); temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"); temporary.replace(target)
def get(trace_id):
    path = _path(trace_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
def list_recent(limit=50, offset=0):
    """
    Paginated: sorts by file mtime (a fast proxy for created_at -- save()
    writes atomically via a temp-file rename, so mtime tracks creation
    order closely) instead of parsing every trace file's JSON just to sort
    them. Only the requested page's files are actually opened and parsed.
    Found live at 764+ accumulated trace files: the old version read and
    JSON-decoded all of them on every single /traces call regardless of
    limit, a cost that only grows as more traces pile up.
    Returns (records, total_count).
    """
    if not TRACE_DIR.exists(): return [], 0
    paths = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = len(paths)
    records = []
    for path in paths[offset:offset + limit]:
        try: records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError): pass
    return records, total
def _contains_source(obj, source):
    """Recursively checks every "source" key anywhere in a trace record's
    nested stage data for a case-insensitive substring match. Needed
    because this app's 3 pipelines (fixed/agent/team) don't store a single
    clean top-level "sources" list in the trace file itself -- source
    filenames only ever appear inside per-stage chunk data (e.g. the
    "retrieval" stage's chunks, a "specialist" stage's result chunks),
    in a different shape per pipeline. Walking the whole structure once
    is simpler and more honest than special-casing each pipeline's stage
    shape separately."""
    source = source.lower()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "source" and isinstance(value, str) and source in value.lower():
                return True
            if _contains_source(value, source):
                return True
        return False
    if isinstance(obj, list):
        return any(_contains_source(item, source) for item in obj)
    return False


def search(query=None, source=None, outcome=None, since=None, limit=50, offset=0):
    """
    Finds a trace from partial information -- the actual "find the refund
    answer nobody logged a trace id for" capability (Week 11): today's
    `/traces` only supports limit/offset pagination, so finding one
    specific trace among 1,668+ accumulated files means paging through
    up to 34 pages of 50 by hand.

    query: case-insensitive substring matched against the ENTIRE trace
    record (serialized once per file) -- covers the question, the final
    answer, every stage's thoughts/observations, and any retrieved chunk
    text, in one pass. A real production system would use a proper search
    index (OpenSearch/Elasticsearch, or a DB with a text index) instead of
    a linear per-file scan -- named here as a known limitation, not
    silently pretended away; honest and simple is the right trade-off at
    this project's actual scale (a few KB per file, ~1,700 files today).

    source/outcome/since are optional additional filters, all ANDed with
    query and with each other when given.

    Returns (records, total_matched) -- same shape as list_recent(), so
    api.py's /traces/search endpoint can mirror /traces exactly.
    """
    if not TRACE_DIR.exists():
        return [], 0
    paths = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    query_lower = query.lower() if query else None
    matches = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if query_lower is not None and query_lower not in json.dumps(record, ensure_ascii=False).lower():
            continue
        if source is not None and not _contains_source(record, source):
            continue
        if outcome is not None and record.get("outcome") != outcome:
            continue
        if since is not None and (record.get("created_at") or "") < since:
            continue
        matches.append(record)

    total = len(matches)
    return matches[offset:offset + limit], total


def annotate(trace_id, values):
    record=get(trace_id)
    if record is None: return None
    allowed={"reviewed","failure_note","problem_type","severity"}
    record.setdefault("annotation",{}).update({key:value for key,value in values.items() if key in allowed})
    note=record["annotation"].get("failure_note") or ""
    if record["annotation"].get("problem_type") and not note.strip(): raise ValueError("Write an open-coded failure note before assigning a problem type.")
    save(record); return record
