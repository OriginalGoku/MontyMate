from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

JSON = Dict[str, Any]


def connect_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row

    # Safer defaults; WAL improves concurrent reads/writes for SQLite.  [oai_citation:4‡SQLite](https://sqlite.org/wal.html?utm_source=chatgpt.com)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS mm_runs (
          run_id TEXT PRIMARY KEY,
          repo_root TEXT NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS mm_events (
          event_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT NOT NULL,
          ts TEXT NOT NULL DEFAULT (datetime('now')),
          event_type TEXT NOT NULL,
          actor_type TEXT NOT NULL,
          severity TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          FOREIGN KEY (run_id) REFERENCES mm_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS mm_llm_calls (
          call_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT NOT NULL,
          step TEXT NOT NULL,
          role TEXT NOT NULL,
          provider TEXT NOT NULL,
          model TEXT NOT NULL,
          ts TEXT NOT NULL DEFAULT (datetime('now')),
          prompt_text TEXT NOT NULL,
          response_text TEXT NOT NULL,
          input_tokens INTEGER,
          output_tokens INTEGER,
          total_tokens INTEGER,
          cost_usd REAL,
          raw_json TEXT,
          FOREIGN KEY (run_id) REFERENCES mm_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS mm_artifacts (
          artifact_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          artifact_type TEXT NOT NULL,
          relpath TEXT NOT NULL,
          sha256 TEXT NOT NULL,
          bytes INTEGER NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          meta_json TEXT NOT NULL,
          FOREIGN KEY (run_id) REFERENCES mm_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS mm_gates (
          gate_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          gate_name TEXT NOT NULL,
          mode TEXT NOT NULL,
          status TEXT NOT NULL,
          requested_at TEXT NOT NULL DEFAULT (datetime('now')),
          decided_at TEXT,
          decision TEXT,
          human_actor TEXT,
          reason TEXT,
          payload_json TEXT NOT NULL,
          FOREIGN KEY (run_id) REFERENCES mm_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_run_ts ON mm_events(run_id, ts);
        CREATE INDEX IF NOT EXISTS idx_calls_run_ts ON mm_llm_calls(run_id, ts);
        CREATE INDEX IF NOT EXISTS idx_artifacts_run ON mm_artifacts(run_id);
        CREATE INDEX IF NOT EXISTS idx_gates_run ON mm_gates(run_id);
        """
    )


def new_run_id() -> str:
    return str(uuid.uuid4())


def now_update_run(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute(
        "UPDATE mm_runs SET updated_at=datetime('now') WHERE run_id=?", (run_id,)
    )


def create_run(
    conn: sqlite3.Connection, *, run_id: str, repo_root: str, status: str
) -> None:
    conn.execute(
        "INSERT INTO mm_runs(run_id, repo_root, status) VALUES(?,?,?)",
        (run_id, repo_root, status),
    )


def set_run_status(conn: sqlite3.Connection, run_id: str, status: str) -> None:
    conn.execute(
        "UPDATE mm_runs SET status=?, updated_at=datetime('now') WHERE run_id=?",
        (status, run_id),
    )


def append_event(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    event_type: str,
    actor_type: str,
    severity: str,
    payload: JSON,
) -> None:
    conn.execute(
        "INSERT INTO mm_events(run_id, event_type, actor_type, severity, payload_json) VALUES(?,?,?,?,?)",
        (
            run_id,
            event_type,
            actor_type,
            severity,
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    now_update_run(conn, run_id)


def record_llm_call(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    step: str,
    role: str,
    provider: str,
    model: str,
    prompt_text: str,
    response_text: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    cost_usd: Optional[float],
    raw: Optional[JSON],
) -> None:
    conn.execute(
        """
        INSERT INTO mm_llm_calls(
          run_id, step, role, provider, model,
          prompt_text, response_text,
          input_tokens, output_tokens, total_tokens, cost_usd, raw_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id,
            step,
            role,
            provider,
            model,
            prompt_text,
            response_text,
            input_tokens,
            output_tokens,
            total_tokens,
            cost_usd,
            json.dumps(raw or {}, ensure_ascii=False),
        ),
    )
    now_update_run(conn, run_id)

def record_artifact(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    artifact_type: str,
    relpath: str,
    content_bytes: bytes,
    meta: Optional[JSON] = None,
) -> str:
    artifact_id = str(uuid.uuid4())
    sha256 = hashlib.sha256(content_bytes).hexdigest()
    nbytes = len(content_bytes)

    conn.execute(
        """
        INSERT INTO mm_artifacts(artifact_id, run_id, artifact_type, relpath, sha256, bytes, meta_json)
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            artifact_id,
            run_id,
            artifact_type,
            relpath,
            sha256,
            nbytes,
            json.dumps(meta or {}, ensure_ascii=False),
        ),
    )
    append_event(
        conn,
        run_id=run_id,
        event_type="artifact.written",
        actor_type="system",
        severity="info",
        payload={
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "relpath": relpath,
            "sha256": sha256,
            "bytes": nbytes,
        },
    )
    return artifact_id
    
def create_gate(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    gate_name: str,
    mode: str,
    payload: JSON,
) -> str:
    gate_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO mm_gates(gate_id, run_id, gate_name, mode, status, payload_json)
        VALUES(?,?,?,?,?,?)
        """,
        (
            gate_id,
            run_id,
            gate_name,
            mode,
            "PENDING",
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    append_event(
        conn,
        run_id=run_id,
        event_type="gate.requested",
        actor_type="system",
        severity="info",
        payload={
            "gate_id": gate_id,
            "gate_name": gate_name,
            "mode": mode,
            "payload": payload,
        },
    )
    return gate_id


def decide_gate(
    conn: sqlite3.Connection,
    *,
    gate_id: str,
    decision: str,
    human_actor: str,
    reason: Optional[str],
) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM mm_gates WHERE gate_id=?", (gate_id,)).fetchone()
    if not row:
        raise ValueError("Unknown gate_id")

    if row["status"] != "PENDING":
        return row

    conn.execute(
        """
        UPDATE mm_gates
        SET status='DECIDED', decided_at=datetime('now'),
            decision=?, human_actor=?, reason=?
        WHERE gate_id=?
        """,
        (decision, human_actor, reason, gate_id),
    )
    append_event(
        conn,
        run_id=row["run_id"],
        event_type=f"gate.{decision.lower()}",
        actor_type="human",
        severity="info" if decision in ("ACK", "APPROVE") else "warn",
        payload={
            "gate_id": gate_id,
            "decision": decision,
            "human_actor": human_actor,
            "reason": reason,
        },
    )
    return conn.execute("SELECT * FROM mm_gates WHERE gate_id=?", (gate_id,)).fetchone()
