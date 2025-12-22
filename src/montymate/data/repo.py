from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from .artifacts import ArtifactRef


def _now_ms() -> int:
    return int(time.time() * 1000)


def new_id() -> str:
    return str(uuid.uuid4())


def create_run(
    conn: sqlite3.Connection,
    *,
    repo_root: str,
    profile_id: str,
    workflow_id: str,
    policy_id: str,
    git_base_sha: str | None,
    git_branch: str | None,
    workflow_sha: str | None = None,
    policy_sha: str | None = None,
    profile_sha: str | None = None,
) -> str:
    run_id = new_id()
    ts = _now_ms()
    conn.execute(
        """
        INSERT INTO mm_runs (
          run_id, created_at, updated_at, status, profile_id, workflow_id, policy_id,
          workflow_sha, policy_sha, profile_sha,
          repo_root, git_base_sha, git_branch
        ) VALUES (?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            ts,
            ts,
            profile_id,
            workflow_id,
            policy_id,
            workflow_sha,
            policy_sha,
            profile_sha,
            repo_root,
            git_base_sha,
            git_branch,
        ),
    )
    return run_id


def start_step(
    conn: sqlite3.Connection, *, run_id: str, step_id: str, attempt_no: int
) -> str:
    step_execution_id = new_id()
    conn.execute(
        """
        INSERT INTO mm_step_executions (
          step_execution_id, run_id, step_id, attempt_no, status, started_at
        ) VALUES (?, ?, ?, ?, 'RUNNING', ?)
        """,
        (step_execution_id, run_id, step_id, attempt_no, _now_ms()),
    )
    return step_execution_id


def finish_step(
    conn: sqlite3.Connection,
    *,
    step_execution_id: str,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE mm_step_executions
        SET status=?, finished_at=?, error_type=?, error_message=?
        WHERE step_execution_id=?
        """,
        (status, _now_ms(), error_type, error_message, step_execution_id),
    )


def insert_artifact_row(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    step_execution_id: str | None,
    artifact_type: str,
    format: str,
    relpath: str,
    sha256: str,
    bytes: int,
    meta: dict[str, Any] | None = None,
    parent_artifact_id: str | None = None,
    artifact_id: str | None = None,
) -> str:
    artifact_id = artifact_id or new_id()
    conn.execute(
        """
        INSERT INTO mm_artifacts (
          artifact_id, run_id, step_execution_id, artifact_type, format, relpath,
          sha256, bytes, created_at, meta_json, parent_artifact_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            artifact_id,
            run_id,
            step_execution_id,
            artifact_type,
            format,
            relpath,
            sha256,
            bytes,
            _now_ms(),
            json.dumps(meta or {}),
            parent_artifact_id,
        ),
    )
    return artifact_id


def append_event(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    step_execution_id: str | None,
    event_type: str,
    actor_type: str,
    severity: str,
    payload: dict[str, Any],
    payload_artifact_id: str | None = None,
    payload_sha256: str | None = None,
) -> int:
    payload_json = json.dumps(payload, ensure_ascii=False)
    cur = conn.execute(
        """
        INSERT INTO mm_events (
          run_id, step_execution_id, ts, event_type, actor_type, severity,
          payload_json, payload_sha256, payload_artifact_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            step_execution_id,
            _now_ms(),
            event_type,
            actor_type,
            severity,
            payload_json,
            payload_sha256,
            payload_artifact_id,
        ),
    )
    return int(cur.lastrowid)


def record_llm_call(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    step_execution_id: str | None,
    role: str,
    provider: str,
    model: str,
    status: str,
    prompt_text: str | None,
    response_text: str | None,
    prompt_artifact_id: str | None,
    response_artifact_id: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    cost_usd: float | None,
    pricing_id: str | None = None,
    template_id: str | None = None,
    request_json: dict[str, Any] | None = None,
    response_json: dict[str, Any] | None = None,
) -> str:
    llm_call_id = new_id()
    now = _now_ms()
    conn.execute(
        """
        INSERT INTO mm_llm_calls (
          llm_call_id, run_id, step_execution_id, role, provider, model,
          request_ts, response_ts, latency_ms, status, template_id,
          prompt_text, response_text, prompt_artifact_id, response_artifact_id,
          request_json, response_json, input_tokens, output_tokens, total_tokens,
          pricing_id, cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            llm_call_id,
            run_id,
            step_execution_id,
            role,
            provider,
            model,
            now,
            now,
            0,
            status,
            template_id,
            prompt_text,
            response_text,
            prompt_artifact_id,
            response_artifact_id,
            json.dumps(request_json or {}),
            json.dumps(response_json or {}),
            input_tokens,
            output_tokens,
            total_tokens,
            pricing_id,
            cost_usd,
        ),
    )
    return llm_call_id


def record_tool_call(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    step_execution_id: str | None,
    tool_name: str,
    tool_type: str,
    runs_in_runtime: bool,
    status: str,
    input_json: dict[str, Any] | None,
    output_json: dict[str, Any] | None,
    input_artifact_id: str | None,
    output_artifact_id: str | None,
    unit_type: str | None,
    units: float | None,
    cost_usd: float | None,
    pricing_id: str | None = None,
) -> str:
    tool_call_id = new_id()
    now = _now_ms()
    conn.execute(
        """
        INSERT INTO mm_tool_calls (
          tool_call_id, run_id, step_execution_id,
          tool_name, tool_type, runs_in_runtime,
          request_ts, response_ts, latency_ms, status,
          input_json, output_json, input_artifact_id, output_artifact_id,
          unit_type, units, pricing_id, cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            tool_call_id,
            run_id,
            step_execution_id,
            tool_name,
            tool_type,
            1 if runs_in_runtime else 0,
            now,
            now,
            0,
            status,
            json.dumps(input_json or {}),
            json.dumps(output_json or {}),
            input_artifact_id,
            output_artifact_id,
            unit_type,
            units,
            pricing_id,
            cost_usd,
        ),
    )
    return tool_call_id
