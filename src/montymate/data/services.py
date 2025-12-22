from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import sqlite3

from .paths import RepoPaths
from .types import (
    ArtifactWriteResult,
    RunContext,
    # StepContext,
    StoragePolicy,
    should_inline,
    summarize_text,
)
from . import repo as repo_ops
from validators import validate_event_envelope

def _now_ms() -> int:
    return int(time.time() * 1000)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class ArtifactManager:
    """
    Writes artifact files and inserts a row into mm_artifacts.
    Enforces immutability by never updating existing artifact rows.
    """
    conn: sqlite3.Connection
    paths: RepoPaths

    def write_bytes(
        self,
        *,
        run_id: str,
        step_execution_id: Optional[str],
        artifact_type: str,
        fmt: str,
        relpath_under_artifacts: str,
        payload: bytes,
        meta: Optional[dict[str, Any]] = None,
        parent_artifact_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> ArtifactWriteResult:
        artifact_id = artifact_id or _uuid()

        out_path = self.paths.artifacts_root / relpath_under_artifacts
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(payload)

        sha = _sha256(payload)
        size = len(payload)

        # IMPORTANT: relpath is stored relative to .ai_montymate/ (not repo root)
        relpath = str(Path("artifacts") / relpath_under_artifacts)

        repo_ops.insert_artifact_row(
            self.conn,
            run_id=run_id,
            step_execution_id=step_execution_id,
            artifact_type=artifact_type,
            format=fmt,
            relpath=relpath,
            sha256=sha,
            bytes=size,
            meta=meta,
            parent_artifact_id=parent_artifact_id,
            artifact_id=artifact_id,
        )
        return ArtifactWriteResult(artifact_id=artifact_id, relpath=relpath, sha256=sha, bytes=size)

    def write_text(
        self,
        *,
        run_id: str,
        step_execution_id: Optional[str],
        artifact_type: str,
        fmt: str,
        relpath_under_artifacts: str,
        text: str,
        meta: Optional[dict[str, Any]] = None,
        parent_artifact_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> ArtifactWriteResult:
        return self.write_bytes(
            run_id=run_id,
            step_execution_id=step_execution_id,
            artifact_type=artifact_type,
            fmt=fmt,
            relpath_under_artifacts=relpath_under_artifacts,
            payload=text.encode(encoding),
            meta=meta,
            parent_artifact_id=parent_artifact_id,
            artifact_id=artifact_id,
        )


@dataclass
class EventLogger:
    """
    Append-only event logger. If payload is too large (per storage policy),
    it writes the payload JSON as an artifact and stores an artifact reference in the event.
    """
    conn: sqlite3.Connection
    artifacts: ArtifactManager
    storage_policy: StoragePolicy

    def append(
        self,
        *,
        run_id: str,
        step_execution_id: Optional[str],
        event_type: str,
        actor_type: str,
        severity: str,
        payload: Mapping[str, Any],
        payload_artifact_type: str = "event_payload",
    ) -> int:
        validate_event_envelope(
            event_type=event_type,
            actor_type=actor_type,
            severity=severity,
            payload=payload,
            strict_event_types=True,
        )
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        payload_bytes = payload_json.encode("utf-8")

        if should_inline(self.storage_policy, len(payload_bytes)):
            return repo_ops.append_event(
                self.conn,
                run_id=run_id,
                step_execution_id=step_execution_id,
                event_type=event_type,
                actor_type=actor_type,
                severity=severity,
                payload=dict(payload),
                payload_artifact_id=None,
                payload_sha256=_sha256(payload_bytes),
            )

        # Store payload as an artifact
        event_payload_id = _uuid()
        relpath = f"events/{run_id}/{event_type}/{event_payload_id}.json"
        aw = self.artifacts.write_text(
            run_id=run_id,
            step_execution_id=step_execution_id,
            artifact_type=payload_artifact_type,
            fmt="JSON",
            relpath_under_artifacts=relpath,
            text=payload_json,
            artifact_id=event_payload_id,
        )

        # Keep payload_json small in mm_events; store pointer + small summary
        stub = {
            "artifact_ref": {
                "artifact_id": aw.artifact_id,
                "relpath": aw.relpath,
                "sha256": aw.sha256,
                "bytes": aw.bytes,
            },
            "summary": summarize_text(payload_json, max_chars=2000),
        }

        return repo_ops.append_event(
            self.conn,
            run_id=run_id,
            step_execution_id=step_execution_id,
            event_type=event_type,
            actor_type=actor_type,
            severity=severity,
            payload=stub,
            payload_artifact_id=aw.artifact_id,
            payload_sha256=aw.sha256,
        )


@dataclass
class CostLedgerWriter:
    """
    Immutable ledger entries. This does NOT attempt to estimate hidden costs.
    It records observed costs (your current preference).
    """
    conn: sqlite3.Connection

    def record(
        self,
        *,
        run_id: str,
        step_execution_id: Optional[str],
        source_type: str,   # 'llm' | 'tool'
        source_id: str,
        amount_usd: float,
        currency: str = "USD",
        unit_type: Optional[str] = None,
        units: Optional[float] = None,
        pricing_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> str:
        ledger_id = _uuid()
        self.conn.execute(
            """
            INSERT INTO mm_cost_ledger (
              ledger_id, run_id, step_execution_id,
              source_type, source_id,
              amount_usd, currency,
              unit_type, units, pricing_id,
              computed_at, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ledger_id, run_id, step_execution_id,
                source_type, source_id,
                float(amount_usd), currency,
                unit_type, units, pricing_id,
                _now_ms(), note
            ),
        )
        return ledger_id


@dataclass
class LLMCallRecorder:
    """
    Records LLM calls + writes large prompt/response to artifacts per storage policy.
    Also emits an event and writes a cost ledger entry when cost_usd is provided.
    """
    conn: sqlite3.Connection
    artifacts: ArtifactManager
    events: EventLogger
    ledger: CostLedgerWriter
    storage_policy: StoragePolicy

    def record_call(
        self,
        *,
        run_id: str,
        step_execution_id: Optional[str],
        role: str,
        provider: str,
        model: str,
        status: str,  # OK|ERROR|TIMEOUT
        prompt_text: str,
        response_text: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        total_tokens: Optional[int],
        cost_usd: Optional[float],
        template_id: Optional[str] = None,
        pricing_id: Optional[str] = None,
        request_json: Optional[dict[str, Any]] = None,
        response_json: Optional[dict[str, Any]] = None,
    ) -> str:
        llm_call_id = _uuid()

        # Decide inline vs artifact for prompt
        p_bytes = prompt_text.encode("utf-8")
        prompt_inline: Optional[str] = None
        prompt_artifact_id: Optional[str] = None
        if should_inline(self.storage_policy, len(p_bytes)):
            prompt_inline = prompt_text
        else:
            awp = self.artifacts.write_text(
                run_id=run_id,
                step_execution_id=step_execution_id,
                artifact_type="llm_prompt",
                fmt="TEXT",
                relpath_under_artifacts=f"llm/{run_id}/{llm_call_id}/prompt.txt",
                text=prompt_text,
                artifact_id=_uuid(),
                meta={"role": role, "provider": provider, "model": model},
            )
            prompt_artifact_id = awp.artifact_id

        # Decide inline vs artifact for response
        r_bytes = response_text.encode("utf-8")
        response_inline: Optional[str] = None
        response_artifact_id: Optional[str] = None
        if should_inline(self.storage_policy, len(r_bytes)):
            response_inline = response_text
        else:
            awr = self.artifacts.write_text(
                run_id=run_id,
                step_execution_id=step_execution_id,
                artifact_type="llm_response",
                fmt="TEXT",
                relpath_under_artifacts=f"llm/{run_id}/{llm_call_id}/response.txt",
                text=response_text,
                artifact_id=_uuid(),
                meta={"role": role, "provider": provider, "model": model},
            )
            response_artifact_id = awr.artifact_id

        # Persist call
        self.conn.execute(
            """
            INSERT INTO mm_llm_calls (
              llm_call_id, run_id, step_execution_id,
              role, provider, model,
              request_ts, response_ts, latency_ms, status, template_id,
              prompt_text, response_text, prompt_artifact_id, response_artifact_id,
              request_json, response_json,
              input_tokens, output_tokens, total_tokens,
              pricing_id, cost_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                llm_call_id, run_id, step_execution_id,
                role, provider, model,
                _now_ms(), _now_ms(), 0, status, template_id,
                prompt_inline, response_inline, prompt_artifact_id, response_artifact_id,
                json.dumps(request_json or {}), json.dumps(response_json or {}),
                input_tokens, output_tokens, total_tokens,
                pricing_id, cost_usd
            )
        )

        # Emit event (coarse)
        self.events.append(
            run_id=run_id,
            step_execution_id=step_execution_id,
            event_type="llm.call",
            actor_type="llm",
            severity="info" if status == "OK" else "error",
            payload={
                "llm_call_id": llm_call_id,
                "role": role,
                "provider": provider,
                "model": model,
                "status": status,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens,
                },
                "cost_usd": cost_usd,
                "prompt": {"inline": prompt_inline is not None, "artifact_id": prompt_artifact_id},
                "response": {"inline": response_inline is not None, "artifact_id": response_artifact_id},
            },
        )

        # Ledger (if cost known)
        if cost_usd is not None:
            self.ledger.record(
                run_id=run_id,
                step_execution_id=step_execution_id,
                source_type="llm",
                source_id=llm_call_id,
                amount_usd=float(cost_usd),
                currency="USD",
                pricing_id=pricing_id,
                note=f"{provider}:{model} role={role}",
            )

        return llm_call_id


@dataclass
class ToolCallRecorder:
    """
    Records tool calls + stores large input/output payloads as artifacts per storage policy.
    Emits a coarse event and writes ledger entries if cost_usd is provided.
    """
    conn: sqlite3.Connection
    artifacts: ArtifactManager
    events: EventLogger
    ledger: CostLedgerWriter
    storage_policy: StoragePolicy

    def record_call(
        self,
        *,
        run_id: str,
        step_execution_id: Optional[str],
        tool_name: str,
        tool_type: str,
        runs_in_runtime: bool,
        status: str,  # OK|ERROR|TIMEOUT
        input_json: Optional[dict[str, Any]],
        output_json: Optional[dict[str, Any]],
        unit_type: Optional[str] = None,
        units: Optional[float] = None,
        cost_usd: Optional[float] = None,
        pricing_id: Optional[str] = None,
    ) -> str:
        tool_call_id = _uuid()

        # Hybrid store input/output as inline json or artifact json
        def store_json(kind: str, obj: Optional[dict[str, Any]]) -> tuple[str, Optional[str], Optional[str]]:
            """
            returns: (inline_json_text, artifact_id, sha256)
            inline_json_text may be '{}' when stored as artifact (to keep DB small)
            """
            if obj is None:
                return "{}", None, None
            txt = json.dumps(obj, ensure_ascii=False, sort_keys=True)
            b = txt.encode("utf-8")
            if should_inline(self.storage_policy, len(b)):
                return txt, None, _sha256(b)

            aw = self.artifacts.write_text(
                run_id=run_id,
                step_execution_id=step_execution_id,
                artifact_type=f"tool_{kind}",
                fmt="JSON",
                relpath_under_artifacts=f"tools/{run_id}/{tool_call_id}/{kind}.json",
                text=txt,
                artifact_id=_uuid(),
                meta={"tool_name": tool_name, "tool_type": tool_type},
            )
            return "{}", aw.artifact_id, aw.sha256

        in_txt, in_art_id, _ = store_json("input", input_json)
        out_txt, out_art_id, _ = store_json("output", output_json)

        self.conn.execute(
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
                tool_call_id, run_id, step_execution_id,
                tool_name, tool_type, 1 if runs_in_runtime else 0,
                _now_ms(), _now_ms(), 0, status,
                in_txt, out_txt, in_art_id, out_art_id,
                unit_type, units, pricing_id, cost_usd
            ),
        )

        self.events.append(
            run_id=run_id,
            step_execution_id=step_execution_id,
            event_type="tool.call",
            actor_type="tool",
            severity="info" if status == "OK" else "error",
            payload={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_type": tool_type,
                "runs_in_runtime": runs_in_runtime,
                "status": status,
                "io": {
                    "input": {"artifact_id": in_art_id, "inline": in_art_id is None},
                    "output": {"artifact_id": out_art_id, "inline": out_art_id is None},
                },
                "billing": {"unit_type": unit_type, "units": units, "cost_usd": cost_usd},
            },
        )

        if cost_usd is not None:
            self.ledger.record(
                run_id=run_id,
                step_execution_id=step_execution_id,
                source_type="tool",
                source_id=tool_call_id,
                amount_usd=float(cost_usd),
                currency="USD",
                unit_type=unit_type,
                units=units,
                pricing_id=pricing_id,
                note=f"{tool_type}:{tool_name}",
            )

        return tool_call_id


@dataclass
class DataServices:
    """
    Convenience bundle used by the runner.
    """
    conn: sqlite3.Connection
    paths: RepoPaths
    ctx: RunContext
    artifacts: ArtifactManager
    events: EventLogger
    ledger: CostLedgerWriter
    llm: LLMCallRecorder
    tools: ToolCallRecorder

    @staticmethod
    def build(conn: sqlite3.Connection, paths: RepoPaths, ctx: RunContext) -> "DataServices":
        artifacts = ArtifactManager(conn=conn, paths=paths)
        ledger = CostLedgerWriter(conn=conn)
        events = EventLogger(conn=conn, artifacts=artifacts, storage_policy=ctx.storage_policy)
        llm = LLMCallRecorder(conn=conn, artifacts=artifacts, events=events, ledger=ledger, storage_policy=ctx.storage_policy)
        tools = ToolCallRecorder(conn=conn, artifacts=artifacts, events=events, ledger=ledger, storage_policy=ctx.storage_policy)
        return DataServices(conn=conn, paths=paths, ctx=ctx, artifacts=artifacts, events=events, ledger=ledger, llm=llm, tools=tools)