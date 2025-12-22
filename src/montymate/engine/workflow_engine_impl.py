from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import sqlite3

from montymate.config.models import ResolvedRunConfig
from montymate.config.resolver import RunConfigResolver
from montymate.core.types import JSON, RunIdentity, RunStatus
from montymate.data.db import transaction
from montymate.data.paths import RepoPaths
from montymate.data.types import StoragePolicy
from montymate.data.services import DataServices
from montymate.data import repo as repo_ops
from montymate.engine.guards import GuardEvaluator
from montymate.engine.state import ExecutionContext, GuardContext, RunHandle
from montymate.events import EventType
from montymate.gates.hitl import HumanGateManager
from montymate.gates.models import GateDecision, GateRequest
from montymate.llm.gateway import LLMGateway
from montymate.llm.models import LLMRequest
from montymate.tools.executor import ToolExecutor
from montymate.tools.models import ToolInvocation


# ---------------------------
# MVP assumptions about workflow YAML
# ---------------------------
# WorkflowGraph:
#   workflow.entry: "<step_id>"
#   steps: { "<step_id>": {"kind": "agent_task"|"tool_task"|"human_gate", ...}, ...}
#   edges: [
#     {"from": "<step_id>", "to": "<step_id>" | null, "guard": {structured_guard}? },
#     ...
#   ]
#
# Step kinds:
# - agent_task:
#     role: "interviewer"|"spec_validator"|...
#     prompt: "string with {placeholders}"
#     output:
#       artifact_type: "module_spec" | "decision_record" | ...
#       format: "YAML"|"JSON"|"MD"|"TEXT"
#       relpath: "spec.yaml" | "decision_record.json" | ...
#       logical_name: "module_spec" | "decision_record" | "analysis_report" | ...
# - tool_task:
#     tool_alias: "run_gates" | "capture_repo_snapshot" | ...
#     input: { ... }  # can contain {placeholders}
# - human_gate:
#     gate_name: "spec_lock" | "architecture_lock" | ...
#     summary: "what the user is approving"
#     required_artifacts: ["module_spec", "architecture_plan"]  # logical names (looked up in artifact index)
#
# Guards:
# - steps may have run_if: <structured guard object>
# - edges may have guard: <structured guard object>
#
# The engine stores durable state as an event "engine.checkpoint" with:
# { next_step_id, artifacts_index, module_spec_inline, decision_record_inline, paused_gate? }


@dataclass(frozen=True)
class CheckpointState:
    next_step_id: Optional[str]
    artifacts_index: Dict[str, str]            # logical_name -> artifact_id
    module_spec_inline: Optional[JSON] = None
    decision_record_inline: Optional[JSON] = None
    paused_gate: Optional[Dict[str, Any]] = None


class DataServicesFactory(Protocol):
    def __call__(
        self,
        *,
        conn: sqlite3.Connection,
        paths: RepoPaths,
        run_id: str,
        repo_root: str,
        profile_id: str,
        workflow_id: str,
        policy_id: str,
        decision_record: Optional[JSON],
    ) -> DataServices:
        ...


def _now_ms() -> int:
    return int(time.time() * 1000)


def _uuid() -> str:
    return str(uuid.uuid4())


def _safe_format(template: str, vars: Dict[str, Any]) -> str:
    """
    Minimal string interpolation for MVP.
    Uses str.format mapping with best-effort fallback.
    """
    try:
        return template.format(**vars)
    except Exception:
        # Don't fail a run because of formatting; keep traceability and fail later if needed.
        return template


def _db_set_run_status(conn: sqlite3.Connection, run_id: str, status: RunStatus) -> None:
    conn.execute(
        "UPDATE mm_runs SET status=?, updated_at=? WHERE run_id=?",
        (status, _now_ms(), run_id),
    )


def _db_get_run_row(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM mm_runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown run_id={run_id}")
    return row


def _db_get_latest_checkpoint(conn: sqlite3.Connection, run_id: str) -> Optional[CheckpointState]:
    row = conn.execute(
        """
        SELECT payload_json
        FROM mm_events
        WHERE run_id=? AND event_type=?
        ORDER BY ts DESC, event_id DESC
        LIMIT 1
        """,
        (run_id, "engine.checkpoint"),
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(row["payload_json"])
    return CheckpointState(
        next_step_id=payload.get("next_step_id"),
        artifacts_index=dict(payload.get("artifacts_index") or {}),
        module_spec_inline=payload.get("module_spec_inline"),
        decision_record_inline=payload.get("decision_record_inline"),
        paused_gate=payload.get("paused_gate"),
    )


def _artifact_id_from_logical(artifacts_index: Dict[str, str], logical_or_id: str) -> str:
    # If user provides a raw artifact id, pass through; else resolve logical name.
    return artifacts_index.get(logical_or_id, logical_or_id)


def _guard_true(guard_eval: GuardEvaluator, guard_obj: Optional[JSON], gctx: GuardContext) -> bool:
    if guard_obj is None:
        return True
    return bool(guard_eval.eval(guard_obj, gctx))


def _choose_next_step(
    *,
    edges: list[JSON],
    from_step: str,
    guard_eval: GuardEvaluator,
    gctx: GuardContext,
) -> Optional[str]:
    """
    Deterministic: pick first matching edge in the workflow YAML order.
    """
    for e in edges:
        if e.get("from") != from_step:
            continue
        guard_obj = e.get("guard")
        if _guard_true(guard_eval, guard_obj, gctx):
            return e.get("to")  # may be None => terminal
    return None


def _gate_mode_for(decision_record: Optional[JSON], gate_name: str, fallback_mode: str = "APPROVE") -> Tuple[str, Optional[int]]:
    """
    decision_record.gate_modes:
      gate_modes:
        spec_lock: { mode: "ACK", ack_window_minutes: 60 }
    """
    if not decision_record:
        return fallback_mode, None
    gm = (decision_record.get("gate_modes") or {}).get(gate_name)
    if not gm:
        return fallback_mode, None
    return str(gm.get("mode", fallback_mode)), gm.get("ack_window_minutes")


def _write_checkpoint(
    *,
    svc: DataServices,
    run_id: str,
    step_execution_id: Optional[str],
    state: CheckpointState,
) -> None:
    svc.events.append(
        run_id=run_id,
        step_execution_id=step_execution_id,
        event_type="engine.checkpoint",
        actor_type="system",
        severity="debug",
        payload={
            "run_id": run_id,
            "next_step_id": state.next_step_id,
            "artifacts_index": state.artifacts_index,
            "module_spec_inline": state.module_spec_inline,
            "decision_record_inline": state.decision_record_inline,
            "paused_gate": state.paused_gate,
        },
    )


class WorkflowEngineV0:
    """
    MVP engine that can execute:
      - agent_task (LLM)
      - tool_task (tools)
      - human_gate (pause/resume)

    Durable state is stored as an append-only checkpoint event (engine.checkpoint),
    so resume can reconstruct where to continue.

    You’ll evolve this later into a richer engine with:
      - structured step inputs/outputs
      - step-group injection
      - stronger schema validation
      - more robust retry policies
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        paths: RepoPaths,
        resolver: RunConfigResolver,
        services_factory: DataServicesFactory,
        llm: LLMGateway,
        tools: ToolExecutor,
        gates: HumanGateManager,
        guards: GuardEvaluator,
    ) -> None:
        self._conn = conn
        self._paths = paths
        self._resolver = resolver
        self._services_factory = services_factory
        self._llm = llm
        self._tools = tools
        self._gates = gates
        self._guards = guards

    # ---------------------------
    # Public API
    # ---------------------------

    def start_run(
        self,
        *,
        identity: RunIdentity,
        config: ResolvedRunConfig,
        initial_inputs: JSON,
    ) -> RunHandle:
        """
        Creates the run row, writes an initial checkpoint, then runs until PAUSED or terminal.
        """
        with transaction(self._conn, write=True):
            run_id = repo_ops.create_run(
                self._conn,
                repo_root=str(self._paths.repo_root),
                profile_id=identity.profile_selector,
                workflow_id=identity.workflow_selector,
                policy_id=identity.policy_selector,
                git_base_sha=None,
                git_branch=None,
                workflow_sha=config.refs.workflow.sha256,
                policy_sha=config.refs.policy.sha256,
                profile_sha=config.refs.profile.sha256,
            )

            # Initial checkpoint: entry step
            svc = self._services_factory(
                conn=self._conn,
                paths=self._paths,
                run_id=run_id,
                repo_root=str(self._paths.repo_root),
                profile_id=identity.profile_selector,
                workflow_id=identity.workflow_selector,
                policy_id=identity.policy_selector,
                decision_record=None,
            )
            svc.events.append(
                run_id=run_id,
                step_execution_id=None,
                event_type=EventType.run_started.value,
                actor_type="system",
                severity="info",
                payload={
                    "run_id": run_id,
                    "profile_id": identity.profile_selector,
                    "workflow_id": identity.workflow_selector,
                    "policy_id": identity.policy_selector,
                },
            )
            _write_checkpoint(
                svc=svc,
                run_id=run_id,
                step_execution_id=None,
                state=CheckpointState(
                    next_step_id=config.workflow.entry,
                    artifacts_index={},
                    module_spec_inline=initial_inputs.get("module_spec"),
                    decision_record_inline=initial_inputs.get("decision_record"),
                ),
            )

        return self.run_until_blocked_or_done(run_id=run_id)

    def resume_run(self, *, run_id: str, human_inputs: Optional[JSON] = None) -> RunHandle:
        """
        If the run is paused for a gate, accept a decision payload then continue.
        human_inputs expected (MVP):
          {
            "gate_id": "...",
            "decision": "APPROVE"|"ACK"|"BLOCK",
            "human_actor": "alice",
            "reason": "...",
            "resume_payload": {...}
          }
        """
        row = _db_get_run_row(self._conn, run_id)
        if row["status"] != "PAUSED":
            return self.run_until_blocked_or_done(run_id=run_id)

        cp = _db_get_latest_checkpoint(self._conn, run_id)
        if not cp or not cp.paused_gate:
            raise RuntimeError("Run is PAUSED but no paused_gate found in checkpoint")

        if not human_inputs:
            raise ValueError("Run is paused; provide human_inputs with gate decision")

        gate_id = human_inputs.get("gate_id")
        decision = human_inputs.get("decision")
        human_actor = human_inputs.get("human_actor")
        reason = human_inputs.get("reason")
        resume_payload = human_inputs.get("resume_payload")

        if not gate_id or not decision or not human_actor:
            raise ValueError("human_inputs must include gate_id, decision, and human_actor")

        with transaction(self._conn, write=True):
            self._gates.decide(
                GateDecision(
                    gate_id=str(gate_id),
                    decision=str(decision),
                    human_actor=str(human_actor),
                    reason=str(reason) if reason else None,
                    resume_payload=resume_payload,
                )
            )
            _db_set_run_status(self._conn, run_id, "RUNNING")

        return self.run_until_blocked_or_done(run_id=run_id)

    def run_until_blocked_or_done(self, *, run_id: str) -> RunHandle:
        """
        Main driver loop.
        Loads config from DB (profile/workflow/policy selectors), reloads latest checkpoint,
        executes steps until PAUSED or terminal.
        """
        row = _db_get_run_row(self._conn, run_id)
        profile_selector = str(row["profile_id"])
        repo_root = str(row["repo_root"])

        config = self._resolver.resolve(repo_root=repo_root, profile_selector=profile_selector)

        cp = _db_get_latest_checkpoint(self._conn, run_id)
        if not cp:
            # No checkpoint => start at entry
            cp = CheckpointState(next_step_id=config.workflow.entry, artifacts_index={})

        next_step = cp.next_step_id
        artifacts_index = dict(cp.artifacts_index)
        module_spec_inline = cp.module_spec_inline
        decision_record_inline = cp.decision_record_inline

        # Stop early if run already terminal
        if row["status"] in ("SUCCEEDED", "FAILED", "CANCELED"):
            return RunHandle(run_id=run_id, status=row["status"])

        while True:
            if next_step is None:
                with transaction(self._conn, write=True):
                    svc = self._services_factory(
                        conn=self._conn,
                        paths=self._paths,
                        run_id=run_id,
                        repo_root=repo_root,
                        profile_id=str(row["profile_id"]),
                        workflow_id=str(row["workflow_id"]),
                        policy_id=str(row["policy_id"]),
                        decision_record=decision_record_inline,
                    )
                    _db_set_run_status(self._conn, run_id, "SUCCEEDED")
                    svc.events.append(
                        run_id=run_id,
                        step_execution_id=None,
                        event_type=EventType.run_succeeded.value,
                        actor_type="system",
                        severity="info",
                        payload={"run_id": run_id},
                    )
                return RunHandle(run_id=run_id, status="SUCCEEDED")

            step_def = config.workflow.steps.get(next_step)
            if step_def is None:
                with transaction(self._conn, write=True):
                    svc = self._services_factory(
                        conn=self._conn,
                        paths=self._paths,
                        run_id=run_id,
                        repo_root=repo_root,
                        profile_id=str(row["profile_id"]),
                        workflow_id=str(row["workflow_id"]),
                        policy_id=str(row["policy_id"]),
                        decision_record=decision_record_inline,
                    )
                    _db_set_run_status(self._conn, run_id, "FAILED")
                    svc.events.append(
                        run_id=run_id,
                        step_execution_id=None,
                        event_type=EventType.run_failed.value,
                        actor_type="system",
                        severity="error",
                        payload={"run_id": run_id, "error_type": "UnknownStep", "error_message": f"Unknown step_id={next_step}"},
                    )
                return RunHandle(run_id=run_id, status="FAILED", last_error=f"Unknown step_id={next_step}")

            # Build contexts
            gctx = GuardContext(
                run_id=run_id,
                step_id=next_step,
                decision_record=decision_record_inline or {},
                artifacts=artifacts_index,
            )

            # Step-level run_if
            if not _guard_true(self._guards, step_def.get("run_if"), gctx):
                # Skip step and advance via edges
                next_step = _choose_next_step(edges=config.workflow.edges, from_step=next_step, guard_eval=self._guards, gctx=gctx)
                continue

            # Execute step attempt 1 (MVP: no retries here; you can add later)
            with transaction(self._conn, write=True):
                step_execution_id = repo_ops.start_step(self._conn, run_id=run_id, step_id=next_step, attempt_no=1)

                svc = self._services_factory(
                    conn=self._conn,
                    paths=self._paths,
                    run_id=run_id,
                    repo_root=repo_root,
                    profile_id=str(row["profile_id"]),
                    workflow_id=str(row["workflow_id"]),
                    policy_id=str(row["policy_id"]),
                    decision_record=decision_record_inline,
                )

                svc.events.append(
                    run_id=run_id,
                    step_execution_id=step_execution_id,
                    event_type=EventType.step_started.value,
                    actor_type="system",
                    severity="info",
                    payload={"run_id": run_id, "step_id": next_step, "step_execution_id": step_execution_id, "attempt_no": 1},
                )

                try:
                    kind = str(step_def.get("kind", "")).strip()

                    if kind == "agent_task":
                        module_spec_inline, decision_record_inline = self._exec_agent_task(
                            svc=svc,
                            config=config,
                            step_id=next_step,
                            step_execution_id=step_execution_id,
                            step_def=step_def,
                            artifacts_index=artifacts_index,
                            module_spec_inline=module_spec_inline,
                            decision_record_inline=decision_record_inline,
                        )

                    elif kind == "tool_task":
                        self._exec_tool_task(
                            svc=svc,
                            config=config,
                            step_id=next_step,
                            step_execution_id=step_execution_id,
                            step_def=step_def,
                            artifacts_index=artifacts_index,
                            module_spec_inline=module_spec_inline,
                            decision_record_inline=decision_record_inline,
                        )

                    elif kind == "human_gate":
                        gate_req = self._exec_human_gate(
                            svc=svc,
                            step_id=next_step,
                            step_execution_id=step_execution_id,
                            step_def=step_def,
                            artifacts_index=artifacts_index,
                            decision_record_inline=decision_record_inline,
                        )

                        # Pause run
                        _db_set_run_status(self._conn, run_id, "PAUSED")
                        repo_ops.finish_step(self._conn, step_execution_id=step_execution_id, status="WAITING_HUMAN")
                        svc.events.append(
                            run_id=run_id,
                            step_execution_id=step_execution_id,
                            event_type=EventType.run_paused.value,
                            actor_type="system",
                            severity="info",
                            payload={"run_id": run_id, "reason": f"waiting_gate:{gate_req.gate_name}"},
                        )

                        # Checkpoint includes paused gate and the step we should continue *after* gate is resolved.
                        # (For MVP: next step is chosen right now; could also choose on resume.)
                        after_gate = _choose_next_step(edges=config.workflow.edges, from_step=next_step, guard_eval=self._guards, gctx=gctx)
                        _write_checkpoint(
                            svc=svc,
                            run_id=run_id,
                            step_execution_id=step_execution_id,
                            state=CheckpointState(
                                next_step_id=after_gate,
                                artifacts_index=artifacts_index,
                                module_spec_inline=module_spec_inline,
                                decision_record_inline=decision_record_inline,
                                paused_gate={
                                    "gate_id": gate_req.gate_id,
                                    "gate_name": gate_req.gate_name,
                                    "step_id": gate_req.step_id,
                                    "mode": gate_req.mode,
                                },
                            ),
                        )

                        return RunHandle(run_id=run_id, status="PAUSED", current_step_id=next_step, waiting_gate=gate_req)

                    else:
                        raise RuntimeError(f"Unsupported step kind '{kind}' at step '{next_step}'")

                    # Step success => choose next step via edges
                    next_step = _choose_next_step(edges=config.workflow.edges, from_step=next_step, guard_eval=self._guards, gctx=gctx)

                    repo_ops.finish_step(self._conn, step_execution_id=step_execution_id, status="SUCCEEDED")
                    svc.events.append(
                        run_id=run_id,
                        step_execution_id=step_execution_id,
                        event_type=EventType.step_finished.value,
                        actor_type="system",
                        severity="info",
                        payload={"run_id": run_id, "step_id": gctx.step_id, "step_execution_id": step_execution_id, "attempt_no": 1, "status": "SUCCEEDED"},
                    )

                    # Durable checkpoint (small & frequent is fine for MVP)
                    _write_checkpoint(
                        svc=svc,
                        run_id=run_id,
                        step_execution_id=step_execution_id,
                        state=CheckpointState(
                            next_step_id=next_step,
                            artifacts_index=artifacts_index,
                            module_spec_inline=module_spec_inline,
                            decision_record_inline=decision_record_inline,
                            paused_gate=None,
                        ),
                    )

                except Exception as e:
                    repo_ops.finish_step(self._conn, step_execution_id=step_execution_id, status="FAILED", error_type=type(e).__name__, error_message=str(e))
                    _db_set_run_status(self._conn, run_id, "FAILED")
                    svc.events.append(
                        run_id=run_id,
                        step_execution_id=step_execution_id,
                        event_type=EventType.step_attempt_failed.value,
                        actor_type="system",
                        severity="error",
                        payload={
                            "run_id": run_id,
                            "step_id": next_step,
                            "step_execution_id": step_execution_id,
                            "attempt_no": 1,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                    )
                    svc.events.append(
                        run_id=run_id,
                        step_execution_id=None,
                        event_type=EventType.run_failed.value,
                        actor_type="system",
                        severity="error",
                        payload={"run_id": run_id, "error_type": type(e).__name__, "error_message": str(e)},
                    )
                    return RunHandle(run_id=run_id, status="FAILED", current_step_id=next_step, last_error=str(e))

    # ---------------------------
    # Step implementations (MVP)
    # ---------------------------

    def _exec_agent_task(
        self,
        *,
        svc: DataServices,
        config: ResolvedRunConfig,
        step_id: str,
        step_execution_id: str,
        step_def: JSON,
        artifacts_index: Dict[str, str],
        module_spec_inline: Optional[JSON],
        decision_record_inline: Optional[JSON],
    ) -> Tuple[Optional[JSON], Optional[JSON]]:
        role = str(step_def["role"])
        template_id = config.prompt_bindings.get(role)

        # Variables available to prompt templates (MVP)
        vars: Dict[str, Any] = {
            "module_spec": module_spec_inline or {},
            "decision_record": decision_record_inline or {},
            "artifacts": artifacts_index,
            "step_id": step_id,
            "run_id": svc.ctx.run_id if hasattr(svc, "ctx") else None,
        }

        prompt = _safe_format(str(step_def.get("prompt", "")), vars)

        req = LLMRequest(
            role=role,
            template_id=template_id,
            messages=[{"role": "system", "content": prompt}],
            metadata={"run_id": svc.ctx.run_id if hasattr(svc, "ctx") else None, "step_id": step_id},
        )

        resp = self._llm.complete(
            run_id=svc.ctx.run_id if hasattr(svc, "ctx") else "",  # safe fallback
            step_execution_id=step_execution_id,
            req=req,
            decision_record=decision_record_inline or {},
        )

        if resp.status != "OK":
            raise RuntimeError(f"LLM call failed for role={role} provider={resp.provider} model={resp.model} status={resp.status}")

        # Write artifact as declared
        out = step_def.get("output") or {}
        artifact_type = str(out.get("artifact_type", "agent_output"))
        fmt = str(out.get("format", "TEXT"))
        relpath = str(out.get("relpath", f"{step_id}.txt"))
        logical_name = str(out.get("logical_name", step_id))

        aw = svc.artifacts.write_text(
            run_id=svc.ctx.run_id,
            step_execution_id=step_execution_id,
            artifact_type=artifact_type,
            fmt=fmt,
            relpath_under_artifacts=relpath,
            text=resp.text,
            meta={"role": role, "provider": resp.provider, "model": resp.model},
        )
        artifacts_index[logical_name] = aw.artifact_id

        svc.events.append(
            run_id=svc.ctx.run_id,
            step_execution_id=step_execution_id,
            event_type=EventType.artifact_written.value,
            actor_type="system",
            severity="info",
            payload={
                "run_id": svc.ctx.run_id,
                "artifact_id": aw.artifact_id,
                "artifact_type": artifact_type,
                "relpath": aw.relpath,
                "sha256": aw.sha256,
                "bytes": aw.bytes,
            },
        )

        # MVP: if this step is known to produce module_spec or decision_record, parse inline JSON from response if possible.
        # In real MontyMate you’ll parse YAML/JSON properly (and validate against schemas).
        if artifact_type == "module_spec":
            # try JSON parse; if fails, keep inline None (artifact is still canonical)
            try:
                module_spec_inline = json.loads(resp.text)
            except Exception:
                module_spec_inline = module_spec_inline

            svc.events.append(
                run_id=svc.ctx.run_id,
                step_execution_id=step_execution_id,
                event_type=EventType.spec_drafted.value,
                actor_type="system",
                severity="info",
                payload={"run_id": svc.ctx.run_id, "artifact_id": aw.artifact_id},
            )

        if artifact_type == "decision_record":
            try:
                decision_record_inline = json.loads(resp.text)
            except Exception:
                decision_record_inline = decision_record_inline

            svc.events.append(
                run_id=svc.ctx.run_id,
                step_execution_id=step_execution_id,
                event_type=EventType.policy_decision_record_created.value,
                actor_type="system",
                severity="info",
                payload={
                    "run_id": svc.ctx.run_id,
                    "artifact_id": aw.artifact_id,
                    "change_class": (decision_record_inline or {}).get("change_class"),
                    "risk_level": (decision_record_inline or {}).get("risk_level"),
                },
            )

        return module_spec_inline, decision_record_inline

    def _exec_tool_task(
        self,
        *,
        svc: DataServices,
        config: ResolvedRunConfig,
        step_id: str,
        step_execution_id: str,
        step_def: JSON,
        artifacts_index: Dict[str, str],
        module_spec_inline: Optional[JSON],
        decision_record_inline: Optional[JSON],
    ) -> None:
        tool_alias = step_def.get("tool_alias")
        tool_name = step_def.get("tool_name")

        if tool_alias and tool_name:
            raise ValueError(f"tool_task '{step_id}' must specify only one of tool_alias or tool_name")

        if tool_alias:
            resolved_tool = config.tool_bindings.get(str(tool_alias))
            if not resolved_tool:
                raise ValueError(f"tool alias '{tool_alias}' not bound in profile")
            tool_name = resolved_tool

        if not tool_name:
            raise ValueError(f"tool_task '{step_id}' missing tool_alias/tool_name")

        tool_meta = config.tool_registry.tools.get(str(tool_name))
        if not tool_meta:
            raise ValueError(f"unknown tool '{tool_name}' (not in tool registry)")

        # Interpolate inputs (MVP)
        vars: Dict[str, Any] = {
            "module_spec": module_spec_inline or {},
            "decision_record": decision_record_inline or {},
            "artifacts": artifacts_index,
            "step_id": step_id,
        }

        raw_input = step_def.get("input") or {}
        tool_input: Dict[str, Any] = {}

        for k, v in raw_input.items():
            if isinstance(v, str):
                tool_input[k] = _safe_format(v, vars)
            else:
                tool_input[k] = v

        inv = ToolInvocation(
            tool_name=str(tool_name),
            tool_type=str(tool_meta.get("tool_type", "unknown")),
            runs_in_runtime=bool(tool_meta.get("runs_in_runtime", False)),
            input=tool_input,
            timeout_s=step_def.get("timeout_s"),
        )

        outcome = self._tools.execute(run_id=svc.ctx.run_id, step_execution_id=step_execution_id, inv=inv)
        if outcome.status != "OK":
            raise RuntimeError(f"tool failed tool={tool_name} status={outcome.status}")

    def _exec_human_gate(
        self,
        *,
        svc: DataServices,
        step_id: str,
        step_execution_id: str,
        step_def: JSON,
        artifacts_index: Dict[str, str],
        decision_record_inline: Optional[JSON],
    ) -> GateRequest:
        gate_name = str(step_def["gate_name"])
        summary = str(step_def.get("summary", f"Approval required: {gate_name}"))

        # Resolve required artifacts (logical names -> artifact ids)
        req_artifacts = []
        for name in (step_def.get("required_artifacts") or []):
            req_artifacts.append(_artifact_id_from_logical(artifacts_index, str(name)))

        mode, ack_window = _gate_mode_for(decision_record_inline, gate_name, fallback_mode=str(step_def.get("mode", "APPROVE")))

        req = GateRequest(
            gate_id=_uuid(),
            run_id=svc.ctx.run_id,
            step_id=step_id,
            gate_name=gate_name,
            mode=mode,  # AUTO|ACK|APPROVE
            ack_window_minutes=ack_window,
            summary=summary,
            required_artifacts=req_artifacts,
            resume_payload_schema=step_def.get("resume_payload_schema"),
        )

        stored = self._gates.request_gate(req)

        # Emit an event too (gate manager may do this, but it’s okay for MVP to do both if you want strict logging)
        svc.events.append(
            run_id=svc.ctx.run_id,
            step_execution_id=step_execution_id,
            event_type=EventType.gate_requested.value,
            actor_type="system",
            severity="info",
            payload={
                "run_id": svc.ctx.run_id,
                "gate_name": gate_name,
                "mode": stored.mode,
                "step_id": step_id,
            },
        )

        return stored