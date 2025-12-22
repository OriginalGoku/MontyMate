from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import sqlite3

from montymate.config.resolver import DefaultRunConfigResolver
from montymate.core.types import JSON, RunIdentity
from montymate.data.db import connect, transaction
from montymate.data.migrate import (
    apply_migrations,
    ensure_application_id,
    load_migrations,
)
from montymate.data.paths import RepoPaths
from montymate.data.types import RunContext, StoragePolicy
from montymate.data.services import DataServices
from montymate.engine.guards_impl import DefaultGuardEvaluator
from montymate.engine.workflow_engine_impl import WorkflowEngineV0
from montymate.events import EventType
from montymate.gates.hitl import HumanGateManager
from montymate.gates.models import GateDecision, GateRequest
from montymate.llm.gateway import LLMGateway
from montymate.llm.models import LLMRequest, LLMResponse
from montymate.tools.executor import ToolExecutor
from montymate.tools.models import ToolInvocation, ToolOutcome
from montymate.llm.gateway_impl import LLMGatewayImpl
from montymate.tools.simple_executor import SimpleLocalToolExecutor

APPLICATION_ID = 1297048389


# ----------------------------
# DataServices factory (used by engine + gateway/tool stubs)
# ----------------------------

def make_services(
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
    # Storage policy comes from decision_record when available, else defaults.
    if decision_record is None:
        storage_policy = StoragePolicy(payload_mode="hybrid", inline_max_bytes=32768, artifact_root=str(paths.ai_root))
    else:
        storage_policy = StoragePolicy.from_decision_record(decision_record)

    ctx = RunContext(
        run_id=run_id,
        repo_root=repo_root,
        profile_id=profile_id,
        workflow_id=workflow_id,
        policy_id=policy_id,
        storage_policy=storage_policy,
    )
    return DataServices.build(conn, paths, ctx)


# ----------------------------
# MVP DB-backed Gate Manager
# ----------------------------

@dataclass
class DBHumanGateManager(HumanGateManager):
    conn: sqlite3.Connection
    paths: RepoPaths

    def request_gate(self, req: GateRequest) -> GateRequest:
        # Persist as events; engine also logs gate_requested, but duplication is okay in MVP.
        svc = make_services(
            conn=self.conn,
            paths=self.paths,
            run_id=req.run_id,
            repo_root=str(self.paths.repo_root),
            profile_id="unknown",
            workflow_id="unknown",
            policy_id="unknown",
            decision_record=None,
        )
        svc.events.append(
            run_id=req.run_id,
            step_execution_id=None,
            event_type=EventType.gate_requested.value,
            actor_type="system",
            severity="info",
            payload={"run_id": req.run_id, "gate_name": req.gate_name, "mode": req.mode, "step_id": req.step_id},
        )
        return req

    def decide(self, decision: GateDecision) -> None:
        # Emit a decision event. If BLOCK, raise so caller can keep run paused/failed.
        # (Engine resume_run wraps in a transaction; raising aborts the resume.)
        # In a later iteration you’ll store gates in a real mm_gates table.
        run_id = self._infer_run_id(decision.gate_id)  # best-effort fallback
        svc = make_services(
            conn=self.conn,
            paths=self.paths,
            run_id=run_id,
            repo_root=str(self.paths.repo_root),
            profile_id="unknown",
            workflow_id="unknown",
            policy_id="unknown",
            decision_record=None,
        )

        if decision.decision == "ACK":
            et = EventType.gate_acknowledged.value
            sev = "info"
        elif decision.decision == "APPROVE":
            et = EventType.gate_approved.value
            sev = "info"
        else:
            et = EventType.gate_blocked.value
            sev = "warn"

        svc.events.append(
            run_id=run_id,
            step_execution_id=None,
            event_type=et,
            actor_type="human",
            severity=sev,
            payload={
                "run_id": run_id,
                "gate_name": "unknown",
                "human_actor": decision.human_actor,
                "reason": decision.reason,
            },
        )

        if decision.decision == "BLOCK":
            raise RuntimeError("Gate BLOCK decision received")

    def get_pending_gate(self, run_id: str) -> Optional[GateRequest]:
        # MVP: engine checkpoint already returns pending gate; keep this as no-op.
        return None

    def _infer_run_id(self, gate_id: str) -> str:
        # MVP fallback: in a real mm_gates table this would be a lookup.
        # Here we cannot infer reliably, so return gate_id as placeholder;
        # the engine also carries run_id and will pass correct context in practice.
        return gate_id


# ----------------------------
# MVP DB-backed LLM Gateway stub
# ----------------------------

@dataclass
class DBLLMGateway(LLMGateway):
    conn: sqlite3.Connection
    paths: RepoPaths

    def complete(self, *, run_id: str, step_execution_id: str, req: LLMRequest, decision_record: JSON) -> LLMResponse:
        # MVP: generate deterministic placeholder output
        # Replace this with a real provider adapter later.
        text = json.dumps(
            {
                "role": req.role,
                "note": "MVP stub response. Replace DBLLMGateway with real provider adapter.",
                "echo_prompt_preview": (req.messages[0].get("content", "")[:200] if req.messages else ""),
            },
            ensure_ascii=False,
            indent=2,
        )

        # Record via DataServices so you get events + cost ledger.
        # We'll treat this as a “free” call (cost_usd=0) for MVP.
        row = self.conn.execute("SELECT * FROM mm_runs WHERE run_id=?", (run_id,)).fetchone()
        profile_id = row["profile_id"] if row else "unknown"
        workflow_id = row["workflow_id"] if row else "unknown"
        policy_id = row["policy_id"] if row else "unknown"
        repo_root = row["repo_root"] if row else str(self.paths.repo_root)

        svc = make_services(
            conn=self.conn,
            paths=self.paths,
            run_id=run_id,
            repo_root=repo_root,
            profile_id=profile_id,
            workflow_id=workflow_id,
            policy_id=policy_id,
            decision_record=decision_record,
        )

        with transaction(self.conn, write=True):
            svc.llm.record_call(
                run_id=run_id,
                step_execution_id=step_execution_id,
                role=req.role,
                provider="stub",
                model="stub-model",
                status="OK",
                prompt_text=json.dumps(req.messages, ensure_ascii=False),
                response_text=text,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                cost_usd=0.0,
                template_id=req.template_id,
                pricing_id=None,
                request_json={"messages": req.messages},
                response_json={"text": text},
            )

        return LLMResponse(
            status="OK",
            provider="stub",
            model="stub-model",
            text=text,
            raw={"text": text},
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            cost_usd=0.0,
        )


# ----------------------------
# MVP DB-backed Tool Executor stub
# ----------------------------

@dataclass
class DBToolExecutor(ToolExecutor):
    conn: sqlite3.Connection
    paths: RepoPaths

    def execute(self, *, run_id: str, step_execution_id: str, inv: ToolInvocation) -> ToolOutcome:
        # MVP: implement only a couple of basic tools as placeholders.
        # Everything is recorded through DataServices.
        row = self.conn.execute("SELECT * FROM mm_runs WHERE run_id=?", (run_id,)).fetchone()
        profile_id = row["profile_id"] if row else "unknown"
        workflow_id = row["workflow_id"] if row else "unknown"
        policy_id = row["policy_id"] if row else "unknown"
        repo_root = row["repo_root"] if row else str(self.paths.repo_root)

        svc = make_services(
            conn=self.conn,
            paths=self.paths,
            run_id=run_id,
            repo_root=repo_root,
            profile_id=profile_id,
            workflow_id=workflow_id,
            policy_id=policy_id,
            decision_record=None,
        )

        output: Dict[str, Any] = {}
        status = "OK"

        # Example minimal tools:
        if inv.tool_name in ("capture_repo_snapshot", "repo_snapshot"):
            output = {"repo_root": repo_root, "note": "MVP snapshot stub"}
        elif inv.tool_name in ("run_gates", "gates"):
            # Stub: pretend tests passed
            output = {"passed": True, "gates": {"pytest": {"passed": True, "stdout": "stub", "stderr": ""}}}
        else:
            status = "ERROR"
            output = {"error": f"Tool not implemented in MVP stub: {inv.tool_name}"}

        with transaction(self.conn, write=True):
            svc.tools.record_call(
                run_id=run_id,
                step_execution_id=step_execution_id,
                tool_name=inv.tool_name,
                tool_type=inv.tool_type,
                runs_in_runtime=inv.runs_in_runtime,
                status=status,
                input_json=inv.input,
                output_json=output,
                input_artifact_id=None,
                output_artifact_id=None,
                unit_type="request",
                units=1.0,
                cost_usd=0.0,
                pricing_id=None,
            )

        if status != "OK":
            return ToolOutcome(status="ERROR", output=output, artifacts=[])

        return ToolOutcome(status="OK", output=output, artifacts=[], cost_usd=0.0, unit_type="request", units=1.0)


# ----------------------------
# CLI
# ----------------------------

def _init_db(paths: RepoPaths) -> sqlite3.Connection:
    conn = connect(paths.db_path)
    # DB identity + migrations (uses PRAGMA application_id/user_version patterns you already adopted)
    ensure_application_id(conn, APPLICATION_ID)
    migrations = load_migrations("montymate", "resources/db/migrations")
    apply_migrations(conn, migrations)
    return conn


def cmd_run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo).resolve()
    paths = RepoPaths.for_repo(repo_root)
    paths.ensure_dirs()

    conn = _init_db(paths)

    resolver = DefaultRunConfigResolver(package="montymate")
    config = resolver.resolve(repo_root=str(repo_root), profile_selector=args.profile)

    guards = DefaultGuardEvaluator()
    # llm = DBLLMGateway(conn=conn, paths=paths)
    # tools = DBToolExecutor(conn=conn, paths=paths)
    llm = LLMGatewayImpl(conn=conn, paths=paths, resolver=resolver, services_factory=make_services)
    tools = SimpleLocalToolExecutor(conn=conn, paths=paths, services_factory=make_services)
    gates = DBHumanGateManager(conn=conn, paths=paths)

    engine = WorkflowEngineV0(
        conn=conn,
        paths=paths,
        resolver=resolver,
        services_factory=make_services,
        llm=llm,
        tools=tools,
        gates=gates,
        guards=guards,
    )

    identity = RunIdentity(
        repo_root=str(repo_root),
        run_id="(ignored_by_engine_v0)",
        profile_selector=args.profile,
        workflow_selector=config.profile.workflow_default,
        policy_selector=config.profile.policy_default,
    )

    handle = engine.start_run(identity=identity, config=config, initial_inputs={})
    print(json.dumps(handle.__dict__, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo).resolve()
    paths = RepoPaths.for_repo(repo_root)
    paths.ensure_dirs()

    conn = _init_db(paths)

    resolver = DefaultRunConfigResolver(package="montymate")
    guards = DefaultGuardEvaluator()
    # llm = DBLLMGateway(conn=conn, paths=paths)
    # tools = DBToolExecutor(conn=conn, paths=paths)
    llm = LLMGatewayImpl(conn=conn, paths=paths, resolver=resolver, services_factory=make_services)
    tools = SimpleLocalToolExecutor(conn=conn, paths=paths, services_factory=make_services)
    gates = DBHumanGateManager(conn=conn, paths=paths)

    engine = WorkflowEngineV0(
        conn=conn,
        paths=paths,
        resolver=resolver,
        services_factory=make_services,
        llm=llm,
        tools=tools,
        gates=gates,
        guards=guards,
    )

    human_inputs: JSON = {
        "gate_id": args.gate_id,
        "decision": args.decision,
        "human_actor": args.human,
        "reason": args.reason,
    }
    handle = engine.resume_run(run_id=args.run_id, human_inputs=human_inputs)
    print(json.dumps(handle.__dict__, ensure_ascii=False, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    # argparse is stdlib and supports subcommands cleanly.  [oai_citation:3‡Python documentation](https://docs.python.org/3/library/argparse.html?utm_source=chatgpt.com)
    p = argparse.ArgumentParser(prog="montymate")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Start a new MontyMate run")
    p_run.add_argument("--repo", default=".", help="Repo root (default: .)")
    p_run.add_argument("--profile", required=True, help="Profile selector, e.g. fastapi_service@2")
    p_run.set_defaults(fn=cmd_run)

    p_resume = sub.add_parser("resume", help="Resume a paused run by providing a gate decision")
    p_resume.add_argument("--repo", default=".", help="Repo root (default: .)")
    p_resume.add_argument("--run-id", required=True, help="Run ID to resume")
    p_resume.add_argument("--gate-id", required=True, help="Gate ID being decided")
    p_resume.add_argument("--decision", required=True, choices=["ACK", "APPROVE", "BLOCK"])
    p_resume.add_argument("--human", required=True, help="Human actor id/name")
    p_resume.add_argument("--reason", default=None)
    p_resume.set_defaults(fn=cmd_resume)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())