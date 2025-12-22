from __future__ import annotations

import os
import subprocess
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from montymate.core.types import JSON
from montymate.data.db import transaction
from montymate.data.paths import RepoPaths
from montymate.data.services import DataServices
from montymate.tools.executor import ToolExecutor
from montymate.tools.models import ToolInvocation, ToolOutcome


@dataclass
class SimpleLocalToolExecutor(ToolExecutor):
    conn: sqlite3.Connection
    paths: RepoPaths
    services_factory: Any  # make_services

    def execute(self, *, run_id: str, step_execution_id: str, inv: ToolInvocation) -> ToolOutcome:
        run_row = self.conn.execute("SELECT * FROM mm_runs WHERE run_id=?", (run_id,)).fetchone()
        if not run_row:
            return ToolOutcome(status="ERROR", output={"error": "run not found"})

        repo_root = str(run_row["repo_root"])
        svc: DataServices = self.services_factory(
            conn=self.conn,
            paths=self.paths,
            run_id=run_id,
            repo_root=repo_root,
            profile_id=str(run_row["profile_id"]),
            workflow_id=str(run_row["workflow_id"]),
            policy_id=str(run_row["policy_id"]),
            decision_record=None,
        )

        if inv.tool_name == "capture_repo_snapshot":
            out = {"repo_root": repo_root, "note": "MVP snapshot"}
            with transaction(self.conn, write=True):
                svc.tools.record_call(
                    run_id=run_id,
                    step_execution_id=step_execution_id,
                    tool_name=inv.tool_name,
                    tool_type=inv.tool_type,
                    runs_in_runtime=inv.runs_in_runtime,
                    status="OK",
                    input_json=inv.input,
                    output_json=out,
                    unit_type="request",
                    units=1.0,
                    cost_usd=0.0,
                    pricing_id=None,
                )
            return ToolOutcome(status="OK", output=out, artifacts=[], cost_usd=0.0, unit_type="request", units=1.0)

        if inv.tool_name == "run_gates":
            # simplest: run pytest only
            cmd = inv.input.get("pytest_cmd") or ["python", "-m", "pytest", "-q"]
            timeout = inv.timeout_s or 600

            try:
                p = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )
                passed = (p.returncode == 0)
                out = {
                    "passed": passed,
                    "pytest": {
                        "cmd": cmd,
                        "returncode": p.returncode,
                        "stdout": p.stdout,
                        "stderr": p.stderr,
                    },
                }
                status = "OK" if passed else "ERROR"
            except subprocess.TimeoutExpired as e:
                out = {"passed": False, "pytest": {"cmd": cmd, "timeout_s": timeout, "stdout": e.stdout, "stderr": e.stderr}}
                status = "TIMEOUT"

            with transaction(self.conn, write=True):
                svc.tools.record_call(
                    run_id=run_id,
                    step_execution_id=step_execution_id,
                    tool_name=inv.tool_name,
                    tool_type=inv.tool_type,
                    runs_in_runtime=inv.runs_in_runtime,
                    status=status,
                    input_json=inv.input,
                    output_json=out,
                    unit_type="command",
                    units=1.0,
                    cost_usd=0.0,
                    pricing_id=None,
                )

            if status == "OK":
                return ToolOutcome(status="OK", output=out, artifacts=[], cost_usd=0.0, unit_type="command", units=1.0)
            if status == "TIMEOUT":
                return ToolOutcome(status="TIMEOUT", output=out, artifacts=[])
            return ToolOutcome(status="ERROR", output=out, artifacts=[])

        # Unknown tool
        out = {"error": f"Tool not implemented: {inv.tool_name}"}
        with transaction(self.conn, write=True):
            svc.tools.record_call(
                run_id=run_id,
                step_execution_id=step_execution_id,
                tool_name=inv.tool_name,
                tool_type=inv.tool_type,
                runs_in_runtime=inv.runs_in_runtime,
                status="ERROR",
                input_json=inv.input,
                output_json=out,
                unit_type="request",
                units=1.0,
                cost_usd=0.0,
                pricing_id=None,
            )
        return ToolOutcome(status="ERROR", output=out, artifacts=[])