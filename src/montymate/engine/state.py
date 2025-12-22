from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from montymate.config.models import ResolvedRunConfig
from montymate.core.types import JSON, RunStatus, StepStatus


@dataclass
class RunHandle:
    run_id: str
    status: RunStatus
    current_step_id: Optional[str] = None
    waiting_gate: Optional["GateRequest"] = None
    last_error: Optional[str] = None


@dataclass(frozen=True)
class GuardContext:
    run_id: str
    step_id: str
    decision_record: JSON
    artifacts: Dict[str, str]  # name->artifact_id or relpath (your choice)


@dataclass(frozen=True)
class ExecutionContext:
    """
    What every step needs to run.
    Keep this stable; it becomes your “engine ABI”.
    """

    run_id: str
    config: ResolvedRunConfig

    decision_record: JSON
    module_spec: Optional[JSON]

    # resolved bindings (from profile)
    tool_bindings: Dict[str, str]  # alias -> concrete tool name
    prompt_bindings: Dict[str, str]  # role -> template_id

    # useful indexes
    artifacts: Dict[str, str]  # logical name -> artifact_id


@dataclass(frozen=True)
class StepResult:
    status: StepStatus
    outputs: Dict[str, Any] | None = None
    produced_artifacts: List[str] | None = None
    pause: Optional["GateRequest"] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
