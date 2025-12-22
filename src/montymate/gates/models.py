from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from montymate.core.types import JSON

Decision = Literal["ACK", "APPROVE", "BLOCK"]
GateMode = Literal["AUTO", "ACK", "APPROVE"]


@dataclass(frozen=True)
class GateRequest:
    gate_id: str
    run_id: str
    step_id: str
    gate_name: str
    mode: GateMode
    ack_window_minutes: Optional[int]
    summary: str

    required_artifacts: List[str]  # artifact_ids the human should review
    resume_payload_schema: Optional[JSON] = None


@dataclass(frozen=True)
class GateDecision:
    gate_id: str
    decision: Decision
    human_actor: str
    reason: Optional[str] = None
    resume_payload: Optional[Dict[str, str]] = None
