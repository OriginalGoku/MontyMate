from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from montymate.core.types import JSON


@dataclass(frozen=True)
class ToolInvocation:
    tool_name: str
    tool_type: str
    runs_in_runtime: bool
    input: JSON
    timeout_s: Optional[int] = None


@dataclass(frozen=True)
class ToolOutcome:
    status: Literal["OK", "ERROR", "TIMEOUT"]
    output: JSON

    stdout: Optional[str] = None
    stderr: Optional[str] = None

    artifacts: List[str] = None  # artifact_ids created by the tool (logs, reports, etc.)

    # billing (optional)
    cost_usd: Optional[float] = None
    unit_type: Optional[str] = None
    units: Optional[float] = None