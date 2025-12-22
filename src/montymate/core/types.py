from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


JSON = Dict[str, Any]

RunStatus = Literal["RUNNING", "PAUSED", "SUCCEEDED", "FAILED", "CANCELED"]
StepStatus = Literal["RUNNING", "WAITING_HUMAN", "SUCCEEDED", "FAILED", "SKIPPED"]


@dataclass(frozen=True)
class ResourceRef:
    """
    Reference to an exact resolved resource used for a run.
    sha256 should be computed from the resolved bytes so provenance is stable.
    """
    id: str            # e.g., "python_unified"
    version: int       # e.g., 2
    sha256: str        # hash of resolved resource content


@dataclass(frozen=True)
class ResolvedRefs:
    workflow: ResourceRef
    profile: ResourceRef
    policy: ResourceRef
    llm_routing: ResourceRef
    tool_registry: ResourceRef
    schemas: Dict[str, ResourceRef]  # name -> ResourceRef (module_spec, decision_record, guard_object...)


@dataclass(frozen=True)
class RunIdentity:
    repo_root: str
    run_id: str

    profile_selector: str   # "fastapi_service@2"
    workflow_selector: str  # "python_unified@2"
    policy_selector: str    # "default_fastapi_policy@2"