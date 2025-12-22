from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from montymate.core.types import JSON, ResolvedRefs


@dataclass(frozen=True)
class WorkflowGraph:
    """
    Parsed workflow YAML.
    For MVP, store raw dict + convenience lookups.
    """

    id: str
    version: int
    entry: str
    raw: JSON

    steps: Mapping[str, JSON]
    edges: List[JSON]  # keep as list for now


@dataclass(frozen=True)
class StepGroupCatalog:
    version: int
    raw: JSON

    # group_id -> definition
    groups: Mapping[str, JSON]


@dataclass(frozen=True)
class ProfileConfig:
    id: str
    version: int
    raw: JSON

    workflow_default: str  # e.g. "python_unified@2"
    policy_default: str  # e.g. "default_fastapi_policy@2"
    tool_bindings: Dict[str, str]  # alias -> concrete tool name
    prompt_bindings: Dict[str, str]  # role -> prompt template id
    allowed_step_groups: List[str]


@dataclass(frozen=True)
class PolicyConfig:
    id: str
    version: int
    raw: JSON


@dataclass(frozen=True)
class ToolRegistry:
    version: int
    raw: JSON

    # concrete tool name -> tool metadata dict (tool_type, runs_in_runtime, costable, etc.)
    tools: Mapping[str, JSON]


@dataclass(frozen=True)
class LLMRoutingConfig:
    version: int
    raw: JSON

    model_catalog: List[JSON]
    roles: Mapping[str, JSON]


@dataclass(frozen=True)
class ResolvedRunConfig:
    """
    Fully resolved configuration needed to execute a run deterministically.
    """

    refs: ResolvedRefs

    workflow: WorkflowGraph
    step_groups_catalog: StepGroupCatalog
    profile: ProfileConfig
    policy: PolicyConfig
    tool_registry: ToolRegistry
    llm_routing: LLMRoutingConfig

    schemas: Dict[str, Any]  # optional: parsed JSONSchema/YAML schema dicts

    # convenience: already-bound
    tool_bindings: Dict[str, str]  # alias -> concrete tool name
    prompt_bindings: Dict[str, str]  # role -> prompt template id
