from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from typing import Optional, Protocol, Sequence, Tuple

import yaml  # PyYAML

from montymate.config.models import (
    LLMRoutingConfig,
    PolicyConfig,
    ProfileConfig,
    ResolvedRunConfig,
    StepGroupCatalog,
    ToolRegistry,
    WorkflowGraph,
)
from montymate.core.types import ResolvedRefs, ResourceRef


@dataclass(frozen=True)
class OverrideSources:
    """
    Precedence idea (you can implement later):
    - user_dir: ~/.config/montymate/resources/...
    - repo_dir:  .montymate/resources/...
    - explicit_files: CLI-provided override YAML files
    """

    user_dir: Optional[str] = None
    repo_dir: Optional[str] = None
    explicit_files: Sequence[str] = ()


class RunConfigResolver(Protocol):
    def resolve(
        self,
        *,
        repo_root: str,
        profile_selector: str,
        overrides: Optional[OverrideSources] = None,
    ) -> ResolvedRunConfig: ...


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _parse_selector(selector: str) -> Tuple[str, int]:
    # "fastapi_service@2" -> ("fastapi_service", 2)
    if "@" not in selector:
        raise ValueError(f"Invalid selector '{selector}', expected 'id@version'")
    rid, ver = selector.split("@", 1)
    return rid, int(ver)


@dataclass
class DefaultRunConfigResolver:
    """
    MVP resolver that loads *shipped defaults* from package resources.

    Use importlib.resources so this works when installed from wheels/zips.  [oai_citation:1‡Python documentation](https://docs.python.org/3/library/importlib.resources.html?utm_source=chatgpt.com)
    """

    package: str = "montymate"

    def resolve(
        self,
        *,
        repo_root: str,
        profile_selector: str,
        overrides: Optional[OverrideSources] = None,
    ) -> ResolvedRunConfig:
        profile_id, profile_ver = _parse_selector(profile_selector)

        profile_raw, profile_ref = self._load_yaml_with_ref(
            f"resources/profiles/{profile_id}_v{profile_ver}.yaml",
            profile_id,
            profile_ver,
        )

        workflow_sel = profile_raw["workflow"]["default"]  # e.g. python_unified@2
        policy_sel = profile_raw["policy"]["default"]  # e.g. default_fastapi_policy@2

        workflow_id, workflow_ver = _parse_selector(workflow_sel)
        policy_id, policy_ver = _parse_selector(policy_sel)

        workflow_raw, workflow_ref = self._load_yaml_with_ref(
            f"resources/workflows/{workflow_id}_v{workflow_ver}.yaml",
            workflow_id,
            workflow_ver,
        )
        catalog_raw, catalog_ref = self._load_yaml_with_ref(
            "resources/workflows/step_groups_catalog_v1.yaml", "step_groups_catalog", 1
        )

        policy_raw, policy_ref = self._load_yaml_with_ref(
            f"resources/policies/{policy_id}_v{policy_ver}.yaml", policy_id, policy_ver
        )
        tools_raw, tools_ref = self._load_yaml_with_ref(
            "resources/tools/tool_registry_v2.yaml", "tool_registry", 2
        )
        llm_raw, llm_ref = self._load_yaml_with_ref(
            "resources/configs/llm_routing_v2.yaml", "llm_routing", 2
        )

        # schemas can be optional in MVP
        schemas = {}
        schema_refs = {}

        refs = ResolvedRefs(
            workflow=workflow_ref,
            profile=profile_ref,
            policy=policy_ref,
            llm_routing=llm_ref,
            tool_registry=tools_ref,
            schemas=schema_refs,
        )

        workflow = WorkflowGraph(
            id=workflow_raw["workflow"]["id"],
            version=int(workflow_raw["workflow"]["version"]),
            entry=workflow_raw["workflow"]["entry"],
            raw=workflow_raw,
            steps=workflow_raw.get("steps", {}),
            edges=workflow_raw.get("edges", []),
        )

        catalog = StepGroupCatalog(
            version=int(catalog_raw["step_groups_catalog"]["version"]),
            raw=catalog_raw,
            groups=catalog_raw["step_groups_catalog"].get("groups", {}),
        )

        profile = ProfileConfig(
            id=profile_raw["profile"]["id"],
            version=int(profile_raw["profile"]["version"]),
            raw=profile_raw,
            workflow_default=workflow_sel,
            policy_default=policy_sel,
            tool_bindings=profile_raw.get("bindings", {}).get("tools", {}),
            prompt_bindings=profile_raw.get("bindings", {}).get("prompts", {}),
            allowed_step_groups=profile_raw.get("allowed_step_groups", []),
        )

        policy = PolicyConfig(id=policy_id, version=policy_ver, raw=policy_raw)

        tool_registry = ToolRegistry(
            version=int(tools_raw["tool_registry"]["version"]),
            raw=tools_raw,
            tools=tools_raw.get("tools", {}),
        )

        llm_routing = LLMRoutingConfig(
            version=int(llm_raw["llm_routing"]["version"]),
            raw=llm_raw,
            model_catalog=llm_raw["llm_routing"].get("model_catalog", []),
            roles=llm_raw["llm_routing"].get("roles", {}),
        )

        return ResolvedRunConfig(
            refs=refs,
            workflow=workflow,
            step_groups_catalog=catalog,
            profile=profile,
            policy=policy,
            tool_registry=tool_registry,
            llm_routing=llm_routing,
            schemas=schemas,
            tool_bindings=profile.tool_bindings,
            prompt_bindings=profile.prompt_bindings,
        )

    def _load_yaml_with_ref(
        self, relpath: str, rid: str, ver: int
    ) -> tuple[dict, ResourceRef]:
        """
        Read resource bytes from the installed package and compute hash for provenance.
        """
        data = resources.files(self.package).joinpath(relpath).read_bytes()
        sha = _sha256_bytes(data)
        parsed = yaml.safe_load(data.decode("utf-8"))
        return parsed, ResourceRef(id=rid, version=ver, sha256=sha)
