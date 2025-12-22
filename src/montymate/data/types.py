from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class PayloadMode(str, Enum):
    INLINE = "inline"
    ARTIFACT_REF = "artifact_ref"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class StoragePolicy:
    payload_mode: PayloadMode
    inline_max_bytes: int
    artifact_root: str  # e.g. ".ai_montymate/"

    @staticmethod
    def from_decision_record(dr: Mapping[str, Any]) -> "StoragePolicy":
        sp = dr.get("storage_policy") or {}
        mode = PayloadMode(sp.get("payload_mode", "hybrid"))
        inline_max_bytes = int(sp.get("inline_max_bytes", 32768))
        artifact_root = str(sp.get("artifact_root", ".ai_montymate/"))
        return StoragePolicy(payload_mode=mode, inline_max_bytes=inline_max_bytes, artifact_root=artifact_root)


@dataclass(frozen=True)
class RunContext:
    run_id: str
    repo_root: str
    profile_id: str
    workflow_id: str
    policy_id: str
    storage_policy: StoragePolicy


@dataclass(frozen=True)
class StepContext:
    step_execution_id: str
    step_id: str
    attempt_no: int


@dataclass(frozen=True)
class ArtifactWriteResult:
    artifact_id: str
    relpath: str
    sha256: str
    bytes: int


def summarize_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n… (truncated; total_chars={len(text)})"


def should_inline(policy: StoragePolicy, payload_bytes: int) -> bool:
    if policy.payload_mode == PayloadMode.INLINE:
        return True
    if policy.payload_mode == PayloadMode.ARTIFACT_REF:
        return False
    # HYBRID
    return payload_bytes <= policy.inline_max_bytes