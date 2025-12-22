from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class ActorType(str, Enum):
    system = "system"
    human = "human"
    llm = "llm"
    tool = "tool"


class Severity(str, Enum):
    debug = "debug"
    info = "info"
    warn = "warn"
    error = "error"


class EventType(str, Enum):
    """
    Stable event taxonomy.
    Keep these names durable; prefer adding new events over renaming.
    """

    # Run lifecycle
    run_started = "run.started"
    run_paused = "run.paused"
    run_resumed = "run.resumed"
    run_succeeded = "run.succeeded"
    run_failed = "run.failed"
    run_canceled = "run.canceled"

    # Step lifecycle
    step_started = "step.started"
    step_finished = "step.finished"
    step_skipped = "step.skipped"
    step_attempt_failed = "step.attempt_failed"

    # Human gate lifecycle (HITL)
    gate_requested = "gate.requested"
    gate_acknowledged = "gate.acknowledged"
    gate_approved = "gate.approved"
    gate_blocked = "gate.blocked"
    gate_timeout = "gate.timeout"

    # Spec pipeline (v2)
    spec_drafted = "spec.drafted"
    spec_validation_passed = "spec.validation_passed"
    spec_validation_failed = "spec.validation_failed"
    spec_locked = "spec.locked"

    # Policy / classification
    policy_decision_record_created = "policy.decision_record_created"
    risk_scored = "policy.risk_scored"

    # Tooling + model calls
    llm_call = "llm.call"
    tool_call = "tool.call"

    # Patch / ChangeSet
    patch_generated = "patch.generated"
    patch_applied = "patch.applied"
    changeset_started = "changeset.started"
    changeset_checkpoint = "changeset.checkpoint"
    changeset_completed = "changeset.completed"

    # Verification gates
    gates_started = "gates.started"
    gates_finished = "gates.finished"
    gate_result = "gates.result"

    # Artifacts + exports
    artifact_written = "artifact.written"
    prompt_bundle_exported = "prompt_bundle.exported"
    run_summary_written = "run_summary.written"

    # Provenance
    provenance_manifest_written = "provenance.manifest_written"
    provenance_committed = "provenance.committed"


@dataclass(frozen=True)
class EventSpec:
    """
    Minimal “shape contract” for payloads.

    We keep this intentionally lightweight:
    - required keys prevent taxonomy drift
    - optional keys allow evolution
    """
    required_keys: Sequence[str]
    optional_keys: Sequence[str] = ()
    allow_extra_keys: bool = True


# Payload contracts:
# - Keep payloads "event-stream friendly": small, structured, and link out to artifacts for bulk content.
# - The EventLogger can store large payloads as artifact refs according to storage_policy.
EVENT_SPECS: Mapping[EventType, EventSpec] = {
    # Run
    EventType.run_started: EventSpec(required_keys=("run_id", "profile_id", "workflow_id", "policy_id")),
    EventType.run_paused: EventSpec(required_keys=("run_id", "reason")),
    EventType.run_resumed: EventSpec(required_keys=("run_id",)),
    EventType.run_succeeded: EventSpec(required_keys=("run_id",)),
    EventType.run_failed: EventSpec(required_keys=("run_id", "error_type", "error_message")),
    EventType.run_canceled: EventSpec(required_keys=("run_id", "reason")),

    # Step
    EventType.step_started: EventSpec(required_keys=("run_id", "step_id", "step_execution_id", "attempt_no")),
    EventType.step_finished: EventSpec(required_keys=("run_id", "step_id", "step_execution_id", "attempt_no", "status")),
    EventType.step_skipped: EventSpec(required_keys=("run_id", "step_id", "reason")),
    EventType.step_attempt_failed: EventSpec(required_keys=("run_id", "step_id", "step_execution_id", "attempt_no", "error_type", "error_message")),

    # Human gate
    EventType.gate_requested: EventSpec(required_keys=("run_id", "gate_name", "mode", "step_id")),
    EventType.gate_acknowledged: EventSpec(required_keys=("run_id", "gate_name", "human_actor")),
    EventType.gate_approved: EventSpec(required_keys=("run_id", "gate_name", "human_actor")),
    EventType.gate_blocked: EventSpec(required_keys=("run_id", "gate_name", "human_actor", "reason")),
    EventType.gate_timeout: EventSpec(required_keys=("run_id", "gate_name", "mode", "ack_window_minutes")),

    # Spec pipeline
    EventType.spec_drafted: EventSpec(required_keys=("run_id", "artifact_id")),
    EventType.spec_validation_passed: EventSpec(required_keys=("run_id", "validator_model", "round_no")),
    EventType.spec_validation_failed: EventSpec(required_keys=("run_id", "validator_model", "round_no", "issue_count")),
    EventType.spec_locked: EventSpec(required_keys=("run_id", "human_gate_mode")),

    # Policy
    EventType.policy_decision_record_created: EventSpec(required_keys=("run_id", "artifact_id", "change_class", "risk_level")),
    EventType.risk_scored: EventSpec(required_keys=("run_id", "risk_score", "risk_level")),

    # Calls
    EventType.llm_call: EventSpec(required_keys=("run_id", "llm_call_id", "role", "provider", "model", "status")),
    EventType.tool_call: EventSpec(required_keys=("run_id", "tool_call_id", "tool_name", "tool_type", "status", "runs_in_runtime")),

    # Patch / ChangeSet
    EventType.patch_generated: EventSpec(required_keys=("run_id", "patch_artifact_id", "patch_kind")),
    EventType.patch_applied: EventSpec(required_keys=("run_id", "commit_sha")),
    EventType.changeset_started: EventSpec(required_keys=("run_id", "mode")),
    EventType.changeset_checkpoint: EventSpec(required_keys=("run_id", "checkpoint_no", "patch_count")),
    EventType.changeset_completed: EventSpec(required_keys=("run_id", "patch_count")),

    # Gates
    EventType.gates_started: EventSpec(required_keys=("run_id",)),
    EventType.gates_finished: EventSpec(required_keys=("run_id", "passed")),
    EventType.gate_result: EventSpec(required_keys=("run_id", "gate_name", "passed")),

    # Artifacts
    EventType.artifact_written: EventSpec(required_keys=("run_id", "artifact_id", "artifact_type", "relpath", "sha256", "bytes")),
    EventType.prompt_bundle_exported: EventSpec(required_keys=("run_id", "artifact_id")),
    EventType.run_summary_written: EventSpec(required_keys=("run_id", "artifact_id")),

    # Provenance
    EventType.provenance_manifest_written: EventSpec(required_keys=("run_id", "artifact_id", "commit_path")),
    EventType.provenance_committed: EventSpec(required_keys=("run_id", "commit_sha", "commit_path")),
}


def is_known_event_type(value: str) -> bool:
    try:
        EventType(value)
        return True
    except ValueError:
        return False