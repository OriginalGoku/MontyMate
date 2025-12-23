# src/montymate/spec_pipeline/data/run_state.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class RunState:
    """Represents the minimal persisted state of a spec-pipeline run.

    The state is intended to support stop/resume flows.
    The state should remain small and stable, and should not embed large artifacts.

    Fields:
    - run_id: Identifies the run folder name.
    - user_prompt: Captures the initial user intent for auditing and resuming.
    - current_round: Tracks the latest completed critic round number (0 means none).
    - status: Represents a coarse run lifecycle marker (e.g., "IN_PROGRESS", "DONE").
    - created_at_iso: Stores a UTC timestamp for run creation.
    """

    run_id: str
    user_prompt: str
    current_round: int = 0
    status: str = "IN_PROGRESS"
    created_at_iso: str = ""

    @staticmethod
    def new(*, run_id: str, user_prompt: str) -> "RunState":
        ts = datetime.now(timezone.utc).isoformat()
        return RunState(run_id=run_id, user_prompt=user_prompt, created_at_iso=ts)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "user_prompt": self.user_prompt,
            "current_round": int(self.current_round),
            "status": self.status,
            "created_at_iso": self.created_at_iso,
        }

    @staticmethod
    def from_dict(d: Mapping[str, object]) -> "RunState":
        run_id = str(d.get("run_id") or "").strip()
        user_prompt = str(d.get("user_prompt") or "")
        current_round_raw = d.get("current_round")
        status = str(d.get("status") or "IN_PROGRESS")
        created_at_iso = str(d.get("created_at_iso") or "")

        current_round = (
            int(current_round_raw) if isinstance(current_round_raw, int) else 0
        )
        return RunState(
            run_id=run_id,
            user_prompt=user_prompt,
            current_round=current_round,
            status=status,
            created_at_iso=created_at_iso,
        )
