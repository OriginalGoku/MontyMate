# src/montymate/spec_pipeline/data/run_state.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping


class RunStatus(str, Enum):
    """Represents the high-level lifecycle state of a spec-pipeline run.

    The status values are designed for a coordinator that:
    - reads run_state.json and other artifacts
    - performs at most one step (or runs until blocked)
    - persists a new RunState after each step

    Status meanings:
    - NEW: The run is initialized, but no authoring has occurred yet.
    - AUTHOR_DRAFTED: A spec draft or revision has been produced and persisted.
    - WAITING_FOR_HUMAN: A critic report exists and human answers are required to proceed.
    - DONE: The critic has passed and the run is complete.
    - ERROR: A malformed artifact or runtime failure has prevented progress.
    """

    NEW = "NEW"
    AUTHOR_DRAFTED = "AUTHOR_DRAFTED"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    DONE = "DONE"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class RunState:
    """Stores the minimal, persisted control-plane state for a run.

    The run state is intended to be serialized as run_state.json and used to
    pause/resume runs deterministically.

    Stored fields:
    - run_id: Unique identifier for the run directory under .ai_module_factory/runs/.
    - status: Current lifecycle status (RunStatus).
    - user_prompt: The original user prompt driving the run. This enables resume without
      re-supplying input externally.
    - round_no: Current human-feedback round number (0 before the first critic failure).
      The round number is intended to increment when the critic fails.
    - last_error: Optional human-readable error message when status=ERROR.
    - step: Optional string used for trace/debug (e.g., "author_draft", "critic_round_1").
    """

    run_id: str
    status: RunStatus
    user_prompt: str
    round_no: int = 0
    last_error: str | None = None
    step: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Converts the run state into a JSON-serializable mapping."""
        out: dict[str, object] = {
            "run_id": self.run_id,
            "status": self.status.value,
            "user_prompt": self.user_prompt,
            "round_no": int(self.round_no),
        }
        if self.last_error is not None:
            out["last_error"] = self.last_error
        if self.step is not None:
            out["step"] = self.step
        return out

    @staticmethod
    def from_dict(d: Mapping[str, object]) -> "RunState":
        """Parses a RunState from a dict-like mapping.

        The parser is intentionally strict for core fields so that malformed
        run_state.json can be detected by the caller.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        run_id = d.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_state.run_id must be a non-empty string")

        raw_status = d.get("status")
        if not isinstance(raw_status, str) or not raw_status.strip():
            raise ValueError("run_state.status must be a non-empty string")
        try:
            status = RunStatus(raw_status.strip())
        except Exception as e:
            raise ValueError(f"run_state.status is invalid: {raw_status!r}") from e

        user_prompt = d.get("user_prompt")
        if not isinstance(user_prompt, str):
            raise ValueError("run_state.user_prompt must be a string")

        raw_round = d.get("round_no", 0)
        if isinstance(raw_round, bool):
            raise ValueError("run_state.round_no must be an int (bool is not allowed)")
        if isinstance(raw_round, int):
            round_no = raw_round
        elif isinstance(raw_round, str) and raw_round.strip().isdigit():
            round_no = int(raw_round.strip())
        else:
            raise ValueError("run_state.round_no must be an int or digit string")

        if round_no < 0:
            raise ValueError("run_state.round_no must be >= 0")

        last_error = d.get("last_error")
        if last_error is not None and not isinstance(last_error, str):
            raise ValueError("run_state.last_error must be a string or null")

        step = d.get("step")
        if step is not None and not isinstance(step, str):
            raise ValueError("run_state.step must be a string or null")

        return RunState(
            run_id=run_id.strip(),
            status=status,
            user_prompt=user_prompt,
            round_no=round_no,
            last_error=last_error,
            step=step,
        )