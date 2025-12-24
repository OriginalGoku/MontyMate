# src/montymate/spec_pipeline/data/context.py
# src/montymate/spec_pipeline/data/context.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class ToolContext:
    """
    Context object passed to every tool call.

    What it does:
    - Carries run identity (`run_id`) so tool outputs and errors can be tagged to a single run.
    - Carries pipeline position (`step`) so artifacts and failures can be attributed to the stage that produced them.
    - Carries `project_root` so paths can be resolved consistently for diagnostics and future expansion.

    Design rule:
    - The context is treated as read-only. Step transitions are represented by creating a new context via `with_step(...)`.
    """

    project_root: Path
    run_id: str
    step: str

    # Optional
    meta: dict[str, JsonValue] | None = None

    def with_step(self, step: str) -> "ToolContext":
        """Returns a new context for the same run with an updated step label."""
        return ToolContext(
            project_root=self.project_root,
            run_id=self.run_id,
            step=step,
            meta=self.meta,
        )
