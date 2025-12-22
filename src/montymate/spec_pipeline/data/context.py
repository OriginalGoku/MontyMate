# src/montymate/spec_pipeline/data/context.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolContext:
    """
    Context object passed to every tool call.

    What it does (and why it exists):
    - Carries *run-level identity* (`run_id`) so every tool can tag its outputs
      (logs, persisted files, error reports) with the run being processed.
    - Carries the *current pipeline step* (`step`) so tool outputs can include
      where they were produced (e.g. "compose", "critic", "refine", "edit").
      This makes debugging much easier because you can correlate artifacts and
      failures to the exact stage that produced them.
    - Carries `project_root` so tools (or helpers) can resolve filesystem paths
      consistently if needed. Most tools shouldn't write files directly, but
      having the root available is useful for diagnostics or future expansion.

    Design rule:
    - Tools should treat ctx as read-only. If you need a new step, create a new
      context via `with_step(...)` rather than mutating anything.
    """

    project_root: Path
    run_id: str
    step: str

    # Optional free-form metadata for debugging or experiments.
    # Keep this small and only use when needed.
    meta: dict[str, Any] | None = None

    def with_step(self, step: str) -> "ToolContext":
        """Return a new context for the same run but a different step."""
        return ToolContext(
            project_root=self.project_root,
            run_id=self.run_id,
            step=step,
            meta=self.meta,
        )
