from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from montymate.core.types import JSON
from montymate.core.types import RunIdentity
from montymate.config.models import ResolvedRunConfig
from montymate.engine.state import RunHandle


class WorkflowEngine(Protocol):
    def start_run(
        self,
        *,
        identity: RunIdentity,
        config: ResolvedRunConfig,
        initial_inputs: JSON,
    ) -> RunHandle:
        """Create DB run row, then run until PAUSED or terminal."""
        ...

    def resume_run(
        self,
        *,
        run_id: str,
        human_inputs: Optional[JSON] = None,
    ) -> RunHandle:
        """Resume from last paused gate with given human inputs."""
        ...

    def run_until_blocked_or_done(self, *, run_id: str) -> RunHandle:
        """Core driver loop: keep executing while runnable; stop on PAUSED or terminal."""
        ...