from __future__ import annotations

from typing import Protocol

from montymate.core.types import JSON
from montymate.engine.state import GuardContext


class GuardEvaluator(Protocol):
    def eval(self, guard_obj: JSON, ctx: GuardContext) -> bool:
        """
        Evaluate a structured guard object.
        Must not use string eval / dynamic code execution.
        """
        ...