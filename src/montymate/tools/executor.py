from __future__ import annotations

from typing import Protocol

from montymate.tools.models import ToolInvocation, ToolOutcome
from montymate.core.types import JSON


class RuntimeClient(Protocol):
    def call(self, *, action: str, payload: JSON) -> JSON:
        """
        Transport for sandbox/runtime execution.
        MVP can be local; later this can be REST to a sandbox, OpenHands-style.  [oai_citation:3‡OpenReview](https://openreview.net/pdf/95990590797cff8b93c33af989ecf4ac58bde9bb.pdf?utm_source=chatgpt.com)
        """
        ...


class ToolExecutor(Protocol):
    def execute(
        self,
        *,
        run_id: str,
        step_execution_id: str,
        inv: ToolInvocation,
    ) -> ToolOutcome:
        """
        Must:
        - execute the requested tool
        - record tool call + cost via your DataServices
        - return normalized outcome
        """
        ...