from __future__ import annotations

from typing import Optional, Protocol

from montymate.gates.models import GateDecision, GateRequest


class HumanGateManager(Protocol):
    def request_gate(self, req: GateRequest) -> GateRequest:
        """Persist the request + emit events; return the stored request (enriched if needed)."""
        ...

    def decide(self, decision: GateDecision) -> None:
        """Persist the decision + emit events; unblock (or block) the run."""
        ...

    def get_pending_gate(self, run_id: str) -> Optional[GateRequest]:
        """If run is paused, return the pending gate request."""
        ...
