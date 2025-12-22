#spec_types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Mapping

class SpecStatus(str, Enum):
    DRAFT = "DRAFT"
    LOCKED = "LOCKED"
    # Add more later if needed (e.g. VALIDATED, NEEDS_WORK, etc.)


@dataclass(frozen=True, slots=True)
class Spec:
    status: SpecStatus = SpecStatus.DRAFT
    goal: str = ""
    functional_requirements: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    security_concerns: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    other_notes: str = ""

    @staticmethod
    def from_dict(d: Mapping[str, Any] |None) -> "Spec":
        d = dict(d or {})

        def as_str(x: Any) -> str:
            return x.strip() if isinstance(x, str) else ""

        def as_list(x: Any) -> list[str]:
            if not isinstance(x, list):
                return []
            out: list[str] = []
            for item in x:
                if item is None:
                    continue
                s = str(item).strip()
                if s:
                    out.append(s)
            return out
            
        raw_status = d.get("status", SpecStatus.DRAFT.value)
        try:
            status = raw_status if isinstance(raw_status, SpecStatus) else SpecStatus(str(raw_status))
        except Exception:
            status = SpecStatus.DRAFT
        

        return Spec(
            status=status,
            goal=as_str(d.get("goal")),
            functional_requirements=as_list(d.get("functional_requirements")),
            constraints=as_list(d.get("constraints")),
            security_concerns=as_list(d.get("security_concerns")),
            assumptions=as_list(d.get("assumptions")),
            other_notes=as_str(d.get("other_notes")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "goal": self.goal,
            "functional_requirements": list(self.functional_requirements),
            "constraints": list(self.constraints),
            "security_concerns": list(self.security_concerns),
            "assumptions": list(self.assumptions),
            "other_notes": self.other_notes,
        }