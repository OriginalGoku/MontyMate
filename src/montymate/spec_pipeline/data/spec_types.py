# spec_types.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum


class SpecStatus(str, Enum):
    DRAFT = "DRAFT"
    LOCKED = "LOCKED"


@dataclass(frozen=True, slots=True)
class Spec:
    status: SpecStatus = SpecStatus.DRAFT
    goal: str = ""
    functional_requirements: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    security_concerns: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    non_goals: list[str] = field(default_factory=list)
    success_metrics: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)

    other_notes: str = ""

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        """Returns the canonical Spec field names derived from the dataclass definition."""
        return tuple(f.name for f in fields(cls))

    @staticmethod
    def from_dict(d: Mapping[str, object] | None) -> "Spec":
        d2: dict[str, object] = dict(d or {})

        def as_str(x: object) -> str:
            return x.strip() if isinstance(x, str) else ""

        def as_list(x: object) -> list[str]:
            if not isinstance(x, (list, tuple)):
                return []
            out: list[str] = []
            for item in x:
                if item is None:
                    continue
                s = str(item).strip()
                if s:
                    out.append(s)
            return out

        def as_other_notes(x: object) -> str:
            """Normalizes other_notes into a single string."""
            if isinstance(x, str):
                return x.strip()
            if isinstance(x, (list, tuple)):
                lines: list[str] = []
                for item in x:
                    if item is None:
                        continue
                    s = str(item).strip()
                    if s:
                        lines.append(s)
                return "\n".join(lines).strip()
            return ""

        raw_status = d2.get("status", SpecStatus.DRAFT.value)
        try:
            status = raw_status if isinstance(raw_status, SpecStatus) else SpecStatus(str(raw_status).upper())
        except Exception:
            status = SpecStatus.DRAFT

        return Spec(
            status=status,
            goal=as_str(d2.get("goal")),
            functional_requirements=as_list(d2.get("functional_requirements")),
            constraints=as_list(d2.get("constraints")),
            security_concerns=as_list(d2.get("security_concerns")),
            assumptions=as_list(d2.get("assumptions")),
            non_goals=as_list(d2.get("non_goals")),
            success_metrics=as_list(d2.get("success_metrics")),
            tradeoffs=as_list(d2.get("tradeoffs")),
            other_notes=as_other_notes(d2.get("other_notes")),
        )

    def to_dict(self) -> dict[str, object]:
        """Converts the spec into a JSON/YAML-serializable mapping."""
        return {
            "status": self.status.value,
            "goal": self.goal,
            "functional_requirements": list(self.functional_requirements),
            "constraints": list(self.constraints),
            "security_concerns": list(self.security_concerns),
            "assumptions": list(self.assumptions),
            "non_goals": list(self.non_goals),
            "success_metrics": list(self.success_metrics),
            "tradeoffs": list(self.tradeoffs),
            "other_notes": self.other_notes,
        }