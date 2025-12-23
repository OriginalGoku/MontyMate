# spec_types.py
from __future__ import annotations

from collections.abc import Mapping as AbcMapping
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

        def as_other_notes(x: object) -> str:
            # Accepts either a string or a list/tuple of strings and normalizes to a single string.
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
            other_notes=as_other_notes(d2.get("other_notes")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "goal": self.goal,
            "functional_requirements": list(self.functional_requirements),
            "constraints": list(self.constraints),
            "security_concerns": list(self.security_concerns),
            "assumptions": list(self.assumptions),
            "other_notes": self.other_notes,
        }