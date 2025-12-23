from __future__ import annotations

from dataclasses import dataclass

JsonDict = dict[str, object]


# src/montymate/spec_pipeline/data/tool_types.py


def _as_str_list(x: object) -> list[str]:
    """Converts a JSON value into list[str] with basic cleanup."""
    if not isinstance(x, list):
        return []
    out: list[str] = []
    for item in x:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


@dataclass(frozen=True, slots=True)
class SpecCriticReport:
    """Structured output produced by SpecCriticTool."""
    passed: bool
    issues: list[str]
    targeted_questions: list[str]

    def to_dict(self) -> JsonDict:
        return {
            "passed": bool(self.passed),
            "issues": list(self.issues),
            "targeted_questions": list(self.targeted_questions),
        }

    @staticmethod
    def from_dict(d: dict[str, object]) -> "SpecCriticReport":
        passed_raw = d.get("passed")
        passed = bool(passed_raw) if isinstance(passed_raw, (bool, int)) else False

        return SpecCriticReport(
            passed=passed,
            issues=_as_str_list(d.get("issues")),
            targeted_questions=_as_str_list(d.get("targeted_questions")),
        )
