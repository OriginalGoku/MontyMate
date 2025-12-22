# src/montymate/spec_pipeline/data/tool_types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class TargetedQuestion:
    text: str

    def to_dict(self) -> JsonDict:
        return {"text": self.text}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TargetedQuestion":
        return TargetedQuestion(text=str(d.get("text") or "").strip())


@dataclass(frozen=True, slots=True)
class SpecCriticReport:
    passed: bool
    issues: list[str]
    targeted_questions: list[TargetedQuestion]

    def to_dict(self) -> JsonDict:
        return {
            "passed": bool(self.passed),
            "issues": list(self.issues),
            "targeted_questions": [q.to_dict() for q in self.targeted_questions],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SpecCriticReport":
        issues_raw = d.get("issues")
        tq_raw = d.get("targeted_questions")

        issues: list[str] = []
        if isinstance(issues_raw, list):
            issues = [str(x).strip() for x in issues_raw if str(x).strip()]

        targeted_questions: list[TargetedQuestion] = []
        if isinstance(tq_raw, list):
            for item in tq_raw:
                if isinstance(item, dict):
                    q = TargetedQuestion.from_dict(item)
                    if q.text:
                        targeted_questions.append(q)
                elif isinstance(item, str) and item.strip():
                    targeted_questions.append(TargetedQuestion(text=item.strip()))

        return SpecCriticReport(
            passed=bool(d.get("passed", False)),
            issues=issues,
            targeted_questions=targeted_questions,
        )