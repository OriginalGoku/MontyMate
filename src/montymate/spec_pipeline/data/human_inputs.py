# human_inpts.py
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

AnswerDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class HumanAnswer:
    """Single answer to a targeted question.

    UI simplification rule:
    - If `answer` is empty/whitespace -> treat as "decide for me".
    """

    question: str
    answer: str = ""

    @property
    def decide_for_me(self) -> bool:
        return self.answer.strip() == ""

    def to_dict(self) -> AnswerDict:
        # Persist decide_for_me too (helpful for debugging)
        return {
            "question": self.question,
            "answer": self.answer,
            "decide_for_me": self.decide_for_me,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "HumanAnswer":
        q = str(d.get("question") or "").strip()
        a = str(d.get("answer") or "")
        return HumanAnswer(question=q, answer=a)


@dataclass(frozen=True, slots=True)
class HumanAnswerBatch:
    """A batch of human answers for a single validation round."""

    round_no: int
    answers: list[HumanAnswer]

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": int(self.round_no),
            "answers": [a.to_dict() for a in self.answers],
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "HumanAnswerBatch":
        round_no = int(d.get("round") or 0)
        raw_answers = d.get("answers")

        answers: list[HumanAnswer] = []
        if isinstance(raw_answers, list):
            for item in raw_answers:
                if isinstance(item, Mapping):
                    # cast to dict-like mapping
                    answers.append(HumanAnswer.from_dict(item))  # type: ignore[arg-type]

        return HumanAnswerBatch(round_no=round_no, answers=answers)


def normalize_answers(
    questions: Iterable[str],
    raw_answers_by_index: Mapping[int, str] | None = None,
) -> list[HumanAnswer]:
    """Align answers to a question list.

    Missing indexes become empty answers => decide_for_me=True.
    """
    raw_answers_by_index = raw_answers_by_index or {}

    out: list[HumanAnswer] = []
    for i, q in enumerate(questions):
        out.append(
            HumanAnswer(
                question=str(q).strip(),
                answer=str(raw_answers_by_index.get(i, "") or ""),
            )
        )
    return out
