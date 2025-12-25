# src/montymate/spec_pipeline/data/human_inputs.py
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


AnswerDict = dict[str, object]


@dataclass(frozen=True, slots=True)
class QuestionAnswer:
    """Represents one answer to a targeted question.

    The instance stores provenance explicitly:
    - decide_for_me indicates that the user delegated the decision.
    - is_llm_generated indicates that the final answer text was produced by an LLM.

    The parsing behavior remains tolerant:
    - If decide_for_me is missing, it is inferred from a blank answer.
    - If is_llm_generated is missing, it defaults to False.
    """

    question: str
    answer: str = ""
    decide_for_me: bool = False
    is_llm_generated: bool = False

    def to_dict(self) -> AnswerDict:
        """Converts the answer into a JSON-serializable mapping."""
        return {
            "question": self.question,
            "answer": self.answer,
            "decide_for_me": bool(self.decide_for_me),
            "is_llm_generated": bool(self.is_llm_generated),
        }

    @staticmethod
    def from_dict(d: Mapping[str, object]) -> "QuestionAnswer":
        """Parses a QuestionAnswer from a dict-like mapping."""
        q = str(d.get("question") or "").strip()
        a = str(d.get("answer") or "")

        raw_dfm = d.get("decide_for_me")
        if isinstance(raw_dfm, bool):
            decide_for_me = raw_dfm
        else:
            decide_for_me = (a.strip() == "")

        # Keep the data consistent: blank answer implies decide_for_me.
        if a.strip() == "":
            decide_for_me = True

        raw_llm = d.get("is_llm_generated")
        is_llm_generated = raw_llm if isinstance(raw_llm, bool) else False

        return QuestionAnswer(
            question=q,
            answer=a,
            decide_for_me=decide_for_me,
            is_llm_generated=is_llm_generated,
        )


@dataclass(frozen=True, slots=True)
class ClarificationBatch:
    """A batch of QuestionAnswers for a single validation round."""

    round_no: int
    answers: list[QuestionAnswer]

    def to_dict(self) -> dict[str, object]:
        """Converts the batch into a JSON-serializable mapping."""
        return {
            "round": int(self.round_no),
            "answers": [a.to_dict() for a in self.answers],
        }

    @staticmethod
    def from_dict(d: Mapping[str, object]) -> "ClarificationBatch":
        """Parses a ClarificationBatch from a dict-like mapping."""
        raw_round = d.get("round")
        if isinstance(raw_round, bool):
            round_no = 0
        elif isinstance(raw_round, int):
            round_no = raw_round
        elif isinstance(raw_round, str) and raw_round.strip().isdigit():
            round_no = int(raw_round.strip())
        else:
            round_no = 0

        raw_answers = d.get("answers")
        answers: list[QuestionAnswer] = []
        if isinstance(raw_answers, list):
            for item in raw_answers:
                if isinstance(item, dict):
                    # dict is compatible with Mapping[str, object] at runtime
                    answers.append(QuestionAnswer.from_dict(item))

        return ClarificationBatch(round_no=int(round_no), answers=answers)


def normalize_answers(
    questions: Iterable[str],
    raw_answers_by_index: Mapping[int, str] | None = None,
) -> list[QuestionAnswer]:
    """Aligns answers to a question list.

    Missing indexes become blank answers and are stored as decide_for_me=True.
    """
    raw_answers_by_index = raw_answers_by_index or {}

    out: list[QuestionAnswer] = []
    for i, q in enumerate(questions):
        qq = str(q).strip()
        if not qq:
            continue

        ans = str(raw_answers_by_index.get(i, "") or "")
        out.append(
            QuestionAnswer(
                question=qq,
                answer=ans,
                decide_for_me=(ans.strip() == ""),
                is_llm_generated=False,
            )
        )

    return out