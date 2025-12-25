# src/montymate/spec_pipeline/tools/spec_answerer.py
from __future__ import annotations

from collections.abc import Mapping

from ..data.component_metadata import ComponentMetadata, OutputFormat
from ..data.human_inputs import QuestionAnswer, ClarificationBatch
from ..data.spec_types import Spec
from ..errors import ToolError
from ..llm.llm_client import LLMClient
from ..llm.prompts.prompts import SpecAnswererPrompt
from .base import BaseTool


def _as_mapping(x: object) -> Mapping[str, object] | None:
    return x if isinstance(x, Mapping) else None


def _as_str_list(x: object) -> list[str]:
    if not isinstance(x, list):
        return []
    out: list[str] = []
    for item in x:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _answers_to_question_payload(answers: list[QuestionAnswer]) -> list[dict[str, object]]:
    """Converts answers into the prompt-facing payload shape."""
    payload: list[dict[str, object]] = []
    for a in answers:
        q = str(a.question).strip()
        if not q:
            continue
        payload.append(
            {
                "question": q,
                "answer": str(a.answer),
                "decide_for_me": bool(a.decide_for_me),
                "is_llm_generated": bool(a.is_llm_generated),
            }
        )
    return payload


def _parse_answers_input(x: object) -> list[QuestionAnswer] | None:
    """Accepts QuestionAnswerBatch | list[QuestionAnswer] | list[dict] and returns answers."""
    if isinstance(x, ClarificationBatch):
        return list(x.answers)

    if isinstance(x, list):
        out: list[QuestionAnswer] = []
        for item in x:
            if isinstance(item, QuestionAnswer):
                out.append(item)
            elif isinstance(item, Mapping):
                out.append(
                    QuestionAnswer.from_dict({str(k): item[k] for k in item.keys()})
                )
        return out

    return None
    

def _parse_qa_blocks(text: str) -> list[tuple[str, str, str]]:
    """Parses SpecAnswerer strict TEXT output into (question, answer, reason) tuples.

    Expected format:
      Q: ...
      A: ...
      R: ...

    Blocks are separated by at least one blank line.
    Raises ToolError if the format is malformed.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    lines = [ln.rstrip("\n") for ln in raw.splitlines()]

    blocks: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if ln.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(ln)
    if cur:
        blocks.append(cur)

    out: list[tuple[str, str, str]] = []
    for b in blocks:
        if len(b) != 3:
            raise ToolError(
                "SpecAnswerer returned malformed text (each block must be exactly 3 lines: Q/A/R).",
                data={"block": b},
            )

        ql, al, rl = b
        if not ql.startswith("Q: "):
            raise ToolError("SpecAnswerer block missing 'Q: ' prefix.", data={"line": ql})
        if not al.startswith("A: "):
            raise ToolError("SpecAnswerer block missing 'A: ' prefix.", data={"line": al})
        if not rl.startswith("R: "):
            raise ToolError("SpecAnswerer block missing 'R: ' prefix.", data={"line": rl})

        q = ql[3:].strip()
        a = al[3:].strip()
        r = rl[3:].strip()

        if not q or not a or not r:
            raise ToolError(
                "SpecAnswerer block fields must be non-empty (Q/A/R).",
                data={"q": q, "a": a, "r": r},
            )

        out.append((q, a, r))

    return out
    
class SpecAnswererTool(BaseTool):
    """Answers targeted questions using an LLM and returns a single guidance text block.

    Inputs (MVP):
    - user_prompt: str (required)
    - answers: ClarificationBatch | list[QuestionAnswer] | list[dict] (recommended)
        - MUST include all questions (even ones the human answered) so the tool can
          prevent the LLM from contradicting the user.
    - questions: list[str] (fallback; treated as all delegated / decide_for_me=True)
    - seed_spec: Spec | Mapping[str, object] (optional)
    - mode: str (optional) "delegate_only" (default) | "answer_all"
    - llm: LLMClient (optional override; used if llm arg not provided)

    Runtime:
    - llm arg overrides inputs["llm"], which overrides the tool default client.

    Output:
    - guidance_text: str
    - raw_text: str
    """

    name = "spec_answerer"
    version = "0.1.0"
    tags = ("llm", "spec", "answer")

    metadata = ComponentMetadata(
        name="SpecAnswerer",
        output_format=OutputFormat.TEXT,
        description=(
            "SpecAnswerer answers clarification questions to unblock spec authoring. "
            "It returns a concise plain-text guidance block with short reasoning and never "
            "contradicts human-provided answers."
        ),
        tags=("spec", "answers", "llm"),
    )

    def __init__(self, *, default_llm: LLMClient | None = None) -> None:
        self.default_llm = default_llm
        self.prompt = SpecAnswererPrompt(metadata=self.metadata)

    def _resolve_llm(self, *, llm: LLMClient | None, inputs: Mapping[str, object]) -> LLMClient:
        candidate = llm or inputs.get("llm") or self.default_llm
        if candidate is None:
            raise ToolError("Missing LLM client", data={"expected": "llm arg or inputs['llm'] or tool.default_llm"})
        if not hasattr(candidate, "chat"):
            raise ToolError("Invalid LLM client (missing chat method)", data={"type": str(type(candidate))})
        return candidate  # type: ignore[return-value]

    def run(self, *, inputs: dict[str, object], ctx: object, llm: LLMClient | None = None) -> dict[str, object]:
        _ = ctx

        BaseTool.require_non_empty(inputs, "user_prompt")
        user_prompt = str(inputs.get("user_prompt") or "").strip()

        mode = str(inputs.get("mode") or "delegate_only").strip().lower()
        if mode not in ("delegate_only", "answer_all"):
            mode = "delegate_only"

        # Preferred: inputs["answers"] includes *all* questions + any human answers.
        answers_list = _parse_answers_input(inputs.get("answers"))

        # Fallback: inputs["questions"] as list[str] (treated as all delegated).
        if answers_list is None:
            questions = _as_str_list(inputs.get("questions"))
            if not questions:
                raise ToolError(
                    "Missing answers/questions",
                    data={"expected": "inputs['answers'] (preferred) or inputs['questions'] (fallback)"},
                )
            if len(questions) > 25:
                raise ToolError("Too many questions for one call (MVP limit=25)", data={"count": len(questions)})

            answers_list = [
                QuestionAnswer(question=q, answer="", decide_for_me=True, is_llm_generated=False)
                for q in questions
            ]
        else:
            if not answers_list:
                raise ToolError("Missing answers (empty list)", data={"expected": "non-empty answers"})
            if len(answers_list) > 25:
                raise ToolError("Too many questions for one call (MVP limit=25)", data={"count": len(answers_list)})

        seed_spec_obj = inputs.get("seed_spec")
        seed_spec: object
        if isinstance(seed_spec_obj, Spec):
            seed_spec = seed_spec_obj
        else:
            seed_spec = _as_mapping(seed_spec_obj) or {}

        # --- Pre-filtering guard (the important change) ---
        # Human constraints: answered by human (or any non-delegated answer), treated as authoritative.
        human_constraints = [
            a for a in answers_list
            if (not a.decide_for_me) and str(a.answer).strip()
        ]

        # Delegated questions: only these are sent as "questions to answer" in delegate_only mode.
        delegated = [a for a in answers_list if a.decide_for_me]

        if mode == "delegate_only" and not delegated:
            # Nothing to answer; avoid an LLM call.
            return {"guidance_text": "", "raw_text": "", "llm_answers": []}

        llm_client = self._resolve_llm(llm=llm, inputs=inputs)

        # In delegate_only, we send only delegated questions as the "questions" list,
        # and send human answers as constraints inside context.
        questions_payload: list[dict[str, object]]
        if mode == "delegate_only":
            questions_payload = _answers_to_question_payload(delegated)
        else:
            questions_payload = _answers_to_question_payload(answers_list)

        context_payload: dict[str, object] = {
            "user_prompt": user_prompt,
            "seed_spec": seed_spec.to_dict() if isinstance(seed_spec, Spec) else dict(seed_spec),
        }
        if human_constraints:
            context_payload["human_answer_constraints"] = _answers_to_question_payload(human_constraints)

        messages = self.prompt.build(
            inputs={
                "mode": mode,
                "context": context_payload,
                "questions": questions_payload,
            }
        )

        result = llm_client.chat(messages=messages)
        raw_text = str(getattr(result, "text", "") or "").strip()
        if not raw_text:
            raise ToolError("SpecAnswerer returned empty text", data={"provider": getattr(result, "provider", None)})
            
        pairs = _parse_qa_blocks(raw_text)
        
        llm_answers: list[QuestionAnswer] = []
        for (q, a, r) in pairs:
            llm_answers.append(
                QuestionAnswer(
                    question=q,
                    answer=f"{a}\nReason: {r}",
                    decide_for_me=True,
                    is_llm_generated=True,
                )
            )
        
        return {
            "guidance_text": raw_text,
            "raw_text": raw_text,
            "llm_answers": [x.to_dict() for x in llm_answers],  # JSON-friendly
        }
        

def main() -> None:
    from pathlib import Path

    from montymate.spec_pipeline.data.context import ToolContext
    from montymate.spec_pipeline.data.human_inputs import QuestionAnswer, ClarificationBatch
    from montymate.spec_pipeline.llm.llm_client import LLMConfig
    from montymate.spec_pipeline.llm.providers.ollama_client import OllamaClient

    llm = OllamaClient(
        config=LLMConfig(
            provider="ollama",
            model="nemotron-3-nano:30b",
            max_tokens=2_000,
            temperature=0.2,
            extra={"base_url": "http://localhost:11434", "timeout_s": 120.0},
        )
    )

    ctx = ToolContext(project_root=Path(".").resolve(), run_id="dev_spec_answerer", step="answerer")

    tool = SpecAnswererTool()

    # IMPORTANT: pass ALL questions in `answers` (even the human-answered ones),
    # so the answerer LLM doesn't contradict the user input.
    batch = ClarificationBatch(
        round_no=1,
        answers=[
            QuestionAnswer(
                question="How should the score be displayed and where in the UI?",
                answer="Display the score in the top-right corner.",
                decide_for_me=False,
                is_llm_generated=False,
            ),
            QuestionAnswer(
                question="Should the game use requestAnimationFrame or setInterval?",
                answer="",  # delegated
                decide_for_me=True,
                is_llm_generated=False,
            ),
        ],
    )

    out = tool(
        inputs={
            "user_prompt": "Create a snake game that can run in a web browser.",
            "answers": batch,          # preferred input
            # "mode": "delegate_only", # default; only answers decide_for_me=True
            "mode": "answer_all",    # optional; answers everything (still respecting human answers)
        },
        ctx=ctx,
        llm=llm,
    )

    print("\n=== GUIDANCE TEXT ===\n")
    print(str(out.get("guidance_text") or ""))


if __name__ == "__main__":
    main()