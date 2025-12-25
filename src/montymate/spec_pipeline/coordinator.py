# src/montymate/spec_pipeline/coordinator.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping

from .data.run_state import RunState, RunStatus
from .data.spec_store import SpecStore
from .data.human_inputs import ClarificationBatch, QuestionAnswer
from .errors import ToolError
from .llm.llm_client import LLMClient


class BlockReason(str, Enum):
    """Describes why the coordinator cannot advance further in the current call."""

    NEEDS_USER_PROMPT = "NEEDS_USER_PROMPT"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    DONE = "DONE"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class StepResult:
    """Represents the outcome of a single coordinator step."""

    advanced: bool
    blocked: bool
    reason: str | None = None
    block_reason: BlockReason | None = None
    state: RunState | None = None


def _as_qa_list(x: object) -> list[QuestionAnswer]:
    """Accepts list[QuestionAnswer] | list[dict] and returns list[QuestionAnswer]."""
    if not isinstance(x, list):
        return []
    out: list[QuestionAnswer] = []
    for item in x:
        if isinstance(item, QuestionAnswer):
            out.append(item)
        elif isinstance(item, Mapping):
            out.append(QuestionAnswer.from_dict({str(k): item[k] for k in item.keys()}))
    return out


def _apply_llm_clarifications_append(
    batch: ClarificationBatch,
    llm_answers: list[QuestionAnswer],
    *,
    drop_delegated_blanks: bool = True,
) -> ClarificationBatch:
    """Append LLM-produced clarifications as additional Q/A items.

    - Keeps human-answered items as-is.
    - Optionally drops delegated blanks (decide_for_me=True and answer empty) to avoid clutter.
    - Appends LLM answers as non-delegated (decide_for_me=False) and is_llm_generated=True
      so they behave like “real answers” downstream and do not re-trigger auto-answer loops.
    """
    kept: list[QuestionAnswer] = []
    for a in batch.answers:
        is_blank = (str(a.answer).strip() == "")
        if drop_delegated_blanks and a.decide_for_me and is_blank:
            continue
        kept.append(a)

    appended: list[QuestionAnswer] = []
    for x in llm_answers:
        q = str(x.question).strip()
        a = str(x.answer).strip()
        if not q or not a:
            continue
        appended.append(
            QuestionAnswer(
                question=q,
                answer=a,
                decide_for_me=False,      # IMPORTANT: now it is an “answered” constraint
                is_llm_generated=True,
            )
        )

    return ClarificationBatch(round_no=batch.round_no, answers=[*kept, *appended])

class SpecPipelineCoordinator:
    def __init__(
        self,
        *,
        store: SpecStore,
        author_tool: object,
        critic_tool: object,
        author_llm: LLMClient,
        critic_llm: LLMClient,
        ctx: object,
        answerer_tool: object | None = None,
        answerer_llm: LLMClient | None = None,
        auto_answer_delegated: bool = True,
    ) -> None:
        self.store = store
        self.author_tool = author_tool
        self.critic_tool = critic_tool
        self.author_llm = author_llm
        self.critic_llm = critic_llm
        self.ctx = ctx

        self.answerer_tool = answerer_tool
        self.answerer_llm = answerer_llm
        self.auto_answer_delegated = bool(auto_answer_delegated)
    
    def run_until_blocked(self, *, user_prompt: str | None = None, max_steps: int = 50) -> StepResult:
        last: StepResult = StepResult(
            advanced=False,
            blocked=True,
            reason="No steps executed.",
            state=self.store.read_state(),
        )
        for _ in range(int(max_steps)):
            r = self.step(user_prompt=user_prompt)
            last = r
            user_prompt = None
            if r.blocked or not r.advanced:
                return r
        return StepResult(
            advanced=False,
            blocked=True,
            reason=f"Reached max_steps={max_steps}.",
            state=self.store.read_state(),
        )
        
    def step(self, *, user_prompt: str | None = None) -> StepResult:
        state = self.store.read_state()

        if state is None:
            if not (user_prompt or "").strip():
                return StepResult(
                    advanced=False,
                    blocked=True,
                    reason="RunState missing and no user_prompt provided.",
                    block_reason=BlockReason.NEEDS_USER_PROMPT,
                    state=None,
                )

            init_state = RunState(
                run_id=str(getattr(self.ctx, "run_id", "") or "run"),
                status=RunStatus.NEW,
                user_prompt=str(user_prompt),
                round_no=0,
                last_error=None,
                step="init",
            )
            self.store.write_state(init_state)
            return StepResult(advanced=True, blocked=False, reason="Initialized run state.", state=init_state)

        if state.status == RunStatus.DONE:
            return StepResult(advanced=False, blocked=True, reason="Run is complete.", block_reason=BlockReason.DONE, state=state)

        if state.status == RunStatus.ERROR:
            return StepResult(advanced=False, blocked=True, reason=state.last_error or "Run is in ERROR state.", block_reason=BlockReason.ERROR, state=state)

        try:
            if state.status == RunStatus.WAITING_FOR_HUMAN:
                batch = self.store.read_answers_batch(state.round_no)
                if batch is None:
                    return StepResult(
                        advanced=False,
                        blocked=True,
                        reason=f"Missing answers for round {state.round_no}.",
                        block_reason=BlockReason.WAITING_FOR_HUMAN,
                        state=state,
                    )

                seed_spec = self.store.read_spec()

                # Auto-answer: append-only (no matching).
                if (
                    self.auto_answer_delegated
                    and self.answerer_tool is not None
                    and self.answerer_llm is not None
                    and any(a.decide_for_me and (str(a.answer).strip() == "") for a in batch.answers)
                ):
                    ans_out = self.answerer_tool(
                        inputs={
                            "user_prompt": state.user_prompt,
                            "seed_spec": seed_spec,
                            "answers": batch,
                            "mode": "delegate_only",
                        },
                        ctx=self.ctx,
                        llm=self.answerer_llm,
                    )

                    llm_answers = _as_qa_list(ans_out.get("llm_answers"))
                    if llm_answers:
                        batch = _apply_llm_clarifications_append(
                            batch,
                            llm_answers,
                            drop_delegated_blanks=True,   # MVP: keep file clean
                        )
                        self.store.write_answers_batch(batch)

                out = self.author_tool(
                    inputs={
                        "user_prompt": state.user_prompt,
                        "seed_spec": seed_spec,
                        "answers": batch,
                    },
                    ctx=self.ctx,
                    llm=self.author_llm,
                )

                spec_dict = out.get("spec")
                if not isinstance(spec_dict, dict):
                    raise ToolError("SpecAuthor output missing 'spec' dict.", data={"keys": list(out.keys())})

                from .data.spec_types import Spec
                self.store.write_spec(Spec.from_dict(spec_dict))

                new_state = RunState(
                    run_id=state.run_id,
                    status=RunStatus.AUTHOR_DRAFTED,
                    user_prompt=state.user_prompt,
                    round_no=state.round_no,
                    last_error=None,
                    step=f"author_revision_round_{state.round_no}",
                )
                self.store.write_state(new_state)
                return StepResult(advanced=True, blocked=False, reason="Authored revision.", state=new_state)
                
            # NEW -> AUTHOR_DRAFTED
            if state.status == RunStatus.NEW:
                seed_spec = self.store.read_spec()
            
                out = self.author_tool(
                    inputs={"user_prompt": state.user_prompt, "seed_spec": seed_spec},
                    ctx=self.ctx,
                    llm=self.author_llm,
                )
            
                spec_dict = out.get("spec")
                if not isinstance(spec_dict, dict):
                    raise ToolError("SpecAuthor output missing 'spec' dict.", data={"keys": list(out.keys())})
            
                from .data.spec_types import Spec
                self.store.write_spec(Spec.from_dict(spec_dict))
            
                new_state = RunState(
                    run_id=state.run_id,
                    status=RunStatus.AUTHOR_DRAFTED,
                    user_prompt=state.user_prompt,
                    round_no=state.round_no,
                    last_error=None,
                    step="author_draft",
                )
                self.store.write_state(new_state)
                return StepResult(advanced=True, blocked=False, reason="Authored initial draft.", state=new_state)
            
            # AUTHOR_DRAFTED -> DONE or WAITING_FOR_HUMAN
            if state.status == RunStatus.AUTHOR_DRAFTED:
                spec = self.store.read_spec()
            
                out = self.critic_tool(
                    inputs={"spec": spec},
                    ctx=self.ctx,
                    llm=self.critic_llm,
                )
            
                report = out.get("report")
                if not isinstance(report, dict):
                    raise ToolError("SpecCritic output missing 'report' dict.", data={"keys": list(out.keys())})
            
                passed = bool(report.get("passed")) is True
                if passed:
                    self.store.write_critic_report(round_no=state.round_no, report=report)
            
                    new_state = RunState(
                        run_id=state.run_id,
                        status=RunStatus.DONE,
                        user_prompt=state.user_prompt,
                        round_no=state.round_no,
                        last_error=None,
                        step=f"critic_round_{state.round_no}",
                    )
                    self.store.write_state(new_state)
                    return StepResult(advanced=True, blocked=False, reason="Critic passed.", state=new_state)
            
                next_round = int(state.round_no) + 1
                self.store.write_critic_report(round_no=next_round, report=report)
            
                new_state = RunState(
                    run_id=state.run_id,
                    status=RunStatus.WAITING_FOR_HUMAN,
                    user_prompt=state.user_prompt,
                    round_no=next_round,
                    last_error=None,
                    step=f"critic_round_{next_round}",
                )
                self.store.write_state(new_state)
                return StepResult(
                    advanced=True,
                    blocked=False,
                    reason=f"Critic failed. Waiting for answers (round {next_round}).",
                    state=new_state,
                )
            
            raise ToolError("Unknown RunStatus.", data={"status": str(state.status)})
        except ToolError as e:
            err_state = RunState(
                run_id=state.run_id,
                status=RunStatus.ERROR,
                user_prompt=state.user_prompt,
                round_no=state.round_no,
                last_error=str(e),
                step="error",
            )
            self.store.write_state(err_state)
            return StepResult(advanced=False, blocked=True, reason=str(e), block_reason=BlockReason.ERROR, state=err_state)

        except Exception as e:
            err_state = RunState(
                run_id=state.run_id,
                status=RunStatus.ERROR,
                user_prompt=state.user_prompt,
                round_no=state.round_no,
                last_error=str(e),
                step="error",
            )
            self.store.write_state(err_state)
            return StepResult(advanced=False, blocked=True, reason=str(e), block_reason=BlockReason.ERROR, state=err_state)
                
                
                