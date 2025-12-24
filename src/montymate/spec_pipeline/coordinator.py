# src/montymate/spec_pipeline/coordinator.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .data.run_state import RunState, RunStatus
from .data.spec_store import SpecStore
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


class SpecPipelineCoordinator:
    """Coordinates the MVP author/critic loop using persisted run artifacts.

    The coordinator:
    - reads and writes RunState via SpecStore
    - runs a single tool action per step()
    - advances until a blocking condition via run_until_blocked()

    The coordinator assumes:
    - SpecAuthorTool and SpecCriticTool follow the BaseTool calling convention:
      tool(inputs=..., ctx=..., llm=...)
    - llm is passed explicitly (ctx remains minimal)
    """

    def __init__(
        self,
        *,
        store: SpecStore,
        author_tool: object,
        critic_tool: object,
        author_llm: LLMClient,
        critic_llm: LLMClient,
        ctx: object,
    ) -> None:
        self.store = store
        self.author_tool = author_tool
        self.critic_tool = critic_tool
        self.author_llm = author_llm
        self.critic_llm = critic_llm
        self.ctx = ctx

    def step(self, *, user_prompt: str | None = None) -> StepResult:
        """Advances the run by one logical action.

        Actions:
        - NEW -> AUTHOR_DRAFTED via SpecAuthorTool
        - AUTHOR_DRAFTED -> (DONE or WAITING_FOR_HUMAN) via SpecCriticTool
        - WAITING_FOR_HUMAN -> AUTHOR_DRAFTED via SpecAuthorTool if answers exist

        A missing RunState is treated as "not initialized yet" and is initialized
        if user_prompt is provided. Initialization is considered a step that only
        writes state (no tool calls).
        """
        state = self.store.read_state()

        # Missing/unreadable state is treated as "not initialized yet".
        if state is None:
            if not (user_prompt or "").strip():
                return StepResult(
                    advanced=False,
                    blocked=True,
                    reason="RunState missing and no user_prompt provided.",
                    block_reason=BlockReason.NEEDS_USER_PROMPT,
                    state=None,
                )

            # Run initialization is purely artifact-driven (state persisted).
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

        # Terminal states
        if state.status == RunStatus.DONE:
            return StepResult(
                advanced=False,
                blocked=True,
                reason="Run is complete.",
                block_reason=BlockReason.DONE,
                state=state,
            )

        if state.status == RunStatus.ERROR:
            return StepResult(
                advanced=False,
                blocked=True,
                reason=state.last_error or "Run is in ERROR state.",
                block_reason=BlockReason.ERROR,
                state=state,
            )

        try:
            # WAITING_FOR_HUMAN -> AUTHOR_DRAFTED if answers exist for current round
            if state.status == RunStatus.WAITING_FOR_HUMAN:
                answers = self.store.read_answers_batch(state.round_no)
                if answers is None:
                    return StepResult(
                        advanced=False,
                        blocked=True,
                        reason=f"Missing human answers for round {state.round_no}.",
                        block_reason=BlockReason.WAITING_FOR_HUMAN,
                        state=state,
                    )

                seed_spec = self.store.read_spec()  # Allows manual edits between rounds.

                out = self.author_tool(
                    inputs={
                        "user_prompt": state.user_prompt,
                        "seed_spec": seed_spec,
                        "answers": answers,
                    },
                    ctx=self.ctx,
                    llm=self.author_llm,
                )

                spec_dict = out.get("spec")
                if not isinstance(spec_dict, dict):
                    raise ToolError("SpecAuthor output missing 'spec' dict.", data={"keys": list(out.keys())})

                # Spec persistence is performed by converting dict->Spec via Spec.from_dict in the store layer.
                # SpecStore.write_spec expects a Spec, so read_spec/write_spec should be used consistently.
                # The coordinator reconstructs Spec through the store read/write boundary by using Spec.from_dict.
                from .data.spec_types import Spec  # Local import to avoid import cycles.

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

            # NEW -> AUTHOR_DRAFTED (draft from scratch)
            if state.status == RunStatus.NEW:
                seed_spec = self.store.read_spec()

                out = self.author_tool(
                    inputs={
                        "user_prompt": state.user_prompt,
                        "seed_spec": seed_spec,
                    },
                    ctx=self.ctx,
                    llm=self.author_llm,
                )

                spec_dict = out.get("spec")
                if not isinstance(spec_dict, dict):
                    raise ToolError("SpecAuthor output missing 'spec' dict.", data={"keys": list(out.keys())})

                from .data.spec_types import Spec  # Local import to avoid import cycles.

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

            # AUTHOR_DRAFTED -> DONE or WAITING_FOR_HUMAN via critic
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

                passed = report.get("passed")
                passed_bool = bool(passed) is True

                if passed_bool:
                    # Passed report is written for current round (MVP behavior).
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

                # Failure increments round_no and blocks for human answers.
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
                    reason=f"Critic failed. Waiting for human answers (round {next_round}).",
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
            return StepResult(
                advanced=False,
                blocked=True,
                reason=str(e),
                block_reason=BlockReason.ERROR,
                state=err_state,
            )

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
            return StepResult(
                advanced=False,
                blocked=True,
                reason=str(e),
                block_reason=BlockReason.ERROR,
                state=err_state,
            )

    def run_until_blocked(self, *, user_prompt: str | None = None, max_steps: int = 50) -> StepResult:
        """Runs step() repeatedly until progress is blocked or max_steps is reached."""
        last: StepResult = StepResult(advanced=False, blocked=True, reason="No steps executed.", state=self.store.read_state())
        for _ in range(int(max_steps)):
            r = self.step(user_prompt=user_prompt)
            last = r

            # user_prompt is only needed for initialization; later steps ignore it.
            user_prompt = None

            if r.blocked:
                return r
            if not r.advanced:
                return r

        return StepResult(
            advanced=False,
            blocked=True,
            reason=f"Reached max_steps={max_steps}.",
            state=self.store.read_state(),
        )