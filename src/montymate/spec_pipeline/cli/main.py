from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, fields
from pathlib import Path
from enum import Enum
from collections.abc import Mapping

import yaml

from montymate.spec_pipeline.data.context import ToolContext
from montymate.spec_pipeline.data.human_inputs import HumanAnswer, HumanAnswerBatch
from montymate.spec_pipeline.data.run_state import RunState, RunStatus
from montymate.spec_pipeline.data.spec_store import FileSpecStore, SpecStore
from montymate.spec_pipeline.llm.llm_client import LLMClient, LLMConfig
from montymate.spec_pipeline.llm.providers.ollama_client import OllamaClient
from montymate.spec_pipeline.llm.providers.openai_client import OpenAIClient
from montymate.spec_pipeline.coordinator import SpecPipelineCoordinator




def _to_primitive(x: object) -> object:
    """Converts objects into YAML/JSON-safe primitives.

    The conversion rules are:
    - dataclasses -> dict of field values (recursively converted)
    - Enum -> its .value
    - Path -> str(path)
    - Mapping -> dict with string keys and recursively converted values
    - Sequence (list/tuple) -> list of recursively converted items
    - everything else -> returned as-is
    """
    if is_dataclass(x):
        return {f.name: _to_primitive(getattr(x, f.name)) for f in fields(x)}

    if isinstance(x, Enum):
        return x.value

    if isinstance(x, Path):
        return str(x)

    if isinstance(x, Mapping):
        return {str(k): _to_primitive(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_to_primitive(v) for v in x]

    return x


def _print_yaml(title: str, payload: object) -> None:
    """Prints a human-readable YAML view of structured data."""
    print(f"\n=== {title} ===\n")
    safe_payload = _to_primitive(payload)
    print(yaml.safe_dump(safe_payload, sort_keys=False, allow_unicode=True))


def _build_llm(
    *,
    provider: str,
    model: str,
    max_tokens: int | None,
    temperature: float | None,
    base_url: str | None,
) -> LLMClient:
    """Builds an LLM client from CLI settings.

    The function keeps the MVP surface area small:
    - provider="ollama" uses a local Ollama server
    - provider="openai" uses OpenAI Chat Completions (expects OPENAI_API_KEY env var)
    """
    p = provider.strip().lower()
    m = model.strip()

    if p == "ollama":
        extra: dict[str, object] = {
            "timeout_s": 120.0,
        }
        if base_url:
            extra["base_url"] = base_url

        return OllamaClient(
            config=LLMConfig(
                provider="ollama",
                model=m,
                max_tokens=max_tokens,
                temperature=temperature,
                extra=extra,
            )
        )

    if p == "openai":
        extra: dict[str, object] = {
            "timeout_s": 60.0,
            # The OpenAI client reads the key from this environment variable name.
            "api_key_env": "OPENAI_API_KEY",
        }
        if base_url:
            extra["base_url"] = base_url

        return OpenAIClient(
            config=LLMConfig(
                provider="openai",
                model=m,
                max_tokens=max_tokens,
                temperature=temperature,
                extra=extra,
            )
        )

    raise SystemExit(f"Unsupported provider: {provider!r}. Supported: ollama, openai")


def _ensure_state_exists(*, store: SpecStore, run_id: str, user_prompt: str) -> RunState:
    """Creates a new RunState when none exists."""
    existing = store.read_state()
    if existing is not None:
        return existing

    state = RunState(
        run_id=run_id,
        status=RunStatus.NEW,
        user_prompt=user_prompt,
        round_no=0,
        last_error=None,
        step="start",
    )
    store.write_state(state)
    return state


def _coordinator_from_cli(
    *,
    store: SpecStore,
    ctx: ToolContext,
    author_provider: str,
    author_model: str,
    critic_provider: str,
    critic_model: str,
    base_url: str | None,
    max_tokens: int | None,
    temperature: float | None,
) -> SpecPipelineCoordinator:
    """Builds the coordinator with tools + LLM clients."""
    from montymate.spec_pipeline.tools.spec_author import SpecAuthorTool
    from montymate.spec_pipeline.tools.spec_critic import SpecCriticTool

    author_llm = _build_llm(
        provider=author_provider,
        model=author_model,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
    )
    critic_llm = _build_llm(
        provider=critic_provider,
        model=critic_model,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
    )

    return SpecPipelineCoordinator(
        store=store,
        author_tool=SpecAuthorTool(),
        critic_tool=SpecCriticTool(),
        author_llm=author_llm,
        critic_llm=critic_llm,
        ctx=ctx,
    )


def cmd_start(args: argparse.Namespace) -> None:
    """Starts a run and advances until blocked."""
    project_root = Path(args.project_root).resolve()
    store = FileSpecStore(project_root=project_root, run_id=args.run_id)

    _ = _ensure_state_exists(store=store, run_id=args.run_id, user_prompt=args.prompt)

    ctx = ToolContext(project_root=project_root, run_id=args.run_id, step="start")

    coord = _coordinator_from_cli(
        store=store,
        ctx=ctx,
        author_provider=args.author_provider,
        author_model=args.author_model,
        critic_provider=args.critic_provider,
        critic_model=args.critic_model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    result = coord.run_until_blocked(max_steps=args.max_steps)

    _print_yaml("COORDINATOR RESULT", asdict(result) if is_dataclass(result) else result)
    print("\n=== RUN ROOT ===\n")
    print(str(store.paths.run_root))


def cmd_resume(args: argparse.Namespace) -> None:
    """Resumes a run and advances until blocked."""
    project_root = Path(args.project_root).resolve()
    store = FileSpecStore(project_root=project_root, run_id=args.run_id)

    state = store.read_state()
    if state is None:
        raise SystemExit("No run_state.json found. Use start first.")

    # The MVP flow benefits from allowing retry after transient errors.
    # Resetting ERROR to NEW keeps pause/resume simple and avoids manual JSON edits.
    if state.status == RunStatus.ERROR:
        state = RunState(
            run_id=state.run_id,
            status=RunStatus.NEW,
            user_prompt=state.user_prompt,
            round_no=state.round_no,
            last_error=None,
            step=state.step,
        )
        store.write_state(state)

    ctx = ToolContext(project_root=project_root, run_id=args.run_id, step="resume")

    coord = _coordinator_from_cli(
        store=store,
        ctx=ctx,
        author_provider=args.author_provider,
        author_model=args.author_model,
        critic_provider=args.critic_provider,
        critic_model=args.critic_model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    result = coord.run_until_blocked(max_steps=args.max_steps)

    _print_yaml("COORDINATOR RESULT", asdict(result) if is_dataclass(result) else result)
    print("\n=== RUN ROOT ===\n")
    print(str(store.paths.run_root))


def cmd_status(args: argparse.Namespace) -> None:
    """Prints run state and a short hint about what blocks progress."""
    project_root = Path(args.project_root).resolve()
    store = FileSpecStore(project_root=project_root, run_id=args.run_id)

    state = store.read_state()
    if state is None:
        raise SystemExit("No run_state.json found. Use start first.")

    _print_yaml("RUN STATE", state)

    if state.status == RunStatus.WAITING_FOR_HUMAN:
        answers_path = store.paths.spec_answers_json(state.round_no)
        report_path = store.paths.spec_critic_report_json(state.round_no)
        print("\n=== NEXT ACTION ===\n")
        print(f"Waiting for answers for round {state.round_no}.")
        print(f"- Critic report: {report_path}")
        print(f"- Answers file: {answers_path}")


def cmd_answer(args: argparse.Namespace) -> None:
    """Interactive wizard that writes HumanAnswerBatch for the current round."""
    project_root = Path(args.project_root).resolve()
    store = FileSpecStore(project_root=project_root, run_id=args.run_id)

    state = store.read_state()
    if state is None:
        raise SystemExit("No run_state.json found. Use start first.")

    round_no = int(args.round_no) if args.round_no is not None else int(state.round_no)

    report = store.read_critic_report(round_no=round_no)
    if report is None:
        raise SystemExit(f"No critic report found for round {round_no}.")

    tq = report.get("targeted_questions")
    if not isinstance(tq, list) or not tq:
        raise SystemExit(f"Critic report for round {round_no} has no targeted_questions.")

    print("\n=== ANSWER ROUND ===\n")
    print(f"Run: {args.run_id}")
    print(f"Round: {round_no}")
    print("Empty answer means: decide for me.\n")

    answers: list[HumanAnswer] = []
    for i, q in enumerate(tq, start=1):
        q_text = str(q).strip()
        if not q_text:
            continue
        print(f"[{i}] {q_text}")
        a = input("> ").rstrip("\n")
        answers.append(HumanAnswer(question=q_text, answer=a))

    batch = HumanAnswerBatch(round_no=round_no, answers=answers)
    store.write_answers_batch(batch)

    print("\nSaved answers:")
    print(str(store.paths.spec_answers_json(round_no)))


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser."""
    p = argparse.ArgumentParser(prog="mm", description="MontyMate MVP CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_llm_flags(sp: argparse.ArgumentParser) -> None:
        # The flags keep the MVP simple but allow switching providers/models.
        sp.add_argument("--author-provider", default="ollama", help="ollama | openai")
        sp.add_argument("--author-model", default="nemotron-3-nano:30b", help="Author model name")
        sp.add_argument("--critic-provider", default="ollama", help="ollama | openai")
        sp.add_argument("--critic-model", default="nemotron-3-nano:30b", help="Critic model name")
        sp.add_argument("--base-url", default=None, help="Provider base URL override")
        sp.add_argument("--max-tokens", type=int, default=18_000, help="Max output tokens")
        sp.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
        sp.add_argument("--max-steps", type=int, default=25, help="Maximum coordinator steps per invocation")

    sp_start = sub.add_parser("start", help="Start a new run and advance until blocked")
    sp_start.add_argument("--project-root", default=".", help="Project root (where .ai_module_factory lives)")
    sp_start.add_argument("--run-id", required=True, help="Run identifier")
    sp_start.add_argument("--prompt", required=True, help="User prompt for the run")
    add_common_llm_flags(sp_start)
    sp_start.set_defaults(func=cmd_start)

    sp_resume = sub.add_parser("resume", help="Resume an existing run and advance until blocked")
    sp_resume.add_argument("--project-root", default=".", help="Project root (where .ai_module_factory lives)")
    sp_resume.add_argument("--run-id", required=True, help="Run identifier")
    add_common_llm_flags(sp_resume)
    sp_resume.set_defaults(func=cmd_resume)

    sp_status = sub.add_parser("status", help="Show current run state")
    sp_status.add_argument("--project-root", default=".", help="Project root (where .ai_module_factory lives)")
    sp_status.add_argument("--run-id", required=True, help="Run identifier")
    sp_status.set_defaults(func=cmd_status)

    sp_answer = sub.add_parser("answer", help="Answer critic questions for a round (interactive)")
    sp_answer.add_argument("--project-root", default=".", help="Project root (where .ai_module_factory lives)")
    sp_answer.add_argument("--run-id", required=True, help="Run identifier")
    sp_answer.add_argument("--round-no", type=int, default=None, help="Round number (default: state.round_no)")
    sp_answer.set_defaults(func=cmd_answer)

    return p


def main() -> None:
    """Runs the CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()