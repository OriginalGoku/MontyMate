from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from montymate.spec_pipeline.data.human_inputs import HumanAnswerBatch
from montymate.spec_pipeline.llm.llm_client import LLMConfig
from montymate.spec_pipeline.llm.providers.ollama_client import OllamaClient
from montymate.spec_pipeline.tools.spec_author import SpecAuthorTool
from montymate.spec_pipeline.tools.spec_critic import SpecCriticTool


@dataclass(slots=True)
class RunCtx:
    """Carries minimal trace metadata for tools.

    The context is intended for tests and local runs.
    """
    project_root: Path
    run_id: str
    step: str


def main() -> None:
    """Runs a 1-round SpecAuthor <-> SpecCritic exchange."""
    model = "nemotron-3-nano:30b"

    llm = OllamaClient(
        config=LLMConfig(
            provider="ollama",
            model=model,
            # Larger budget reduces the “thinking-only, empty content” failure mode.
            max_tokens=4_800,
            temperature=0.2,
            extra={
                "base_url": "http://localhost:11434",
                "timeout_s": 180.0,
            },
        )
    )

    ctx = RunCtx(
        project_root=Path(".").resolve(),
        run_id="round1_author_critic",
        step="round1",
    )

    author = SpecAuthorTool()
    critic = SpecCriticTool()

    user_prompt = (
        "Create a CLI tool that renames files using regex find/replace, supports --dry-run, "
        "and optionally recurses with --recursive."
    )

    # -------------------------
    # 1) Author drafts spec
    # -------------------------
    author_out_1 = author(
        inputs={"user_prompt": user_prompt},
        ctx=ctx,
        llm=llm,
    )

    raw_yaml_1 = str(author_out_1.get("raw_yaml", "") or "")
    spec_1 = author_out_1.get("spec")

    print("\n" + "=" * 80)
    print("AUTHOR — DRAFT")
    print("=" * 80)
    print("\n=== RAW YAML ===\n")
    print(raw_yaml_1.strip() or "<empty>")
    print("\n=== SPEC (dict) ===\n")
    print(yaml.safe_dump(spec_1 if isinstance(spec_1, dict) else {}, sort_keys=False, allow_unicode=True))

    if not isinstance(spec_1, dict):
        raise SystemExit("Author did not return spec as dict.")

    # -------------------------
    # 2) Critic reviews spec
    # -------------------------
    ctx.step = "round1_critic"
    critic_out = critic(
        inputs={"spec": spec_1},
        ctx=ctx,
        llm=llm,
    )

    raw_json = str(critic_out.get("raw_json", "") or "")
    report = critic_out.get("report")

    print("\n" + "=" * 80)
    print("CRITIC — REPORT")
    print("=" * 80)
    print("\n=== RAW JSON ===\n")
    print(raw_json.strip() or "<empty>")
    print("\n=== REPORT (dict) ===\n")
    print(yaml.safe_dump(report if isinstance(report, dict) else {}, sort_keys=False, allow_unicode=True))

    if not isinstance(report, dict):
        raise SystemExit("Critic did not return report as dict.")

    targeted_questions = report.get("targeted_questions")
    if not isinstance(targeted_questions, list):
        targeted_questions = []

    # -------------------------
    # 3) Author revises once
    #    (empty answers => decide-for-me behavior)
    # -------------------------
    ctx.step = "round1_author_revision"
    batch = HumanAnswerBatch.from_dict(
        {
            "round": 1,
            "answers": [{"question": str(q), "answer": ""} for q in targeted_questions if isinstance(q, str) and q.strip()],
        }
    )

    author_out_2 = author(
        inputs={
            "user_prompt": user_prompt,
            "seed_spec": spec_1,
            "answers": batch,
        },
        ctx=ctx,
        llm=llm,
    )

    raw_yaml_2 = str(author_out_2.get("raw_yaml", "") or "")
    spec_2 = author_out_2.get("spec")

    print("\n" + "=" * 80)
    print("AUTHOR — REVISION (ROUND 1)")
    print("=" * 80)
    print("\n=== RAW YAML ===\n")
    print(raw_yaml_2.strip() or "<empty>")
    print("\n=== SPEC (dict) ===\n")
    print(yaml.safe_dump(spec_2 if isinstance(spec_2, dict) else {}, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()