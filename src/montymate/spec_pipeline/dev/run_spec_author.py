# src/montymate/spec_pipeline/dev/run_spec_author.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from montymate.spec_pipeline.llm.llm_client import LLMConfig
from montymate.spec_pipeline.llm.providers.ollama_client import OllamaClient
from montymate.spec_pipeline.tools.spec_author import SpecAuthorTool


@dataclass(slots=True)
class RunCtx:
    """Carries minimal trace metadata for tools.

    The context is intended for tests and local runs.
    """
    project_root: Path
    run_id: str
    step: str


def main() -> None:
    """Runs a smoke test for SpecAuthorTool using an Ollama-backed LLM client."""
    model = "nemotron-3-nano:30b"

    llm = OllamaClient(
        config=LLMConfig(
            provider="ollama",
            model=model,
            max_tokens=4_000,
            temperature=0.2,
            extra={
                "base_url": "http://localhost:11434",
                "timeout_s": 120.0,
            },
        )
    )

    ctx = RunCtx(
        project_root=Path(".").resolve(),
        run_id="manual_ollama_run",
        step="spec_author_smoke",
    )

    tool = SpecAuthorTool()

    out = tool(
        inputs={
            "user_prompt": (
                "Create a CLI tool that renames files using regex find/replace, "
                "supports --dry-run, and optionally recurses with --recursive."
            ),
        },
        ctx=ctx,
        llm=llm,  # Explicit LLM injection for this run
    )

    raw_yaml = str(out.get("raw_yaml", "") or "")
    spec_dict = out.get("spec")

    print("\n=== RAW YAML ===\n")
    print(raw_yaml)

    print("\n=== SPEC (dict) ===\n")
    if isinstance(spec_dict, dict):
        print(yaml.safe_dump(spec_dict, sort_keys=False, allow_unicode=True))
    else:
        print(spec_dict)


if __name__ == "__main__":
    main()