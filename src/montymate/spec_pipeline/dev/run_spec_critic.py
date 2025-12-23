# File: src/montymate/spec_pipeline/dev/run_spec_critic.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from montymate.spec_pipeline.data.spec_types import Spec
from montymate.spec_pipeline.llm.llm_client import LLMConfig
from montymate.spec_pipeline.llm.providers.ollama_client import OllamaClient
from montymate.spec_pipeline.tools.spec_critic import SpecCriticTool


@dataclass(slots=True)
class RunCtx:
    """Carries minimal trace metadata for tools."""
    project_root: Path
    run_id: str
    step: str


def main() -> None:
    model = "nemotron-3-nano:30b"

    llm = OllamaClient(
        config=LLMConfig(
            provider="ollama",
            model=model,
            max_tokens=4_000,
            temperature=0.2,
            extra={"base_url": "http://localhost:11434", "timeout_s": 120.0},
        )
    )

    ctx = RunCtx(project_root=Path(".").resolve(), run_id="manual_ollama_run", step="spec_critic_smoke")

    tool = SpecCriticTool()

    seed = Spec.from_dict(
        {
            "status": "DRAFT",
            "goal": "Create a CLI tool that renames files using regex find/replace with dry-run.",
            "functional_requirements": ["Support --dry-run", "Support regex find/replace"],
            "constraints": ["Python 3.11+"],
            "security_concerns": ["Avoid accidental overwrites"],
            "assumptions": ["User has filesystem permissions"],
            "other_notes": "",
        }
    )

    out = tool(inputs={"spec": seed}, ctx=ctx, llm=llm)

    print("\n=== RAW JSON ===\n")
    print(str(out.get("raw_json") or ""))

    print("\n=== REPORT (dict) ===\n")
    report = out.get("report")
    if isinstance(report, dict):
        print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    else:
        print(report)


if __name__ == "__main__":
    main()