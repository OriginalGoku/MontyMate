#src/montymate/spec_pipeline/dev/run_coordinator_test.py
from __future__ import annotations

from pathlib import Path

from montymate.spec_pipeline.data.context import ToolContext
from montymate.spec_pipeline.data.spec_store import FileSpecStore
from montymate.spec_pipeline.llm.llm_client import LLMConfig
from montymate.spec_pipeline.llm.providers.ollama_client import OllamaClient
from montymate.spec_pipeline.tools.spec_author import SpecAuthorTool
from montymate.spec_pipeline.tools.spec_critic import SpecCriticTool

from montymate.spec_pipeline.coordinator import SpecPipelineCoordinator


def main() -> None:
    """Runs a smoke test that should stop after critic, blocked on human answers."""
    project_root = Path(".").resolve()
    run_id = "smoke_coordinator_round1"

    author_llm = OllamaClient(
        config=LLMConfig(
            provider="ollama",
            model="nemotron-3-nano:30b",
            max_tokens=4_000,
            temperature=0.2,
            extra={"base_url": "http://localhost:11434", "timeout_s": 120.0},
        )
    )
    critic_llm = OllamaClient(
        config=LLMConfig(
            provider="ollama",
            model="nemotron-3-nano:30b",
            max_tokens=4_000,
            temperature=0.2,
            extra={"base_url": "http://localhost:11434", "timeout_s": 120.0},
        )
    )

    store = FileSpecStore(project_root=project_root, run_id=run_id)

    ctx = ToolContext(
        project_root=project_root,
        run_id=run_id,
        step="coordinator_smoke",
    )

    coordinator = SpecPipelineCoordinator(
        store=store,
        author_tool=SpecAuthorTool(),
        critic_tool=SpecCriticTool(),
        author_llm=author_llm,
        critic_llm=critic_llm,
        ctx=ctx,
    )

    user_prompt = (
        "Create a CLI tool that renames files using regex find/replace, "
        "supports --dry-run, and optionally recurses with --recursive."
    )

    result = coordinator.run_until_blocked(user_prompt=user_prompt)

    run_root = project_root / ".ai_module_factory" / "runs" / run_id
    print("\n=== COORDINATOR RESULT ===\n")
    print(result)
    print("\n=== RUN ROOT ===\n")
    print(run_root)


if __name__ == "__main__":
    main()