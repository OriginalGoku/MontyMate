# src/montymate/spec_pipeline/tools/spec_author.py
from __future__ import annotations

from collections.abc import Mapping

import yaml

from .base import BaseTool
from ..data.component_metadata import ComponentMetadata, OutputFormat
from ..data.human_inputs import HumanAnswerBatch
from ..data.spec_types import Spec
from ..data.context import ToolContext
from ..errors import ToolError
from ..llm.llm_client import LLMClient
from ..llm.prompts.prompts import SpecAuthorPrompt
from ..utils.llm_text_utils import strip_code_fences


def _as_mapping(x: object) -> Mapping[str, object] | None:
    """Returns x as a Mapping if it is Mapping-like, otherwise returns None."""
    return x if isinstance(x, Mapping) else None


def _as_llm_client(x: object) -> LLMClient | None:
    """Returns x as an LLMClient if it exposes a callable chat() method."""
    chat = getattr(x, "chat", None)
    if callable(chat):
        # Structural check only; Protocol is compile-time, not runtime.
        return x  # type: ignore[return-value]
    return None


class SpecAuthorTool(BaseTool):
    """Produces an updated Spec from a user prompt and optional context.

    Inputs:
    - user_prompt: str (required)
    - seed_spec: Spec | Mapping[str, object] (optional)
    - answers: HumanAnswerBatch | Mapping[str, object] (optional)
    - llm: LLMClient (optional override; typically used in tests/dev)

    Runtime:
    - llm argument overrides inputs["llm"], which overrides the tool default client.
    """

    name = "spec_author"
    version = "0.1.0"
    tags = ("llm", "spec")

    metadata = ComponentMetadata(
        name="SpecAuthor",
        output_format=OutputFormat.YAML,
        description=(
            "SpecAuthor produces and maintains a structured software specification. "
            "It outputs a complete specification object in YAML form and keeps the spec internally consistent."
        ),
        tags=("spec", "author"),
    )

    def __init__(self, *, default_llm: LLMClient | None = None) -> None:
        self.default_llm: LLMClient | None = default_llm
        # The tool owns metadata; the prompt reads it from the tool metadata.
        # SpecAuthorPrompt must accept metadata in __init__ for this to work.
        self.prompt = SpecAuthorPrompt(metadata=self.metadata)

    def _resolve_llm(self, *, llm: LLMClient | None, inputs: Mapping[str, object]) -> LLMClient:
        """Resolves the LLMClient to use for this run."""
        if llm is not None:
            return llm

        from_inputs = _as_llm_client(inputs.get("llm")) if "llm" in inputs else None
        if from_inputs is not None:
            return from_inputs

        if self.default_llm is not None:
            return self.default_llm

        raise ToolError(
            "Missing LLM client",
            data={"expected": "llm arg or inputs['llm'] or tool.default_llm"},
        )

    def run(self, *, inputs: dict[str, object], ctx: ToolContext, llm: LLMClient | None = None) -> dict[str, object]:
        _ = ctx  # The context is used indirectly by BaseTool for tracing metadata.

        BaseTool.require_non_empty(inputs, "user_prompt")
        user_prompt = str(inputs.get("user_prompt") or "").strip()

        seed_spec_obj = inputs.get("seed_spec")
        answers_obj = inputs.get("answers")

        seed_spec: object = seed_spec_obj if isinstance(seed_spec_obj, Spec) else (_as_mapping(seed_spec_obj) or {})
        answers: object = answers_obj if isinstance(answers_obj, HumanAnswerBatch) else (_as_mapping(answers_obj) or {})

        llm_client = self._resolve_llm(llm=llm, inputs=inputs)

        messages = self.prompt.build(
            inputs={
                "user_prompt": user_prompt,
                "seed_spec": seed_spec,
                "answers": answers,
            }
        )

        result = llm_client.chat(messages=messages)
        raw_yaml = strip_code_fences(result.text)

        try:
            parsed = yaml.safe_load(raw_yaml) or {}
        except Exception as e:
            raise ToolError("SpecAuthor returned invalid YAML", 
                data={"error": str(e), 
                    "raw": raw_yaml}) from e

        if not isinstance(parsed, dict):
            raise ToolError("SpecAuthor YAML must be a mapping/object", 
                data={"raw": raw_yaml})

        spec = Spec.from_dict(parsed)

        return {
            "spec": spec.to_dict(),
            "raw_yaml": raw_yaml,
        }