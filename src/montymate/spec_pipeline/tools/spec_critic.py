# File: src/montymate/spec_pipeline/tools/spec_critic.py
from __future__ import annotations

import json
from collections.abc import Mapping

from .base import BaseTool
from ..errors import ToolError
from ..data.component_metadata import ComponentMetadata, OutputFormat
from ..data.spec_types import Spec
from ..data.tool_types import SpecCriticReport
from ..llm.llm_client import LLMClient
from ..llm.prompts.prompts import SpecCriticPrompt
from ..utils.llm_text_utils import strip_code_fences


def _as_mapping(x: object) -> Mapping[str, object] | None:
    return x if isinstance(x, Mapping) else None


class SpecCriticTool(BaseTool):
    """Evaluates a spec and produces a targeted-question report.

    Inputs:
    - spec: Spec | Mapping[str, object] (required)
    - llm: LLMClient (optional override; used if llm arg is not provided)

    Runtime:
    - llm argument overrides inputs["llm"], which overrides the tool default client.

    Outputs:
    - report: dict[str, object]  (SpecCriticReport.to_dict())
    - raw_json: str              (debug)
    """

    name = "spec_critic"
    version = "0.1.0"
    tags = ("llm", "spec", "critic")

    metadata = ComponentMetadata(
        name="SpecCritic",
        output_format=OutputFormat.JSON,
        description=(
            "SpecCritic reviews a software specification for clarity, completeness, and internal consistency. "
            "It returns a JSON report containing issues and targeted questions that reduce ambiguity."
        ),
        tags=("spec", "critic"),
    )

    def __init__(self, *, default_llm: LLMClient | None = None) -> None:
        self.default_llm = default_llm
        self.prompt = SpecCriticPrompt(metadata=self.metadata)

    def _resolve_llm(self, *, llm: LLMClient | None, inputs: Mapping[str, object]) -> LLMClient:
        candidate = llm or inputs.get("llm") or self.default_llm
        if candidate is None:
            raise ToolError("Missing LLM client", data={"expected": "llm arg or inputs['llm'] or tool.default_llm"})
        if not hasattr(candidate, "chat"):
            raise ToolError("Invalid LLM client (missing chat method)", data={"type": str(type(candidate))})
        return candidate  # type: ignore[return-value]

    def run(self, *, inputs: dict[str, object], ctx: object, llm: LLMClient | None = None) -> dict[str, object]:
        _ = ctx  # ctx is used indirectly by BaseTool for tracing metadata

        BaseTool.require(inputs, "spec")

        spec_obj = inputs.get("spec")
        if isinstance(spec_obj, Spec):
            spec_payload: dict[str, object] = {str(k): v for k, v in spec_obj.to_dict().items()}
        else:
            spec_map = _as_mapping(spec_obj)
            if spec_map is None:
                raise ToolError("Invalid spec input (expected Spec or mapping)", data={"type": str(type(spec_obj))})
            spec_payload = {str(k): spec_map[k] for k in spec_map.keys()}

        llm_client = self._resolve_llm(llm=llm, inputs=inputs)

        messages = self.prompt.build(inputs={"spec": spec_payload})

        result = llm_client.chat(messages=messages)
        raw_text = str(result.text or "")
        raw_json = strip_code_fences(raw_text).strip()

        if not raw_json:
            raise ToolError("SpecCritic returned empty output", data={"tool": self.name})

        try:
            parsed = json.loads(raw_json)
        except Exception as e:
            raise ToolError("SpecCritic returned invalid JSON", data={"error": str(e), "raw": raw_json}) from e

        if not isinstance(parsed, dict):
            raise ToolError("SpecCritic JSON must be an object", data={"raw": raw_json})

        report = SpecCriticReport.from_dict(parsed)

        return {
            "report": report.to_dict(),
            "raw_json": raw_json,
        }