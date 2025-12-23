# src/montymate/spec_pipeline/llm/prompts/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from ...data.component_metadata import ComponentMetadata
from ..llm_client import ChatMessages


class BasePrompt(ABC):
    """Builds chat-style prompt messages for spec-pipeline components.

    The base prompt standardizes:
    - a consistent system message header (component description + strict output rules)
    - a stable message shape (system + user)
    - a single entrypoint (build) that tools can call deterministically

    Metadata is injected by the owning tool to keep naming, description, and
    output-format authoritative at the tool level.
    """

    def __init__(self, *, metadata: ComponentMetadata) -> None:
        self.metadata = metadata

    def build(self, *, inputs: Mapping[str, object]) -> ChatMessages:
        """Returns chat messages as [system, user]."""
        return [
            {"role": "system", "content": self._system_message(inputs=inputs)},
            {"role": "user", "content": self.user_message(inputs=inputs)},
        ]

    def _system_message(self, *, inputs: Mapping[str, object]) -> str:
        rules = [
            self.metadata.description.strip(),
            "",
            f"- Output MUST be valid {self.metadata.output_format.value.upper()} only.",
            "- No markdown fences.",
            "- No extra commentary.",
        ]
        extras = self.system_extras(inputs=inputs).strip()
        if extras:
            rules.append(extras)
        return "\n".join(rules).strip() + "\n"

    def system_extras(self, *, inputs: Mapping[str, object]) -> str:
        """Returns extra system constraints specific to the prompt."""
        _ = inputs
        return ""

    @abstractmethod
    def user_message(self, *, inputs: Mapping[str, object]) -> str:
        """Returns the user-facing context block assembled by subclasses."""
        raise NotImplementedError