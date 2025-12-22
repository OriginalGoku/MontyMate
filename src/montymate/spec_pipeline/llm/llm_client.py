# src/montymate/spec_pipeline/llm/llm_client.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


ChatMessage = dict[str, str]
ChatMessages = list[ChatMessage]


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Standard return type for all LLM clients."""
    status: Literal["OK", "ERROR"]
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    raw: Any | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LLMError(RuntimeError):
    """Raised for expected LLM failures (network, invalid response, provider issues)."""

    message: str
    data: JsonDict | None = None

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Represents the fixed settings for one configured client instance."""
    provider: str               # "openai" | "anthropic" | "ollama" | "lmstudio" | etc.
    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # provider-specific fields


class LLMClient(Protocol):
    """Defines the minimal interface used by spec-pipeline tools.

    Each concrete client instance is configured (provider + model + defaults).
    The chat() call uses those defaults and returns the model text + metadata.
    """

    config: LLMConfig

    def chat(self, *, messages: ChatMessages) -> LLMResult:
        ...