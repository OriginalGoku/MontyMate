from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

JSON = Dict[str, Any]


@dataclass(frozen=True)
class ProviderConfig:
    name: str                       # "openai" | "ollama" | "lmstudio"
    kind: Literal["openai_chat", "openai_compat_chat", "ollama_chat"]
    base_url: str                   # e.g. "https://api.openai.com/v1" or "http://localhost:1234/v1"
    api_key_env: Optional[str] = None
    api_key_value: Optional[str] = None


@dataclass(frozen=True)
class ProviderResponse:
    status: Literal["OK", "ERROR", "TIMEOUT"]
    provider: str
    model: str
    text: str
    raw: JSON

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None