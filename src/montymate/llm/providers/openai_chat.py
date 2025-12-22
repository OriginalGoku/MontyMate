from __future__ import annotations

import os
from typing import Dict, List, Optional

from montymate.core.types import JSON
from .http import post_json
from .types import ProviderConfig, ProviderResponse


def openai_chat_complete(
    *,
    cfg: ProviderConfig,
    model: str,
    messages: List[JSON],
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    timeout_s: int = 120,
) -> ProviderResponse:
    base = cfg.base_url.rstrip("/")
    url = f"{base}/chat/completions"

    api_key = cfg.api_key_value
    if not api_key and cfg.api_key_env:
        api_key = os.environ.get(cfg.api_key_env)

    if not api_key:
        return ProviderResponse(
            status="ERROR",
            provider=cfg.name,
            model=model,
            text="Missing API key",
            raw={"error": "Missing API key", "expected_env": cfg.api_key_env},
        )

    headers = {"Authorization": f"Bearer {api_key}"}

    payload: JSON = {"model": model, "messages": messages}
    if max_output_tokens is not None:
        payload["max_tokens"] = int(max_output_tokens)
    if temperature is not None:
        payload["temperature"] = float(temperature)

    r = post_json(url=url, payload=payload, headers=headers, timeout_s=timeout_s)

    if r.status < 200 or r.status >= 300:
        return ProviderResponse(status="ERROR", provider=cfg.name, model=model, text="", raw=r.body)

    # OpenAI Chat Completions response: choices[0].message.content, usage fields when available  [oai_citation:4‡OpenAI Platform](https://platform.openai.com/docs/api-reference/chat?utm_source=chatgpt.com)
    choices = r.body.get("choices") or []
    content = ""
    if choices and isinstance(choices, list):
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content") or ""

    usage = r.body.get("usage") or {}
    return ProviderResponse(
        status="OK",
        provider=cfg.name,
        model=model,
        text=content,
        raw=r.body,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )