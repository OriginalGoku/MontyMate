from __future__ import annotations

from typing import List, Optional

from montymate.core.types import JSON
from .http import post_json
from .types import ProviderConfig, ProviderResponse


def openai_compat_chat_complete(
    *,
    cfg: ProviderConfig,
    model: str,
    messages: List[JSON],
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    timeout_s: int = 120,
) -> ProviderResponse:
    base = cfg.base_url.rstrip("/")
    # LM Studio supports /v1/chat/completions as OpenAI-compatible endpoint  [oai_citation:7‡LM Studio](https://lmstudio.ai/docs/developer/openai-compat?utm_source=chatgpt.com)
    url = f"{base}/chat/completions"

    headers = {}
    # LM Studio typically does not require an API key; but if you want, you can still pass one.  [oai_citation:8‡LM Studio](https://lmstudio.ai/docs/developer?utm_source=chatgpt.com)
    if cfg.api_key_value:
        headers["Authorization"] = f"Bearer {cfg.api_key_value}"

    payload: JSON = {"model": model, "messages": messages}
    if max_output_tokens is not None:
        payload["max_tokens"] = int(max_output_tokens)
    if temperature is not None:
        payload["temperature"] = float(temperature)

    r = post_json(url=url, payload=payload, headers=headers, timeout_s=timeout_s)

    if r.status < 200 or r.status >= 300:
        return ProviderResponse(status="ERROR", provider=cfg.name, model=model, text="", raw=r.body)

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