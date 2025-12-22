from __future__ import annotations

from typing import List, Optional

from montymate.core.types import JSON
from .http import post_json
from .types import ProviderConfig, ProviderResponse


def ollama_chat_complete(
    *,
    cfg: ProviderConfig,
    model: str,
    messages: List[JSON],
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    timeout_s: int = 120,
) -> ProviderResponse:
    base = cfg.base_url.rstrip("/")
    url = f"{base}/api/chat"  # Ollama chat endpoint  [oai_citation:10‡Ollama Docs](https://docs.ollama.com/api/chat?utm_source=chatgpt.com)

    options: JSON = {}
    if temperature is not None:
        options["temperature"] = float(temperature)
    if max_output_tokens is not None:
        options["num_predict"] = int(max_output_tokens)

    payload: JSON = {
        "model": model,
        "messages": messages,
        "stream": False,  # disable streaming so we get one JSON response  [oai_citation:11‡ollama.readthedocs.io](https://ollama.readthedocs.io/en/api/?utm_source=chatgpt.com)
    }
    if options:
        payload["options"] = options

    r = post_json(url=url, payload=payload, headers=None, timeout_s=timeout_s)

    if r.status < 200 or r.status >= 300:
        return ProviderResponse(status="ERROR", provider=cfg.name, model=model, text="", raw=r.body)

    # Ollama chat response includes: { message: { role, content }, ... }  [oai_citation:12‡Ollama Docs](https://docs.ollama.com/api/chat?utm_source=chatgpt.com)
    msg = r.body.get("message") or {}
    content = msg.get("content") or ""

    return ProviderResponse(
        status="OK",
        provider=cfg.name,
        model=model,
        text=content,
        raw=r.body,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
    )