from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

JSON = Dict[str, Any]


@dataclass(frozen=True)
class ProviderConfig:
    name: str  # "lmstudio" | "ollama" | "openai"
    kind: str  # "openai_compat_chat" | "ollama_chat" | "openai_chat"
    base_url: str
    api_key_env: Optional[str] = None


@dataclass(frozen=True)
class ProviderResult:
    status: str  # "OK" | "ERROR"
    provider: str
    model: str
    text: str
    raw: JSON
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


def _post_json(
    url: str,
    payload: JSON,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: int = 120,
) -> Tuple[int, JSON]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            return int(e.code), (json.loads(raw) if raw.strip() else {"raw": raw})
        except Exception:
            return int(e.code), {"raw": raw}


def call_openai_chat(
    cfg: ProviderConfig,
    model: str,
    messages: List[JSON],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> ProviderResult:
    api_key = os.environ.get(cfg.api_key_env or "OPENAI_API_KEY")
    if not api_key:
        return ProviderResult(
            status="ERROR",
            provider=cfg.name,
            model=model,
            text="Missing OPENAI_API_KEY",
            raw={"error": "missing api key"},
        )

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload: JSON = {"model": model, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if temperature is not None:
        payload["temperature"] = float(temperature)

    status, body = _post_json(url, payload, headers=headers)
    if status < 200 or status >= 300:
        return ProviderResult(
            status="ERROR", provider=cfg.name, model=model, text="", raw=body
        )

    choices = body.get("choices") or []
    content = ""
    if choices:
        content = (((choices[0] or {}).get("message") or {}).get("content")) or ""

    usage = body.get("usage") or {}
    return ProviderResult(
        status="OK",
        provider=cfg.name,
        model=model,
        text=content,
        raw=body,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def call_openai_compat_chat(
    cfg: ProviderConfig,
    model: str,
    messages: List[JSON],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> ProviderResult:
    # LM Studio: OpenAI-compatible /v1/chat/completions.  [oai_citation:8‡Streamlit Docs](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader?utm_source=chatgpt.com)
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload: JSON = {"model": model, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if temperature is not None:
        payload["temperature"] = float(temperature)

    status, body = _post_json(url, payload, headers=None)
    if status < 200 or status >= 300:
        return ProviderResult(
            status="ERROR", provider=cfg.name, model=model, text="", raw=body
        )

    choices = body.get("choices") or []
    content = ""
    if choices:
        content = (((choices[0] or {}).get("message") or {}).get("content")) or ""

    usage = body.get("usage") or {}
    return ProviderResult(
        status="OK",
        provider=cfg.name,
        model=model,
        text=content,
        raw=body,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def call_ollama_chat(
    cfg: ProviderConfig,
    model: str,
    messages: List[JSON],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> ProviderResult:
    # Ollama: /api/chat, streaming disabled via {"stream": false}.  [oai_citation:9‡Ollama](https://ollama.readthedocs.io/en/api/?utm_source=chatgpt.com)
    url = cfg.base_url.rstrip("/") + "/api/chat"
    payload: JSON = {"model": model, "messages": messages, "stream": False}
    options: JSON = {}
    if max_tokens is not None:
        options["num_predict"] = int(max_tokens)
    if temperature is not None:
        options["temperature"] = float(temperature)
    if options:
        payload["options"] = options

    status, body = _post_json(url, payload, headers=None)
    if status < 200 or status >= 300:
        return ProviderResult(
            status="ERROR", provider=cfg.name, model=model, text="", raw=body
        )

    msg = body.get("message") or {}
    content = msg.get("content") or ""
    return ProviderResult(
        status="OK", provider=cfg.name, model=model, text=content, raw=body
    )


def call_provider(
    cfg: ProviderConfig,
    model: str,
    messages: List[JSON],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> ProviderResult:
    if cfg.kind == "openai_chat":
        return call_openai_chat(cfg, model, messages, max_tokens, temperature)
    if cfg.kind == "openai_compat_chat":
        return call_openai_compat_chat(cfg, model, messages, max_tokens, temperature)
    if cfg.kind == "ollama_chat":
        return call_ollama_chat(cfg, model, messages, max_tokens, temperature)
    return ProviderResult(
        status="ERROR",
        provider=cfg.name,
        model=model,
        text="",
        raw={"error": f"unsupported kind {cfg.kind}"},
    )
