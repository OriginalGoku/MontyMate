# src/montymate/spec_pipeline/llm/providers/ollama_client.py
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from ..llm_client import ChatMessages, LLMConfig, LLMError, LLMResult

JsonDict = dict[str, object]


def _post_json(*, url: str, payload: JsonDict, timeout_s: float = 60.0) -> JsonDict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body) if body else {}
    return obj if isinstance(obj, dict) else {"raw": obj}


@dataclass(slots=True)
class OllamaClient:
    config: LLMConfig

    def chat(self, *, messages: ChatMessages) -> LLMResult:
        base_url = str(self.config.extra.get("base_url") or "http://localhost:11434").rstrip("/")
        timeout_s = float(self.config.extra.get("timeout_s") or 60.0)
        url = f"{base_url}/api/chat"

        payload: JsonDict = {"model": self.config.model, "messages": list(messages), "stream": False}

        options: JsonDict = {}
        extra_options = self.config.extra.get("options")
        if isinstance(extra_options, dict):
            options.update(extra_options)

        if self.config.temperature is not None:
            options.setdefault("temperature", float(self.config.temperature))
        if self.config.max_tokens is not None:
            options.setdefault("num_predict", int(self.config.max_tokens))

        if options:
            payload["options"] = options

        extra_request = self.config.extra.get("request")
        if isinstance(extra_request, dict):
            payload.update(extra_request)

        try:
            raw = _post_json(url=url, payload=payload, timeout_s=timeout_s)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise LLMError("Ollama HTTPError", data={"code": getattr(e, "code", None), "body": body}) from e
        except urllib.error.URLError as e:
            raise LLMError("Ollama URLError", data={"error": str(e)}) from e

        err = raw.get("error")
        if isinstance(err, str) and err.strip():
            raise LLMError("Ollama error", data={"error": err})

        msg = raw.get("message")
        content = ""
        thinking = ""
        if isinstance(msg, dict):
            c = msg.get("content")
            t = msg.get("thinking")
            if isinstance(c, str):
                content = c.strip()
            if isinstance(t, str):
                thinking = t.strip()

        # If the model hit length before emitting "final", Ollama can return thinking-only.
        text = content or thinking
        if not text:
            raise LLMError(
                "Ollama returned empty response text",
                data={"done_reason": raw.get("done_reason"), "eval_count": raw.get("eval_count")},
            )

        input_tokens = raw.get("prompt_eval_count")
        output_tokens = raw.get("eval_count")
        total_tokens = None
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            total_tokens = input_tokens + output_tokens

        return LLMResult(
            status="OK",
            text=text,
            provider=self.config.provider,
            model=self.config.model,
            input_tokens=int(input_tokens) if isinstance(input_tokens, int) else 0,
            output_tokens=int(output_tokens) if isinstance(output_tokens, int) else 0,
            total_tokens=int(total_tokens) if isinstance(total_tokens, int) else 0,
            raw=raw,
            error=None,
        )