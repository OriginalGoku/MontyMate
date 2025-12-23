# src/montymate/spec_pipeline/llm/providers/ollama_client.py
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..llm_client import ChatMessages, LLMConfig, LLMResult


JsonDict = dict[str, Any]


def _post_json(*, url: str, payload: JsonDict, timeout_s: float = 60.0) -> JsonDict:
    """Sends a JSON POST request and returns a parsed JSON object."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body) if body else {}
    return obj if isinstance(obj, dict) else {"raw": obj}


@dataclass(slots=True)
class OllamaClient:
    """Calls the Ollama /api/chat endpoint (typically local).

    This client keeps provider configuration on the instance (LLMConfig).
    Provider-specific settings are read from config.extra.

    Supported config.extra keys (optional):
    - base_url: str (default: http://localhost:11434)
    - timeout_s: float (default: 60.0)
    - options: dict (merged into Ollama "options")
    - request: dict (merged into request payload)
    """

    config: LLMConfig

    def chat(self, *, messages: ChatMessages) -> LLMResult:
        base_url = str(self.config.extra.get("base_url") or "http://localhost:11434").rstrip("/")
        timeout_s = float(self.config.extra.get("timeout_s") or 60.0)
        url = f"{base_url}/api/chat"

        payload: JsonDict = {
            "model": self.config.model,
            "messages": list(messages),
            "stream": False,
        }

        options: JsonDict = {}
        extra_options = self.config.extra.get("options")
        if isinstance(extra_options, dict):
            options.update(extra_options)

        if self.config.temperature is not None:
            options.setdefault("temperature", float(self.config.temperature))
        if self.config.max_tokens is not None:
            # Ollama uses num_predict for output token budget.
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
            raise RuntimeError(f"Ollama HTTPError {getattr(e, 'code', None)}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama URLError: {e}") from e
        
        if raw.get("error"):
            raise RuntimeError(f"Ollama error: {raw['error']}")
        
        
        # Ollama returns {"message": {"role": "...", "content": "..."}, ...}
        text = ""
        msg = raw.get("message")
        
        if isinstance(msg, dict):
            content = msg.get("content")
            text = content.strip() if isinstance(content, str) else ""
        
        if not text:
            print(f"ollama_client.py raw output -> {raw}")
            raise RuntimeError(f"Ollama error: No content in response")
            
        # Ollama sometimes returns eval_count / prompt_eval_count instead of OpenAI usage
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
            input_tokens=int(input_tokens) if isinstance(input_tokens, int) else None,
            output_tokens=int(output_tokens) if isinstance(output_tokens, int) else None,
            total_tokens=int(total_tokens) if isinstance(total_tokens, int) else None,
            raw=raw,
        )

def main() -> None:
    from montymate.spec_pipeline.llm.llm_client import LLMConfig

    client = OllamaClient(
        config=LLMConfig(
            provider="ollama",
            model="Osmosis/Osmosis-Structure-0.6B:latest", 
            max_tokens=64,
            temperature=0.0,
            extra={"base_url": "http://localhost:11434"},
        )
    )

    r = client.chat(
        messages=[
            {"role": "system", "content": "The assistant replies with a short greeting."},
            {"role": "user", "content": "Hello world"},
        ]
    )

    print("status:", r.status)
    print("provider:", r.provider)
    print("model:", r.model)
    print("text:", r.text.strip())
    print("input_tokens:", r.input_tokens)
    print("output_tokens:", r.output_tokens)
    print("total_tokens:", r.total_tokens)


if __name__ == "__main__":
    main()