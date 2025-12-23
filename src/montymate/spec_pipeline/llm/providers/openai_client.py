# src/montymate/spec_pipeline/llm/providers/openai_client.py
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..llm_client import ChatMessages, LLMConfig, LLMResult

JsonDict = dict[str, Any]


def _post_json(*, url: str, headers: dict[str, str], payload: JsonDict, timeout_s: float = 60.0) -> JsonDict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body) if body else {}
    return obj if isinstance(obj, dict) else {"raw": obj}


@dataclass(slots=True)
class OpenAIClient:
    config: LLMConfig

    def chat(self, *, messages: ChatMessages) -> LLMResult:
        base_url = str(self.config.extra.get("base_url") or "https://api.openai.com").rstrip("/")
        api_key_env = str(self.config.extra.get("api_key_env") or "OPENAI_API_KEY")
        timeout_s = float(self.config.extra.get("timeout_s") or 60.0)

        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {api_key_env}")

        url = f"{base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: JsonDict = {"model": self.config.model, "messages": list(messages)}
        if self.config.temperature is not None:
            payload["temperature"] = float(self.config.temperature)
        if self.config.max_tokens is not None:
            payload["max_tokens"] = int(self.config.max_tokens)

        extra_request = self.config.extra.get("request")
        if isinstance(extra_request, dict):
            payload.update(extra_request)

        try:
            raw = _post_json(url=url, headers=headers, payload=payload, timeout_s=timeout_s)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"OpenAI HTTPError {getattr(e, 'code', None)}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenAI URLError: {e}") from e

        text = ""
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict):
                    text = str(msg.get("content") or "")

        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        input_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        output_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None

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
    from dotenv import load_dotenv

    load_dotenv()

    print("Starting", flush=True)

    model = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()

    client = OpenAIClient(
        config=LLMConfig(
            provider="openai",
            model=model,
            max_tokens=64,
            temperature=0.2,
            extra={
                "api_key_env": "OPENAI_API_KEY",
            },
        )
    )

    result = client.chat(
        messages=[
            {"role": "system", "content": "The assistant replies concisely."},
            {"role": "user", "content": "Say hello world."},
        ]
    )

    print(f"Here is result.text: {result.text!r}", flush=True)


if __name__ == "__main__":
    main()