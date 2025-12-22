# src/montymate/spec_pipeline/llm/providers/openai_client.py
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from streamlit import form

from ..llm_client import ChatMessages, LLMConfig, LLMResult


JsonDict = dict[str, Any]


def _post_json(*, url: str, headers: dict[str, str], payload: JsonDict, timeout_s: float = 60.0) -> JsonDict:
    """Sends a JSON POST request and returns a parsed JSON object."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body) if body else {}
    return obj if isinstance(obj, dict) else {"raw": obj}


@dataclass(slots=True)
class OpenAIClient:
    """Calls the OpenAI Chat Completions endpoint.

    This client keeps provider configuration on the instance (LLMConfig).
    Provider-specific settings are read from config.extra.

    Supported config.extra keys (optional):
    - base_url: str (default: https://api.openai.com)
    - api_key_env: str (default: OPENAI_API_KEY)
    - timeout_s: float (default: 60.0)
    - request: dict (merged into request payload)
    """

    config: LLMConfig

    def chat(self, *, messages: ChatMessages) -> LLMResult:
        base_url = str(self.config.extra.get("base_url") or "https://api.openai.com").rstrip("/")
        api_key_env = str(self.config.extra.get("api_key_env"))
        timeout_s = float(self.config.extra.get("timeout_s") or 60.0)

        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {api_key_env}")

        url = f"{base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: JsonDict = {
            "model": self.config.model,
            "messages": list(messages),
        }
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
    """Runs a minimal smoke test against OpenAI and prints the returned text."""
    from dotenv import load_dotenv

    _ = load_dotenv()

    import os
    import sys
    openai_key = os.environ.get("OPENAI_API_KEY")
    print("Starting")
    # This check keeps the failure message friendly.
    if not (openai_key or "").strip():
        print("OPENAI_API_KEY environment variable is required.", file=sys.stderr)
        raise SystemExit(2)

    model = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()

    client = OpenAIClient(
        config=LLMConfig(
            provider="openai",
            model=model,
            max_tokens=None,
            temperature=None,
            extra={
                # This client reads the key from the environment.
                "api_key_env": openai_key,
                # Optional overrides:
                # "base_url": "https://api.openai.com",
                # "timeout_s": 60.0,
            },
        )
    )

    result = client.chat(
        messages=[
            {"role": "system", "content": "The assistant replies concisely."},
            {"role": "user", "content": "Say hello world."},
        ]
    )

    print(f'Here are the result.text: {result.text}')

    
