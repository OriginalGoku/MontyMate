from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

JSON = Dict[str, Any]


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: JSON
    raw_text: str


def post_json(
    *,
    url: str,
    payload: JSON,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: int = 120,
) -> HttpResult:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            body = json.loads(raw) if raw.strip() else {}
            return HttpResult(status=resp.status, body=body, raw_text=raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            body = json.loads(raw) if raw.strip() else {}
        except Exception:
            body = {"raw": raw}
        return HttpResult(status=int(e.code), body=body, raw_text=raw)