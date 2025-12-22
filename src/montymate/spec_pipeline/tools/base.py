# src/montymate/spec_pipeline/tools/base.py
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class ToolError(RuntimeError):
    def __init__(self, message: str, data: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.data = data

    def __str__(self) -> str:
        return self.message


class BaseTool(ABC):
    """
    Base class for all spec-pipeline tools.

    Common behaviors:
    - Stable identity: name/version/tags
    - Single entrypoint: __call__ wraps run() with timing + consistent ToolError
    - Shared input guards: require(), require_non_empty()
    """

    # Override in subclasses
    name: str = "base_tool"
    version: str = "0.1.0"
    tags: tuple[str, ...] = ()

    @abstractmethod
    def run(self, *, inputs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        """
        Tool implementation.

        - inputs: plain dict payload
        - ctx: context object (at minimum should include run_id + step)
        - returns: plain dict output
        """
        raise NotImplementedError

    def __call__(self, *, inputs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        """
        Safe wrapper around run():
        - adds timing metadata
        - normalizes unexpected exceptions into ToolError
        """
        t0 = time.perf_counter()
        try:
            out = self.run(inputs=inputs, ctx=ctx)
            if not isinstance(out, dict):
                raise ToolError(
                    f"{self.name} Tool returned non-dict output",
                    data={"tool": self.name, "type": str(type(out))},
                )
            return {
                **out,
                "_meta": {
                    "tool": self.name,
                    "version": self.version,
                    "tags": list(self.tags),
                    "run_id": getattr(ctx, "run_id", None),
                    "step": getattr(ctx, "step", None),
                    "duration_ms": int((time.perf_counter() - t0) * 1000),
                },
            }
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(
                "Tool crashed",
                data={
                    "tool": self.name,
                    "error": str(e),
                    "run_id": getattr(ctx, "run_id", None),
                    "step": getattr(ctx, "step", None),
                },
            ) from e

    # --------------------
    # Input guards
    # --------------------
    @staticmethod
    def require(inputs: Mapping[str, Any], *keys: str) -> None:
        missing = [k for k in keys if k not in inputs]
        if missing:
            raise ToolError("Missing required inputs", data={"missing": missing})

    @staticmethod
    def require_non_empty(inputs: Mapping[str, Any], *keys: str) -> None:
        """
        Requires keys exist AND their values are not empty.

        Empty rules:
        - None -> empty
        - str -> empty if strip() == ""
        - list/tuple/set/dict -> empty if len == 0
        - everything else -> empty if falsy
        """
        BaseTool.require(inputs, *keys)

        empty: list[str] = []
        for k in keys:
            v = inputs.get(k)

            if v is None:
                empty.append(k)
                continue

            if isinstance(v, str):
                if v.strip() == "":
                    empty.append(k)
                continue

            if isinstance(v, (list, tuple, set, dict)):
                if len(v) == 0:
                    empty.append(k)
                continue

        if empty:
            raise ToolError("Required inputs must be non-empty", data={"empty": empty})
