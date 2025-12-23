# src/montymate/spec_pipeline/tools/base.py

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from ..errors import ToolError
from ..llm.llm_client import LLMClient
from ..utils.input_guards import require as _require
from ..utils.input_guards import require_non_empty as _require_non_empty


class BaseTool(ABC):
    """Defines the common execution contract for spec-pipeline tools.

    The base class provides:
    - stable identity (name/version/tags)
    - a single entrypoint (__call__) that wraps run() with timing and consistent errors
    - convenience wrappers for input guards

    The LLM client is passed explicitly to keep tool execution deterministic and easy
    to test. Tools may still support per-call override and/or a default client at the
    concrete tool level.
    """

    name: str = "base_tool"
    version: str = "0.1.0"
    tags: tuple[str, ...] = ()

    @abstractmethod
    def run(
        self,
        *,
        inputs: dict[str, object],
        ctx: object,
        llm: LLMClient | None = None,
    ) -> dict[str, object]:
        """Runs the tool implementation."""
        raise NotImplementedError

    def __call__(
        self,
        *,
        inputs: dict[str, object],
        ctx: object,
        llm: LLMClient | None = None,
    ) -> dict[str, object]:
        """Executes the tool with consistent timing and error handling."""
        t0 = time.perf_counter()
        try:
            out = self.run(inputs=inputs, ctx=ctx, llm=llm)
            if not isinstance(out, dict):
                raise ToolError(
                    f"{self.name} returned non-dict output",
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
                    "error": f"{type(e).__name__}: {e}",
                    "run_id": getattr(ctx, "run_id", None),
                    "step": getattr(ctx, "step", None),
                },
            ) from e

    # --------------------
    # Guard wrappers - removable once the code stabilizes
    # --------------------
    @staticmethod
    def require(inputs: dict[str, object], *keys: str) -> None:
        _require(inputs, *keys)

    @staticmethod
    def require_non_empty(inputs: dict[str, object], *keys: str) -> None:
        _require_non_empty(inputs, *keys)