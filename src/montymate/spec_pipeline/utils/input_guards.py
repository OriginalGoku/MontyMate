# src/montymate/spec_pipeline/utils/input_guards.py
# This file is used mostly by Tools for checking input values
from __future__ import annotations

from collections.abc import Mapping

from ..errors import ToolError


def require(inputs: Mapping[str, object], *keys: str) -> None:
    """Ensures required keys exist in the inputs mapping."""
    missing = [k for k in keys if k not in inputs]
    if missing:
        raise ToolError("Missing required inputs", data={"missing": missing})


def require_non_empty(inputs: Mapping[str, object], *keys: str) -> None:
    """Ensures required keys exist and their values are non-empty.

    Empty rules:
    - None is empty
    - str is empty when strip() == ""
    - list/tuple/set/dict is empty when len == 0
    - everything else is empty when falsy
    """
    require(inputs, *keys)

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

        if not v:
            empty.append(k)

    if empty:
        raise ToolError("Required inputs must be non-empty", data={"empty": empty})
