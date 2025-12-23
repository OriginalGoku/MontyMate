from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ToolError(RuntimeError):
    """Represents an expected, structured failure raised by a tool.

    The error is intended for cases such as invalid inputs, invalid outputs,
    or other non-crash failures that should be reported consistently.

    The optional data payload is intended to carry machine-readable context
    (e.g., missing keys, validation details, partial outputs).
    """

    message: str
    data: dict[str, object] | None = None

    def __str__(self) -> str:
        return self.message
