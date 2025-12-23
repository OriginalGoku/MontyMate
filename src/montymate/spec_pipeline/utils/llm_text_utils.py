# src/montymate/spec_pipeline/utils/llm_text_utils.py
from __future__ import annotations


def strip_code_fences(text: str) -> str:
    """Removes markdown code fences from a model response.

    The function supports responses that start with ``` or ```yaml/```json and end with ```.
    The function returns the original text if no fences are present.
    """
    s = (text or "").strip()
    if not s.startswith("```"):
        return s

    lines = s.splitlines()

    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()