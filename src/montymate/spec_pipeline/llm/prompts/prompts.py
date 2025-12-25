# src/montymate/spec_pipeline/llm/prompts/prompts.py
from __future__ import annotations

import json
from collections.abc import Mapping

import yaml

from .base import BasePrompt
from ...data.component_metadata import ComponentMetadata
from ...data.human_inputs import ClarificationBatch
from ...data.spec_types import Spec


def _get_str(inputs: Mapping[str, object], key: str) -> str:
    v = inputs.get(key)
    return v.strip() if isinstance(v, str) else ""


def _get_spec_dict(inputs: Mapping[str, object], key: str) -> dict[str, object]:
    v = inputs.get(key)
    if isinstance(v, Spec):
        return dict(v.to_dict())
    if isinstance(v, Mapping):
        return {str(k): v[k] for k in v.keys()}
    return {}


def _get_answers_payload(inputs: Mapping[str, object], key: str) -> dict[str, object]:
    """Normalizes answers into a JSON-serializable dict."""
    v = inputs.get(key)
    if isinstance(v, ClarificationBatch):
        return dict(v.to_dict())
    if isinstance(v, Mapping):
        return {str(k): v[k] for k in v.keys()}
    return {}


def _is_non_empty_seed(seed: Mapping[str, object]) -> bool:
    """Treats a seed as non-empty if any non-status field has meaningful content."""
    for k, v in seed.items():
        if str(k) == "status":
            continue

        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return True
        if isinstance(v, dict) and len(v) > 0:
            return True

    return False


def _has_answers_payload(payload: Mapping[str, object]) -> bool:
    """Treats answers as present if an answers list exists and contains at least one item."""
    answers = payload.get("answers")
    return isinstance(answers, list) and len(answers) > 0

def _get_spec_payload(inputs: Mapping[str, object], key: str) -> dict[str, object]:
    v = inputs.get(key)
    if isinstance(v, Spec):
        return dict(v.to_dict())
    if isinstance(v, Mapping):
        return {str(k): v[k] for k in v.keys()}
    return {}

class SpecAuthorPrompt(BasePrompt):
    """Builds the prompt messages used by the SpecAuthor tool.

    Responsibilities:
    - Draft phase: if seed spec is empty -> draft a full spec from scratch.
    - Revision phase: if seed spec is meaningful -> revise with minimal churn.
    - Answer integration: if answers are present -> integrate them, prioritizing human answers.
    """

    def __init__(self, *, metadata: ComponentMetadata) -> None:
        self.metadata = metadata

    def system_extras(self, *, inputs: Mapping[str, object]) -> str:
        seed_spec = _get_spec_dict(inputs, "seed_spec")
        answers_payload = _get_answers_payload(inputs, "answers")

        seed_non_empty = _is_non_empty_seed(seed_spec)
        answers_present = _has_answers_payload(answers_payload)

        keys_line = ", ".join(Spec.keys())

        lines: list[str] = [
            # Output contract
            f"- Output MUST contain ONLY these keys: {keys_line}",
            "- Output MUST be valid YAML only.",
            "- No markdown fences.",
            "- No extra commentary.",
            "- Output MUST be a complete spec (all keys present).",
            "- All list items MUST be YAML strings. If a list item contains ':' it MUST be quoted.",
            "",
            # Global consistency rules
            "- Keep the spec internally consistent (no contradictions).",
            "- Do not invent unrelated requirements; stay aligned with the user prompt and any provided answers.",
            "- Prefer concrete, testable requirements over vague language.",
        ]

        if not seed_non_empty:
            lines.extend(
                [
                    "",
                    "Phase: DRAFT",
                    "- No meaningful seed spec exists: draft the specification from scratch.",
                    "- Infer a reasonable MVP scope from the user prompt.",
                    "- When details are missing, make reasonable assumptions and record them under assumptions.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Phase: REVISION",
                    "- A meaningful seed spec exists: revise it rather than rewriting.",
                    "- Minimize churn: keep unchanged list items unchanged unless new info requires a change.",
                    "- Do not drop existing requirements without a clear reason.",
                ]
            )

            if answers_present:
                lines.extend(
                    [
                        "",
                        "Answer integration:",
                        "- Answers are present and MUST be integrated.",
                        "- Answers have higher priority than the seed spec.",
                        "- If an answer has decide_for_me=true or is blank, make a reasonable decision and update the spec accordingly.",
                        "- If an answer is_llm_generated=true, treat it as helpful guidance but still subordinate to any human-provided answers.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        "No answers present:",
                        "- Improve clarity, completeness, and consistency without unnecessary restructuring.",
                        "- You may reword items for clarity, but avoid reorganizing lists unless needed for correctness.",
                    ]
                )

        return "\n".join(lines)

    def user_message(self, *, inputs: Mapping[str, object]) -> str:
        user_prompt = _get_str(inputs, "user_prompt")
        seed_spec = _get_spec_dict(inputs, "seed_spec")
        answers_payload = _get_answers_payload(inputs, "answers")

        parts: list[str] = [f"USER PROMPT:\n{user_prompt}\n"]

        if _is_non_empty_seed(seed_spec):
            parts.append(
                "\nSEED SPEC:\n"
                + yaml.safe_dump(seed_spec, sort_keys=False, allow_unicode=True).rstrip()
                + "\n"
            )

        if _has_answers_payload(answers_payload):
            parts.append(
                "\nCLARIFICATIONS / ANSWERS:\n"
                + json.dumps(answers_payload, ensure_ascii=False, indent=2).rstrip()
                + "\n"
            )

        return "".join(parts).strip() + "\n"





class SpecAnswererPrompt(BasePrompt):
    """Builds the prompt used by the SpecAnswerer component.

    Purpose:
    - Receives targeted questions plus optional human-provided answers/constraints.
    - Produces strict plain-text Q/A/R blocks.
    - Never contradicts authoritative human answers.

    Modes:
    - delegate_only (default): answer only items with decide_for_me=true.
    - answer_all: answer every question; if a human answer exists, repeat it verbatim.
    """

    def system_extras(self, *, inputs: Mapping[str, object]) -> str:
        mode = str(inputs.get("mode") or "delegate_only").strip().lower()
        if mode not in ("delegate_only", "answer_all"):
            mode = "delegate_only"

        lines: list[str] = [
            "- Output MUST be valid TEXT only.",
            "- No markdown fences.",
            "- No headings, no bullet lists, no numbering, no JSON/YAML.",
            "- Output MUST contain ONLY Q/A/R blocks as specified below.",
            "",
            "Authoritativeness rules:",
            "- Any input item where decide_for_me=false AND answer is non-empty is an authoritative human constraint.",
            "- You MUST NOT contradict an authoritative human constraint.",
            "- If answering a delegated question depends on a human constraint, align with it and mention the alignment in R.",
            "",
            "Output format rules (STRICT):",
            "- Output MUST be a sequence of blocks separated by EXACTLY ONE blank line.",
            "- Each block MUST be exactly 3 non-empty lines in this order:",
            "  Q: <question text copied EXACTLY from the input (character-for-character)>",
            "  A: <answer text (no leading/trailing whitespace)>",
            "  R: <short reason, 1–2 sentences>",
            "- Do NOT add extra lines. Do NOT wrap lines. Do NOT add numbering.",
            "- Do NOT paraphrase or edit the Q line.",
            "",
            "Answer content rules:",
            "- A must be direct and implementation-oriented.",
            "- R must be short (1–2 sentences) and may state assumptions when needed.",
            "- If the question cannot be determined from context, choose a reasonable default and state the assumption in R.",
        ]

        if mode == "delegate_only":
            lines.extend(
                [
                    "",
                    "Scope rules (delegate_only):",
                    "- Only output blocks for items with decide_for_me=true.",
                    "- Do not output blocks for items that already have a non-empty answer in the input.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Scope rules (answer_all):",
                    "- Output one block for every input question (in the same order as provided).",
                    "- If a question already has a non-empty answer in the input, copy that answer EXACTLY into the A line (do not change it) and provide a reason in R.",
                ]
            )

        return "\n".join(lines)

    def user_message(self, *, inputs: Mapping[str, object]) -> str:
        # Expected inputs:
        # - inputs["context"] : Mapping, typically {"user_prompt": str, "seed_spec": {...}, "human_answer_constraints": [...]}
        # - inputs["questions"]: list of dicts like:
        #   {"question": "...", "answer": "...", "decide_for_me": bool, "is_llm_generated": bool}

        questions = inputs.get("questions")
        if not isinstance(questions, list):
            questions = []

        context = inputs.get("context")
        if not isinstance(context, Mapping):
            context = {}

        return (
            "CONTEXT:\n"
            + json.dumps(dict(context), ensure_ascii=False, indent=2).rstrip()
            + "\n\n"
            "QUESTIONS:\n"
            + json.dumps(list(questions), ensure_ascii=False, indent=2).rstrip()
            + "\n"
        )
        
class SpecCriticPrompt(BasePrompt):
    """Builds the prompt used by the SpecCritic tool.

    Responsibilities:
    - Evaluate whether the spec is implementable without further clarification.
    - If not, produce concrete issues + targeted questions that unblock implementation.
    """

    def __init__(self, *, metadata: ComponentMetadata) -> None:
        self.metadata = metadata

    def system_extras(self, *, inputs: Mapping[str, object]) -> str:
        _ = inputs

        return "\n".join(
            [
                # Output contract
                "- Output MUST be a single JSON object only.",
                "- Do NOT wrap in markdown fences.",
                "- Do NOT include any extra keys or commentary.",
                "- Output MUST contain ONLY these keys: passed, issues, targeted_questions.",
                "",
                "Schema (strict):",
                "- passed: boolean",
                "- issues: list of strings",
                "- targeted_questions: list of strings",
                "",
                "Evaluation rules:",
                "- passed is true only if the spec is clear, internally consistent, and implementable without asking any questions.",
                "- If passed is true: issues MUST be empty and targeted_questions MUST be empty.",
                "- If passed is false: issues MUST contain at least 1 item, and targeted_questions MUST contain at least 1 item.",
                "",
                "Issue rules:",
                "- issues should describe missing/ambiguous/contradictory requirements.",
                "- Each issue should be one sentence and refer to a specific gap (avoid generic statements).",
                "",
                "Question rules:",
                "- targeted_questions MUST be specific and directly answerable by a human.",
                "- Questions should be minimal: only ask what is necessary to implement correctly.",
                "- Prefer 3–7 targeted_questions; never exceed 10.",
                "- Avoid yes/no unless it truly resolves ambiguity; otherwise ask for concrete choices/values.",
                "- Do not propose implementation details unless the spec needs a decision (e.g., fixed timestep vs rAF).",
            ]
        )

    def user_message(self, *, inputs: Mapping[str, object]) -> str:
        spec_payload = _get_spec_payload(inputs, "spec")
        return (
            "MODULE SPEC (JSON):\n"
            + json.dumps(spec_payload, ensure_ascii=False, indent=2).rstrip()
            + "\n"
        )