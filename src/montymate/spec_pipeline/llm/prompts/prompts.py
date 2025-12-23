# src/montymate/spec_pipeline/llm/prompts/prompts.py
from __future__ import annotations

import json
from collections.abc import Mapping

import yaml

from .base import BasePrompt
from ...data.component_metadata import ComponentMetadata
from ...data.human_inputs import HumanAnswerBatch
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
    """Normalizes human answers into a JSON-serializable dict."""
    v = inputs.get(key)
    if isinstance(v, HumanAnswerBatch):
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
    if isinstance(v, Mapping):
        return {str(k): v[k] for k in v.keys()}
    return {}

class SpecAuthorPrompt(BasePrompt):
    """Builds the prompt messages used by the SpecAuthor tool.

    The prompt supports:
    - Draft phase: no meaningful seed spec exists, so the specification is drafted from scratch.
    - Revision phase: a meaningful seed spec exists, so the specification is revised with minimal churn.
    - Answer integration: when human answers are present, answers are integrated and take priority over the seed.
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
            f"- Output MUST contain ONLY these keys: {keys_line}",
            "- Output MUST be a complete spec (all keys present).",
            "- All list items MUST be YAML strings. If a list item contains ':' it MUST be quoted.",
        ]

        if not seed_non_empty:
            lines.extend(
                [
                    "- No meaningful seed spec exists, so the task is to draft the specification from scratch.",
                    "- The spec should be inferred from the user prompt and written as a practical, testable draft.",
                    "- Reasonable assumptions should be made when details are missing, and those assumptions should be placed in assumptions.",
                ]
            )
        else:
            lines.extend(
                [
                    "- A meaningful seed spec exists, so the task is to revise the seed spec rather than rewrite it.",
                    "- Churn should be minimized: unchanged items should remain unchanged unless new information requires updates.",
                    "- The revised spec should remain internally consistent and should not drop existing requirements without a clear reason.",
                ]
            )

            if answers_present:
                lines.extend(
                    [
                        "- Human answers are present and must be integrated into the spec.",
                        "- If an answer is empty, the component should decide reasonably and update the spec accordingly.",
                        "- Conflicts between the seed spec and answers should be resolved by updating the spec to reflect the new information. Answers have a higher priority.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "- No new human answers are present, so revisions should focus on clarity, consistency, and completeness.",
                        "- Wording improvements are allowed, but unnecessary restructuring should be avoided.",
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
                "\nHUMAN ANSWERS:\n"
                + json.dumps(answers_payload, ensure_ascii=False, indent=2).rstrip()
                + "\n"
            )

        return "".join(parts).strip() + "\n"





class SpecCriticPrompt(BasePrompt):
    """Builds the prompt used by the SpecCritic component.

    The prompt:
    - receives a spec (dict-like)
    - asks for a strict JSON report with issues + targeted questions
    """

    def system_extras(self, *, inputs: Mapping[str, object]) -> str:
        _ = inputs
        return (
            "- Output MUST be a single JSON object only.\n"
            "- Output MUST contain ONLY these keys: passed, issues, targeted_questions.\n"
            "- Schema:\n"
            "  - passed: boolean\n"
            "  - issues: list of strings\n"
            "  - targeted_questions: list of strings\n"
            "- passed is true only if the spec is clear, complete, and implementable without further questions.\n"
            "- If passed is true, issues MUST be empty and targeted_questions MUST be empty.\n"
            "- targeted_questions MUST be specific and directly answerable by a human.\n"
            "- Prefer 3–7 targeted_questions max.\n"
        )
        
    def user_message(self, *, inputs: Mapping[str, object]) -> str:
        spec_payload = _get_spec_payload(inputs, "spec")
        return (
            "MODULE SPEC:\n"
            + json.dumps(spec_payload, ensure_ascii=False, indent=2).rstrip()
            + "\n"
        )
        
        
from montymate.spec_pipeline.data.component_metadata import ComponentMetadata, OutputFormat
from montymate.spec_pipeline.data.human_inputs import HumanAnswerBatch
from montymate.spec_pipeline.data.spec_types import Spec
from montymate.spec_pipeline.llm.llm_client import ChatMessages


def _print_messages(title: str, messages: ChatMessages) -> None:
    """Prints prompt messages in a readable, deterministic form."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, m in enumerate(messages, start=1):
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        print(f"\n--- message #{i} ({role}) ---\n{content}")


def main() -> None:
    """Runs smoke tests that demonstrate how the prompt changes across phases."""
    metadata = ComponentMetadata(
        name="SpecAuthor",
        output_format=OutputFormat.YAML,
        description=(
            "SpecAuthor produces and maintains a structured software specification. "
            "It outputs a complete specification object in YAML form and keeps the spec internally consistent."
        ),
        tags=("spec", "author"),
    )

    prompt = SpecAuthorPrompt(metadata=metadata)

    # Case 1: Draft phase (no seed, no answers)
    inputs_case_1: dict[str, object] = {
        "user_prompt": "Build a CLI that renames files in a folder using a pattern and supports dry-run.",
    }
    _print_messages(
        "CASE 1 — Draft phase (user_prompt only)",
        prompt.build(inputs=inputs_case_1),
    )

    # Case 2: Revision phase (seed exists, no answers)
    seed_spec_2 = Spec.from_dict(
        {
            "status": "DRAFT",
            "goal": "Rename files in a folder using a pattern.",
            "functional_requirements": ["Support renaming by regex find/replace", "Provide a preview mode"],
            "constraints": ["Python 3.11+"],
            "security_concerns": [],
            "assumptions": ["User has filesystem permissions"],
            "other_notes": "",
        }
    )
    inputs_case_2: dict[str, object] = {
        "user_prompt": "Build a CLI that renames files in a folder using a pattern and supports dry-run.",
        "seed_spec": seed_spec_2,
    }
    _print_messages(
        "CASE 2 — Revision phase (seed_spec, no answers)",
        prompt.build(inputs=inputs_case_2),
    )

    # Case 3: Revision phase (seed exists + human answers)
    batch_3 = HumanAnswerBatch.from_dict(
        {
            "round": 1,
            "answers": [
                {"question": "Should the tool support recursive folders?", "answer": "Yes, optional flag --recursive."},
                {"question": "Should it overwrite existing files?", "answer": "No, refuse unless --force is provided."},
                {"question": "What should happen if answer is empty?", "answer": ""},
            ],
        }
    )
    inputs_case_3: dict[str, object] = {
        "user_prompt": "Build a CLI that renames files in a folder using a pattern and supports dry-run.",
        "seed_spec": seed_spec_2,
        "answers": batch_3,
    }
    _print_messages(
        "CASE 3 — Revision phase (seed_spec + answers)",
        prompt.build(inputs=inputs_case_3),
    )


if __name__ == "__main__":
    main()