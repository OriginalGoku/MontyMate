# src/montymate/spec_pipeline/dev/answer_round.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from montymate.spec_pipeline.data.human_inputs import HumanAnswer, HumanAnswerBatch
from montymate.spec_pipeline.data.spec_store import FileSpecStore


def _as_str_list(x: object) -> list[str]:
    if not isinstance(x, list):
        return []
    out: list[str] = []
    for item in x:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _confirm(prompt: str) -> bool:
    """Returns True when the user confirms, False otherwise."""
    while True:
        resp = input(f"{prompt} [y/N]: ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("", "n", "no"):
            return False
        print("Please answer y or n.")


def main() -> None:
    """Collects a HumanAnswerBatch for the current run round and persists it to disk.

    The command is intended to be used after the coordinator blocks on WAITING_FOR_HUMAN.
    It reads:
    - run_state.json to infer the round number (unless overridden)
    - spec_critic_report_round_<N>.json to obtain targeted questions
    It writes:
    - spec_answers_round_<N>.json
    """
    parser = argparse.ArgumentParser(prog="answer_round", description="Answer SpecCritic targeted questions for a run.")
    parser.add_argument("--run-id", required=True, help="Run identifier under .ai_module_factory/runs/<run_id>/")
    parser.add_argument("--project-root", default=".", help="Project root (default: current directory)")
    parser.add_argument("--round", type=int, default=None, help="Round number override (default: run_state.round_no)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing answers file for that round")

    args = parser.parse_args()

    store = FileSpecStore(project_root=Path(args.project_root), run_id=str(args.run_id))

    state = store.read_state()
    if state is None:
        print("No run_state.json found (or failed to parse).", file=sys.stderr)
        raise SystemExit(2)

    round_no = int(args.round) if args.round is not None else int(getattr(state, "round_no", 0))
    if round_no <= 0:
        print("Invalid round number. run_state.round_no must be >= 1, or pass --round.", file=sys.stderr)
        raise SystemExit(2)

    report = store.read_critic_report(round_no=round_no)
    if report is None:
        print(f"No critic report found for round {round_no}.", file=sys.stderr)
        raise SystemExit(2)

    issues = _as_str_list(report.get("issues"))
    questions = _as_str_list(report.get("targeted_questions"))

    if not questions:
        print(f"No targeted_questions found in critic report for round {round_no}.", file=sys.stderr)
        raise SystemExit(2)

    existing = store.read_answers_batch(round_no)
    if existing is not None and not bool(args.force):
        print(f"An answers file already exists for round {round_no}.")
        if not _confirm("Overwrite it?"):
            print("Aborted without changes.")
            raise SystemExit(0)

    print("\n" + "=" * 80)
    print(f"RUN: {args.run_id}")
    print(f"ROUND: {round_no}")
    print("=" * 80)

    if issues:
        print("\nCRITIC ISSUES:\n")
        for i, issue in enumerate(issues, start=1):
            print(f"{i}. {issue}")

    print("\nTARGETED QUESTIONS:")
    print("Blank answer means 'decide for me'.")
    print("-" * 80)

    answers: list[HumanAnswer] = []
    try:
        for i, q in enumerate(questions, start=1):
            print(f"\nQ{i}: {q}")
            a = input("> ")
            a = a.rstrip("\n")
            answers.append(
                HumanAnswer(
                    question=q,
                    answer=a,
                    decide_for_me=(a.strip() == ""),
                    is_llm_generated=False,
                )
            )
    except KeyboardInterrupt:
        print("\nInterrupted. No answers were written.")
        raise SystemExit(130)

    batch = HumanAnswerBatch(round_no=round_no, answers=answers)
    store.write_answers_batch(batch)

    run_root = store.paths.run_root
    print("\nSaved answers:")
    print(f"  {run_root / f'spec_answers_round_{round_no}.json'}")
    print("\nNext step:")
    print(f"  Run the coordinator again for run_id={args.run_id}.")


if __name__ == "__main__":
    main()