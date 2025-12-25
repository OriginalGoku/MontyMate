# spec_store.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import yaml

from .constants import (
    FACTORY_DIR_NAME,
    RUNS_DIR_NAME,
    MODULE_SPEC_FILE_NAME,
    RUN_STATE_FILE_NAME,
    SPEC_ANSWERS_ROUND,
    SPEC_CRITIC_REPORT_ROUND,
)
from .run_state import RunState
from .human_inputs import ClarificationBatch
from .spec_types import Spec


@dataclass(frozen=True, slots=True)
class SpecStorePaths:
    """Defines the file layout for a single run (paths only)."""

    project_root: Path
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.project_root / FACTORY_DIR_NAME / RUNS_DIR_NAME / self.run_id

    @property
    def run_state_json(self) -> Path:
        return self.run_root / RUN_STATE_FILE_NAME

    @property
    def module_spec_yaml(self) -> Path:
        return self.run_root / MODULE_SPEC_FILE_NAME

    def spec_answers_json(self, round_no: int) -> Path:
        return self.run_root / f"{SPEC_ANSWERS_ROUND}{int(round_no)}.json"

    def spec_critic_report_json(self, round_no: int) -> Path:
        return self.run_root / f"{SPEC_CRITIC_REPORT_ROUND}{int(round_no)}.json"


class SpecStore:
    """Defines the persistence contract for the spec pipeline (MVP)."""

    def read_state(self) -> RunState | None: ...
    def write_state(self, state: RunState) -> None: ...

    def read_spec(self) -> Spec: ...
    def write_spec(self, spec: Spec, *, tag: str | None = None) -> None: ...

    def write_answers_batch(self, batch: ClarificationBatch) -> None: ...
    def read_answers_batch(self, round_no: int) -> ClarificationBatch | None: ...

    def write_critic_report(self, *, round_no: int, report: dict[str, object]) -> None: ...
    def read_critic_report(self, *, round_no: int) -> dict[str, object] | None: ...


class FileSpecStore:
    """Persists run artifacts to disk for pause/resume of runs.

    Layout:
      <project_root>/.ai_module_factory/runs/<run_id>/
        run_state.json
        module_spec.yaml
        spec_answers_round_<N>.json
        spec_critic_report_round_<N>.json
    """

    def __init__(self, *, project_root: str | Path, run_id: str) -> None:
        self.paths = SpecStorePaths(project_root=Path(project_root).resolve(), run_id=run_id)

    def _ensure_dirs(self) -> None:
        self.paths.run_root.mkdir(parents=True, exist_ok=True)

    # --------------------
    # Run state
    # --------------------
    def read_state(self) -> RunState | None:
        p = self.paths.run_state_json
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return RunState.from_dict(raw) if isinstance(raw, dict) else None
        except Exception:
            return None

    def write_state(self, state: RunState) -> None:
        self._ensure_dirs()
        self.paths.run_state_json.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # --------------------
    # Spec
    # --------------------
    def read_spec(self) -> Spec:
        p = self.paths.module_spec_yaml
        if not p.exists():
            return Spec()
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return Spec.from_dict(raw) if isinstance(raw, dict) else Spec()
        except Exception:
            return Spec()

    def write_spec(self, spec: Spec, *, tag: str | None = None) -> None:
        self._ensure_dirs()
        text = yaml.safe_dump(spec.to_dict(), sort_keys=False, allow_unicode=True)
        self.paths.module_spec_yaml.write_text(text, encoding="utf-8")

        if tag:
            (self.paths.run_root / f"module_spec_{tag}.yaml").write_text(text, encoding="utf-8")

    # --------------------
    # Answers
    # --------------------
    def write_answers_batch(self, batch: ClarificationBatch) -> None:
        self._ensure_dirs()
        self.paths.spec_answers_json(batch.round_no).write_text(
            json.dumps(batch.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read_answers_batch(self, round_no: int) -> ClarificationBatch | None:
        p = self.paths.spec_answers_json(round_no)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return ClarificationBatch.from_dict(raw) if isinstance(raw, dict) else None
        except Exception:
            return None

    # --------------------
    # Critic report
    # --------------------
    def write_critic_report(self, *, round_no: int, report: dict[str, object]) -> None:
        self._ensure_dirs()
        p = self.paths.spec_critic_report_json(round_no)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_critic_report(self, *, round_no: int) -> dict[str, object] | None:
        p = self.paths.spec_critic_report_json(round_no)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else None
        except Exception:
            return None


class MemorySpecStore:
    """Stores run artifacts in memory for unit tests and local experiments."""

    def __init__(self) -> None:
        self._state: RunState | None = None
        self._spec: Spec = Spec()
        self._answers: dict[int, ClarificationBatch] = {}
        self._critic_reports: dict[int, dict[str, object]] = {}
        self._snapshots: dict[str, Spec] = {}

    def read_state(self) -> RunState | None:
        return self._state

    def write_state(self, state: RunState) -> None:
        self._state = state

    def read_spec(self) -> Spec:
        return self._spec

    def write_spec(self, spec: Spec, *, tag: str | None = None) -> None:
        self._spec = spec
        if tag:
            self._snapshots[str(tag)] = spec

    def write_answers_batch(self, batch: ClarificationBatch) -> None:
        self._answers[int(batch.round_no)] = batch

    def read_answers_batch(self, round_no: int) -> ClarificationBatch | None:
        return self._answers.get(int(round_no))

    def write_critic_report(self, *, round_no: int, report: dict[str, object]) -> None:
        self._critic_reports[int(round_no)] = dict(report)

    def read_critic_report(self, *, round_no: int) -> dict[str, object] | None:
        r = self._critic_reports.get(int(round_no))
        return dict(r) if isinstance(r, dict) else None