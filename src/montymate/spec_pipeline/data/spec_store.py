# spec_store.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import yaml

from .constants import (
    FACTORY_DIR_NAME,
    MODULE_SPEC_FILE_NAME,
    RUNS_DIR_NAME,
    SPEC_ANSWERS_ROUND,
    SPEC_VALIDATION_REPORT_FILE_NAME,
)
from .human_inputs import HumanAnswerBatch
from .spec_types import Spec


@dataclass(frozen=True, slots=True)
class SpecStorePaths:
    """Filesystem layout for a single run (paths only)."""

    project_root: Path
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.project_root / FACTORY_DIR_NAME / RUNS_DIR_NAME / self.run_id

    @property
    def module_spec_yaml(self) -> Path:
        return self.run_root / MODULE_SPEC_FILE_NAME

    @property
    def spec_validation_report_json(self) -> Path:
        return self.run_root / SPEC_VALIDATION_REPORT_FILE_NAME

    def spec_answers_json(self, round_no: int) -> Path:
        return self.run_root / f"{SPEC_ANSWERS_ROUND}{int(round_no)}.json"


class SpecStore(Protocol):
    def read_spec(self) -> Spec: ...
    def write_spec(self, spec: Spec, *, tag: Optional[str] = None) -> None: ...

    def read_validation_report(self) -> Dict[str, Any]: ...
    def write_validation_report(self, report: Dict[str, Any]) -> None: ...

    def write_answers_batch(self, batch: HumanAnswerBatch) -> None: ...
    def read_answers_batch(self, round_no: int) -> Optional[HumanAnswerBatch]: ...


class FileSpecStore:
    """File-backed store.

    Layout:
      <project_root>/.ai_module_factory/runs/<run_id>/
        module_spec.yaml
        spec_validation_report.json
        spec_answers_round_<N>.json
    """

    def __init__(self, *, project_root: str | Path, run_id: str):
        self.paths = SpecStorePaths(
            project_root=Path(project_root).resolve(), run_id=run_id
        )

    def _ensure_dirs(self) -> None:
        self.paths.run_root.mkdir(parents=True, exist_ok=True)

    def read_spec(self) -> Spec:
        p = self.paths.module_spec_yaml
        if not p.exists():
            return Spec()
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return Spec.from_dict(raw) if isinstance(raw, dict) else Spec()
        except Exception:
            return Spec()

    def write_spec(self, spec: Spec, *, tag: Optional[str] = None) -> None:
        self._ensure_dirs()
        text = yaml.safe_dump(spec.to_dict(), sort_keys=False, allow_unicode=True)
        self.paths.module_spec_yaml.write_text(text, encoding="utf-8")

        if tag:
            (self.paths.run_root / f"module_spec_{tag}.yaml").write_text(
                text, encoding="utf-8"
            )

    def read_validation_report(self) -> Dict[str, Any]:
        p = self.paths.spec_validation_report_json
        if not p.exists():
            return {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def write_validation_report(self, report: Dict[str, Any]) -> None:
        self._ensure_dirs()
        _ = self.paths.spec_validation_report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_answers_batch(self, batch: HumanAnswerBatch) -> None:
        self._ensure_dirs()
        _ = self.paths.spec_answers_json(batch.round_no).write_text(
            json.dumps(batch.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read_answers_batch(self, round_no: int) -> HumanAnswerBatch | None:
        p = self.paths.spec_answers_json(round_no)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return HumanAnswerBatch.from_dict(raw) if isinstance(raw, dict) else None
        except Exception:
            return None


class MemorySpecStore:
    def __init__(self):
        self._spec: Spec = Spec()
        self._report: Dict[str, Any] = {}
        self._answers: Dict[int, HumanAnswerBatch] = {}
        self._snapshots: Dict[str, Spec] = {}

    def read_spec(self) -> Spec:
        return self._spec

    def write_spec(self, spec: Spec, *, tag: Optional[str] = None) -> None:
        self._spec = spec
        if tag:
            self._snapshots[str(tag)] = spec

    def read_validation_report(self) -> Dict[str, Any]:
        return dict(self._report)

    def write_validation_report(self, report: Dict[str, Any]) -> None:
        self._report = dict(report)

    def write_answers_batch(self, batch: HumanAnswerBatch) -> None:
        self._answers[int(batch.round_no)] = batch

    def read_answers_batch(self, round_no: int) -> Optional[HumanAnswerBatch]:
        return self._answers.get(int(round_no))
