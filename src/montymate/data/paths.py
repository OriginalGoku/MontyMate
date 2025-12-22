from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    ai_root: Path               # gitignored
    db_path: Path               # sqlite file in ai_root
    artifacts_root: Path        # ai_root/artifacts

    @staticmethod
    def for_repo(repo_root: Path) -> "RepoPaths":
        repo_root = repo_root.resolve()
        ai_root = repo_root / ".ai_montymate"
        db_path = ai_root / "montymate.sqlite3"
        artifacts_root = ai_root / "artifacts"
        return RepoPaths(repo_root=repo_root, ai_root=ai_root, db_path=db_path, artifacts_root=artifacts_root)

    def ensure_dirs(self) -> None:
        self.ai_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)