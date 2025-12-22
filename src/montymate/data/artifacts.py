from __future__ import annotations

import hashlib
# import json
# import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    relpath: str
    sha256: str
    bytes: int


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_artifact(
    *,
    artifacts_root: Path,
    relpath: str,
    payload: bytes,
) -> ArtifactRef:
    artifact_id = str(uuid.uuid4())
    out_path = artifacts_root / relpath
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    sha = _sha256_bytes(payload)
    return ArtifactRef(
        artifact_id=artifact_id,
        relpath=str(Path("artifacts") / relpath),
        sha256=sha,
        bytes=len(payload),
    )


def choose_inline_or_artifact(
    *,
    text: str,
    inline_max_bytes: int,
    artifacts_root: Path,
    relpath: str,
) -> tuple[Optional[str], Optional[ArtifactRef]]:
    b = text.encode("utf-8")
    if len(b) <= inline_max_bytes:
        return text, None
    ref = write_artifact(artifacts_root=artifacts_root, relpath=relpath, payload=b)
    return None, ref