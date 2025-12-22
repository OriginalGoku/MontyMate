# src/montymate/mvp/pipeline.py
from __future__ import annotations

import json
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .db import (
    append_event,
    create_gate,
    create_run,
    decide_gate,
    record_artifact,
    record_llm_call,
    set_run_status,
)
from .providers import call_provider
from .routing import RoutingConfig
from montymate.constants import INTERVIEWER_MAX_NO_QUESTIONS, CANVAS_FIELDS

JSON = Dict[str, Any]

def _spec_defaults() -> JSON:
    # Keep types stable: strings are "", lists are [].
    return {
        "status": "DRAFT",
        "goal": "",
        "functional_requirements": [],
        "constraints": [],
        "security_concerns": [],
        "assumptions": [],
        "other_notes": "",
    }


def _normalize_spec(spec: Optional[JSON]) -> JSON:
    """
    Ensure all canvas keys exist with correct types, even if empty.
    This reduces LLM drift and makes UI + locking deterministic.
    """
    base = _spec_defaults()
    incoming = dict(spec or {})

    # Always keep status present
    if not isinstance(incoming.get("status"), str) or not incoming.get("status"):
        incoming["status"] = base["status"]

    # Strings
    for k in ("goal", "other_notes"):
        v = incoming.get(k)
        incoming[k] = v if isinstance(v, str) else base[k]

    # Lists
    for k in ("functional_requirements", "constraints", "security_concerns", "assumptions"):
        v = incoming.get(k)
        if isinstance(v, list):
            # Optional: coerce list items to strings
            incoming[k] = [str(x) for x in v if x is not None]
        else:
            incoming[k] = list(base[k])  # new empty list

    return incoming
def _load_json_file(p: Path) -> Optional[JSON]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _next_round_no(conn, run_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM mm_artifacts WHERE run_id=? AND artifact_type=?",
        (run_id, "spec_answers"),
    ).fetchone()
    return int(row["c"] or 0) + 1
    
def _default_locks() -> JSON:
    return {k: False for k in CANVAS_FIELDS}

def _normalize_locks(locks: Optional[JSON]) -> JSON:
    d = _default_locks()
    if isinstance(locks, dict):
        for k in CANVAS_FIELDS:
            if k in locks:
                d[k] = bool(locks[k])
    return d

def _enforce_locked_fields(prev_spec: JSON, new_spec: JSON, locks: JSON) -> JSON:
    """
    Deterministic enforcement: if a field is locked, preserve the previous value.
    """
    out = dict(new_spec)
    for k in CANVAS_FIELDS:
        if locks.get(k) is True and k in prev_spec:
            out[k] = prev_spec[k]
    return out
    
def _strip_markdown_code_fence(text: str) -> str:
    """
    If LLM returns ```yaml ... ``` (or ```json ... ```), strip the fences.
    """
    s = text.strip()

    if not s.startswith("```"):
        return text

    lines = s.splitlines()
    # Drop first fence line: ``` or ```yaml or ```json etc.
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    # Drop last fence line if present
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _run_root(repo_root: str, run_id: str) -> Path:
    return Path(repo_root) / ".ai_module_factory" / "runs" / run_id


def _write_artifact_indexed(
    *,
    conn,
    repo_root: str,
    run_id: str,
    artifact_type: str,
    relpath: str,
    content: bytes,
    meta: Optional[JSON] = None,
) -> Path:
    root = _run_root(repo_root, run_id)
    p = root / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)

    record_artifact(
        conn,
        run_id=run_id,
        artifact_type=artifact_type,
        relpath=relpath,
        content_bytes=content,
        meta=meta or {},
    )
    return p


@dataclass
class PipelineResult:
    status: str  # "DONE" | "PAUSED" | "FAILED"
    run_id: str
    gate_id: Optional[str] = None
    gate_name: Optional[str] = None
    message: Optional[str] = None


def _call_role(
    *,
    conn,
    routing: RoutingConfig,
    run_id: str,
    step: str,
    role: str,
    messages: List[JSON],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Tuple[str, JSON]:
    if role not in routing.roles:
        raise ValueError(f"No routing for role '{role}'")

    last_err: Optional[JSON] = None
    for hop in routing.roles[role].chain:
        if hop.provider not in routing.providers:
            last_err = {"error": f"Unknown provider {hop.provider}"}
            continue

        cfg = routing.providers[hop.provider]
        try:
            r = call_provider(
            cfg, hop.model, messages, max_tokens=max_tokens, temperature=temperature
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
               # append_event(...) here so we can see it in the UI
               append_event(
                   conn,
                   run_id=run_id,
                   event_type="llm.call_failed",
                   actor_type="system",
                   severity="error",
                   payload={
                       "role": role,
                       "step": step,
                       "provider": getattr(cfg, "provider", None) or getattr(hop, "provider", None),
                       "model": getattr(hop, "model", None),
                       "error": str(e),
                   },
               )
               raise RuntimeError(f"LLM call failed for role={role}, step={step}: {e}") from e
        record_llm_call(
            conn,
            run_id=run_id,
            step=step,
            role=role,
            provider=hop.provider,
            model=hop.model,
            prompt_text=json.dumps(messages, ensure_ascii=False),
            response_text=r.text,
            input_tokens=getattr(r, "input_tokens", None),
            output_tokens=getattr(r, "output_tokens", None),
            total_tokens=getattr(r, "total_tokens", None),
            cost_usd=None,
            raw=getattr(r, "raw", None),
        )

        if getattr(r, "status", None) == "OK":
            return r.text, {"provider": hop.provider, "model": hop.model, "raw": getattr(r, "raw", None)}

        last_err = getattr(r, "raw", None)

    raise RuntimeError(f"All models failed for role={role}: {last_err}")


def start_run(
    *,
    conn,
    repo_root: str,
    routing: RoutingConfig,
    user_prompt: str,
    default_gate_mode: str = "ACK",
    initial_module_spec: Optional[JSON] = None,
    initial_spec_locks: Optional[JSON] = None,
) -> PipelineResult:
    run_id = __import__("uuid").uuid4().hex
    create_run(conn, run_id=run_id, repo_root=repo_root, status="RUNNING")

    append_event(
        conn,
        run_id=run_id,
        event_type="run.started",
        actor_type="system",
        severity="info",
        payload={"repo_root": repo_root},
    )
    append_event(
        conn,
        run_id=run_id,
        event_type="input.prompt",
        actor_type="human",
        severity="info",
        payload={"prompt": user_prompt},
    )
    
    seed_spec: JSON = _normalize_spec(initial_module_spec)
    seed_spec.setdefault("status", "DRAFT")
    locks = _normalize_locks(initial_spec_locks)

    # Write locks artifact early so UI + later steps can read it
    _write_artifact_indexed(
        conn=conn,
        repo_root=repo_root,
        run_id=run_id,
        artifact_type="module_spec_locks",
        relpath="artifacts/module_spec_locks.json",
        content=json.dumps(locks, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    append_event(
        conn,
        run_id=run_id,
        event_type="spec.locks_written",
        actor_type="system",
        severity="info",
        payload={"locks": locks},
    )
    # STEP 1: Interviewer -> module_spec.yaml
    # Build interviewer prompt with context pack (seed spec + locks)
    interview_prompt = [
        {
            "role": "system",
            "content": (
                "You are a Software Engineer interviewing a user to clarify requirements.\n"
                "- No code.\n"
                f"- Ask up to {INTERVIEWER_MAX_NO_QUESTIONS} clarifying questions, then output YAML ONLY.\n"
                "- Output MUST be valid YAML ONLY (no Markdown fences like ```yaml).\n"
                "- You MUST output ONLY these keys:\n"
                "  status, goal, functional_requirements, constraints, security_concerns, assumptions, other_notes\n"
                "- IMPORTANT: Some fields may be locked. Do not change locked fields.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "### USER PROMPT\n"
                f"{user_prompt}\n\n"
                "### CURRENT MODULE_SPEC (DRAFT)\n"
                + yaml.safe_dump(seed_spec, sort_keys=False)
                + "\n"
                "### FIELD LOCKS (true means immutable)\n"
                + json.dumps(locks, ensure_ascii=False, indent=2)
                + "\n"
            ),
        },
    ]

    # spec_yaml, _meta = _call_role(...)

    spec_yaml, _meta = _call_role(
        conn=conn,
        routing=routing,
        run_id=run_id,
        step="interview",
        role="interviewer",
        messages=interview_prompt,
    )
    
    
    # Parse YAML; if parse fails, store raw and fail fast.
    try:
        spec_yaml_clean = _strip_markdown_code_fence(spec_yaml)
        module_spec = yaml.safe_load(spec_yaml_clean) or {}
        module_spec = _normalize_spec(module_spec if isinstance(module_spec, dict) else {})
        # module_spec["status"] = "LOCKED"
        if not isinstance(module_spec, dict):
            raise ValueError("module_spec must be a mapping")
        module_spec.setdefault("status", "DRAFT")
        # Enforce locks deterministically (AFTER parse, BEFORE write)
        module_spec = _normalize_spec(module_spec)
        module_spec = _enforce_locked_fields(seed_spec, module_spec, locks)

    except Exception as e:
        _write_artifact_indexed(
            conn=conn,
            repo_root=repo_root,
            run_id=run_id,
            artifact_type="module_spec_invalid",
            relpath="artifacts/module_spec_invalid.yaml",
            content=spec_yaml.encode("utf-8", errors="replace"),
            meta={"error": str(e)},
        )
        append_event(
            conn,
            run_id=run_id,
            event_type="module_spec.invalid",
            actor_type="system",
            severity="error",
            payload={"error": str(e)},
        )
        set_run_status(conn, run_id, "FAILED")
        return PipelineResult(
            status="FAILED", run_id=run_id, message=f"module_spec YAML invalid: {e}"
        )

    _write_artifact_indexed(
        conn=conn,
        repo_root=repo_root,
        run_id=run_id,
        artifact_type="module_spec",
        relpath="artifacts/module_spec.yaml",
        content=yaml.safe_dump(module_spec, sort_keys=False).encode("utf-8"),
    )
    append_event(
        conn,
        run_id=run_id,
        event_type="module_spec.written",
        actor_type="system",
        severity="info",
        payload={"status": module_spec.get("status")},
    )

    # STEP 2: Spec Validator -> spec_validation_report.json
    validate_prompt = [
        {
            "role": "system",
            "content": "You are MontyMate Spec Validator. No code. Check completeness and contradictions.",
        },
        {
            "role": "user",
            "content": "Here is module_spec YAML:\n\n"
            + yaml.safe_dump(module_spec, sort_keys=False),
        },
        {
            "role": "system",
            "content": "Return JSON with keys: passed (bool), issues (list), contradictions (list), targeted_questions (list).",
        },
    ]
    report_text, _ = _call_role(
        conn=conn,
        routing=routing,
        run_id=run_id,
        step="spec_validate",
        role="spec_validator",
        messages=validate_prompt,
    )

    try:
        # report = json.loads(report_text)
        report_text_clean = _strip_markdown_code_fence(report_text)
        report = json.loads(report_text_clean)
        if not isinstance(report, dict):
            raise ValueError("spec_validation_report must be an object")
        report.setdefault("passed", False)
        report.setdefault("issues", [])
        report.setdefault("contradictions", [])
        report.setdefault("targeted_questions", [])
    except Exception as e:
        report = {
            "passed": False,
            "issues": [f"Invalid JSON from spec_validator: {e}"],
            "contradictions": [],
            "targeted_questions": [],
        }

    _write_artifact_indexed(
        conn=conn,
        repo_root=repo_root,
        run_id=run_id,
        artifact_type="spec_validation_report",
        relpath="artifacts/spec_validation_report.json",
        content=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    append_event(
        conn,
        run_id=run_id,
        event_type="spec_validation.done",
        actor_type="system",
        severity="info",
        payload={"passed": bool(report.get("passed"))},
    )

    if not report.get("passed"):
        gate_id = create_gate(
            conn,
            run_id=run_id,
            gate_name="spec_validation",
            mode=default_gate_mode,
            payload={
                "user_prompt": user_prompt,
                "passed": False,
                "issues": report.get("issues"),
                "targeted_questions": report.get("targeted_questions"),
            },
        )
        set_run_status(conn, run_id, "PAUSED")
        return PipelineResult(
            status="PAUSED",
            run_id=run_id,
            gate_id=gate_id,
            gate_name="spec_validation",
            message="Spec validation failed; gate required.",
        )

    # STEP 3: Spec Lock gate (ACK default)
    gate_id = create_gate(
        conn,
        run_id=run_id,
        gate_name="spec_lock",
        mode=default_gate_mode,
        payload={
            "user_prompt": user_prompt,
            "module_spec_status": module_spec.get("status"),
            "action": "set status=LOCKED",
        },
    )
    set_run_status(conn, run_id, "PAUSED")
    return PipelineResult(
        status="PAUSED",
        run_id=run_id,
        gate_id=gate_id,
        gate_name="spec_lock",
        message="Spec lock gate required.",
    )

def resume_run(
    *,
    conn,
    repo_root: str,
    routing: RoutingConfig,
    run_id: str,
    gate_id: str,
    decision: str,
    human_actor: str,
    reason: Optional[str],
    # Only used when gate_name == "spec_validation"
    spec_answers: Optional[list[JSON]] = None,
    updated_module_spec: Optional[JSON] = None,
    updated_spec_locks: Optional[JSON] = None,
    # Only used when gate_name == "spec_lock"
    revalidate_before_lock: bool = False,
    ai_polish_before_lock: bool = False,
    default_gate_mode: str = "ACK",
) -> PipelineResult:
    # ----------------------------
    # Helper functions (local)
    # ----------------------------
    def _load_prev_spec_and_locks() -> tuple[JSON, JSON, Path]:
        root = _run_root(repo_root, run_id)
        art_dir = root / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)

        spec_path = art_dir / "module_spec.yaml"
        prev_spec_raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
        prev_spec = _normalize_spec(prev_spec_raw if isinstance(prev_spec_raw, dict) else {})

        locks_path = art_dir / "module_spec_locks.json"
        file_locks = _load_json_file(locks_path) or {}
        locks = _normalize_locks(updated_spec_locks if isinstance(updated_spec_locks, dict) else file_locks)

        return prev_spec, locks, art_dir

    def _write_locks_snapshot(locks: JSON, meta: Optional[JSON] = None) -> None:
        _write_artifact_indexed(
            conn=conn,
            repo_root=repo_root,
            run_id=run_id,
            artifact_type="module_spec_locks",
            relpath="artifacts/module_spec_locks.json",
            content=json.dumps(locks, ensure_ascii=False, indent=2).encode("utf-8"),
            meta=meta or {},
        )

    def _write_spec(new_spec: JSON, *, round_no: Optional[int] = None, tag: Optional[str] = None) -> None:
        spec_bytes = yaml.safe_dump(new_spec, sort_keys=False).encode("utf-8")
        if round_no is not None:
            _write_artifact_indexed(
                conn=conn,
                repo_root=repo_root,
                run_id=run_id,
                artifact_type="module_spec",
                relpath=f"artifacts/module_spec_round_{round_no}.yaml",
                content=spec_bytes,
                meta={"round": round_no, "tag": tag} if tag else {"round": round_no},
            )
        if tag:
            _write_artifact_indexed(
                conn=conn,
                repo_root=repo_root,
                run_id=run_id,
                artifact_type="module_spec",
                relpath=f"artifacts/module_spec_{tag}.yaml",
                content=spec_bytes,
                meta={"tag": tag},
            )
        _write_artifact_indexed(
            conn=conn,
            repo_root=repo_root,
            run_id=run_id,
            artifact_type="module_spec",
            relpath="artifacts/module_spec.yaml",
            content=spec_bytes,
            meta={"current": True, "tag": tag} if tag else {"current": True},
        )

    def _run_validator(spec: JSON, *, step: str, meta: Optional[JSON] = None) -> JSON:
        validate_prompt = [
            {"role": "system", "content": "You are MontyMate Spec Validator. No code. Check completeness and contradictions."},
            {"role": "user", "content": "Here is module_spec YAML:\n\n" + yaml.safe_dump(spec, sort_keys=False)},
            {"role": "system", "content": "Return JSON with keys: passed (bool), issues (list), contradictions (list), targeted_questions (list)."},
        ]
        report_text, _ = _call_role(
            conn=conn,
            routing=routing,
            run_id=run_id,
            step=step,
            role="spec_validator",
            messages=validate_prompt,
        )

        try:
            report_text_clean = _strip_markdown_code_fence(report_text)
            report = json.loads(report_text_clean)
            if not isinstance(report, dict):
                raise ValueError("spec_validation_report must be an object")
            report.setdefault("passed", False)
            report.setdefault("issues", [])
            report.setdefault("contradictions", [])
            report.setdefault("targeted_questions", [])
        except Exception as e:
            report = {
                "passed": False,
                "issues": [f"Invalid JSON from spec_validator: {e}"],
                "contradictions": [],
                "targeted_questions": [],
            }

        _write_artifact_indexed(
            conn=conn,
            repo_root=repo_root,
            run_id=run_id,
            artifact_type="spec_validation_report",
            relpath="artifacts/spec_validation_report.json",
            content=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
            meta=meta or {},
        )
        return report

    def _parse_llm_yaml_or_fail(text: str, *, artifact_relpath_on_fail: str, meta_on_fail: Optional[JSON] = None) -> JSON:
        try:
            clean = _strip_markdown_code_fence(text)
            raw = yaml.safe_load(clean) or {}
            if not isinstance(raw, dict):
                raise ValueError("YAML must be a mapping/object")
            return _normalize_spec(raw)
        except Exception as e:
            _write_artifact_indexed(
                conn=conn,
                repo_root=repo_root,
                run_id=run_id,
                artifact_type="module_spec_invalid",
                relpath=artifact_relpath_on_fail,
                content=text.encode("utf-8", errors="replace"),
                meta={"error": str(e), **(meta_on_fail or {})},
            )
            append_event(
                conn,
                run_id=run_id,
                event_type="module_spec.invalid",
                actor_type="system",
                severity="error",
                payload={"error": str(e), **(meta_on_fail or {})},
            )
            raise

    def _make_spec_lock_gate(spec: JSON, user_prompt: str) -> PipelineResult:
        gid = create_gate(
            conn,
            run_id=run_id,
            gate_name="spec_lock",
            mode=default_gate_mode,
            payload={
                "user_prompt": user_prompt,
                "module_spec_status": spec.get("status", "DRAFT"),
                "action": "review and set status=LOCKED",
            },
        )
        set_run_status(conn, run_id, "PAUSED")
        return PipelineResult(
            status="PAUSED",
            run_id=run_id,
            gate_id=gid,
            gate_name="spec_lock",
            message="Spec is valid. Proceed to final review (spec_lock).",
        )

    def _make_spec_validation_gate(user_prompt: str, report: JSON, msg: str) -> PipelineResult:
        gid = create_gate(
            conn,
            run_id=run_id,
            gate_name="spec_validation",
            mode=default_gate_mode,
            payload={
                "user_prompt": user_prompt,
                "passed": False,
                "issues": report.get("issues"),
                "targeted_questions": report.get("targeted_questions"),
            },
        )
        set_run_status(conn, run_id, "PAUSED")
        return PipelineResult(status="PAUSED", run_id=run_id, gate_id=gid, gate_name="spec_validation", message=msg)

    def _continue_pipeline_after_lock(locked_spec: JSON, user_prompt: str) -> PipelineResult:
        # Architecture
        plan_prompt = [
            {"role": "system", "content": "You are MontyMate Architect. No code. Produce an architecture_plan markdown with headings and clear interfaces."},
            {"role": "user", "content": "User request:\n" + user_prompt},
            {"role": "user", "content": "module_spec:\n\n" + yaml.safe_dump(locked_spec, sort_keys=False)},
        ]
        plan_md, _ = _call_role(
            conn=conn, routing=routing, run_id=run_id, step="architecture", role="architect", messages=plan_prompt
        )
        _write_artifact_indexed(
            conn=conn,
            repo_root=repo_root,
            run_id=run_id,
            artifact_type="architecture_plan",
            relpath="artifacts/architecture_plan.md",
            content=plan_md.encode("utf-8", errors="replace"),
        )
        append_event(conn, run_id=run_id, event_type="architecture.written", actor_type="system", severity="info", payload={})

        # Audit
        audit_prompt = [
            {"role": "system", "content": "You are MontyMate Reviewer. No code. Produce an audit_report markdown. Start with a short YAML frontmatter: decision: APPROVE|BLOCK and rationale."},
            {"role": "user", "content": "User request:\n" + user_prompt},
            {"role": "user", "content": "module_spec:\n\n" + yaml.safe_dump(locked_spec, sort_keys=False)},
            {"role": "user", "content": "architecture_plan:\n\n" + plan_md},
        ]
        audit_md, _ = _call_role(
            conn=conn, routing=routing, run_id=run_id, step="audit", role="reviewer", messages=audit_prompt
        )
        _write_artifact_indexed(
            conn=conn,
            repo_root=repo_root,
            run_id=run_id,
            artifact_type="audit_report",
            relpath="artifacts/audit_report.md",
            content=audit_md.encode("utf-8", errors="replace"),
        )
        append_event(conn, run_id=run_id, event_type="audit.written", actor_type="system", severity="info", payload={})

        # Audit gate
        gid = create_gate(
            conn,
            run_id=run_id,
            gate_name="audit_gate",
            mode="ACK",
            payload={"user_prompt": user_prompt, "note": "Review audit_report.md"},
        )
        set_run_status(conn, run_id, "PAUSED")
        return PipelineResult(status="PAUSED", run_id=run_id, gate_id=gid, gate_name="audit_gate", message="Audit gate required.")

    # ----------------------------
    # Gate decision + basics
    # ----------------------------
    gate_row = decide_gate(conn, gate_id=gate_id, decision=decision, human_actor=human_actor, reason=reason)
    gate_name = gate_row["gate_name"]
    payload = json.loads(gate_row["payload_json"] or "{}")
    user_prompt = payload.get("user_prompt") or ""

    if decision == "BLOCK":
        set_run_status(conn, run_id, "FAILED")
        return PipelineResult(status="FAILED", run_id=run_id, message=f"Blocked at gate: {gate_name}")

    prev_spec, locks, art_dir = _load_prev_spec_and_locks()

    # ----------------------------
    # 1) spec_validation
    # ----------------------------
    if gate_name == "spec_validation":
        if spec_answers is None:
            set_run_status(conn, run_id, "FAILED")
            append_event(conn, run_id=run_id, event_type="spec.answers.missing", actor_type="system", severity="error", payload={"gate_id": gate_id})
            return PipelineResult(status="FAILED", run_id=run_id, message="spec_validation resume requires spec_answers from UI.")

        round_no = _next_round_no(conn, run_id)

        # Canvas edits + enforce locks
        seed_from_ui = _normalize_spec(updated_module_spec if isinstance(updated_module_spec, dict) else prev_spec)
        seed_spec = _enforce_locked_fields(prev_spec, seed_from_ui, locks)

        # Persist answers
        _write_artifact_indexed(
            conn=conn,
            repo_root=repo_root,
            run_id=run_id,
            artifact_type="spec_answers",
            relpath=f"artifacts/spec_answers_round_{round_no}.json",
            content=json.dumps({"round": round_no, "answers": spec_answers}, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        append_event(conn, run_id=run_id, event_type="spec.answers_submitted", actor_type="human", severity="info", payload={"round": round_no, "n_answers": len(spec_answers)})

        # Persist locks snapshot
        _write_locks_snapshot(locks, meta={"round": round_no})
        append_event(conn, run_id=run_id, event_type="spec.locks_updated", actor_type="system", severity="info", payload={"round": round_no, "locks": locks})

        # Interviewer call
        interview_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a Software Engineer interviewing a user to clarify requirements.\n"
                    "- No code.\n"
                    f"- Ask up to {INTERVIEWER_MAX_NO_QUESTIONS} clarifying questions ONLY if still necessary.\n"
                    "- Output MUST be valid YAML ONLY (no Markdown fences like ```yaml).\n"
                    "- You MUST output ONLY these keys:\n"
                    "  status, goal, functional_requirements, constraints, security_concerns, assumptions, other_notes\n"
                    "- IMPORTANT: Some fields are locked. Do not change locked fields.\n"
                    "- Incorporate the user's answers below. If decide_for_me=true, you decide.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "### ORIGINAL USER PROMPT\n"
                    f"{user_prompt}\n\n"
                    "### CURRENT MODULE_SPEC (DRAFT)\n"
                    + yaml.safe_dump(seed_spec, sort_keys=False)
                    + "\n"
                    "### FIELD LOCKS\n"
                    + json.dumps(locks, ensure_ascii=False, indent=2)
                    + "\n\n"
                    "### USER ANSWERS (this round)\n"
                    + json.dumps(spec_answers, ensure_ascii=False, indent=2)
                    + "\n"
                ),
            },
        ]

        spec_yaml, _ = _call_role(
            conn=conn, routing=routing, run_id=run_id, step=f"interview_round_{round_no}", role="interviewer", messages=interview_prompt
        )

        try:
            new_spec = _parse_llm_yaml_or_fail(
                spec_yaml,
                artifact_relpath_on_fail=f"artifacts/module_spec_invalid_round_{round_no}.yaml",
                meta_on_fail={"round": round_no},
            )
            new_spec = _enforce_locked_fields(seed_spec, new_spec, locks)
        except Exception as e:
            set_run_status(conn, run_id, "FAILED")
            return PipelineResult(status="FAILED", run_id=run_id, message=f"module_spec YAML invalid: {e}")

        _write_spec(new_spec, round_no=round_no)
        append_event(conn, run_id=run_id, event_type="module_spec.updated", actor_type="system", severity="info", payload={"round": round_no})

        report = _run_validator(new_spec, step=f"spec_validate_round_{round_no}", meta={"round": round_no})
        append_event(conn, run_id=run_id, event_type="spec_validation.done", actor_type="system", severity="info", payload={"round": round_no, "passed": bool(report.get("passed"))})

        if not report.get("passed"):
            return _make_spec_validation_gate(user_prompt, report, "Spec validation failed again; answer the next set of targeted questions.")

        return _make_spec_lock_gate(new_spec, user_prompt)

    # ----------------------------
    # 2) spec_lock (final review)
    # ----------------------------
    if gate_name == "spec_lock":
        # Apply UI edits (respecting locks)
        seed_from_ui = _normalize_spec(updated_module_spec if isinstance(updated_module_spec, dict) else prev_spec)
        reviewed_spec = _enforce_locked_fields(prev_spec, seed_from_ui, locks)

        _write_locks_snapshot(locks, meta={"stage": "spec_lock"})

        # 2A) Improve (new!)
        improve_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a requirements improver.\n"
                    "- Output MUST be YAML ONLY.\n"
                    "- Keep keys: status, goal, functional_requirements, constraints, security_concerns, assumptions, other_notes.\n"
                    "- Improve clarity/completeness and remove contradictions.\n"
                    "- Do NOT expand scope.\n"
                    "- Do NOT change locked fields.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "USER PROMPT:\n"
                    f"{user_prompt}\n\n"
                    "CURRENT SPEC:\n"
                    + yaml.safe_dump(reviewed_spec, sort_keys=False)
                    + "\n"
                    "LOCKS:\n"
                    + json.dumps(locks, ensure_ascii=False, indent=2)
                ),
            },
        ]

        improved_yaml, _ = _call_role(
            conn=conn,
            routing=routing,
            run_id=run_id,
            step="spec_improve",
            role="spec_improver",
            messages=improve_prompt,
        )

        try:
            improved_spec = _parse_llm_yaml_or_fail(
                improved_yaml,
                artifact_relpath_on_fail="artifacts/module_spec_improved_invalid.yaml",
                meta_on_fail={"stage": "spec_lock"},
            )
            improved_spec = _enforce_locked_fields(reviewed_spec, improved_spec, locks)

            reviewed_spec = improved_spec
            _write_spec(reviewed_spec, tag="improved")
            append_event(conn, run_id=run_id, event_type="spec.improved", actor_type="system", severity="info", payload={})
        except Exception as e:
            append_event(conn, run_id=run_id, event_type="spec.improve_failed", actor_type="system", severity="error", payload={"error": str(e)})

        # 2B) Optional polish (existing behavior)
        if ai_polish_before_lock:
            polish_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a senior product engineer polishing a software module spec.\n"
                        "- Output MUST be valid YAML ONLY.\n"
                        "- Keep the same keys: status, goal, functional_requirements, constraints, security_concerns, assumptions, other_notes.\n"
                        "- Improve clarity, remove ambiguity, add missing details, but do NOT expand scope.\n"
                        "- Do NOT change locked fields.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "ORIGINAL USER PROMPT:\n"
                        f"{user_prompt}\n\n"
                        "CURRENT SPEC:\n"
                        + yaml.safe_dump(reviewed_spec, sort_keys=False)
                        + "\n"
                        "LOCKS:\n"
                        + json.dumps(locks, ensure_ascii=False, indent=2)
                    ),
                },
            ]

            polished_yaml, _ = _call_role(
                conn=conn,
                routing=routing,
                run_id=run_id,
                step="spec_polish",
                role="spec_polisher",
                messages=polish_prompt,
            )

            try:
                polished_spec = _parse_llm_yaml_or_fail(
                    polished_yaml,
                    artifact_relpath_on_fail="artifacts/module_spec_polished_invalid.yaml",
                    meta_on_fail={"stage": "spec_lock"},
                )
                polished_spec = _enforce_locked_fields(reviewed_spec, polished_spec, locks)
                reviewed_spec = polished_spec
                _write_spec(reviewed_spec, tag="polished")
                append_event(conn, run_id=run_id, event_type="spec.polished", actor_type="system", severity="info", payload={})
            except Exception as e:
                append_event(conn, run_id=run_id, event_type="spec.polish_failed", actor_type="system", severity="error", payload={"error": str(e)})

        # 2C) Optional revalidate (and bounce back to spec_validation if needed)
        if revalidate_before_lock:
            report = _run_validator(reviewed_spec, step="spec_validate_final", meta={"stage": "spec_lock_final"})
            append_event(conn, run_id=run_id, event_type="spec_validation.done", actor_type="system", severity="info", payload={"round": "final", "passed": bool(report.get("passed"))})

            if not report.get("passed"):
                return _make_spec_validation_gate(user_prompt, report, "Final validation failed; returning to spec_validation.")

        # 2D) Lock and continue pipeline
        locked_spec = dict(reviewed_spec)
        locked_spec["status"] = "LOCKED"

        _write_spec(locked_spec, tag="locked")
        append_event(conn, run_id=run_id, event_type="spec.locked", actor_type="system", severity="info", payload={"status": "LOCKED"})

        return _continue_pipeline_after_lock(locked_spec, user_prompt)

    # ----------------------------
    # 3) Unknown gate
    # ----------------------------
    append_event(conn, run_id=run_id, event_type="gate.unhandled", actor_type="system", severity="error", payload={"gate_id": gate_id, "gate_name": gate_name})
    set_run_status(conn, run_id, "FAILED")
    return PipelineResult(status="FAILED", run_id=run_id, message=f"Unhandled gate: {gate_name}")
    
def finalize_after_audit(*, conn, repo_root: str, run_id: str) -> PipelineResult:
    calls = conn.execute(
        "SELECT ts, step, role, provider, model, input_tokens, output_tokens, total_tokens "
        "FROM mm_llm_calls WHERE run_id=? ORDER BY ts ASC",
        (run_id,),
    ).fetchall()

    bundle = {"run_id": run_id, "llm_calls": [dict(r) for r in calls]}
    _write_artifact_indexed(
        conn=conn,
        repo_root=repo_root,
        run_id=run_id,
        artifact_type="prompt_bundle",
        relpath="artifacts/prompt_bundle.json",
        content=json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8"),
    )

    summary = [
        "# MontyMate Run Summary",
        f"- run_id: `{run_id}`",
        f"- artifacts root: `.ai_module_factory/runs/{run_id}/artifacts/`",
        "",
        "## Artifacts",
        "- module_spec.yaml",
        "- spec_validation_report.json",
        "- architecture_plan.md",
        "- audit_report.md",
        "- prompt_bundle.json",
        "- run_summary.md",
        "",
        "## LLM Calls",
        f"- {len(calls)} calls recorded in SQLite",
        "",
    ]
    _write_artifact_indexed(
        conn=conn,
        repo_root=repo_root,
        run_id=run_id,
        artifact_type="run_summary",
        relpath="artifacts/run_summary.md",
        content="\n".join(summary).encode("utf-8"),
    )
    append_event(
        conn,
        run_id=run_id,
        event_type="run_summary.written",
        actor_type="system",
        severity="info",
        payload={},
    )

    set_run_status(conn, run_id, "DONE")
    append_event(
        conn,
        run_id=run_id,
        event_type="run.done",
        actor_type="system",
        severity="info",
        payload={},
    )
    return PipelineResult(status="DONE", run_id=run_id, message="Run completed.")
