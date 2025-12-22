from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st
import yaml

from montymate.config.models import (
    LLMRoutingConfig,
    PolicyConfig,
    ProfileConfig,
    ResolvedRunConfig,
    StepGroupCatalog,
    ToolRegistry,
    WorkflowGraph,
)
from montymate.core.types import JSON, ResourceRef, ResolvedRefs, RunIdentity
from montymate.data.db import connect, transaction
from montymate.data.migration import apply_migrations, ensure_application_id, load_migrations
from montymate.data.paths import RepoPaths
from montymate.data.services import DataServices
from montymate.data.types import RunContext, StoragePolicy
from montymate.engine.guards_impl import DefaultGuardEvaluator
from montymate.engine.workflow_engine_impl import WorkflowEngineV0
from montymate.gates.hitl import HumanGateManager
from montymate.gates.models import GateDecision, GateRequest
from montymate.llm.gateway_impl import LLMGatewayImpl
from montymate.tools.simple_executor import SimpleLocalToolExecutor


APP_ID = 1297048389  # keep aligned with your CLI


# -----------------------------
# Helpers
# -----------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_yaml_from_path(p: str) -> Tuple[dict, bytes]:
    data = Path(p).read_bytes()
    return yaml.safe_load(data.decode("utf-8")), data


def read_yaml_from_upload(uploaded) -> Tuple[dict, bytes]:
    data = uploaded.getvalue()
    return yaml.safe_load(data.decode("utf-8")), data


def yaml_preview(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def _parse_selector(selector: str) -> Tuple[str, int]:
    if "@" not in selector:
        raise ValueError(f"Invalid selector '{selector}', expected 'id@version'")
    rid, ver = selector.split("@", 1)
    return rid, int(ver)


@dataclass(frozen=True)
class FixedRunConfigResolver:
    """
    Resolver used by the engine during run_until_blocked_or_done().
    It ignores repo_root/profile_selector and always returns the pre-built config.
    """
    cfg: ResolvedRunConfig

    def resolve(self, *, repo_root: str, profile_selector: str, overrides=None) -> ResolvedRunConfig:
        return self.cfg


def make_services(*, conn: sqlite3.Connection, paths: RepoPaths, run_id: str, repo_root: str,
                  profile_id: str, workflow_id: str, policy_id: str, decision_record: Optional[JSON]) -> DataServices:
    # For MVP, keep a simple storage policy. You can later switch based on decision_record.storage_policy.
    storage_policy = StoragePolicy(payload_mode="hybrid", inline_max_bytes=32768, artifact_root=str(paths.ai_root))
    ctx = RunContext(
        run_id=run_id,
        repo_root=repo_root,
        profile_id=profile_id,
        workflow_id=workflow_id,
        policy_id=policy_id,
        storage_policy=storage_policy,
    )
    return DataServices.build(conn, paths, ctx)


@dataclass
class DBHumanGateManager(HumanGateManager):
    """
    Minimal DB-backed gate manager:
    - request_gate: logs event with gate_id + run_id
    - decide: logs decision event and returns (engine controls run status)
    """
    conn: sqlite3.Connection
    paths: RepoPaths

    def request_gate(self, req: GateRequest) -> GateRequest:
        svc = make_services(
            conn=self.conn,
            paths=self.paths,
            run_id=req.run_id,
            repo_root=str(self.paths.repo_root),
            profile_id="unknown",
            workflow_id="unknown",
            policy_id="unknown",
            decision_record=None,
        )
        with transaction(self.conn, write=True):
            svc.events.append(
                run_id=req.run_id,
                step_execution_id=None,
                event_type="gate.requested",
                actor_type="system",
                severity="info",
                payload={
                    "run_id": req.run_id,
                    "gate_id": req.gate_id,
                    "gate_name": req.gate_name,
                    "mode": req.mode,
                    "step_id": req.step_id,
                    "required_artifacts": req.required_artifacts,
                    "summary": req.summary,
                },
            )
        return req

    def decide(self, decision: GateDecision) -> None:
        run_id = self._find_run_id_for_gate(decision.gate_id)
        if not run_id:
            # still allow resume; engine controls state. We just lose the decision event.
            return

        event_type = "gate.acknowledged" if decision.decision == "ACK" else \
                     "gate.approved" if decision.decision == "APPROVE" else \
                     "gate.blocked"

        svc = make_services(
            conn=self.conn,
            paths=self.paths,
            run_id=run_id,
            repo_root=str(self.paths.repo_root),
            profile_id="unknown",
            workflow_id="unknown",
            policy_id="unknown",
            decision_record=None,
        )
        with transaction(self.conn, write=True):
            svc.events.append(
                run_id=run_id,
                step_execution_id=None,
                event_type=event_type,
                actor_type="human",
                severity="info" if decision.decision in ("ACK", "APPROVE") else "warn",
                payload={
                    "run_id": run_id,
                    "gate_id": decision.gate_id,
                    "decision": decision.decision,
                    "human_actor": decision.human_actor,
                    "reason": decision.reason,
                    "resume_payload": decision.resume_payload,
                },
            )

        if decision.decision == "BLOCK":
            raise RuntimeError("Gate decision was BLOCK")

    def get_pending_gate(self, run_id: str) -> Optional[GateRequest]:
        return None

    def _find_run_id_for_gate(self, gate_id: str) -> Optional[str]:
        # Try to find the most recent gate.requested with this gate_id
        row = self.conn.execute(
            """
            SELECT payload_json
            FROM mm_events
            WHERE event_type = 'gate.requested'
            ORDER BY ts DESC, event_id DESC
            LIMIT 500
            """
        ).fetchall()

        for r in row:
            try:
                payload = json.loads(r["payload_json"])
                if payload.get("gate_id") == gate_id:
                    return payload.get("run_id")
            except Exception:
                continue
        return None


def init_db(conn: sqlite3.Connection) -> None:
    ensure_application_id(conn, APP_ID)
    migrations = load_migrations("montymate", "resources/db/migrations")
    apply_migrations(conn, migrations)


def normalize_llm_routing(raw: dict) -> dict:
    """
    Accept either:
      A) new format: llm_routing.providers + roles.<role>.chain
      B) old format: model_catalog + roles.<role>.primary/fallbacks (IDs)
    Convert B -> A in-memory if needed.

    NOTE: Your LLMGatewayImpl expects roles.<role>.chain entries with {provider, model}.
    """
    lr = raw.get("llm_routing", {})
    if "providers" in lr and isinstance(lr.get("roles", {}), dict) and "chain" in next(iter(lr["roles"].values()), {}):
        return raw  # already new-style

    # Old-style conversion
    catalog = {m["id"]: m for m in lr.get("model_catalog", []) if isinstance(m, dict) and "id" in m}
    roles = lr.get("roles", {}) or {}
    providers = lr.get("providers", {}) or {}

    new_roles: Dict[str, Any] = {}
    for role, cfg in roles.items():
        chain = []
        primary_id = cfg.get("primary")
        fallback_ids = cfg.get("fallbacks", []) or []
        for mid in [primary_id] + list(fallback_ids):
            if not mid or mid not in catalog:
                continue
            m = catalog[mid]
            chain.append({"provider": m.get("provider"), "model": m.get("model")})
        new_roles[role] = {"chain": chain}

    lr["providers"] = providers  # may be empty; you should supply providers in your YAML
    lr["roles"] = new_roles
    raw["llm_routing"] = lr
    return raw


def build_resolved_config(
    *,
    workflow_raw: dict,
    profile_raw: dict,
    policy_raw: dict,
    tool_registry_raw: dict,
    llm_routing_raw: dict,
    workflow_bytes: bytes,
    profile_bytes: bytes,
    policy_bytes: bytes,
    tool_bytes: bytes,
    llm_bytes: bytes,
) -> ResolvedRunConfig:
    # Refs (hashes for provenance)
    wf_id = workflow_raw["workflow"]["id"]
    wf_ver = int(workflow_raw["workflow"]["version"])
    prof_id = profile_raw["profile"]["id"]
    prof_ver = int(profile_raw["profile"]["version"])
    pol_id = policy_raw["policy"]["id"]
    pol_ver = int(policy_raw["policy"]["version"])

    wf_ref = ResourceRef(id=wf_id, version=wf_ver, sha256=sha256_bytes(workflow_bytes))
    prof_ref = ResourceRef(id=prof_id, version=prof_ver, sha256=sha256_bytes(profile_bytes))
    pol_ref = ResourceRef(id=pol_id, version=pol_ver, sha256=sha256_bytes(policy_bytes))

    tool_ref = ResourceRef(id="tool_registry", version=int(tool_registry_raw["tool_registry"]["version"]), sha256=sha256_bytes(tool_bytes))
    llm_ref = ResourceRef(id="llm_routing", version=int(llm_routing_raw["llm_routing"]["version"]), sha256=sha256_bytes(llm_bytes))

    refs = ResolvedRefs(
        workflow=wf_ref,
        profile=prof_ref,
        policy=pol_ref,
        llm_routing=llm_ref,
        tool_registry=tool_ref,
        schemas={},
    )

    workflow = WorkflowGraph(
        id=wf_id,
        version=wf_ver,
        entry=workflow_raw["workflow"]["entry"],
        raw=workflow_raw,
        steps=workflow_raw.get("steps", {}),
        edges=workflow_raw.get("edges", []),
    )

    # Step groups catalog is optional in this UI MVP
    catalog = StepGroupCatalog(version=1, raw={"step_groups_catalog": {"version": 1, "groups": {}}}, groups={})

    profile = ProfileConfig(
        id=prof_id,
        version=prof_ver,
        raw=profile_raw,
        workflow_default=profile_raw["workflow"]["default"],
        policy_default=profile_raw["policy"]["default"],
        tool_bindings=profile_raw.get("bindings", {}).get("tools", {}),
        prompt_bindings=profile_raw.get("bindings", {}).get("prompts", {}),
        allowed_step_groups=profile_raw.get("allowed_step_groups", []),
    )

    policy = PolicyConfig(id=pol_id, version=pol_ver, raw=policy_raw)

    tool_registry = ToolRegistry(
        version=int(tool_registry_raw["tool_registry"]["version"]),
        raw=tool_registry_raw,
        tools=tool_registry_raw.get("tools", {}),
    )

    llm_routing = LLMRoutingConfig(
        version=int(llm_routing_raw["llm_routing"]["version"]),
        raw=llm_routing_raw,
        model_catalog=llm_routing_raw["llm_routing"].get("model_catalog", []),
        roles=llm_routing_raw["llm_routing"].get("roles", {}),
    )

    return ResolvedRunConfig(
        refs=refs,
        workflow=workflow,
        step_groups_catalog=catalog,
        profile=profile,
        policy=policy,
        tool_registry=tool_registry,
        llm_routing=llm_routing,
        schemas={},
        tool_bindings=profile.tool_bindings,
        prompt_bindings=profile.prompt_bindings,
    )


def list_runs(conn: sqlite3.Connection) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT run_id, status, created_at, updated_at, profile_id, workflow_id FROM mm_runs ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def list_events(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT ts, event_type, actor_type, severity, payload_json FROM mm_events WHERE run_id=? ORDER BY ts ASC, event_id ASC LIMIT 2000",
            (run_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d.pop("payload_json"))
            except Exception:
                d["payload"] = d.pop("payload_json")
            out.append(d)
        return out
    except Exception:
        return []


def list_artifacts(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT artifact_id, artifact_type, relpath, sha256, bytes FROM mm_artifacts WHERE run_id=? ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="MontyMate MVP Runner", layout="wide")
st.title("MontyMate MVP Runner")

st.caption(
    "Load workflow/policy/profile/tool/LLM YAML, pick a SQLite DB, then run one prompt end-to-end "
    "(including pause/resume on human gates)."
)

with st.sidebar:
    st.header("Repo + DB")

    repo_root = st.text_input("Repo root path", value=str(Path(".").resolve()))
    repo_root_path = Path(repo_root).resolve()

    # RepoPaths decides where the DB *usually* lives; user can override.
    default_paths = RepoPaths.for_repo(repo_root_path)
    default_db_path = str(default_paths.db_path)

    db_path = st.text_input("SQLite DB path", value=default_db_path)

    st.divider()
    st.header("Resource loading")

    load_mode = st.radio(
        "How to load YAML resources?",
        ["From file paths", "Upload YAML files"],
        index=0,
    )

    st.caption("Use Streamlit forms to avoid reruns on every keystroke.  [oai_citation:1‡docs.streamlit.io](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form?utm_source=chatgpt.com)")


def load_yaml_bundle() -> Optional[dict]:
    """
    Returns dict with keys: workflow, profile, policy, tool_registry, llm_routing and their bytes.
    """
    if load_mode == "From file paths":
        with st.form("yaml_paths_form"):
            st.subheader("YAML paths")
            wf_p = st.text_input("Workflow YAML path", value=str(repo_root_path / "resources/workflows/python_unified_v1.yaml"))
            prof_p = st.text_input("Profile YAML path", value=str(repo_root_path / "resources/profiles/fastapi_service_v1.yaml"))
            pol_p = st.text_input("Policy YAML path", value=str(repo_root_path / "resources/policies/default_fastapi_policy_v1.yaml"))
            tool_p = st.text_input("Tool registry YAML path", value=str(repo_root_path / "resources/tools/tool_registry_v1.yaml"))
            llm_p = st.text_input("LLM routing YAML path", value=str(repo_root_path / "resources/configs/llm_routing_v1.yaml"))
            submit = st.form_submit_button("Load YAMLs")

        if not submit:
            return None

        try:
            wf_raw, wf_b = read_yaml_from_path(wf_p)
            prof_raw, prof_b = read_yaml_from_path(prof_p)
            pol_raw, pol_b = read_yaml_from_path(pol_p)
            tool_raw, tool_b = read_yaml_from_path(tool_p)
            llm_raw, llm_b = read_yaml_from_path(llm_p)
            llm_raw = normalize_llm_routing(llm_raw)
            llm_b = yaml_preview(llm_raw).encode("utf-8")
            return {
                "workflow_raw": wf_raw, "workflow_bytes": wf_b,
                "profile_raw": prof_raw, "profile_bytes": prof_b,
                "policy_raw": pol_raw, "policy_bytes": pol_b,
                "tool_raw": tool_raw, "tool_bytes": tool_b,
                "llm_raw": llm_raw, "llm_bytes": llm_b,
            }
        except yaml.YAMLError as e:
            st.error(f"YAML parse error: {e}")
            return None
        except Exception as e:
            st.error(f"Failed to load YAMLs: {e}")
            return None

    # Upload mode
    st.subheader("Upload YAMLs")
    st.caption("Uploaded files are available via st.file_uploader.  [oai_citation:2‡docs.streamlit.io](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader?utm_source=chatgpt.com)")

    wf_u = st.file_uploader("Workflow YAML", type=["yaml", "yml"], key="wf_u")
    prof_u = st.file_uploader("Profile YAML", type=["yaml", "yml"], key="prof_u")
    pol_u = st.file_uploader("Policy YAML", type=["yaml", "yml"], key="pol_u")
    tool_u = st.file_uploader("Tool registry YAML", type=["yaml", "yml"], key="tool_u")
    llm_u = st.file_uploader("LLM routing YAML", type=["yaml", "yml"], key="llm_u")

    if not all([wf_u, prof_u, pol_u, tool_u, llm_u]):
        return None

    try:
        wf_raw, wf_b = read_yaml_from_upload(wf_u)
        prof_raw, prof_b = read_yaml_from_upload(prof_u)
        pol_raw, pol_b = read_yaml_from_upload(pol_u)
        tool_raw, tool_b = read_yaml_from_upload(tool_u)
        llm_raw, llm_b = read_yaml_from_upload(llm_u)
        llm_raw = normalize_llm_routing(llm_raw)
        llm_b = yaml_preview(llm_raw).encode("utf-8")
        return {
            "workflow_raw": wf_raw, "workflow_bytes": wf_b,
            "profile_raw": prof_raw, "profile_bytes": prof_b,
            "policy_raw": pol_raw, "policy_bytes": pol_b,
            "tool_raw": tool_raw, "tool_bytes": tool_b,
            "llm_raw": llm_raw, "llm_bytes": llm_b,
        }
    except yaml.YAMLError as e:
        st.error(f"YAML parse error: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to read uploads: {e}")
        return None


bundle = load_yaml_bundle()

colA, colB = st.columns([1, 1])
with colA:
    st.subheader("Prompt")
    prompt = st.text_area(
        "One prompt to run through the whole system",
        height=160,
        placeholder="e.g., 'Add a new FastAPI endpoint /health that returns build info...'",
    )  # st.text_area widget  [oai_citation:3‡docs.streamlit.io](https://docs.streamlit.io/develop/api-reference/widgets/st.text_area?utm_source=chatgpt.com)

with colB:
    st.subheader("DB + Run controls")
    st.caption("DB is initialized with migrations if needed.")
    init_clicked = st.button("Initialize DB / Apply Migrations")
    run_clicked = st.button("Start run")
    st.caption("Use download buttons to export artifacts.  [oai_citation:4‡docs.streamlit.io](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button?utm_source=chatgpt.com)")


# DB open/init
conn: Optional[sqlite3.Connection] = None
try:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    if init_clicked:
        init_db(conn)
        st.success("DB initialized / migrations applied.")
except Exception as e:
    st.error(f"DB error: {e}")

# Preview YAML
if bundle:
    st.subheader("Loaded YAML preview")
    t1, t2 = st.columns(2)
    with t1:
        st.code(yaml_preview(bundle["workflow_raw"]), language="yaml")
        st.code(yaml_preview(bundle["profile_raw"]), language="yaml")
        st.code(yaml_preview(bundle["policy_raw"]), language="yaml")
    with t2:
        st.code(yaml_preview(bundle["tool_raw"]), language="yaml")
        st.code(yaml_preview(bundle["llm_raw"]), language="yaml")

# Start run
if run_clicked:
    if not conn:
        st.error("DB connection not available.")
    elif not bundle:
        st.error("Load YAMLs first.")
    elif not prompt.strip():
        st.error("Enter a prompt.")
    else:
        try:
            init_db(conn)

            cfg = build_resolved_config(
                workflow_raw=bundle["workflow_raw"],
                profile_raw=bundle["profile_raw"],
                policy_raw=bundle["policy_raw"],
                tool_registry_raw=bundle["tool_raw"],
                llm_routing_raw=bundle["llm_raw"],
                workflow_bytes=bundle["workflow_bytes"],
                profile_bytes=bundle["profile_bytes"],
                policy_bytes=bundle["policy_bytes"],
                tool_bytes=bundle["tool_bytes"],
                llm_bytes=bundle["llm_bytes"],
            )

            # Fixed resolver so engine uses exactly the YAML you loaded
            resolver = FixedRunConfigResolver(cfg=cfg)

            paths = RepoPaths.for_repo(repo_root_path)
            paths.ensure_dirs()

            gates = DBHumanGateManager(conn=conn, paths=paths)
            guards = DefaultGuardEvaluator()

            # Real LLM gateway + simple local tool executor
            llm = LLMGatewayImpl(conn=conn, paths=paths, resolver=resolver, services_factory=make_services)
            tools = SimpleLocalToolExecutor(conn=conn, paths=paths, services_factory=make_services)

            engine = WorkflowEngineV0(
                conn=conn,
                paths=paths,
                resolver=resolver,
                services_factory=make_services,
                llm=llm,
                tools=tools,
                gates=gates,
                guards=guards,
            )

            # For MVP: pass the prompt via module_spec.initial_prompt and reference it in workflow prompts as:
            #   {module_spec[initial_prompt]}
            identity = RunIdentity(
                repo_root=str(repo_root_path),
                run_id="(ignored_by_engine_v0)",
                profile_selector=f"{cfg.profile.id}@{cfg.profile.version}",
                workflow_selector=f"{cfg.workflow.id}@{cfg.workflow.version}",
                policy_selector=f"{cfg.policy.id}@{cfg.policy.version}",
            )

            handle = engine.start_run(
                identity=identity,
                config=cfg,
                initial_inputs={"module_spec": {"initial_prompt": prompt}},
            )

            st.session_state["last_run_id"] = handle.run_id
            st.session_state["last_gate"] = getattr(handle, "waiting_gate", None)

            st.success(f"Run started. run_id={handle.run_id} status={handle.status}")
        except Exception as e:
            st.error(f"Run failed to start: {e}")

# Resume controls (if paused)
last_run_id = st.session_state.get("last_run_id")
if conn and last_run_id:
    runs = list_runs(conn)
    st.subheader("Runs (latest 50)")
    st.json(runs)

    st.subheader("Selected run")
    selected_run = st.text_input("Run ID to inspect/resume", value=last_run_id)

    events = list_events(conn, selected_run)
    arts = list_artifacts(conn, selected_run)

    st.subheader("Events")
    st.json(events[:200])  # keep UI snappy

    st.subheader("Artifacts")
    st.json(arts)

    # Offer artifact download by reading from artifact paths if your artifacts are written to disk.
    # If your DataServices stores artifact contents in DB, you can adapt this.
    if arts:
        st.caption("Tip: implement artifact content lookup to support direct downloads from UI. ")

    st.subheader("Resume (Human gate)")
    gate = st.session_state.get("last_gate")
    if gate:
        st.info(f"Pending gate: {gate.gate_name} (mode={gate.mode}) gate_id={gate.gate_id}")

        decision = st.selectbox("Decision", ["APPROVE", "ACK", "BLOCK"], index=0)
        human = st.text_input("Human actor", value=os.getenv("USER", "you"))
        reason = st.text_input("Reason (optional)", value="")

        if st.button("Submit decision + resume"):
            try:
                paths = RepoPaths.for_repo(repo_root_path)
                resolver = FixedRunConfigResolver(cfg=build_resolved_config(
                    workflow_raw=bundle["workflow_raw"],
                    profile_raw=bundle["profile_raw"],
                    policy_raw=bundle["policy_raw"],
                    tool_registry_raw=bundle["tool_raw"],
                    llm_routing_raw=bundle["llm_raw"],
                    workflow_bytes=bundle["workflow_bytes"],
                    profile_bytes=bundle["profile_bytes"],
                    policy_bytes=bundle["policy_bytes"],
                    tool_bytes=bundle["tool_bytes"],
                    llm_bytes=bundle["llm_bytes"],
                ))
                gates = DBHumanGateManager(conn=conn, paths=paths)
                guards = DefaultGuardEvaluator()
                llm = LLMGatewayImpl(conn=conn, paths=paths, resolver=resolver, services_factory=make_services)
                tools = SimpleLocalToolExecutor(conn=conn, paths=paths, services_factory=make_services)

                engine = WorkflowEngineV0(
                    conn=conn,
                    paths=paths,
                    resolver=resolver,
                    services_factory=make_services,
                    llm=llm,
                    tools=tools,
                    gates=gates,
                    guards=guards,
                )

                handle = engine.resume_run(
                    run_id=selected_run,
                    human_inputs={
                        "gate_id": gate.gate_id,
                        "decision": decision,
                        "human_actor": human,
                        "reason": reason or None,
                    },
                )
                st.session_state["last_run_id"] = handle.run_id
                st.session_state["last_gate"] = getattr(handle, "waiting_gate", None)
                st.success(f"Resumed. status={handle.status}")
            except Exception as e:
                st.error(f"Resume failed: {e}")
    else:
        st.caption("No pending gate recorded in session. Start a run and pause on a gate to resume here.")

st.caption(
    "Notes: Use st.session_state to persist selected YAML/DB/run across reruns. "
)