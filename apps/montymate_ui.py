# apps/montymate_ui.py
from __future__ import annotations

import yaml
import json
from pathlib import Path
from typing import Optional

import streamlit as st


from montymate.mvp.db import connect_db, init_schema, decide_gate
from montymate.mvp.routing import load_llm_routing_yaml
from montymate.mvp.pipeline import start_run, resume_run, finalize_after_audit


CANVAS = [
    ("goal", "Goal", "str"),
    ("functional_requirements", "Functional requirements (one per line)", "list"),
    ("constraints", "Constraints (one per line)", "list"),
    ("security_concerns", "Security concerns (one per line)", "list"),
    ("assumptions", "Assumptions (one per line)", "list"),
    ("other_notes", "Other notes", "str"),
]

def _artifacts_dir(repo_root_path: Path, run_id: str) -> Path:
    return repo_root_path / ".ai_module_factory" / "runs" / run_id / "artifacts"

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

def _read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        obj = json.loads(_read_text(p))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _read_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        obj = yaml.safe_load(_read_text(p))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}
        
def _list_run_ids(repo_root_path: Path) -> list[str]:
    runs_dir = repo_root_path / ".ai_module_factory" / "runs"
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []
    return sorted(
        [p.name for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith(".")],
        reverse=True,  # newest-ish first if ids are time-ish; otherwise just shows recent at top
    )
    
def _run_artifacts_dir(repo_root_path: Path, run_id: str) -> Path:
    return repo_root_path / ".ai_module_factory" / "runs" / run_id / "artifacts"


def _load_json_file(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _load_yaml_file(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _spec_list_to_text(x) -> str:
    if isinstance(x, list):
        return "\n".join(str(i).strip() for i in x if str(i).strip())
    return ""


def _fill_canvas_from_artifacts(repo_root_path: Path, run_id: str, prefix: str = "mm_") -> None:
    art = _run_artifacts_dir(repo_root_path, run_id)

    spec = _load_yaml_file(art / "module_spec.yaml")
    locks = _load_json_file(art / "module_spec_locks.json")

    # ---- text fields (mm_goal, mm_constraints, ...) ----
    st.session_state[f"{prefix}goal"] = (spec.get("goal") or "").strip()
    st.session_state[f"{prefix}functional_requirements"] = _spec_list_to_text(spec.get("functional_requirements"))
    st.session_state[f"{prefix}constraints"] = _spec_list_to_text(spec.get("constraints"))
    st.session_state[f"{prefix}security_concerns"] = _spec_list_to_text(spec.get("security_concerns"))
    st.session_state[f"{prefix}assumptions"] = _spec_list_to_text(spec.get("assumptions"))
    st.session_state[f"{prefix}other_notes"] = (spec.get("other_notes") or "").strip()

    # ---- lock fields (mm_lock_goal, ...) ----
    for k in ["goal", "functional_requirements", "constraints", "security_concerns", "assumptions", "other_notes"]:
        st.session_state[f"{prefix}lock_{k}"] = bool(locks.get(k, False))
        

def render_canvas(prefix: str = "mm_") -> None:
    st.markdown("### Spec canvas (optional)")
    st.caption("Fill what you know. Lock fields you consider final. Unlocked fields may be refined by the interviewer.")

    for field, label, kind in CANVAS:
        left, right = st.columns([0.86, 0.14], vertical_alignment="center")

        # IMPORTANT: keep your current key naming exactly
        lock_key = f"{prefix}lock_{field}"
        text_key = f"{prefix}{field}"

        with right:
            st.checkbox("Lock", key=lock_key)

        disabled = bool(st.session_state.get(lock_key, False))

        with left:
            st.text_area(
                label,
                key=text_key,
                height=110 if kind == "list" else 80,
                disabled=disabled,
            )

st.set_page_config(page_title="MontyMate MVP", layout="wide")
st.title("MontyMate MVP")



def _collect_spec_answers(run_id: str, gate_id: str, questions: list[str]) -> list[dict]:
    out: list[dict] = []
    for i, q in enumerate(questions):
        decide_key = f"mm_decide_{run_id}_{gate_id}_{i}"
        answer_key = f"mm_answer_{run_id}_{gate_id}_{i}"
        decide_ui = bool(st.session_state.get(decide_key, False))
        ans = (st.session_state.get(answer_key, "") or "").strip()
        
        # Default behavior: if empty answer -> decide_for_me=True
        decide = decide_ui or (ans == "")
        out.append(
            {
                "question": q,
                "decide_for_me": decide,
                "answer": "" if decide else ans,
            }
        )
    return out
    
# -----------------------------
# Sidebar: repo/db/routing
# -----------------------------
with st.sidebar:
    st.header("Repo + DB")

    repo_root = st.text_input("Repo root", value=str(Path(".").resolve()))
    repo_root_path = Path(repo_root).resolve()
    
    resources_root_path = Path(
        st.sidebar.text_input("Resources root", value=str(repo_root_path / "src" / "montymate" / "resources"))
    ).resolve()
    
    default_db = repo_root_path / ".ai_module_factory" / "montymate.sqlite3"
    db_path = st.text_input("SQLite DB path", value=str(default_db))
    
    
    # -----------------------------
    # DB init
    # -----------------------------
    conn = connect_db(db_path)
    init_schema(conn)

    st.divider()
    st.header("LLM Routing")
    

    default_routing_path = resources_root_path / "configs" / "llm_routing_v1.yaml"
    routing_mode = st.radio("Routing source", ["Use file from repo", "Upload YAML"], index=0)

    routing_bytes: Optional[bytes] = None

    if routing_mode == "Use file from repo":
        st.caption(f"Expected path: {default_routing_path}")
        if default_routing_path.exists():
            routing_bytes = default_routing_path.read_bytes()
            st.success("Loaded routing YAML from repo.")
        else:
            st.error("Routing YAML not found at the expected path.")
    else:
        uploaded = st.file_uploader("Upload llm_routing_v1.yaml", type=["yaml", "yml"])  #  [oai_citation:4‡Streamlit Docs](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader?utm_source=chatgpt.com)
        if uploaded is not None:
            routing_bytes = uploaded.getvalue()
            st.success("Uploaded routing YAML loaded.")

    st.divider()
    st.header("Gate defaults")
    default_gate_mode = st.selectbox("Default gate mode", ["ACK", "APPROVE"], index=0)
    
    st.divider()
    st.header("Load existing run")
    
    run_ids = _list_run_ids(repo_root_path)
    
    selected_run_id = st.selectbox(
        "Run to load",
        options=[""] + run_ids,
        index=0,
    )
    
    if st.button("Load run"):
        if selected_run_id:
            st.session_state.run_id = selected_run_id
    
            row = conn.execute(
                """
                SELECT gate_id, gate_name
                FROM mm_gates
                WHERE run_id=? AND status='PENDING'
                ORDER BY gate_id DESC
                LIMIT 1
                """,
                # ORDER BY created_at DESC
                (st.session_state.run_id,),
            ).fetchone()
    
            if row:
                st.session_state.gate_id = row["gate_id"]
                st.session_state.gate_name = row["gate_name"]
    
                payload_row = conn.execute(
                    "SELECT payload_json FROM mm_gates WHERE gate_id=?",
                    (row["gate_id"],),
                ).fetchone()
                payload = json.loads(payload_row["payload_json"]) if payload_row else {}
                st.session_state["mm_user_prompt"] = payload.get("user_prompt", "")
    
                # _fill_canvas_from_artifacts(repo_root_path, st.session_state.run_id, prefix="mm_")
                st.session_state["mm_hydrate_run_id"] = st.session_state.run_id
                st.rerun()
                # st.success(f"Loaded run {selected_run_id}, pending gate: {row['gate_name']}")
            else:
                st.session_state.gate_id = None
                st.session_state.gate_name = None
                st.session_state.run_id = None
                st.info("Run loaded, but no pending gate found.")
        else:
            st.error("Pick a run_id")

# Must have routing to proceed
if not routing_bytes:
    st.info("Provide an LLM routing YAML (repo file or upload) to continue.")
    st.stop()

routing = load_llm_routing_yaml(routing_bytes)
# --- One-shot canvas hydration (MUST run before render_canvas creates widgets) ---
# hydrate_run_id = st.session_state.pop("mm_hydrate_run_id", None)
# if hydrate_run_id:
#     _fill_canvas_from_artifacts(repo_root_path, hydrate_run_id, prefix="mm_")


# --- One-shot full reset (MUST run before widgets are created) ---
# do_reset = st.session_state.pop("mm_do_reset", False)
# if do_reset:
#     st.session_state["run_id"] = None
#     st.session_state["gate_id"] = None
#     st.session_state["gate_name"] = None

#     st.session_state["mm_user_prompt"] = ""
#     for k in ["goal","functional_requirements","constraints","security_concerns","assumptions","other_notes"]:
#         st.session_state[f"mm_{k}"] = ""
#         st.session_state[f"mm_lock_{k}"] = False
     
# --- One-shot full reset (MUST run before widgets are created) ---
do_reset = st.session_state.pop("mm_do_reset", False)
if do_reset:
    st.session_state["run_id"] = None
    st.session_state["gate_id"] = None
    st.session_state["gate_name"] = None

    st.session_state["mm_user_prompt"] = ""
    for k in ["goal","functional_requirements","constraints","security_concerns","assumptions","other_notes"]:
        st.session_state[f"mm_{k}"] = ""
        st.session_state[f"mm_lock_{k}"] = False

# --- One-shot canvas hydration (MUST run before render_canvas creates widgets) ---
hydrate_run_id = st.session_state.pop("mm_hydrate_run_id", None)
if hydrate_run_id:
    _fill_canvas_from_artifacts(repo_root_path, hydrate_run_id, prefix="mm_")
    
def _lines_to_list(s: str) -> list[str]:
    return [ln.strip() for ln in (s or "").splitlines() if ln.strip()]

def _collect_initial_module_spec(prefix: str = "mm_") -> dict:
    return {
        "status": "DRAFT",
        "goal": (st.session_state.get(f"{prefix}goal", "") or "").strip(),
        "functional_requirements": _lines_to_list(st.session_state.get(f"{prefix}functional_requirements", "")),
        "constraints": _lines_to_list(st.session_state.get(f"{prefix}constraints", "")),
        "security_concerns": _lines_to_list(st.session_state.get(f"{prefix}security_concerns", "")),
        "assumptions": _lines_to_list(st.session_state.get(f"{prefix}assumptions", "")),
        "other_notes": (st.session_state.get(f"{prefix}other_notes", "") or "").strip(),
    }

def _collect_initial_spec_locks(prefix: str = "mm_") -> dict:
    return {
        "goal": bool(st.session_state.get(f"{prefix}lock_goal", False)),
        "functional_requirements": bool(st.session_state.get(f"{prefix}lock_functional_requirements", False)),
        "constraints": bool(st.session_state.get(f"{prefix}lock_constraints", False)),
        "security_concerns": bool(st.session_state.get(f"{prefix}lock_security_concerns", False)),
        "assumptions": bool(st.session_state.get(f"{prefix}lock_assumptions", False)),
        "other_notes": bool(st.session_state.get(f"{prefix}lock_other_notes", False)),
    }    
# -----------------------------
# Session state
# -----------------------------
def _init_state():
    st.session_state.setdefault("run_id", None)
    st.session_state.setdefault("gate_id", None)
    st.session_state.setdefault("gate_name", None)

    # Canvas text values
    st.session_state.setdefault("mm_goal", "")
    st.session_state.setdefault("mm_functional_requirements", "")
    st.session_state.setdefault("mm_constraints", "")
    st.session_state.setdefault("mm_security_concerns", "")
    st.session_state.setdefault("mm_assumptions", "")
    st.session_state.setdefault("mm_other_notes", "")

    # Locks
    st.session_state.setdefault("mm_lock_goal", False)
    st.session_state.setdefault("mm_lock_functional_requirements", False)
    st.session_state.setdefault("mm_lock_constraints", False)
    st.session_state.setdefault("mm_lock_security_concerns", False)
    st.session_state.setdefault("mm_lock_assumptions", False)
    st.session_state.setdefault("mm_lock_other_notes", False)

def _clear_run_state():
    st.session_state["mm_do_reset"] = True
    
_init_state()

# -----------------------------
# Main: Start run form
# -----------------------------
is_active_run = bool(st.session_state.run_id and st.session_state.gate_id and st.session_state.gate_name)
if not is_active_run:
    
    st.subheader("Start a run")
    
    prompt = st.text_area("One prompt to run through MontyMate", height=150, key="mm_user_prompt")
    
    render_canvas(prefix="mm_")
    
    # NOTE: widgets support disabled=; this is exactly how you make “locked fields” read-only.  [oai_citation:2‡Streamlit Docs](https://docs.streamlit.io/develop/api-reference/widgets/st.text_input?utm_source=chatgpt.com)
    
    if st.button("Start Run", disabled=is_active_run):
        if not (prompt or "").strip():
            st.error("Please enter a prompt.")
        else:
            initial_spec = _collect_initial_module_spec()
            initial_locks = _collect_initial_spec_locks()
    
            try:
                r = start_run(
                    conn=conn,
                    repo_root=str(repo_root_path),
                    routing=routing,
                    user_prompt=prompt,
                    default_gate_mode=default_gate_mode,
                    initial_module_spec=initial_spec,        # <-- NEW
                    initial_spec_locks=initial_locks,        # <-- NEW
                )
                st.session_state.run_id = r.run_id
                st.session_state.gate_id = r.gate_id
                st.session_state.gate_name = r.gate_name
                st.session_state["mm_hydrate_run_id"] = r.run_id
                st.success(f"{r.status}: run_id={r.run_id} pending_gate={r.gate_name}")
                st.rerun()
            except Exception as e:
                st.exception(e)
                st.stop()
                
else:
    st.text_area("Original prompt", st.session_state.get("mm_user_prompt", ""), disabled=True, height=120)
    st.info(f"Run in progress: {st.session_state.run_id} (pending gate: {st.session_state.gate_name})")

# Quick controls
col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("New run"):
        _clear_run_state()
        # st.session_state["mm_do_reset"] = True
        st.rerun()
        # st.success("Ready for a new run.")

with col_b:
    if st.session_state.run_id:
        st.caption(f"Current run_id: {st.session_state.run_id}")

st.divider()

# -----------------------------
# Gate handling
# -----------------------------
run_id = st.session_state.run_id
gate_id = st.session_state.gate_id
gate_name = st.session_state.gate_name

if run_id and gate_id and gate_name:
    st.subheader(f"Pending gate : {gate_name}", text_alignment="center")
    # st.write({"run_id": run_id, "gate_name": gate_name, "gate_id": gate_id})
    gate_row = conn.execute(
        "SELECT payload_json, mode, status FROM mm_gates WHERE gate_id=?",
        (gate_id,),
    ).fetchone()
    payload = {}
    if gate_row:
        try:
            payload = json.loads(gate_row["payload_json"])
        except Exception:
            payload = {}
    
    st.caption(f"Mode: {gate_row['mode']} | Status: {gate_row['status']}" if gate_row else "Gate not found")
    
    if gate_name == "spec_validation":
    
        questions = payload.get("targeted_questions") or []
        issues = payload.get("issues") or []
    
        if issues:
            st.divider()
            st.markdown("### Validator issues")
            st.dataframe(issues)
    
        st.subheader("Answer targeted questions")
        
        
        st.markdown("### Questions")
        
        for i, q in enumerate(questions):
            decide_key = f"mm_decide_{run_id}_{gate_id}_{i}"
            answer_key = f"mm_answer_{run_id}_{gate_id}_{i}"
        
            cols = st.columns([0.15, 0.85], vertical_alignment="center")
            with cols[0]:
                # Default True, but don't override once it exists
                if decide_key not in st.session_state:
                    st.session_state[decide_key] = True
                st.checkbox("Decide for me", key=decide_key)
        
            with cols[1]:
                st.text_area(
                    q,
                    key=answer_key,
                    height=80,
                    # disabled=st.session_state.get(decide_key, True),
                )
        
        render_canvas(prefix="mm_")
        
        gate_decision = st.selectbox(
            "Decision for this round",
            ["APPROVE", "ACK", "BLOCK"],
            index=0,
            key=f"mm_sv_decision_{run_id}_{gate_id}",
        )
        human_actor = st.text_input("Human actor", value="you", key=f"mm_sv_actor_{run_id}_{gate_id}")
        reason = st.text_input("Reason (optional)", value="", key=f"mm_sv_reason_{run_id}_{gate_id}")
        
        submitted_answers = st.button("Submit answers + continue")
        if submitted_answers:
            spec_answers = _collect_spec_answers(run_id, gate_id, questions)
            updated_spec = _collect_initial_module_spec()
            updated_locks = _collect_initial_spec_locks()
        
            r = resume_run(
                conn=conn,
                repo_root=str(repo_root_path),
                routing=routing,
                run_id=run_id,
                gate_id=gate_id,
                decision=gate_decision,
                human_actor=human_actor,
                reason=reason or None,
                spec_answers=spec_answers,
                updated_module_spec=updated_spec,
                updated_spec_locks=updated_locks,
                default_gate_mode=default_gate_mode,
            )
        
            st.session_state.gate_id = r.gate_id
            st.session_state.gate_name = r.gate_name
        
            if r.status == "FAILED":
                st.error(r.message or "Run failed.")
                _clear_run_state()
            else:
                st.success(f"{r.status}: {r.message}")
        
            st.rerun()
        st.stop()
            
    
    if gate_name == "spec_lock":
        st.subheader("Review and lock spec")
    
        # 1) Ensure canvas is populated from current artifacts (only once per gate)
        hydrate_key = f"mm_hydrated_{run_id}_{gate_id}"
        if not st.session_state.get(hydrate_key, False):
            st.session_state["mm_hydrate_run_id"] = run_id
            st.session_state[hydrate_key] = True
            st.rerun()
    
        # 2) Show current spec + allow edits in canvas
        art = _artifacts_dir(repo_root_path, run_id)
        st.markdown("### Current module_spec.yaml")
        st.code(_read_text(art / "module_spec.yaml"), language="yaml")
    
        st.divider()
        st.markdown("### Edit spec in canvas")
        render_canvas(prefix="mm_")
    
        # 3) Optional: revalidate one last time (recommended)
        col1, col2 = st.columns(2)
        with col1:
            do_revalidate = st.checkbox("Re-run spec validator before locking", value=True)
        with col2:
            do_ai_polish = st.checkbox("Run AI polish suggestions (optional)", value=False)
    
        if st.button("Lock spec and continue"):
            updated_spec = _collect_initial_module_spec()
            updated_locks = _collect_initial_spec_locks()
    
            # IMPORTANT: this requires a small pipeline change (next section)
            r = resume_run(
                conn=conn,
                repo_root=str(repo_root_path),
                routing=routing,
                run_id=run_id,
                gate_id=gate_id,
                decision="ACK",
                human_actor="you",
                reason=None,
                updated_module_spec=updated_spec,
                updated_spec_locks=updated_locks,
                revalidate_before_lock=do_revalidate,
                ai_polish_before_lock=do_ai_polish,
                default_gate_mode=default_gate_mode,
            )
            st.session_state.gate_id = r.gate_id
            st.session_state.gate_name = r.gate_name
            st.success(f"{r.status}: {r.message}")
            st.rerun()
    # Show gate payload
    
    if gate_row:
        # st.caption(f"Mode: {gate_row['mode']} | Status: {gate_row['status']}")
        try:
            st.json(json.loads(gate_row["payload_json"]))
        except Exception:
            st.code(gate_row["payload_json"])

    decision = st.selectbox("Decision", ["ACK", "APPROVE", "BLOCK"], index=0)
    human_actor = st.text_input("Human actor", value="you")
    reason = st.text_input("Reason (optional)", value="")

    if st.button("Submit decision + continue"):
        if gate_name == "audit_gate":
            # audit_gate is terminal in MVP: decide it, then finalize.
            decide_gate(
                conn,
                gate_id=gate_id,
                decision=decision,
                human_actor=human_actor,
                reason=reason or None,
            )
            if decision == "BLOCK":
                st.error("Run blocked at audit_gate.")
            else:
                r2 = finalize_after_audit(conn=conn, repo_root=str(repo_root_path), run_id=run_id)
                st.success(f"{r2.status}: {r2.message}")
                _clear_run_state()
                st.rerun()
        else:
            r = resume_run(
                conn=conn,
                repo_root=str(repo_root_path),
                routing=routing,
                run_id=run_id,
                gate_id=gate_id,
                decision=decision,
                human_actor=human_actor,
                reason=reason or None,
            )
            st.session_state.gate_id = r.gate_id
            st.session_state.gate_name = r.gate_name
            if r.status == "FAILED":
                st.error(r.message or "Run failed.")
                _clear_run_state()
                st.rerun()
            else:
                st.success(f"{r.status}: {r.message}")

st.divider()

run_id = st.session_state.run_id
# -----------------------------
# Inspect: events + artifacts
# -----------------------------
run_id = st.session_state.run_id


# -----------------------------
# Inspect: run summary + key artifacts
# -----------------------------
if run_id:
    art = _artifacts_dir(repo_root_path, run_id)

    st.subheader("Run summary")
    st.write(
        {
            "run_id": run_id,
            "pending_gate": st.session_state.get("gate_name"),
            "gate_id": st.session_state.get("gate_id"),
        }
    )

    # --- Key artifacts (the ones humans actually care about) ---
    spec = _read_yaml(art / "module_spec.yaml")
    locks = _read_json(art / "module_spec_locks.json")
    report = _read_json(art / "spec_validation_report.json")

    if report:
        st.markdown("### Latest spec validation report")
        st.json(report)

    if spec:
        st.markdown("### Current module_spec.yaml")
        st.code(yaml.safe_dump(spec, sort_keys=False), language="yaml")

    if locks:
        st.markdown("### Current module_spec_locks.json")
        st.json(locks)

    # Optional: show other “endgame” artifacts if present
    plan_md = _read_text(art / "architecture_plan.md")
    if plan_md:
        with st.expander("Architecture plan"):
            st.markdown(plan_md)

    audit_md = _read_text(art / "audit_report.md")
    if audit_md:
        with st.expander("Audit report"):
            st.markdown(audit_md)

    st.divider()

    # --- Generic artifact browser (keep it, but move below) ---
    st.subheader("Artifacts (browser)")

    arts = conn.execute(
        """
        SELECT artifact_id, artifact_type, relpath, sha256, bytes, created_at
        FROM mm_artifacts
        WHERE run_id=?
        ORDER BY created_at ASC
        """,
        (run_id,),
    ).fetchall()

    if not arts:
        st.info("No artifacts recorded yet.")
    else:
        options = {
            f'{r["artifact_type"]} — {r["relpath"]} ({r["bytes"]} bytes)': r["artifact_id"]
            for r in arts
        }
        label = st.selectbox("Select an artifact", list(options.keys()))
        artifact_id = options[label]

        row = conn.execute(
            "SELECT artifact_type, relpath FROM mm_artifacts WHERE artifact_id=?",
            (artifact_id,),
        ).fetchone()

        abs_path = repo_root_path / ".ai_module_factory" / "runs" / run_id / row["relpath"]
        st.caption(f"Path: {abs_path}")

        if abs_path.exists():
            data = abs_path.read_bytes()
            suffix = abs_path.suffix.lower()
            mime = "application/octet-stream"

            if suffix in (".md", ".markdown"):
                st.markdown(data.decode("utf-8", errors="replace"))
                mime = "text/markdown"
            elif suffix in (".yaml", ".yml"):
                st.code(data.decode("utf-8", errors="replace"), language="yaml")
                mime = "application/x-yaml"
            elif suffix == ".json":
                try:
                    st.json(json.loads(data.decode("utf-8", errors="replace")))
                except Exception:
                    st.code(data.decode("utf-8", errors="replace"))
                mime = "application/json"
            else:
                st.code(data.decode("utf-8", errors="replace"))

            st.download_button(
                label="Download artifact",
                data=data,
                file_name=abs_path.name,
                mime=mime,
            )
        else:
            st.error("Artifact file missing on disk (DB row exists but file not found).")

    # --- Events become “debug”, not the main UI ---
    with st.expander("Debug: events (latest 50)"):
        events = conn.execute(
            "SELECT ts, event_type, actor_type, severity, payload_json "
            "FROM mm_events WHERE run_id=? ORDER BY event_id DESC LIMIT 50",
            (run_id,),
        ).fetchall()
        st.json(
            [
                {
                    "ts": r["ts"],
                    "type": r["event_type"],
                    "actor": r["actor_type"],
                    "severity": r["severity"],
                    "payload": json.loads(r["payload_json"]),
                }
                for r in events
            ]
        )
else:
    st.info("Start a run to see summary and artifacts.")