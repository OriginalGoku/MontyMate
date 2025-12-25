# apps/spec_pipeline_ui.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import yaml

from montymate.spec_pipeline.data.context import ToolContext
from montymate.spec_pipeline.data.human_inputs import HumanAnswerBatch, normalize_answers
from montymate.spec_pipeline.data.run_state import RunState, RunStatus
from montymate.spec_pipeline.data.spec_store import FileSpecStore, SpecStore
from montymate.spec_pipeline.llm.llm_client import LLMClient, LLMConfig
from montymate.spec_pipeline.llm.providers.ollama_client import OllamaClient

# Tool imports are expected to exist in the project.
from montymate.spec_pipeline.tools.spec_author import SpecAuthorTool
from montymate.spec_pipeline.tools.spec_critic import SpecCriticTool

from montymate.spec_pipeline.coordinator import SpecPipelineCoordinator  # type: ignore[reportMissingImports]


@dataclass(frozen=True, slots=True)
class CoordinatorWiring:
    """Bundles runtime wiring for the coordinator."""
    store: SpecStore
    coordinator: object
    run_root: Path


def _project_root_from_text(text: str) -> Path:
    p = Path(text.strip() or ".").expanduser().resolve()
    return p


def _make_ollama_client(*, model: str, base_url: str, timeout_s: float, max_tokens: int, temperature: float) -> LLMClient:
    return OllamaClient(
        config=LLMConfig(
            provider="ollama",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            extra={"base_url": base_url, "timeout_s": timeout_s},
        )
    )


def _build_wiring(*, project_root: Path, run_id: str) -> CoordinatorWiring:
    store = FileSpecStore(project_root=project_root, run_id=run_id)

    # LLM settings are configurable from the UI.
    base_url = st.session_state.get("ollama_base_url", "http://localhost:11434")
    author_model = st.session_state.get("author_model", "nemotron-3-nano:30b")
    critic_model = st.session_state.get("critic_model", "nemotron-3-nano:30b")
    timeout_s = float(st.session_state.get("ollama_timeout_s", 180.0))
    max_tokens = int(st.session_state.get("ollama_max_tokens", 1200))
    temperature = float(st.session_state.get("ollama_temperature", 0.2))

    author_llm = _make_ollama_client(
        model=author_model,
        base_url=base_url,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    critic_llm = _make_ollama_client(
        model=critic_model,
        base_url=base_url,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    author_tool = SpecAuthorTool()
    critic_tool = SpecCriticTool()

    ctx = ToolContext(project_root=project_root, run_id=run_id, step="ui")

    coordinator = SpecPipelineCoordinator(
        store=store,
        author_tool=author_tool,
        critic_tool=critic_tool,
        author_llm=author_llm,
        critic_llm=critic_llm,
        and self.answerer_llm is not None=ctx,
    )

    run_root = project_root / ".ai_module_factory" / "runs" / run_id
    return CoordinatorWiring(store=store, coordinator=coordinator, run_root=run_root)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_json_dumps(obj: object) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return json.dumps({"error": "Failed to JSON-serialize object"}, indent=2)


def _ensure_run_state(*, store: SpecStore, run_id: str, user_prompt: str) -> RunState:
    existing = store.read_state()
    if existing is not None:
        return existing

    state = RunState(
        run_id=run_id,
        status=RunStatus.NEW,
        user_prompt=user_prompt,
        round_no=0,
        last_error=None,
        step="init",
    )
    store.write_state(state)
    return state


def _run_until_blocked(w: CoordinatorWiring) -> object:
    # Coordinator API is expected to expose run_until_blocked().
    return w.coordinator.run_until_blocked()  # type: ignore[attr-defined]


def _step_once(w: CoordinatorWiring) -> object:
    # Coordinator API is expected to expose step().
    return w.coordinator.step()  # type: ignore[attr-defined]


def _render_state_panel(*, w: CoordinatorWiring) -> None:
    st.subheader("Run status")

    state = w.store.read_state()
    if state is None:
        st.info("No run_state.json found yet for this run.")
        return

    st.code(yaml.safe_dump(state.to_dict(), sort_keys=False, allow_unicode=True), language="yaml")
    st.caption(f"Run root: {w.run_root}")


def _render_spec_panel(*, w: CoordinatorWiring) -> None:
    st.subheader("module_spec.yaml")

    spec_path = w.run_root / "module_spec.yaml"
    raw = _read_text(spec_path)

    if not raw.strip():
        st.info("module_spec.yaml not found yet.")
        return

    st.code(raw, language="yaml")

    with st.expander("Edit and save module_spec.yaml (manual intervention)", expanded=False):
        edited = st.text_area("Spec YAML", value=raw, height=360)
        if st.button("Save spec YAML"):
            try:
                parsed = yaml.safe_load(edited) or {}
                if not isinstance(parsed, dict):
                    st.error("Spec YAML must be a mapping at the top level.")
                    return
                # Spec validation is delegated to Spec.from_dict() via store.write_spec().
                from montymate.spec_pipeline.data.spec_types import Spec  # local import to avoid cycles during dev

                spec_obj = Spec.from_dict(parsed)
                w.store.write_spec(spec_obj, tag="manual_edit")
                st.success("Saved module_spec.yaml and a snapshot module_spec_manual_edit.yaml.")
            except Exception as e:
                st.error(f"Failed to parse/save YAML: {e}")


def _render_critic_panel(*, w: CoordinatorWiring) -> None:
    st.subheader("Critic report (latest round)")

    state = w.store.read_state()
    if state is None:
        st.info("No state yet.")
        return

    if state.round_no <= 0:
        st.info("No critic round yet.")
        return

    report = w.store.read_critic_report(round_no=state.round_no)
    if report is None:
        st.info(f"No critic report found for round {state.round_no}.")
        return

    st.code(_safe_json_dumps(report), language="json")


def _render_answers_panel(*, w: CoordinatorWiring) -> None:
    st.subheader("Answer targeted questions")

    state = w.store.read_state()
    if state is None:
        st.info("No state yet.")
        return

    if state.status != RunStatus.WAITING_FOR_HUMAN:
        st.info("Run is not waiting for human answers.")
        return

    round_no = int(state.round_no)
    report = w.store.read_critic_report(round_no=round_no)
    if report is None:
        st.warning(f"Critic report for round {round_no} is missing.")
        return

    questions_obj = report.get("targeted_questions")
    questions: list[str] = []
    if isinstance(questions_obj, list):
        for q in questions_obj:
            if isinstance(q, str) and q.strip():
                questions.append(q.strip())

    if not questions:
        st.info("No targeted questions present.")
        return

    existing = w.store.read_answers_batch(round_no)
    existing_by_index: dict[int, str] = {}
    if existing is not None:
        for idx, a in enumerate(existing.answers):
            existing_by_index[idx] = a.answer

    st.caption("Blank answer means: decide-for-me")

    with st.form(key=f"answers_round_{round_no}"):
        answers: list[str] = []
        for i, q in enumerate(questions, start=1):
            default_val = existing_by_index.get(i - 1, "")
            ans = st.text_area(f"Q{i}: {q}", value=default_val, height=80)
            answers.append(ans)

        submitted = st.form_submit_button("Save answers")
        if submitted:
            normalized = normalize_answers(questions, {i: answers[i] for i in range(len(answers))})
            batch = HumanAnswerBatch(round_no=round_no, answers=normalized)
            w.store.write_answers_batch(batch)
            st.success(f"Saved answers for round {round_no}.")


def main() -> None:
    st.set_page_config(page_title="Spec Pipeline MVP", layout="wide")
    st.title("Spec Pipeline MVP (Author ↔ Critic ↔ Human)")

    with st.sidebar:
        st.header("Run selection")

        project_root_text = st.text_input("project_root", value=str(Path(".").resolve()))
        run_id = st.text_input("run_id", value=st.session_state.get("run_id", "demo_ui"))

        st.divider()
        st.header("Ollama settings")

        st.text_input("ollama_base_url", value=st.session_state.get("ollama_base_url", "http://localhost:11434"), key="ollama_base_url")
        st.text_input("author_model", value=st.session_state.get("author_model", "nemotron-3-nano:30b"), key="author_model")
        st.text_input("critic_model", value=st.session_state.get("critic_model", "nemotron-3-nano:30b"), key="critic_model")
        st.number_input("timeout_s", min_value=10.0, max_value=600.0, value=float(st.session_state.get("ollama_timeout_s", 180.0)), step=10.0, key="ollama_timeout_s")
        st.number_input("max_tokens", min_value=64, max_value=20_000, value=int(st.session_state.get("ollama_max_tokens", 12000)), step=64, key="ollama_max_tokens")
        st.number_input("temperature", min_value=0.0, max_value=2.0, value=float(st.session_state.get("ollama_temperature", 0.2)), step=0.1, key="ollama_temperature")

        st.divider()
        user_prompt = st.text_area("user_prompt (used on start if no state exists)", height=120)

        st.session_state["run_id"] = run_id

    project_root = _project_root_from_text(project_root_text)
    if not run_id.strip():
        st.error("run_id must be non-empty.")
        return

    w = _build_wiring(project_root=project_root, run_id=run_id)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("Start (init state if missing)"):
            _ = _ensure_run_state(store=w.store, run_id=run_id, user_prompt=user_prompt)
            st.success("Run initialized (or already existed).")

    with col_b:
        if st.button("Step once"):
            try:
                result = _step_once(w)
                st.session_state["last_result"] = result
                st.success("Step executed.")
            except Exception as e:
                st.error(f"Step failed: {e}")

    with col_c:
        if st.button("Resume (run_until_blocked)"):
            try:
                _ = _ensure_run_state(store=w.store, run_id=run_id, user_prompt=user_prompt)
                result = _run_until_blocked(w)
                st.session_state["last_result"] = result
                st.success("Coordinator advanced until blocked.")
            except Exception as e:
                st.error(f"Resume failed: {e}")

    if "last_result" in st.session_state:
        st.subheader("Last coordinator result")
        st.code(_safe_json_dumps(st.session_state["last_result"]), language="json")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["State", "Spec", "Critic", "Answers"])
    with tab1:
        _render_state_panel(w=w)
    with tab2:
        _render_spec_panel(w=w)
    with tab3:
        _render_critic_panel(w=w)
    with tab4:
        _render_answers_panel(w=w)


if __name__ == "__main__":
    main()