# Spec Pipeline Contracts
Version: v1 - Dec 21 2025

## Purpose
`montymate.spec_pipeline` builds and finalizes a **Module Spec** from a user prompt through an interactive loop:

1) **SpecComposer** produces a draft spec  
2) **SpecCritic** evaluates the spec and emits targeted questions if needed  
3) Human answers those questions (or leaves blank for “decide for me”)  
4) **SpecRefiner** integrates answers and resolves issues into a stronger spec  
5) Optional **SpecEditor** improves clarity/style without changing intent  
6) **SpecLocker** finalizes the spec (status becomes `LOCKED`)  
7) Output is a locked spec artifact that later feeds the Architect module

Each component can be tested independently, then composed by a single orchestrator: `spec_agent.py`.

---

## Package and File Layout

Recommended layout:
src/
  montymate/
    spec_pipeline/
      __init__.py
      spec_agent.py

      tools/
        __init__.py
        tool_protocol.py
        composer.py
        critic.py
        refiner.py
        editor.py
        locker.py

      data/
        __init__.py
        spec_types.py
        spec_store.py
        human_inputs.py

      prompts/
        __init__.py
        prompts.py

      tests/
        test_composer.py
        test_critic.py
        test_refiner.py
        test_editor.py
        test_locker.py
        test_spec_store.py
        test_agent_flow.py

---

## Core Domain Objects

### Spec
A spec is the single source of truth for “what we intend to build”.

**Canonical fields**
- `status`: `"DRAFT" | "LOCKED"`
- `goal`: `str`
- `functional_requirements`: `list[str]`
- `constraints`: `list[str]`
- `security_concerns`: `list[str]`
- `assumptions`: `list[str]`
- `other_notes`: `str`

**Rules**
- Spec must always contain all keys (empty string/list if unknown).
- Spec must be serializable to/from YAML.
- Spec should be normalized on every boundary (tool in/out, store read/write).
- Tools must output only these keys.

---

### CritiqueReport
Produced by SpecCritic.

**Fields**
- `passed`: `bool`
- `issues`: `list[str]`
- `contradictions`: `list[str]`
- `targeted_questions`: `list[str]`

**Rules**
- If `passed=False`, `targeted_questions` should be non-empty (unless the critic is truly blocked; in that case include an issue explaining why).
- Questions must be actionable and minimal.

---

### HumanAnswer
Simplified human input.

**Fields**
- `question`: `str`
- `answer`: `str`
- derived: `decide_for_me = (answer.strip() == "")`

**Rules**
- If `decide_for_me=True`, downstream tools should choose a reasonable default and document it in `assumptions` if appropriate.

---

## Tool Protocol (Class Inheritance)

All components are “tools” that can be run in isolation and wired together.

### Base Tool Contract
Every tool:
- has a name
- takes typed input
- returns typed output
- is deterministic in *structure* (even if LLM output varies)
- does not do persistence directly (the orchestrator/store handles I/O)

---

## Components

### 1) SpecComposer
**Goal:** Produce an initial draft spec from the user prompt + optional seed spec.

**Input**
- `user_prompt: str`
- `seed_spec: Spec` (optional; defaults to empty normalized Spec)

**Output**
- `spec: Spec` (status should remain `DRAFT`)

**Responsibilities**
- Fill as much as possible from the prompt without inventing scope.
- If unknown, write explicit assumptions in `assumptions`.
- Keep only canonical keys.

---

### 2) SpecCritic
**Goal:** Evaluate the spec for completeness, ambiguity, contradictions. Generate targeted questions if needed.

**Input**
- `user_prompt: str`
- `spec: Spec`

**Output**
- `report: CritiqueReport`

**Responsibilities**
- Identify missing information, contradictions, unclear requirements.
- Ask the smallest set of questions needed to unblock.
- If spec is good enough, return `passed=True` with minimal/no questions.

---

### 3) SpecRefiner
**Goal:** Update the spec using critique + human answers. Resolve issues and make choices when answers are blank.

**Input**
- `user_prompt: str`
- `spec: Spec` (current draft)
- `report: CritiqueReport`
- `answers: list[HumanAnswer]`

**Output**
- `spec: Spec` (still `DRAFT`)

**Responsibilities**
- Incorporate answers into the spec.
- If an answer is blank, decide a reasonable default; record it in `assumptions` when relevant.
- Resolve contradictions flagged by the critic where possible.
- Do not expand scope beyond the original prompt.

---

### 4) SpecEditor (Optional)
**Goal:** Improve clarity/wording/formatting of the spec without changing meaning or scope.

**Input**
- `user_prompt: str`
- `spec: Spec`

**Output**
- `spec: Spec` (still `DRAFT`)

**Responsibilities**
- Rewrite for clarity and precision.
- Remove ambiguity where possible *without* adding new requirements.
- Keep canonical keys and stable types.

---

### 5) SpecLocker
**Goal:** Finalize the spec once it is acceptable.

**Input**
- `spec: Spec`

**Output**
- `spec: Spec` with `status="LOCKED"`

**Responsibilities**
- Set `status="LOCKED"`.
- Optionally perform final normalization checks (types/keys).
- No semantic changes besides status.

---

## SpecStore

### Purpose
A thin persistence layer that reads/writes spec artifacts from the run directory.

**Responsibilities**
- Read current spec (`module_spec.yaml`) if present.
- Write updated spec (`module_spec.yaml`) and optionally versioned snapshots.
- Read/write latest critique report (`spec_critique_report.json`) and optionally round snapshots.

**Recommended artifacts**
- `artifacts/module_spec.yaml` (current)
- `artifacts/module_spec_round_{n}.yaml` (optional history)
- `artifacts/spec_critique_report.json` (current)
- `artifacts/spec_critique_report_round_{n}.json` (optional history)
- `artifacts/spec_answers_round_{n}.json` (history of human answers)

---

## Orchestrator: spec_agent.py

### Goal
Wire tools together into a controllable state machine that can pause for humans.

### Core flow
1. Load/initialize current spec via SpecStore
2. If no spec exists → run **SpecComposer**
3. Run **SpecCritic**
4. If `passed=False` → create/return a “needs human answers” state containing `targeted_questions`
5. When answers arrive → run **SpecRefiner** → then back to **SpecCritic**
6. Optional **SpecEditor** pass before lock (toggleable)
7. When accepted → run **SpecLocker** and return locked spec

### Stop conditions
- Locked spec produced (`status="LOCKED"`)
- Human blocked the run (outside this module’s scope)
- Tool failure (invalid YAML/JSON): store raw output and return failure state

---

## Glossary

- **Spec**: The structured YAML-backed requirements document.
- **Composer**: Generates the first draft spec.
- **Critic**: Reviews spec and asks targeted questions.
- **Refiner**: Applies human answers + resolves critique into a stronger draft.
- **Editor**: Optional clarity/wording pass.
- **Locker**: Marks the spec final by setting `status="LOCKED"`.
- **SpecStore**: Persistence for spec artifacts and round history.
- **Spec Agent**: The orchestrator that runs the loop and pauses for humans.
