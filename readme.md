# MontyMate 🐍🤝

MontyMate is a playful sidekick for software engineers: it helps you build and modify code with LLMs **without losing intent, control, or receipts**.

Instead of “prompt → code dump,” MontyMate runs a **spec-first, evidence-first** workflow:

**Interview → Spec Validator → Spec Lock → Policy Decision → (Optional) Repo Analysis → (Optional) Research → Design → Audit → ChangeSet Build (diffs) → Verification Gates → Reports + Provenance**

If you’ve ever asked an agent for something simple (like a calculator) and gotten code immediately—with no questions, no explicit plan, and no recorded verification—MontyMate is built to do the opposite: it **slows down early** (to clarify intent) so you can **move fast later** (with confidence).

---

## Why MontyMate exists

LLM coding is powerful, but common pain points are familiar:

- The model guesses your intent and starts coding immediately.
- Changes are too large to review comfortably.
- Tests/linting are skipped or not captured as evidence.
- You can’t answer: **what changed, why, and how much did it cost?**

MontyMate is designed to work **alongside** a software engineer. It’s not a vibes-based coder—it’s a workflow system that enforces gates, captures decisions, and keeps everything auditable.

---

## What MontyMate produces

MontyMate generates both **human-readable artifacts** and a **queryable run database**.

### Files (human-readable artifacts)

**Tracked (committed)**
- `.montymate/provenance/`
  - `provenance_manifest.json` — run provenance you can trust in git history (workflow/policy/profile versions + hashes, model routing snapshot, patch list, gate results summary, produced artifacts)

**Untracked (gitignored)**
- `.ai_montymate/` — run workspace + artifacts (safe to delete; reproducible from provenance + DB)
  - `spec.yaml` (canonical) + `spec_validation_report.json`
  - `decision_record.json` (canonical)
  - `analysis_report.md` (optional)
  - `research_report.md` (optional)
  - `architecture_plan.md`
  - `audit_report.md`
  - `integration_guide.md`
  - `patches/*.diff` (unified diffs)
  - `patch_metadata.json` (includes ChangeSet state)
  - `gates/gate_report.json` + `gates/gate_log.txt`
  - `checkpoints/checkpoint_report.md` (ChangeSet checkpoints)
  - `prompt_bundle.json` (export of prompts/responses used in the run)
  - `run_summary.md`

### SQLite (machine-readable, queryable)

One SQLite DB per repo stores:

- runs, steps, attempts (retries)
- append-only event stream (timeline)
- artifact index (paths + hashes)
- prompts + responses (versioned, queryable)
- LLM usage + **cost per call** (computed at call time)
- tool-call usage + fees (e.g., web search / runtime commands)
- unified cost ledger (cost per step/role/phase)

**Storage policy (new):**
MontyMate supports a `storage_policy` in the Decision Record:
- `inline` — store payloads in SQLite
- `artifact_ref` — store payloads as files; DB stores pointers + hashes
- `hybrid` (default) — small payloads inline; large payloads as artifacts

---

## How MontyMate works

MontyMate splits “build software” into roles with strict contracts:

- **Interviewer**: asks clarifying questions (no code) → drafts `spec.yaml`
- **Spec Validator (cheap/fast)**: checks completeness + contradictions → emits targeted follow-up questions  
  - if spec fails validation: MontyMate loops back to Interview automatically (up to a limit)  
- **Policy Engine**: outputs `decision_record.json` (risk, gates, limits, approvals, storage policy, model policy, ChangeSet policy)
- **Repo Analyzer** (optional): maps current code → `analysis_report.md`
- **Researcher** (optional): evaluates packages → `research_report.md`
- **Architect**: proposes structure & interfaces → `architecture_plan.md`
- **Reviewer**: critique-only review (no code) → `audit_report.md`
- **Builder**: produces unified diffs only → `patches/*.diff`
- **QA**: runs verification gates and records outputs → `gates/*`

### Multi-LLM by design (now policy-governed)
Each role can use a different model/provider. MontyMate adds **model governance** on top:

- role SLAs (e.g., reviewer must be “high” quality on high-risk runs)
- allowed model tags by risk level (`low|medium|high`)
- routing with primary + fallbacks per role (`llm_routing.yaml`)

---

## Deterministic workflow graph + policy-driven decisions

### Workflow DSL (YAML → graph)
MontyMate runs workflows described as a graph:

- conditional branches
- loops (spec-validation loop, audit revisions, test-fix cycles)
- human gates (pause/resume)
- structured guards (no string-eval)

### Gate Modes (new): fewer interrupts, same control
Not every gate must be a hard stop. MontyMate supports:

- **AUTO**: recorded, non-blocking
- **ACK**: proceeds unless someone blocks within an approval window
- **APPROVE**: hard stop until approved

Policy sets gate modes per run via `decision_record.gate_modes`.

### Decision Record (the guard driver)
Instead of stuffing logic into workflow YAML, MontyMate uses a single canonical JSON output:

`decision_record.json` drives:
- change class (`new_module`, `modify_endpoint`, …)
- risk level (`low|medium|high`)
- required steps (repo analysis? research? failing test first? spec validation?)
- verification gates (pytest/ruff/mypy/security scan…)
- patch limits (LOC/files, allow/deny paths)
- approvals (deps, scope expansion, high-risk changes)
- **gate modes** (AUTO/ACK/APPROVE)
- **workflow overrides** (deterministic step-group activation)
- **storage policy** (inline vs artifact refs)
- **model policy** (role SLAs + allowed tags by risk)
- **ChangeSet policy** (multi-patch series + checkpoint cadence)
- **provenance policy** (commit provenance manifest to repo)

This keeps guards clean and workflows reusable.

---

## Deterministic dynamic expansion (new)

Real work needs flexibility—but MontyMate stays reproducible.

MontyMate supports **deterministic dynamic expansion** using a fixed catalog of step-groups:

- Policy outputs: `decision_record.workflow_overrides.add_groups: [...]`
- Runner validates groups against `resources/workflows/step_groups_catalog_v1.yaml`
- Workflow conditionally runs group steps (or runner injects them deterministically)

No arbitrary branching, no mystery behavior.

---

## ChangeSets (new): production-friendly refactors without mega-diffs

Instead of forcing everything into one patch, MontyMate supports **ChangeSets**:

- a series of small unified diffs
- checkpoint reports every N patches
- optional human checkpoint gate (mode depends on risk)

This makes large refactors reviewable *and* traceable.

---

## Provenance Manifest (new): “who/what produced this change?”
Every run writes `provenance_manifest.json` and commits it to:

- `.montymate/provenance/`

This ties code changes to:
- workflow/policy/profile versions + hashes
- model routing snapshot
- verification results summary
- patch list / commit SHAs
- key artifacts produced

Reproducibility is no longer “best effort.”

---

## Profiles: support FastAPI, CLI, libraries without duplicating workflows

MontyMate uses **one unified workflow graph** and a **profile system** that binds:

- tool aliases (e.g., `repo_analyze` → `repo_analyze_fastapi`)
- prompt templates per role (FastAPI vs CLI vs library tone/structure)
- policy defaults/overrides (gates, limits, gate modes)

This keeps the workflow stable while adapting behavior per project type.

---

## Feature comparison

✅ built-in / first-class  
◐ partial / depends on configuration / not a primary focus  
❌ not a focus

| Capability | MontyMate | OpenHands | SWE-agent | Aider | LangGraph |
|---|---:|---:|---:|---:|---:|
| Spec-first (interview → spec before code) | ✅ | ◐ | ◐ | ❌ | ❌ |
| **Spec Validator loop** (cheap completeness/consistency check) | ✅ | ◐ | ◐ | ❌ | ❌ |
| Deterministic workflow graph (YAML → guarded transitions) | ✅ | ◐ | ◐ | ❌ | ✅ |
| **Gate modes** (AUTO / ACK / APPROVE) | ✅ | ◐ | ◐ | ◐ | ◐ |
| Human-in-the-loop pause/resume (approval gates) | ✅ | ◐ | ◐ | ◐ | ✅ |
| Sandbox/runtime separation + execution API | ◐ (pluggable) | ✅ | ◐ | ❌ | ❌ |
| Event-stream-first run logging (“trajectory”) | ✅ | ✅ | ◐ | ◐ | ◐ |
| Patch-centric (unified diffs as unit of work) | ✅ | ◐ | ✅ | ✅ | ❌ |
| **ChangeSets** (multi-patch + checkpoints) | ✅ | ◐ | ◐ | ◐ | ❌ |
| Verification gates as first-class outputs (logs + reports) | ✅ | ◐ | ✅ | ◐ | ❌ |
| Multi-LLM per role (interviewer/reviewer/builder/QA) | ✅ | ◐ | ◐ | ◐ | ❌ |
| Prompts + responses stored & exportable | ✅ | ◐ | ◐ | ◐ | ❌ |
| Cost ledger (LLM + tool fees) per step/role/phase | ✅ | ◐ | ◐ | ◐ | ❌ |
| **Provenance manifest committed to repo** | ✅ | ◐ | ◐ | ◐ | ❌ |

---

## Repo layout (shipped defaults)

MontyMate ships default resources under:

- `resources/`
  - `workflows/` (graph YAML + step-group catalog)
  - `policies/` (policy YAML)
  - `profiles/` (profile YAML)
  - `tools/` (tool registry YAML)
  - `configs/` (LLM routing + model catalog YAML)
  - `schemas/` (Decision Record schema, etc.)

When installed as a Python package, MontyMate should load these as **package resources** (not relative paths), so defaults work from wheels/zips too.

---

## Roadmap ideas

- Timeline UI: events, diffs, gate results, cost dashboards
- More policy packs: security fix, perf work, refactor-only invariance
- Deeper repo intelligence: symbol-level blast radius, coverage deltas
- Pluggable tools: scanners, benchmark runners, package analyzers
- Export/import runs (for team reviews and CI artifacts)

---

## References (project inspirations / ecosystem)

- Python FAQ (name origin): https://docs.python.org/3/faq/general.html
- OpenHands runtime architecture: https://docs.openhands.dev/usage/architecture/runtime
- OpenHands (paper): https://openreview.net/forum?id=OJd3ayDDoF
- LangGraph interrupts + durable execution:
  - https://docs.langchain.com/oss/python/langgraph/interrupts
  - https://docs.langchain.com/oss/python/langgraph/durable-execution
- SQLite JSON1: https://sqlite.org/json1.html
- SWE-agent: https://github.com/SWE-agent/SWE-agent
- Aider: https://github.com/Aider-AI/aider
- LangGraph: https://github.com/langchain-ai/langgraph
