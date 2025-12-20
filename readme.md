# MontyMate 🐍🤝

MontyMate is a playful sidekick for software engineers: it helps you build and modify code with LLMs **without losing intent, control, or receipts**.

Instead of “prompt → code dump,” MontyMate runs a **spec-first, evidence-first** workflow:

**Interview → Spec Lock → Policy Decision → (Optional) Repo Analysis → (Optional) Research → Design → Audit → Patch-based Build → Verification → Reports + Cost Ledger**

If you’ve ever asked an agent for something simple (like a calculator) and gotten code immediately—with no questions, no explicit plan, and no recorded verification—MontyMate is built to do the opposite: it **slows down early** (to clarify intent) so you can **move fast later** (with confidence).

---

## Why MontyMate exists

LLM coding is powerful, but common pain points are familiar:

- The model guesses your intent and starts coding immediately.
- Changes are too large to review comfortably.
- Tests/linting are skipped or not captured as evidence.
- You can’t answer: **what changed, why, and how much did it cost?**

MontyMate is designed to work **alongside** a software engineer. It’s not a “vibes-based coder”—it’s a workflow system that enforces gates, captures decisions, and keeps everything auditable.

---

## What MontyMate produces

MontyMate generates both **human-readable** artifacts and a **queryable** run database.

### Files (human-readable artifacts)



### SQLite (machine-readable, queryable)

One SQLite DB per repo stores:

- runs, steps, attempts (retries)
- append-only event stream (timeline)
- artifact index (paths + hashes)
- **full prompts + full responses** (versioned)
- LLM usage + **cost per call** (computed at call time)
- tool-call usage + fees (e.g., web search)
- unified cost ledger (cost per step/role/phase)

---

## How MontyMate works

MontyMate splits “build software” into roles with strict contracts:

- **Interviewer**: asks clarifying questions (no code) → drafts `spec.yaml`
- **Policy Engine**: outputs `decision_record.json` (risk, gates, limits, approvals, budgets)
- **Repo Analyzer** (optional): maps current code → `analysis_report.md`
- **Researcher** (optional): evaluates packages → `research_report.md`
- **Architect**: proposes structure & interfaces → `architecture_plan.md`
- **Reviewer**: critique-only review (no code) → `audit_report.md`
- **Builder**: produces unified diffs only → `patches/*.diff`
- **QA**: runs gates and records outputs → `verification/*`

### Multi-LLM by design
Each role can use a different model/provider. A separate **reviewer model** (or even two) is a core feature—not an afterthought.

---

## Deterministic workflow graph + policy-driven decisions

### Workflow DSL (YAML → graph)
MontyMate runs workflows described as a graph:

- conditional branches
- loops (audit revisions, test-fix cycles)
- human gates (pause/resume)
- structured guards (no string-eval)

### Decision Record (the guard driver)
Instead of stuffing logic into workflow YAML, MontyMate uses a single canonical JSON output:

`decision_record.json` drives:
- change class (`new_module`, `modify_endpoint`, …)
- risk level (`low|medium|high`)
- required steps (repo analysis? research? failing test first?)
- verification gates (pytest/ruff/mypy/security scan…)
- patch limits (LOC/files, allow/deny paths)
- approvals (deps, scope expansion, high-risk changes)
- budgets (warn/stop thresholds)

This keeps guards clean and workflows reusable.

---

## Borrowed platform ideas (and why)

MontyMate borrows a few battle-tested ideas from the agent ecosystem and then adds strong governance on top:

### Sandbox/runtime separation + action execution API (OpenHands-inspired)
Actions like “run tests,” “execute commands,” and “apply patches” should happen in an isolated runtime. MontyMate follows the sandboxed runtime pattern and exposes execution via an API boundary.

### Event-stream-first logging / “trajectory”
MontyMate treats the event stream as the source of truth: every action, observation, patch, and verification run is an append-only event.

### SDK boundary (Python + REST)
MontyMate is built to be embedded:
- Python SDK (library usage)
- REST API (CI, remote runners, internal tooling)

### Conditional graphs + HITL pause/resume (LangGraph-inspired)
Human approvals (spec lock, architecture lock, audit acceptance, dependency approvals) are first-class and resumable.

---

## Profiles: support FastAPI, CLI, libraries without duplicating workflows

MontyMate uses **one unified workflow graph** and a **profile system** that binds:

- tool aliases (e.g., `repo_analyze` → `repo_analyze_fastapi`)
- prompt templates per role (FastAPI vs CLI vs library tone/structure)
- policy defaults/overrides (gates, limits, budgets)

This keeps the workflow stable while adapting behavior per project type.

---

## Feature comparison

✅ built-in / first-class  
◐ partial / depends on configuration / not a primary focus  
❌ not a focus

| Capability | MontyMate | OpenHands | SWE-agent | Aider | LangGraph |
|---|---:|---:|---:|---:|---:|
| Spec-first (interview → locked spec before code) | ✅ | ◐ | ◐ | ❌ | ❌ |
| Deterministic workflow graph (YAML → guarded transitions) | ✅ | ◐ | ◐ | ❌ | ✅ |
| Human-in-the-loop pause/resume (approval gates) | ✅ | ◐ | ◐ | ◐ | ✅ |
| Sandbox/runtime separation + execution API | ✅ | ✅ | ◐ | ❌ | ❌ |
| Event-stream-first run logging (“trajectory”) | ✅ | ✅ | ◐ | ◐ | ◐ |
| Patch-centric (unified diffs as the unit of work) | ✅ | ◐ | ✅ | ✅ | ❌ |
| Verification gates as first-class outputs (logs + reports) | ✅ | ◐ | ✅ | ◐ | ❌ |
| Multi-LLM per role (interviewer/reviewer/builder/QA) | ✅ | ◐ | ◐ | ◐ | ❌ |
| Prompts + responses stored in DB (versioned, queryable) | ✅ | ◐ | ◐ | ◐ | ❌ |
| Cost ledger (LLM + tool fees) per step/role/phase | ✅ | ◐ | ◐ | ◐ | ❌ |

---

## “What happens when I ask for a feature?”

| Phase | MontyMate |
|---|---|
| Requirements | Mandatory interview → `spec.yaml` (locked) |
| Constraints | Captured and enforced via `decision_record.json` |
| Planning | Architecture plan + independent audit |
| Changes | Unified diffs (validated, size-limited) |
| Verification | Gates must run; results + logs stored |
| Reporting | Analysis, research, audit, integration guide, run summary |
| Accountability | SQLite event timeline + prompts/responses + cost ledger |

---

## Roadmap ideas

- UI: timeline, diff provenance, gate results, cost dashboards
- More policy packs: security fix, perf work, refactor-only invariance
- Deeper repo intelligence: symbol-level blast radius, coverage deltas
- Pluggable tools: scanners, benchmark runners, package analyzers
- Export/import runs (for team reviews and CI artifacts)

---

## References (project inspirations / ecosystem)

- Python FAQ (name origin): https://docs.python.org/3/faq/general.html
- OpenHands Runtime Architecture: https://docs.openhands.dev/openhands/usage/architecture/runtime
- OpenHands paper / Action Execution API (ArXiv/OpenReview):
  - https://arxiv.org/html/2407.16741v3
  - https://openreview.net/pdf/95990590797cff8b93c33af989ecf4ac58bde9bb.pdf
- LangGraph durable execution + interrupts:
  - https://docs.langchain.com/oss/python/langgraph/durable-execution
  - https://docs.langchain.com/oss/python/langgraph/interrupts
- Aider (git-centric diffs/commit workflow): https://aider.chat/
