---

## How MontyMate loads resources

At runtime, MontyMate resolves a complete “run configuration” by combining:

1) **Workflow**: the graph of steps, edges, guards, gates  
2) **Step-group catalog**: the allowlisted set of deterministic “optional groups” a policy can enable  
3) **Profile**: binds tool aliases + prompt template IDs + default policy  
4) **Policy**: turns `module_spec` into `decision_record` (risk, gates, gate modes, patch limits, model policy, storage policy, ChangeSets, provenance)  
5) **LLM Routing Config**: chooses models/providers per role (with primary + fallbacks + tags)  
6) **Tool Registry**: defines action tools, runtime boundary (orchestrator vs runtime), and costability  
7) **Schemas**: validate spec/record/guard structures

### Resource access (installed package)
When MontyMate is installed as a Python package, **load these files using `importlib.resources`** (not filesystem-relative paths). This is the standard library mechanism designed for reading resources from packages (including wheels/zip installs).  [oai_citation:0‡Python documentation](https://docs.python.org/3/library/importlib.resources.html?utm_source=chatgpt.com)

### Shipping these YAML files
If you ship defaults inside the Python distribution, ensure they are included as **package data** (setuptools documents multiple supported approaches, including `pyproject.toml` configuration).  [oai_citation:1‡Setuptools](https://setuptools.pypa.io/en/latest/userguide/datafiles.html?utm_source=chatgpt.com)

---

## Conventions used in this repo

### Versioning
Every resource file is versioned:
- workflows: `workflow.version`
- policies: `policy.version`
- profiles: `profile.version`
- registries/configs/schemas/catalogs: top-level `version`

This supports backward compatibility and clean upgrades.

### IDs and selectors
Resources are referenced as `id@version`, for example:
- `python_unified@2`
- `default_fastapi_policy@2`
- `fastapi_service@2`

### Tool aliasing
Workflows reference tool *aliases* (e.g., `repo_analyze`) instead of hardcoding concrete tools.  
Profiles bind aliases to concrete tools from the registry (e.g., `repo_analyze_fastapi`). This prevents workflow duplication across project types.

### Deterministic dynamic expansion (step-groups)
Policy may set:
- `decision_record.workflow_overrides.add_groups: [...]`

Runner must validate:
- every requested group exists in `step_groups_catalog_v1.yaml`
- optionally: group is allowed by the current profile

This gives flexibility **without** turning execution into arbitrary branching.

### Gate modes
MontyMate supports gate modes to reduce “gate fatigue” while keeping engineers in control:
- **AUTO**: recorded, non-blocking
- **ACK**: proceeds unless blocked within an approval window
- **APPROVE**: hard stop until human approves

Gate modes are set by policy under `decision_record.gate_modes.*`.

### Storage policy (DB vs artifacts)
MontyMate supports a policy-driven storage strategy:
- `inline`: store payloads in SQLite
- `artifact_ref`: store payloads as files; DB stores pointers/hashes
- `hybrid`: inline small payloads; artifact refs for larger payloads

SQLite remains query-friendly for structured metadata (JSON1 is available for JSON functions).  [oai_citation:2‡SQLite](https://sqlite.org/json1.html?utm_source=chatgpt.com)

---

# `resources/workflows/`

Workflows define the deterministic state machine (graph) for a run.

**Files**
- `python_unified_v2.yaml` — unified workflow with:
  - **Spec Validator** stage + loop-back to interview
  - gate modes (AUTO/ACK/APPROVE) on spec/arch/audit + checkpoints
  - ChangeSets (multi-patch series + checkpoints)
  - provenance manifest generation + commit
- `step_groups_catalog_v1.yaml` — allowlisted step-groups for deterministic expansion

**What workflows contain**
- `workflow`: id, version, entry step
- `artifact_types`: known artifact types produced by the workflow
- `steps`: agent/tool/human gate steps
- `edges`: transitions, including loops and guarded branches

**Key concepts**
- Agent steps are role-driven (interviewer, spec_validator, researcher, architect, reviewer, builder, qa)
- Tool steps call named tools from the registry
- Human gates pause/resume execution (HITL)
- Guards are structured (no string-eval)

---

# `resources/policies/`

Policies transform `module_spec` (intent + constraints) into canonical `decision_record.json`.

**Files**
- `default_fastapi_policy_v2.yaml`
- `default_python_cli_policy_v2.yaml`
- `default_python_library_policy_v2.yaml`

**What a policy produces**
`decision_record.json` contains:
- `change_class` (`new_module`, `modify_endpoint`, …)
- `risk_level` (`low|medium|high`)
- `requires.*` booleans (which steps must run)
- `gates.*` map (pytest/ruff/mypy/security scan…)
- `gate_modes.*` (AUTO/ACK/APPROVE + ack window)
- `patch_limits` (max files/LOC + allow/deny paths)
- `workflow_overrides` (step-groups to enable)
- `storage_policy` (inline vs artifact refs)
- `model_policy` (role SLAs + allowed tags by risk + spec validator behavior)
- `changeset_policy` (single patch vs ChangeSet + checkpoints)
- `provenance_policy` (commit manifest to `.montymate/provenance/`)

**Policy structure**
- `defaults`: baseline record
- `classify_change`: `module_spec.intent.kind` → `change_class`
- `risk_signals`: deterministic risk scoring rules
- `risk_levels`: thresholds to map score → low/medium/high
- `rules`: overrides based on change class + risk level

**Policy design goal**
Keep workflows generic. Put “what is required and how strict to be” in policy.

---

# `resources/profiles/`

Profiles select “how MontyMate behaves for this project type” without duplicating workflows.

**Files**
- `fastapi_service_v2.yaml`
- `python_cli_v2.yaml`
- `python_library_v2.yaml`

**What a profile does**
A profile binds:
- workflow (which graph to run)
- policy defaults (and optional overrides)
- tool bindings (alias → concrete tool)
- prompt template IDs per role (templates live in SQLite; profile references them)

**Why profiles exist**
FastAPI vs CLI vs library usually doesn’t need a new workflow—just different repo analyzers, integration-guide tools, and prompt tone/structure.

---

# `resources/tools/`

The tool registry is the catalog of all tools MontyMate can invoke.

**File**
- `tool_registry_v2.yaml`

**What it contains**
Per tool:
- `tool_type` (for classification + pricing lookup)
- `runs_in_runtime` (orchestrator vs runtime boundary)
- `costable` + `cost_unit` (for cost ledger)
- description/metadata

**Why this matters**
This is the “action execution API surface” for the runner. The workflow stays stable while tools evolve behind aliases.

---

# `resources/configs/`

Configs are shipped defaults that aren’t “workflow” or “policy.”

**File**
- `llm_routing_v2.yaml`

**What it does**
Defines:
- per-role primary + fallback models
- a model catalog with stable tags (quality/cost/latency)
- routing selection rules (policy enforcement + provenance recording)

**Important**
This file must **not** contain API keys. Keys should come from environment variables or a secret manager.

---

# `resources/schemas/`

Schemas define the contracts that keep the system deterministic and guard-friendly.

**Files**
- `decision_record_v2.yaml` — canonical schema for the guard driver
- `module_spec_v1.yaml` — schema for the interviewer’s `spec.yaml` output (if present in repo)
- `guard_object_v1.yaml` — supported structured guard operators and reference syntax (if present in repo)

**Why schemas matter**
Schemas prevent drift:
- validate step outputs early
- keep guard references predictable
- prevent “mystery fields” from creeping into policies/workflows

---

## Override strategy (recommended)

MontyMate ships these defaults, but users should be able to override without forking.

Recommended precedence order:
1) explicit CLI flags (e.g., `--profile python_cli@2`)
2) user override folder (e.g., `~/.config/montymate/resources/...`)
3) repo override folder (optional): `.montymate/resources/...`
4) shipped package defaults (`resources/...`)

---

## What NOT to put in `resources/`

Avoid putting:
- API keys, tokens, credentials
- environment endpoints that differ between dev/staging/prod
- customer data or logs

Keep shipped defaults clean and portable; load secrets from the environment or secret management tooling.
