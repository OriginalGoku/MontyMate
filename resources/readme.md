---

## How MontyMate loads resources

At runtime, MontyMate resolves a complete “run configuration” by combining:

1. **Workflow**: the graph of steps, edges, guards, and human gates  
2. **Profile**: binds tool aliases + prompt templates + default policy  
3. **Policy**: turns `module_spec` into `decision_record` (risk, gates, patch limits, approvals, budgets)  
4. **LLM Routing Config**: chooses models/providers per role (interviewer/reviewer/builder/QA, etc.)  
5. **Tool Registry**: defines what tools exist, where they execute (runtime vs orchestrator), and whether they incur tool fees  
6. **Schemas**: validate that specs/records/guards are structurally correct

### Resource access (installed package)
When MontyMate is installed as a Python package, resources should be loaded via `importlib.resources` rather than relying on filesystem-relative paths. This is the recommended stdlib approach for reading “package resources.” 

### Shipping these YAML files
If you ship these defaults inside the Python distribution, you must ensure they’re included as package data (setuptools supports multiple ways of including data files). 

---

## Conventions used in this repo

### Versioning
Every YAML resource file includes a `version` field:
- workflows: `workflow.version`
- policies: `policy.version`
- profiles: `profile.version`
- registries/configs/schemas: top-level `version`

These versions allow MontyMate to support multiple installed defaults and keep backward compatibility.

### IDs and selectors
Resources are referenced using an ID + version selector convention, e.g.:
- `python_unified@1`
- `default_fastapi_policy@1`
- `fastapi_service@1`

### Tool aliasing
Workflows reference tool *aliases* (e.g., `repo_analyze`) instead of hardcoding concrete implementations. Profiles bind aliases to concrete tools (e.g., `repo_analyze_fastapi`). This prevents workflow duplication across project types.

---

# resources/workflows/

Workflows describe the deterministic state machine (graph) that runs a job.

**File(s):**
- `python_unified_v1.yaml`

### What it contains
- `workflow`: id, version, entry step
- `artifact_types`: known artifact types used by the workflow
- `steps`: each step definition (agent/tool/human gate)
- `edges`: transitions, including guards and loops

### Key concepts
- **Agent steps** are role-driven (interviewer, researcher, architect, reviewer, builder, QA).
- **Tool steps** call named tools from the registry.
- **Human gates** pause execution and require user input/approval to proceed.
- **Structured guards** are used everywhere (no string-eval).

### Why this matters
Workflows are your “process contract.” The workflow graph should stay stable, while **profiles and policies** tune behavior.

---

# resources/policies/

Policies transform `module_spec` (intent + constraints) into a canonical `decision_record`.

**Files:**
- `default_fastapi_policy_v1.yaml`
- `default_python_cli_policy_v1.yaml`
- `default_python_library_policy_v1.yaml`

### What a policy produces
A `decision_record.json` containing:
- `change_class` (`new_module`, `modify_endpoint`, etc.)
- `risk_level` (`low|medium|high`)
- `requires.*` booleans (which steps must run)
- `gates.*` map (pytest/ruff/mypy/security scan requirements)
- `patch_limits` (max files/LOC + allow/deny paths)
- `approvals` (dependency/scope/high-risk approvals)
- `cost_policy` (warn/stop thresholds, optional)

### Policy structure
- `defaults`: baseline decision record fields
- `classify_change`: maps `module_spec.intent.kind` → `change_class`
- `risk_signals`: deterministic risk scoring rules
- `risk_levels`: thresholds to map score → low/medium/high
- `rules`: overrides based on change class + risk level
- optional approval triggers (e.g., dependency additions require human approval)

### Policy design goal
Keep workflows generic. Put “what is required” and “how strict should we be” in policy.

---

# resources/profiles/

Profiles select “how MontyMate behaves for this kind of project” without duplicating workflows.

**Files:**
- `fastapi_service_v1.yaml`
- `python_cli_v1.yaml`
- `python_library_v1.yaml`

### What a profile does
A profile binds:
- **workflow**: which workflow graph to run
- **policy**: which policy to use (and optional overrides)
- **tool bindings**: alias → concrete tool implementation
- **prompt template IDs**: role → prompt template name/version (stored in SQLite, referenced by ID)

### Example bindings
- `repo_analyze` → `repo_analyze_fastapi`
- `generate_integration_guide` → `generate_integration_guide_cli`

### Why profiles exist
Framework differences (FastAPI vs CLI vs library) usually don’t require new workflows—just different repo analysis tools, docs generators, prompt templates, and policy defaults.

---

# resources/tools/

The tool registry is the catalog of all tools MontyMate can invoke.

**File:**
- `tool_registry_v1.yaml`

### What it contains
For each tool:
- `tool_type`: used for classification + pricing lookup
- `runs_in_runtime`: whether the tool must run inside the sandbox/runtime environment
- `costable`: whether tool usage should be recorded in cost ledger
- optional `cost_unit`: how pricing is measured (request/command/etc.)
- description and metadata

### Runtime boundary
A central MontyMate design is separating:
- **orchestrator** (workflow engine, DB logging, policy evaluation)
- **runtime/sandbox** (commands, tests, git apply, repo inspection)

This file makes that boundary explicit per tool.

---

# resources/configs/

Configs are shipped defaults that aren’t “workflow” or “policy.”

**File:**
- `llm_routing_v1.yaml`

### What it does
Defines default model/provider selection per role:
- interviewer / researcher / architect / reviewer / builder / QA
- includes primary + fallback chains

### Important: “routing” is not secrets
This file should not contain API keys. Keys should come from environment variables or your secrets manager. 

---

# resources/schemas/

Schemas define the contracts that keep the system deterministic and guard-friendly.

**Files:**
- `module_spec_v1.yaml`: what the interviewer must output in `spec.yaml`
- `decision_record_v1.yaml`: structure of the decision record produced by policy
- `guard_object_v1.yaml`: supported structured guard operators and reference syntax

### Why schemas matter
Schemas let MontyMate:
- validate step outputs (catch drift early)
- keep guard references clean and predictable
- prevent “mystery fields” from spreading into policies and workflows

---

## How to extend MontyMate (common tasks)

### Add a new project type (e.g., “data_pipeline”)
1. Add a new profile: `profiles/data_pipeline_v1.yaml`
2. Add/update a policy: `policies/default_data_pipeline_policy_v1.yaml`
3. Add tool implementations in registry (repo analysis, integration guide)
4. Add prompt templates in SQLite and reference IDs in the profile

No new workflow needed unless the lifecycle truly differs.

### Add a new gate (e.g., `bandit`)
1. Extend the policy `defaults.gates` map to include `bandit`
2. Ensure `run_gates` tool supports executing it and recording output

### Add a new change_class
1. Add it to the policy’s `classify_change`
2. Add rules for requires/gates/limits
3. Make sure module_spec schema allows intent kind if you want it validated

---

## Packaging notes (shipping these defaults)

If you intend to ship these YAML files with MontyMate:
- include them as **package data** via your build system (setuptools supports several approaches) 
- read them using `importlib.resources` at runtime (stdlib-supported, works across installation formats) 

---

## Override strategy (recommended)

MontyMate ships these defaults, but users should be able to override without forking the repo.

Recommended precedence order:
1. explicit CLI flags (e.g., `--profile python_cli@1`)
2. user override folder (e.g., `~/.config/montymate/resources/...`)
3. repo override folder (optional): `.montymate/resources/...`
4. shipped package defaults (`resources/...`)

This keeps defaults stable while supporting customization.

---

## What NOT to put in `resources/`

Avoid putting:
- API keys, tokens, credentials
- environment endpoints that differ between dev/staging/prod
- customer data or logs

Those belong in environment configuration (or secret management), not shipped defaults.
