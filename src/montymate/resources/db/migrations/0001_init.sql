BEGIN;

PRAGMA application_id = 1297048389;
PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS mm_runs (
  run_id       TEXT PRIMARY KEY,
  created_at   INTEGER NOT NULL,
  updated_at   INTEGER NOT NULL,
  status       TEXT NOT NULL CHECK (status IN ('RUNNING','PAUSED','SUCCEEDED','FAILED','CANCELED')),
  profile_id   TEXT NOT NULL,
  workflow_id  TEXT NOT NULL,
  policy_id    TEXT NOT NULL,
  workflow_sha TEXT,
  policy_sha   TEXT,
  profile_sha  TEXT,

  repo_root    TEXT NOT NULL,
  git_base_sha TEXT,
  git_branch   TEXT
);

CREATE TABLE IF NOT EXISTS mm_step_executions (
  step_execution_id TEXT PRIMARY KEY,
  run_id            TEXT NOT NULL,
  step_id           TEXT NOT NULL,
  attempt_no        INTEGER NOT NULL,
  status            TEXT NOT NULL CHECK (status IN ('RUNNING','WAITING_HUMAN','SUCCEEDED','FAILED','SKIPPED')),
  started_at        INTEGER NOT NULL,
  finished_at       INTEGER,
  error_type        TEXT,
  error_message     TEXT,
  FOREIGN KEY (run_id) REFERENCES mm_runs(run_id) ON DELETE CASCADE,
  UNIQUE (run_id, step_id, attempt_no)
);

CREATE TABLE IF NOT EXISTS mm_artifacts (
  artifact_id       TEXT PRIMARY KEY,
  run_id            TEXT NOT NULL,
  step_execution_id TEXT,
  artifact_type     TEXT NOT NULL,
  format            TEXT NOT NULL CHECK (format IN ('YAML','JSON','MD','DIFF','TEXT','BIN')),
  relpath           TEXT NOT NULL,
  sha256            TEXT NOT NULL,
  bytes             INTEGER NOT NULL,
  created_at        INTEGER NOT NULL,
  meta_json         TEXT,
  parent_artifact_id TEXT,
  FOREIGN KEY (run_id) REFERENCES mm_runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (step_execution_id) REFERENCES mm_step_executions(step_execution_id) ON DELETE SET NULL,
  FOREIGN KEY (parent_artifact_id) REFERENCES mm_artifacts(artifact_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run_type_time ON mm_artifacts(run_id, artifact_type, created_at);

CREATE TABLE IF NOT EXISTS mm_events (
  event_id          INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id            TEXT NOT NULL,
  step_execution_id TEXT,
  ts                INTEGER NOT NULL,
  event_type        TEXT NOT NULL,
  actor_type        TEXT NOT NULL CHECK (actor_type IN ('system','human','llm','tool')),
  severity          TEXT NOT NULL CHECK (severity IN ('debug','info','warn','error')),

  -- Keep payload_json “small”; if large, write payload to an artifact and set payload_artifact_id.
  payload_json       TEXT NOT NULL,
  payload_sha256     TEXT,
  payload_artifact_id TEXT,

  FOREIGN KEY (run_id) REFERENCES mm_runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (step_execution_id) REFERENCES mm_step_executions(step_execution_id) ON DELETE SET NULL,
  FOREIGN KEY (payload_artifact_id) REFERENCES mm_artifacts(artifact_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_ts ON mm_events(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_run_type ON mm_events(run_id, event_type);

CREATE TABLE IF NOT EXISTS mm_prompt_templates (
  template_id  TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  role         TEXT NOT NULL,
  version      INTEGER NOT NULL,
  content      TEXT NOT NULL,
  sha256       TEXT NOT NULL,
  created_at   INTEGER NOT NULL,
  UNIQUE(name, role, version)
);

CREATE TABLE IF NOT EXISTS mm_llm_pricing (
  pricing_id     TEXT PRIMARY KEY,
  provider       TEXT NOT NULL,
  model          TEXT NOT NULL,
  currency       TEXT NOT NULL,
  input_per_1k   REAL NOT NULL,
  output_per_1k  REAL NOT NULL,
  effective_from INTEGER NOT NULL,
  effective_to   INTEGER,
  meta_json      TEXT
);

CREATE TABLE IF NOT EXISTS mm_llm_calls (
  llm_call_id       TEXT PRIMARY KEY,
  run_id            TEXT NOT NULL,
  step_execution_id TEXT,
  role              TEXT NOT NULL,
  provider          TEXT NOT NULL,
  model             TEXT NOT NULL,
  request_ts        INTEGER NOT NULL,
  response_ts       INTEGER,
  latency_ms        INTEGER,
  status            TEXT NOT NULL CHECK (status IN ('OK','ERROR','TIMEOUT')),
  template_id       TEXT,

  -- Hybrid storage:
  prompt_text       TEXT,
  response_text     TEXT,
  prompt_artifact_id   TEXT,
  response_artifact_id TEXT,

  request_json      TEXT,
  response_json     TEXT,
  input_tokens      INTEGER,
  output_tokens     INTEGER,
  total_tokens      INTEGER,
  pricing_id        TEXT,
  cost_usd          REAL,

  FOREIGN KEY (run_id) REFERENCES mm_runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (step_execution_id) REFERENCES mm_step_executions(step_execution_id) ON DELETE SET NULL,
  FOREIGN KEY (template_id) REFERENCES mm_prompt_templates(template_id) ON DELETE SET NULL,
  FOREIGN KEY (pricing_id) REFERENCES mm_llm_pricing(pricing_id) ON DELETE SET NULL,
  FOREIGN KEY (prompt_artifact_id) REFERENCES mm_artifacts(artifact_id) ON DELETE SET NULL,
  FOREIGN KEY (response_artifact_id) REFERENCES mm_artifacts(artifact_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON mm_llm_calls(run_id, request_ts);

CREATE TABLE IF NOT EXISTS mm_tool_pricing (
  pricing_id     TEXT PRIMARY KEY,
  tool_name      TEXT,
  tool_type      TEXT,
  currency       TEXT NOT NULL,
  unit_type      TEXT NOT NULL,
  per_unit       REAL NOT NULL,
  effective_from INTEGER NOT NULL,
  effective_to   INTEGER,
  meta_json      TEXT
);

CREATE TABLE IF NOT EXISTS mm_tool_calls (
  tool_call_id      TEXT PRIMARY KEY,
  run_id            TEXT NOT NULL,
  step_execution_id TEXT,
  tool_name         TEXT NOT NULL,
  tool_type         TEXT NOT NULL,
  runs_in_runtime   INTEGER NOT NULL CHECK (runs_in_runtime IN (0,1)),
  request_ts        INTEGER NOT NULL,
  response_ts       INTEGER,
  latency_ms        INTEGER,
  status            TEXT NOT NULL CHECK (status IN ('OK','ERROR','TIMEOUT')),

  input_json        TEXT,
  output_json       TEXT,
  input_artifact_id  TEXT,
  output_artifact_id TEXT,

  unit_type         TEXT,
  units             REAL,
  pricing_id        TEXT,
  cost_usd          REAL,

  FOREIGN KEY (run_id) REFERENCES mm_runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (step_execution_id) REFERENCES mm_step_executions(step_execution_id) ON DELETE SET NULL,
  FOREIGN KEY (pricing_id) REFERENCES mm_tool_pricing(pricing_id) ON DELETE SET NULL,
  FOREIGN KEY (input_artifact_id) REFERENCES mm_artifacts(artifact_id) ON DELETE SET NULL,
  FOREIGN KEY (output_artifact_id) REFERENCES mm_artifacts(artifact_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON mm_tool_calls(run_id, request_ts);

CREATE TABLE IF NOT EXISTS mm_cost_ledger (
  ledger_id         TEXT PRIMARY KEY,
  run_id            TEXT NOT NULL,
  step_execution_id TEXT,
  source_type       TEXT NOT NULL CHECK (source_type IN ('llm','tool')),
  source_id         TEXT NOT NULL,
  amount_usd        REAL NOT NULL,
  currency          TEXT NOT NULL,
  unit_type         TEXT,
  units             REAL,
  pricing_id        TEXT,
  computed_at       INTEGER NOT NULL,
  note              TEXT,
  FOREIGN KEY (run_id) REFERENCES mm_runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (step_execution_id) REFERENCES mm_step_executions(step_execution_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_cost_run ON mm_cost_ledger(run_id);

-- Immutability / append-only
CREATE TRIGGER IF NOT EXISTS trg_events_no_update
BEFORE UPDATE ON mm_events
BEGIN SELECT RAISE(ABORT, 'mm_events is append-only'); END;

CREATE TRIGGER IF NOT EXISTS trg_events_no_delete
BEFORE DELETE ON mm_events
BEGIN SELECT RAISE(ABORT, 'mm_events is append-only'); END;

CREATE TRIGGER IF NOT EXISTS trg_artifacts_no_update
BEFORE UPDATE ON mm_artifacts
BEGIN SELECT RAISE(ABORT, 'mm_artifacts is immutable'); END;

CREATE TRIGGER IF NOT EXISTS trg_artifacts_no_delete
BEFORE DELETE ON mm_artifacts
BEGIN SELECT RAISE(ABORT, 'mm_artifacts is immutable'); END;

CREATE TRIGGER IF NOT EXISTS trg_cost_no_update
BEFORE UPDATE ON mm_cost_ledger
BEGIN SELECT RAISE(ABORT, 'mm_cost_ledger is immutable'); END;

CREATE TRIGGER IF NOT EXISTS trg_cost_no_delete
BEFORE DELETE ON mm_cost_ledger
BEGIN SELECT RAISE(ABORT, 'mm_cost_ledger is immutable'); END;

COMMIT;
