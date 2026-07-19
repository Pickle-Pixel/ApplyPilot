-- ===========================================================================
-- Distributed residential fleet v3 schema (Postgres).
--
-- Layered ON TOP of apply/fleet_schema.sql (which creates apply_queue,
-- fleet_config, fleet_assets). Run fleet_schema.sql FIRST, then this file.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS, CREATE ... IF NOT EXISTS, enum + value
-- guards. Safe to run on every broker/home startup. In Topology A these are
-- coordination tables; in Topology B they are brain tables (identical SQL).
--
-- See docs/superpowers/specs/2026-06-26-distributed-residential-fleet-design.md
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Extend apply_queue with residential + governance columns (R1, R6, R9, R11).
-- ---------------------------------------------------------------------------
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS worker_home_ip TEXT;   -- sending residential IP
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS target_host    TEXT;   -- effective apply host (governor key)
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS lane           TEXT NOT NULL DEFAULT 'ats';
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS dedup_key      TEXT;   -- (company, normalized_role) -- R9
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS approved_batch TEXT;   -- owner approval token -- R11
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS liveness_required BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS liveness_status TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS liveness_reason TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS liveness_checked_at TIMESTAMPTZ;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS liveness_check_owner TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS liveness_check_expires_at TIMESTAMPTZ;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS liveness_check_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS liveness_consecutive_uncertain INTEGER NOT NULL DEFAULT 0;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS eligibility_required BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS eligibility_status TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS eligibility_reason TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS eligibility_checked_at TIMESTAMPTZ;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS infrastructure_failure_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS infrastructure_last_failure_at TIMESTAMPTZ;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS session_required BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS tenant_profile_id TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS routing_required BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS execution_route TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS host_policy TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS decision_id TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS policy_version TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS decision_action TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS qualification_verdict TEXT;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS qualification_score REAL;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS qualification_floor REAL;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS preference_score REAL;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS outcome_score REAL;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS final_score REAL;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS decision_confidence REAL;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS decision_created_at TIMESTAMPTZ;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS decision_expires_at TIMESTAMPTZ;
ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS input_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_apply_queue_dedup ON apply_queue (dedup_key);
CREATE INDEX IF NOT EXISTS idx_apply_queue_approved
    ON apply_queue (score DESC) WHERE status = 'queued' AND approved_batch IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_apply_queue_liveness
    ON apply_queue (liveness_checked_at, score DESC)
    WHERE status = 'queued' AND liveness_required;
CREATE INDEX IF NOT EXISTS idx_apply_queue_infrastructure_pending
    ON apply_queue (infrastructure_last_failure_at DESC)
    WHERE apply_status = 'infrastructure_pending';

-- ---------------------------------------------------------------------------
-- Extend fleet_config (single-row id=1) with v3 controls (R11, R12, R14).
-- ---------------------------------------------------------------------------
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS approval_threshold     REAL;            -- auto-approve fit floor (NULL = batch-only)
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS approval_policy        JSONB;           -- {min_fit, min_confidence, exclude_flags[]} -- R11
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS approval_sampling_rate REAL NOT NULL DEFAULT 0.0;  -- audit sample fraction
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS cost_cap_daily_usd     NUMERIC(10,2) NOT NULL DEFAULT 0;  -- 0 = no cap -- R14
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS cost_cap_total_usd     NUMERIC(10,2) NOT NULL DEFAULT 0;  -- 0 = no cap
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS pinned_worker_version  TEXT;            -- R12 fleet version
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS canary_version         TEXT;            -- R12 staged update
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS canary_worker_id       TEXT;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS last_window_roll_at    TIMESTAMPTZ;     -- nightly window roll guard
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS ats_apply_mode          TEXT NOT NULL DEFAULT 'stopped'; -- 'stopped'|'canary'|'steady'
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS canary_enabled         BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS canary_remaining        INTEGER;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS linkedin_apply_mode     TEXT NOT NULL DEFAULT 'stopped'; -- 'stopped'|'canary'|'steady'
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS linkedin_canary_enabled  BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS linkedin_canary_remaining INTEGER;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS daily_apply_target       INTEGER;        -- optional operator target; NULL = unconfigured
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS ats_policy_version TEXT;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS linkedin_policy_version TEXT;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS ats_policy_lane TEXT
    GENERATED ALWAYS AS ('ats'::text) STORED;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS linkedin_policy_lane TEXT
    GENERATED ALWAYS AS ('linkedin'::text) STORED;

UPDATE fleet_config SET ats_apply_mode='stopped'
WHERE ats_apply_mode IS NULL OR ats_apply_mode NOT IN ('stopped', 'canary', 'steady');
UPDATE fleet_config SET linkedin_apply_mode='stopped'
WHERE linkedin_apply_mode IS NULL OR linkedin_apply_mode NOT IN ('stopped', 'canary', 'steady');

-- ---------------------------------------------------------------------------
-- Status enum shared by the compute + search-task queues.
-- ---------------------------------------------------------------------------
DO $$ BEGIN
    CREATE TYPE fleet_task_status AS ENUM (
        'queued',      -- eligible to lease (search: AND next_due_at <= now())
        'leased',      -- a worker holds it
        'done',        -- terminal success (compute: result written / search: run complete)
        'failed',      -- terminal failure
        'quarantined'  -- poison: pulled from the pool (R7)
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------------------------------------------------------------------------
-- compute_queue: score / audit / tailor / enrich -- IP-free, cost-governed (§8, R14).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS compute_queue (
    url               TEXT NOT NULL,
    task              TEXT NOT NULL,                 -- 'score'|'audit'|'tailor'|'enrich'
    payload           JSONB,                         -- minimal context for the task
    status            fleet_task_status NOT NULL DEFAULT 'queued',
    lease_owner       TEXT,
    lease_expires_at  TIMESTAMPTZ,
    attempts          INTEGER NOT NULL DEFAULT 0,
    result            JSONB,                         -- advisory score/audit/tailored-resume ref
    est_cost_usd      NUMERIC(10,4) NOT NULL DEFAULT 0,
    synced_to_home_at TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (url, task)
);

DO $$
DECLARE
    pk_cols TEXT[];
BEGIN
    SELECT array_agg(a.attname ORDER BY u.ordinality)
      INTO pk_cols
    FROM pg_constraint c
    JOIN unnest(c.conkey) WITH ORDINALITY u(attnum, ordinality) ON true
    JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = u.attnum
    WHERE c.conrelid = 'compute_queue'::regclass
      AND c.contype = 'p';

    IF pk_cols = ARRAY['url'] THEN
        ALTER TABLE compute_queue DROP CONSTRAINT compute_queue_pkey;
        ALTER TABLE compute_queue ADD CONSTRAINT compute_queue_pkey PRIMARY KEY (url, task);
    ELSIF pk_cols IS NULL THEN
        ALTER TABLE compute_queue ADD CONSTRAINT compute_queue_pkey PRIMARY KEY (url, task);
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS idx_compute_lease ON compute_queue (status) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS idx_compute_reclaim ON compute_queue (lease_expires_at) WHERE status = 'leased';
CREATE INDEX IF NOT EXISTS idx_compute_unsynced ON compute_queue (updated_at)
    WHERE status IN ('done','failed') AND synced_to_home_at IS NULL;

-- discovered_postings: raw JobSpy postings staged by lean discovery workers (no local brain).
-- The home box ingests these into the shared SQLite brain via store_jobspy_results (one write path).
CREATE TABLE IF NOT EXISTS discovered_postings (
    id                BIGSERIAL PRIMARY KEY,
    task_id           TEXT,
    source_label      TEXT,
    posting           JSONB NOT NULL,
    worker_id         TEXT,
    discovered_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    synced_to_home_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_discovered_unsynced ON discovered_postings (discovered_at)
    WHERE synced_to_home_at IS NULL;

-- ---------------------------------------------------------------------------
-- search_tasks: distributed, RECURRING discovery (R3-disc / RF3).
-- One task = (query x board x location). Re-runs when next_due_at passes.
-- IP/account-sensitive: governed by board:<name> scope (RF2).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS search_tasks (
    task_id           TEXT PRIMARY KEY,              -- hash(query|board|location)
    query             TEXT NOT NULL,
    board             TEXT NOT NULL,                 -- 'indeed'|'linkedin'|'greenhouse'|... (governor key)
    location          TEXT,
    params            JSONB,                         -- results_wanted, hours_old, filters, ...
    status            fleet_task_status NOT NULL DEFAULT 'queued',
    lease_owner       TEXT,
    lease_expires_at  TIMESTAMPTZ,
    next_due_at       TIMESTAMPTZ NOT NULL DEFAULT now(),  -- claimable when <= now()
    cadence_seconds   INTEGER NOT NULL DEFAULT 21600,      -- re-run cadence (6h default)
    attempts          INTEGER NOT NULL DEFAULT 0,
    last_run_at       TIMESTAMPTZ,
    result_count      INTEGER,                        -- postings found on last run (coverage view)
    last_error        TEXT,
    enabled           BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_search_due
    ON search_tasks (board, next_due_at) WHERE status = 'queued' AND enabled;
CREATE INDEX IF NOT EXISTS idx_search_reclaim ON search_tasks (lease_expires_at) WHERE status = 'leased';

-- ---------------------------------------------------------------------------
-- linkedin_queue: ONE owner IP, N coordinated machines, governor-serialized (R1).
-- Same shape as apply_queue (separate lane).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS linkedin_queue (LIKE apply_queue INCLUDING DEFAULTS INCLUDING INDEXES);
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS worker_home_ip TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS target_host TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS lane TEXT NOT NULL DEFAULT 'linkedin';
ALTER TABLE linkedin_queue ALTER COLUMN lane SET DEFAULT 'linkedin';
UPDATE linkedin_queue SET lane='linkedin' WHERE lane IS DISTINCT FROM 'linkedin';
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS dedup_key TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS approved_batch TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS decision_id TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS policy_version TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS decision_action TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS qualification_verdict TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS qualification_score REAL;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS qualification_floor REAL;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS preference_score REAL;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS outcome_score REAL;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS final_score REAL;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS decision_confidence REAL;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS decision_created_at TIMESTAMPTZ;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS decision_expires_at TIMESTAMPTZ;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS input_hash TEXT;
-- apply-time channel recorder (zero LinkedIn scraping): how the apply actually happened.
--   apply_channel: 'easy_apply' (stayed on linkedin.com) | 'external' (redirected to an ATS)
--   apply_external_host: the ATS base host when external (e.g. 'ashbyhq.com', 'greenhouse.io')
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS apply_channel TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS apply_external_host TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS linkedin_resolve_status TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS linkedin_resolved_at TIMESTAMPTZ;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS linkedin_resolve_error TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS linkedin_unresolved_kind TEXT;
ALTER TABLE linkedin_queue ADD COLUMN IF NOT EXISTS linkedin_next_action TEXT;

DO $$
DECLARE
    queue_name TEXT;
    target_oid OID;
    schema_oid OID := (SELECT oid FROM pg_namespace WHERE nspname = current_schema());
BEGIN
    FOREACH queue_name IN ARRAY ARRAY['apply_queue', 'linkedin_queue'] LOOP
        target_oid := to_regclass(format('%I.%I', current_schema(), queue_name));

        EXECUTE format('ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I', queue_name, queue_name || '_decision_action_ck');
        EXECUTE format('ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I', queue_name, queue_name || '_qualification_verdict_ck');
        EXECUTE format('ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I', queue_name, queue_name || '_confidence_ck');
        EXECUTE format('ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I', queue_name, queue_name || '_expiry_ck');
        EXECUTE format('ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I', queue_name, queue_name || '_decision_action_check');
        EXECUTE format('ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I', queue_name, queue_name || '_qualification_verdict_check');
        EXECUTE format('ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I', queue_name, queue_name || '_decision_confidence_check');

        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = queue_name || '_canonical_provenance_ck'
              AND connamespace = schema_oid AND conrelid = target_oid
        ) THEN
            EXECUTE format($sql$
                ALTER TABLE %I ADD CONSTRAINT %I CHECK (
                    (
                        decision_id IS NULL AND policy_version IS NULL AND decision_action IS NULL
                        AND qualification_verdict IS NULL AND qualification_score IS NULL
                        AND qualification_floor IS NULL AND preference_score IS NULL
                        AND outcome_score IS NULL AND final_score IS NULL
                        AND decision_confidence IS NULL AND decision_created_at IS NULL
                        AND decision_expires_at IS NULL AND input_hash IS NULL
                    ) OR (
                        decision_id IS NOT NULL AND btrim(decision_id) <> ''
                        AND policy_version IS NOT NULL AND btrim(policy_version) <> ''
                        AND decision_action IN ('apply', 'review', 'reject')
                        AND qualification_verdict IN ('qualified', 'uncertain', 'unqualified')
                        AND qualification_score IS NOT NULL AND qualification_floor IS NOT NULL
                        AND preference_score IS NOT NULL AND outcome_score IS NOT NULL
                        AND final_score IS NOT NULL AND decision_confidence IS NOT NULL
                        AND qualification_score > '-Infinity'::REAL AND qualification_score < 'Infinity'::REAL
                        AND qualification_floor > '-Infinity'::REAL AND qualification_floor < 'Infinity'::REAL
                        AND preference_score > '-Infinity'::REAL AND preference_score < 'Infinity'::REAL
                        AND outcome_score > '-Infinity'::REAL AND outcome_score < 'Infinity'::REAL
                        AND final_score > '-Infinity'::REAL AND final_score < 'Infinity'::REAL
                        AND score = final_score
                        AND decision_confidence >= 0 AND decision_confidence <= 1
                        AND decision_created_at IS NOT NULL AND decision_expires_at IS NOT NULL
                        AND decision_expires_at > decision_created_at
                        AND input_hash IS NOT NULL AND btrim(input_hash) <> ''
                    )
                ) NOT VALID
            $sql$, queue_name, queue_name || '_canonical_provenance_ck');
            EXECUTE format('ALTER TABLE %I VALIDATE CONSTRAINT %I', queue_name, queue_name || '_canonical_provenance_ck');
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = queue_name || '_policy_lane_fk'
              AND connamespace = schema_oid AND conrelid = target_oid
        ) THEN
            EXECUTE format('ALTER TABLE %I ADD CONSTRAINT %I FOREIGN KEY (policy_version, lane) REFERENCES fleet_decision_policies(policy_version, lane) NOT VALID', queue_name, queue_name || '_policy_lane_fk');
            EXECUTE format('ALTER TABLE %I VALIDATE CONSTRAINT %I', queue_name, queue_name || '_policy_lane_fk');
        END IF;
    END LOOP;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'apply_queue_lane_ck' AND connamespace = schema_oid
          AND conrelid = to_regclass(format('%I.apply_queue', current_schema()))
    ) THEN
        ALTER TABLE apply_queue ADD CONSTRAINT apply_queue_lane_ck
            CHECK (policy_version IS NULL OR lane = 'ats') NOT VALID;
        ALTER TABLE apply_queue VALIDATE CONSTRAINT apply_queue_lane_ck;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'linkedin_queue_lane_ck' AND connamespace = schema_oid
          AND conrelid = to_regclass(format('%I.linkedin_queue', current_schema()))
    ) THEN
        ALTER TABLE linkedin_queue ADD CONSTRAINT linkedin_queue_lane_ck
            CHECK (policy_version IS NULL OR lane = 'linkedin') NOT VALID;
        ALTER TABLE linkedin_queue VALIDATE CONSTRAINT linkedin_queue_lane_ck;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fleet_config_ats_policy_fk' AND connamespace = schema_oid
          AND conrelid = to_regclass(format('%I.fleet_config', current_schema()))
    ) THEN
        ALTER TABLE fleet_config ADD CONSTRAINT fleet_config_ats_policy_fk
            FOREIGN KEY (ats_policy_version, ats_policy_lane)
            REFERENCES fleet_decision_policies(policy_version, lane) NOT VALID;
        ALTER TABLE fleet_config VALIDATE CONSTRAINT fleet_config_ats_policy_fk;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fleet_config_linkedin_policy_fk' AND connamespace = schema_oid
          AND conrelid = to_regclass(format('%I.fleet_config', current_schema()))
    ) THEN
        ALTER TABLE fleet_config ADD CONSTRAINT fleet_config_linkedin_policy_fk
            FOREIGN KEY (linkedin_policy_version, linkedin_policy_lane)
            REFERENCES fleet_decision_policies(policy_version, lane) NOT VALID;
        ALTER TABLE fleet_config VALIDATE CONSTRAINT fleet_config_linkedin_policy_fk;
    END IF;
END $$;

DO $$
DECLARE
    index_predicate TEXT;
BEGIN
    SELECT pg_get_expr(i.indpred, i.indrelid)
      INTO index_predicate
    FROM pg_index i
    JOIN pg_class idx ON idx.oid = i.indexrelid
    JOIN pg_class rel ON rel.oid = i.indrelid
    JOIN pg_namespace n ON n.oid = idx.relnamespace AND n.oid = rel.relnamespace
    WHERE n.nspname = current_schema() AND rel.relname = 'apply_queue'
      AND idx.relname = 'idx_apply_queue_canonical_lease';
    IF index_predicate IS NOT NULL
       AND NOT (
           index_predicate LIKE '%policy_version IS NOT NULL%'
           AND index_predicate LIKE '%qualification_score IS NOT NULL%'
           AND index_predicate LIKE '%decision_created_at IS NOT NULL%'
           AND index_predicate LIKE '%decision_expires_at IS NOT NULL%'
           AND index_predicate LIKE '%input_hash IS NOT NULL%'
       ) THEN
        DROP INDEX idx_apply_queue_canonical_lease;
    END IF;

    SELECT pg_get_expr(i.indpred, i.indrelid)
      INTO index_predicate
    FROM pg_index i
    JOIN pg_class idx ON idx.oid = i.indexrelid
    JOIN pg_class rel ON rel.oid = i.indrelid
    JOIN pg_namespace n ON n.oid = idx.relnamespace AND n.oid = rel.relnamespace
    WHERE n.nspname = current_schema() AND rel.relname = 'linkedin_queue'
      AND idx.relname = 'idx_linkedin_queue_canonical_lease';
    IF index_predicate IS NOT NULL
       AND NOT (
           index_predicate LIKE '%policy_version IS NOT NULL%'
           AND index_predicate LIKE '%qualification_score IS NOT NULL%'
           AND index_predicate LIKE '%decision_created_at IS NOT NULL%'
           AND index_predicate LIKE '%decision_expires_at IS NOT NULL%'
           AND index_predicate LIKE '%input_hash IS NOT NULL%'
       ) THEN
        DROP INDEX idx_linkedin_queue_canonical_lease;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_apply_queue_canonical_lease
    ON apply_queue (policy_version, decision_action, decision_expires_at, score DESC)
    WHERE status = 'queued' AND decision_id IS NOT NULL AND btrim(decision_id) <> ''
      AND policy_version IS NOT NULL AND btrim(policy_version) <> ''
      AND decision_action = 'apply' AND qualification_verdict = 'qualified'
      AND qualification_score IS NOT NULL AND qualification_floor IS NOT NULL
      AND preference_score IS NOT NULL AND outcome_score IS NOT NULL AND final_score IS NOT NULL
      AND decision_confidence IS NOT NULL AND decision_created_at IS NOT NULL
      AND decision_expires_at IS NOT NULL AND input_hash IS NOT NULL AND btrim(input_hash) <> '';
CREATE INDEX IF NOT EXISTS idx_linkedin_queue_canonical_lease
    ON linkedin_queue (policy_version, decision_action, decision_expires_at, score DESC)
    WHERE status = 'queued' AND decision_id IS NOT NULL AND btrim(decision_id) <> ''
      AND policy_version IS NOT NULL AND btrim(policy_version) <> ''
      AND decision_action = 'apply' AND qualification_verdict = 'qualified'
      AND qualification_score IS NOT NULL AND qualification_floor IS NOT NULL
      AND preference_score IS NOT NULL AND outcome_score IS NOT NULL AND final_score IS NOT NULL
      AND decision_confidence IS NOT NULL AND decision_created_at IS NOT NULL
      AND decision_expires_at IS NOT NULL AND input_hash IS NOT NULL AND btrim(input_hash) <> '';

-- ---------------------------------------------------------------------------
-- rate_governor: outcome-aware + adaptive circuit-breaker (R6, R1, RF2).
-- scope_key examples:
--   'global' | 'host:boards.greenhouse.io' | 'board:linkedin'
--   'home_ip:1.2.3.4' | 'account:linkedin' (the LinkedIn mutex)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rate_governor (
    scope_key       TEXT PRIMARY KEY,
    window_start    TIMESTAMPTZ NOT NULL DEFAULT now(),   -- rolling 24h anchor
    count_24h       INTEGER NOT NULL DEFAULT 0,           -- CAP counter (++ on confirmed terminal)
    daily_cap       INTEGER NOT NULL DEFAULT 1000000,     -- per-scope cap (large = effectively none)
    last_applied_at TIMESTAMPTZ,                          -- per-scope min-gap / mutex stamp
    min_gap_seconds INTEGER NOT NULL DEFAULT 90,
    -- outcome counters (++ on their own terminal class -- the deliberate v2/v3 change):
    success_24h     INTEGER NOT NULL DEFAULT 0,
    captcha_24h     INTEGER NOT NULL DEFAULT 0,
    block_24h       INTEGER NOT NULL DEFAULT 0,
    challenge_rate  REAL GENERATED ALWAYS AS (
        CASE WHEN (success_24h + captcha_24h + block_24h) = 0 THEN 0
             ELSE (captcha_24h + block_24h)::real
                  / (success_24h + captcha_24h + block_24h) END
    ) STORED,
    breaker_state   TEXT NOT NULL DEFAULT 'ok',           -- 'ok'|'throttled'|'paused'|'demoted'
    breaker_until   TIMESTAMPTZ,                          -- auto-recover time
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Pristine min-gap, so a 'throttled' breaker can widen the gap (base * multiplier)
-- WITHOUT compounding across throttle->recover->throttle cycles, and restore it
-- exactly on recovery. NULL is treated as "base == current min_gap_seconds".
ALTER TABLE rate_governor ADD COLUMN IF NOT EXISTS base_min_gap_seconds INTEGER;
ALTER TABLE rate_governor ADD COLUMN IF NOT EXISTS halted_until TIMESTAMPTZ;
-- A3: a recency stamp for EVERY lease/outcome (success + captcha + block), distinct from
-- last_applied_at (which is stamped ONLY on a confirmed apply). The ATS apply lease gates the
-- min-gap / Doctor floor off COALESCE(last_applied_at, last_attempt_at) so a never-succeeded
-- (hard-blocking) host -- whose last_applied_at is forever NULL -- is still spaced by the breaker
-- gap AND the Doctor's doctor_min_gap_floor, instead of leasing back-to-back at zero spacing.
ALTER TABLE rate_governor ADD COLUMN IF NOT EXISTS last_attempt_at TIMESTAMPTZ;

-- ---------------------------------------------------------------------------
-- llm_usage cost ledger (R14). The CAP lives in fleet_config; this is the spend log.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_usage (
    id            BIGSERIAL PRIMARY KEY,
    worker_id     TEXT,
    machine_owner TEXT,
    task          TEXT,
    model         TEXT,
    tokens_in     INTEGER,
    tokens_out    INTEGER,
    cost_usd      NUMERIC(10,6) NOT NULL DEFAULT 0,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_llm_usage_ts ON llm_usage (ts);
ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS provider TEXT;
-- provider indexed for the rolling per-agent apply-spend sum (agent_budget).
CREATE INDEX IF NOT EXISTS idx_llm_usage_provider_ts ON llm_usage (provider, ts);

-- ---------------------------------------------------------------------------
-- agent_availability: FLEET-WIDE apply-agent block state (one row per agent).
-- Two writers, one channel: (1) a worker that hits a usage/session wall records
-- blocked_until = the reset time so the WHOLE fleet skips that agent (not just the
-- one worker that discovered it -- the fleet-wide upgrade over per-worker memory);
-- (2) the predictive monitor (agent_budget.evaluate_soft_blocks) pre-emptively blocks
-- an agent whose rolling apply spend crosses its soft cap, BEFORE it walls. Workers
-- read this each tick and feed it into their AgentSwitcher. Layered ON TOP of the
-- per-worker reactive switch; never replaces it.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_availability (
    agent         TEXT PRIMARY KEY,          -- 'claude' | 'codex' | 'deepseek'
    blocked_until TIMESTAMPTZ,               -- NULL or past = available now
    reason        TEXT,                      -- 'usage_limit_wall' | 'predictive_spend'
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- applied_set: cross-board posting-level dedup / double-apply guard (R9).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS applied_set (
    dedup_key        TEXT PRIMARY KEY,               -- (company, normalized_role) -- board-agnostic
    company          TEXT,
    normalized_role  TEXT,
    first_applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    applied_url      TEXT,                            -- the one URL we applied through
    got_response     BOOLEAN NOT NULL DEFAULT FALSE   -- set by the outcome loop (R8); display/feedback
);
-- Phase 2.3: belt-and-suspenders for a live DB whose applied_set predates got_response
-- (CREATE TABLE IF NOT EXISTS above is a no-op on an already-existing table).
ALTER TABLE applied_set ADD COLUMN IF NOT EXISTS got_response BOOLEAN NOT NULL DEFAULT FALSE;

-- ---------------------------------------------------------------------------
-- answer_bank: screening-question answers, broker-served, defer-on-unknown (R10).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS answer_bank (
    q_norm     TEXT PRIMARY KEY,                      -- normalized question key
    q_raw      TEXT,
    answer     TEXT,                                  -- owner-vetted answer (NULL while unknown)
    kind       TEXT,                                  -- 'work_auth'|'years_x'|'salary'|'eeo'|'why_us'|...
    status     TEXT NOT NULL DEFAULT 'unknown_deferred',  -- 'known'|'unknown_deferred'
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- auth_challenge: captcha inbox / human-in-the-loop (R3). Surrogate id PK so the
-- SAME url can wall on a friend box AND be re-attempted on the owner box (C4).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS auth_challenge (
    id             BIGSERIAL PRIMARY KEY,
    url            TEXT NOT NULL,
    worker_id      TEXT,
    machine_owner  TEXT,
    home_ip        TEXT,
    kind           TEXT,                              -- classifier: 'visible_captcha'|'email_otp'|'sms_otp'
                                                      --  |'login_gate'|'invisible_block'|'cf'|...
    route          TEXT,                              -- 'owner_inbox'|'owner_tray'|'skip'|'remote_assist'
    screenshot_url TEXT,                              -- broker-served blob ref (NOT inline BYTEA)
    raised_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at    TIMESTAMPTZ,
    outcome        TEXT                               -- 'solved'|'skipped'|'deferred'|'rerouted_owner'
);
CREATE INDEX IF NOT EXISTS idx_challenge_open ON auth_challenge (url) WHERE resolved_at IS NULL;

-- ---------------------------------------------------------------------------
-- otp_request: Gmail relay bookkeeping (R4). The code is persisted only for the
-- short interval between the home responder answering and the worker consuming it.
-- This row is also the consumed audit trail; matched_message_id prevents one email
-- from being delivered to multiple requests, including across responder cycles.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS otp_request (
    id               BIGSERIAL PRIMARY KEY,
    worker_id        TEXT,
    url              TEXT,
    sender_hint      TEXT,                            -- ATS/sender domain the worker expects
    requested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    matched_email_ts TIMESTAMPTZ,                     -- timestamp of the email the broker matched
    consumed_at      TIMESTAMPTZ
);

-- Relay transport columns (2026-07-03): the CODE lives here only for the seconds
-- between the home responder answering and the worker consuming it, then is nulled.
ALTER TABLE otp_request ADD COLUMN IF NOT EXISTS code        TEXT;
ALTER TABLE otp_request ADD COLUMN IF NOT EXISTS code_kind   TEXT;   -- 'code' | 'magic_link'
ALTER TABLE otp_request ADD COLUMN IF NOT EXISTS expires_at  TIMESTAMPTZ;
ALTER TABLE otp_request ADD COLUMN IF NOT EXISTS answered_at TIMESTAMPTZ;
ALTER TABLE otp_request ADD COLUMN IF NOT EXISTS matched_message_id TEXT;
ALTER TABLE otp_request ADD COLUMN IF NOT EXISTS wait_started_at TIMESTAMPTZ;
-- The responder's pending-scan: unanswered, unconsumed requests.
CREATE INDEX IF NOT EXISTS idx_otp_pending ON otp_request (requested_at)
    WHERE code IS NULL AND consumed_at IS NULL;
-- DeadMan active-demand summary: skip expired/legacy history and cover all
-- timestamps needed by the count/oldest-wait query without visiting those rows.
CREATE INDEX IF NOT EXISTS idx_otp_active_wait
    ON otp_request (expires_at, requested_at, wait_started_at)
    WHERE code IS NULL AND consumed_at IS NULL
      AND wait_started_at IS NOT NULL AND expires_at IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_otp_matched_message_unique
    ON otp_request (matched_message_id) WHERE matched_message_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- inbox_events: outcome tracking / feedback loop (R8). Written by the inbox
-- scanner on the TRUSTED spot (owner box / broker), keyed by dedup_key.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inbox_events (
    id          BIGSERIAL PRIMARY KEY,
    dedup_key   TEXT,                                 -- ties response -> application (R9 key)
    url         TEXT,
    event_type  TEXT,                                 -- 'interview'|'rejection'|'assessment'|'ack'
    sender      TEXT,
    received_at TIMESTAMPTZ,
    raw_snippet TEXT,
    surfaced    BOOLEAN NOT NULL DEFAULT FALSE         -- interview requests surfaced loudly
);
CREATE INDEX IF NOT EXISTS idx_inbox_dedup ON inbox_events (dedup_key);
CREATE INDEX IF NOT EXISTS idx_inbox_unsurfaced ON inbox_events (received_at)
    WHERE event_type = 'interview' AND NOT surfaced;

-- ---------------------------------------------------------------------------
-- inbox_outcomes: thin per-EMAIL application-outcome summary pushed from the home
-- brain's email_events (SQLite outcomes tracker). One row per Gmail message_id
-- (idempotency anchor). Carries the R9 dedup_key so an outcome ties back to the
-- application cross-board. Read-only mirror; no body_text / PII crosses.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inbox_outcomes (
    message_id    TEXT PRIMARY KEY,
    dedup_key     TEXT,
    job_url       TEXT,
    company       TEXT,
    title         TEXT,
    stage         TEXT,
    outcome       TEXT,
    sender_domain TEXT,
    confidence    TEXT,
    occurred_at   TIMESTAMPTZ,
    pushed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_inbox_outcomes_dedup ON inbox_outcomes (dedup_key);
CREATE INDEX IF NOT EXISTS idx_inbox_outcomes_stage ON inbox_outcomes (stage, occurred_at);

-- ---------------------------------------------------------------------------
-- workers: enrollment / registration + capability flags + canary validation
-- (R4 token auth, R7 health, R15 canary, lane split §4).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS workers (
    worker_id     TEXT PRIMARY KEY,
    machine_owner TEXT,
    token_hash    TEXT,                               -- sha256 of the broker bearer token
    public_ip     TEXT,                               -- registered egress IP (LinkedIn lane gate)
    capabilities  JSONB,                              -- {can_ats, can_linkedin, can_compute, can_discover}
    validated     BOOLEAN NOT NULL DEFAULT FALSE,     -- passed the canary (R15) -> may do live applies
    enrolled_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at    TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- worker_heartbeat: fleet health substrate (R5, R7). Updated ~every 20s.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS worker_heartbeat (
    worker_id       TEXT PRIMARY KEY,
    machine_owner   TEXT,
    home_ip         TEXT,
    role            TEXT,                             -- 'apply'|'compute'|'discover'|'both'
    state           TEXT,                             -- 'idle'|'applying'|'challenge_pending'|'paused'
    current_job     TEXT,
    job_started_at  TIMESTAMPTZ,                      -- over-max-duration detection
    success_today   INTEGER NOT NULL DEFAULT 0,
    captcha_today   INTEGER NOT NULL DEFAULT 0,
    block_today     INTEGER NOT NULL DEFAULT 0,
    spend_today_usd NUMERIC(10,4) NOT NULL DEFAULT 0,
    cpu_pct         REAL,
    ram_pct         REAL,
    browser_count   INTEGER,                          -- WORKERS=N concurrency (R5)
    sw_version      TEXT,                             -- reported version (R12)
    last_beat       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_stale ON worker_heartbeat (last_beat);
-- Crash + log visibility (shipped on every beat, OVERWRITE per beat). Both are
-- already SCRUBBED of secrets by the worker before the UPSERT; the console scrubs
-- again on read (defense in depth). Capped at write time (last_error <= 4000,
-- recent_log <= 8000) so this one-row-per-worker table cannot grow unbounded.
ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS last_error  TEXT;
ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS recent_log  TEXT;
ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS current_agent TEXT;
ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS current_model TEXT;
ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS agent_chain TEXT;
ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS last_agent_switch_at TIMESTAMPTZ;
ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS last_agent_switch_reason TEXT;

-- ---------------------------------------------------------------------------
-- poison_jobs: quarantine for jobs that crash/hang whoever claims them (R7).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS poison_jobs (
    url            TEXT PRIMARY KEY,
    crash_count    INTEGER NOT NULL DEFAULT 0,
    last_worker    TEXT,
    quarantined_at TIMESTAMPTZ,
    reason         TEXT,
    reviewed       BOOLEAN NOT NULL DEFAULT FALSE
);

-- ---------------------------------------------------------------------------
-- remote_commands: owner -> machine control (R7 remote-restart, R12 self-update).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS remote_commands (
    id             BIGSERIAL PRIMARY KEY,
    worker_id      TEXT,                              -- target worker_id ('*' = fleet-wide)
    command        TEXT NOT NULL,                     -- 'restart'|'pause'|'resume'|'self_update'|'drain'
    target_version TEXT,                              -- for 'self_update' (R12 canary)
    issued_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    acked_at       TIMESTAMPTZ                        -- hard global close (direct commands)
);
CREATE INDEX IF NOT EXISTS idx_commands_open ON remote_commands (worker_id) WHERE acked_at IS NULL;

-- ---------------------------------------------------------------------------
-- fleet_machine_blackout: central, expiring operator control for all fleet work
-- on selected machine labels. Launchers/agents read this before starting apply,
-- discovery, or compute work. Expired rows remain as audit history.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fleet_machine_blackout (
    id             BIGSERIAL PRIMARY KEY,
    name           TEXT NOT NULL,
    active         BOOLEAN NOT NULL DEFAULT TRUE,
    starts_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at     TIMESTAMPTZ NOT NULL,
    allow_patterns TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    block_patterns TEXT[] NOT NULL DEFAULT ARRAY['*']::TEXT[],
    reason         TEXT,
    created_by     TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    cleared_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_fleet_machine_blackout_active
    ON fleet_machine_blackout (active, starts_at, expires_at);

-- command_acks: per-worker ack of a command, so a fleet-wide ('*') broadcast is
-- delivered to EVERY worker -- not consumed by whoever acks first (R7). poll_commands
-- excludes a command this worker has already acked here; a DIRECT command also closes
-- via remote_commands.acked_at. No FK (keeps the test TRUNCATE order-independent).
CREATE TABLE IF NOT EXISTS command_acks (
    command_id BIGINT      NOT NULL,
    worker_id  TEXT        NOT NULL,
    acked_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (command_id, worker_id)
);

-- ===========================================================================
-- FLEET DOCTOR v1 (this file's only auto-remediation layer).
--
-- The Doctor reads the centralized failure data (apply_queue.apply_error,
-- apply_status, target_host, worker_id) and applies BOUNDED, REVERSIBLE,
-- MONOTONICALLY-CONSERVATIVE auto-fixes. Per-worker recent_log/last_error are consumed
-- by the console and the Fleet Diagnoser (fleet/diagnoser.py), not the Doctor. Every auto action it takes can ONLY make
-- the fleet MORE conservative (skip a host, un-approve queued rows, quarantine a
-- poison url, pace down / pause, or RAISE a timeout within a ceiling); it can never
-- un-pause, re-approve, raise the spend cap, lower the gap, or touch LinkedIn. The
-- two tables below are its KNOBS (active conservative state) + its AUDIT LOG.
-- ===========================================================================

-- fleet_config: a NEW, bounded apply-agent timeout override. NULL = use the
-- env/default (APPLYPILOT_AGENT_TIMEOUT). The Doctor only ever RAISES it within a
-- ceiling (timeout_bump); a longer timeout is conservative (it lets a slow page
-- finish instead of being killed + retried, which would re-hit the host). The apply
-- worker prefers this value over the env when it is set.
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS agent_timeout_override INTEGER;

-- ===========================================================================
-- FLEET DOCTOR HARDENING (red-team H1-H19): EFFECT-level monotonicity.
-- ===========================================================================

-- H1 (CATASTROPHE FIX): an ATS-ONLY pause flag. The Doctor's lane-pause writes THIS,
-- never fleet_config.paused. The apply lease/worker honor ats_paused; the LinkedIn lane
-- (_LEASE_LINKEDIN + linkedin_should_halt) NEVER reads it, so a Doctor pause can never
-- halt the LinkedIn catastrophe lane. fleet_config.paused stays the shared operator/cost
-- kill switch (still read by BOTH lanes' should_halt); the Doctor is forbidden to touch it.
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS ats_paused BOOLEAN NOT NULL DEFAULT FALSE;

-- H2 (AGGREGATE BUDGET): per-day blast-radius counters the Doctor decrements against. The
-- date anchor rolls the counters once per UTC day. NULL anchor / mismatched day == fresh.
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS doctor_budget_day        DATE;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS doctor_host_skips_today  INTEGER NOT NULL DEFAULT 0;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS doctor_pace_actions_today INTEGER NOT NULL DEFAULT 0;

-- H18 (above-the-fold signal): the Doctor's last-pass timestamp + active auto-fix snapshot,
-- so /api/status (the 4s poll) can show a "Doctor" card + fire a toast on a new auto-fix
-- WITHOUT folding the heavy diagnostics blob into the fast poll.
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS doctor_last_pass_at TIMESTAMPTZ;

-- H5/H8 (pause debounce + provenance): persist when the Doctor armed a lane-pause (2-pass
-- debounce) and remember that the ACTIVE ats_paused was set by the Doctor (so it auto-reverts
-- only its OWN pause and the console can label provenance). pause_source: NULL|'doctor'.
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS doctor_pause_armed_at TIMESTAMPTZ;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS ats_pause_source      TEXT;

-- A6: consecutive-systemic-pass counter. The N4 systemic classifier emits ONE de-duplicated
-- distinct-severity alert only on the Nth consecutive systemic pass (so a single flagged home_IP
-- mis-classified as systemic doesn't spam, and a permanently-systemic fleet emits one escalation
-- rather than latching recommend-only silently). Reset to 0 on the first non-systemic pass.
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS doctor_systemic_streak INTEGER NOT NULL DEFAULT 0;

-- DeadMan monitor (Task 2): persisted alert state for the read-only fleet dead-man
-- detector (applypilot.fleet.deadman). deadman_alert is a '|'-joined "kind: detail"
-- summary of the currently-active alerts (NULL when healthy); deadman_alert_at is
-- when it was last set; deadman_hot_streak persists the running_hot consecutive-check
-- counter across invocations (deadman_check is pure and takes/returns it explicitly).
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS deadman_alert TEXT;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS deadman_alert_at TIMESTAMPTZ;
ALTER TABLE fleet_config ADD COLUMN IF NOT EXISTS deadman_hot_streak INTEGER NOT NULL DEFAULT 0;

-- N2 (DOCTOR-OWNED, NON-SHARED ACTUATORS): two columns the Doctor SOLELY owns on a host:<h>
-- governor scope, so the watchdog breaker (which owns min_gap_seconds/base_min_gap_seconds/
-- breaker_state) and the Doctor never clobber each other.
--   doctor_min_gap_floor  -- H4: the lease uses GREATEST(min_gap_seconds, doctor_min_gap_floor)
--                            so a Doctor pace is monotone-by-construction and the breaker's
--                            min_gap restore can never wipe it.
--   doctor_skip_until     -- H6: host_skip becomes a self-expiring leasable FILTER (the lease
--                            adds AND COALESCE(doctor_skip_until,'-infinity') < now()) instead
--                            of NULLing approved_batch -- vetted approval is preserved.
ALTER TABLE rate_governor ADD COLUMN IF NOT EXISTS doctor_min_gap_floor INTEGER;
ALTER TABLE rate_governor ADD COLUMN IF NOT EXISTS doctor_skip_until    TIMESTAMPTZ;
-- NOTE: the fleet_diagnoses audit columns (H19/H13) are added AFTER that table is created,
-- at the bottom of this file -- ADD COLUMN here would fail (the table doesn't exist yet).

-- fleet_knobs: ACTIVE conservative state the Doctor sets (and a human can REVERSE).
-- A host_skip knob makes a re-push/approve respect the skip; a timeout_bump knob is
-- the audit twin of fleet_config.agent_timeout_override. TTL'd via expires_at so a
-- transient host block self-heals; the sweep deactivates expired rows each run.
CREATE TABLE IF NOT EXISTS fleet_knobs (
    id          BIGSERIAL PRIMARY KEY,
    knob_type   TEXT NOT NULL,                      -- 'host_skip'|'timeout_bump'|'quarantine'|'pace_or_pause'
    scope_key   TEXT,                               -- host / lane / url the knob applies to
    value_text  TEXT,                               -- e.g. the new timeout, the new min_gap, 'paused'
    reason      TEXT,
    created_by  TEXT NOT NULL DEFAULT 'doctor',
    active      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_fleet_knobs_active
    ON fleet_knobs (active, knob_type, scope_key);

-- H9 (singleton / race-proof idempotency): at most ONE active knob per (knob_type, scope_key).
-- With INSERT ... ON CONFLICT DO NOTHING this makes a duplicate-knob write from two racing
-- Doctor passes a no-op at the DB level (the advisory lock in run_doctor is the first line of
-- defense; this index is the backstop). A12: the de-dup of a NULL scope_key works ONLY because
-- of COALESCE(scope_key,''): Postgres treats two NULLs as DISTINCT in a UNIQUE index (NULLs do
-- NOT collide), so WITHOUT the COALESCE two active (knob_type, NULL) knobs would both be allowed
-- and the H9 race would re-open for any NULL-scope knob. The COALESCE(...,'') is therefore
-- LOAD-BEARING -- it maps NULL -> '' so they collide. Do NOT "simplify" it away.
CREATE UNIQUE INDEX IF NOT EXISTS uq_fleet_knobs_one_active
    ON fleet_knobs (knob_type, COALESCE(scope_key, '')) WHERE active;

-- fleet_diagnoses: the Doctor's AUDIT LOG + the human RECOMMENDATION queue. Every
-- auto action writes one row (what + why + evidence + how_to_reverse + expires_at);
-- everything the Doctor finds that is NOT one of the four conservative auto-fixes
-- becomes a status='recommended' row for a human (never auto-applied).
CREATE TABLE IF NOT EXISTS fleet_diagnoses (
    id             BIGSERIAL PRIMARY KEY,
    cluster_key    TEXT,                             -- reason|host|machine|lane signature (idempotency key)
    reason         TEXT,
    host           TEXT,
    machine        TEXT,
    lane           TEXT,
    sample_count   INT,
    severity       TEXT,                             -- 'info'|'warn'|'severe'
    diagnosis      TEXT,                             -- rule-templated, NO LLM in v1 (hook reserved)
    recommendation TEXT,
    auto_action    TEXT,                             -- the knob_type applied, or NULL for a recommendation
    how_to_reverse TEXT,
    status         TEXT NOT NULL DEFAULT 'open',     -- open|auto_applied|recommended|applied|dismissed|reverted|expired
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_fleet_diagnoses_status
    ON fleet_diagnoses (status, created_at);
CREATE INDEX IF NOT EXISTS idx_fleet_diagnoses_cluster
    ON fleet_diagnoses (cluster_key) WHERE status IN ('auto_applied','recommended','open');

-- fleet_console_audit: append-only operator console action trail. This table records
-- existing allow-listed console actions only; it never stores request bodies, tokens,
-- DSNs, prompts, browser logs, resume/profile data, or raw secrets.
CREATE TABLE IF NOT EXISTS fleet_console_audit (
    id         BIGSERIAL PRIMARY KEY,
    action     TEXT NOT NULL,
    actor      TEXT,
    lane       TEXT,
    target     TEXT,
    message    TEXT,
    ok         BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_fleet_console_audit_created
    ON fleet_console_audit (created_at DESC);

-- autotriage_actions: audit trail for autonomous bounded LLM/rule triage.
-- The LLM may choose only from a fixed action menu, and the executor validates
-- each action before mutating fleet state. This table records both applied and
-- rejected choices plus manual-review deferrals so the loop is inspectable and reversible.
CREATE TABLE IF NOT EXISTS autotriage_actions (
    id                 BIGSERIAL PRIMARY KEY,
    url                TEXT,
    worker_id          TEXT,
    chosen_action      TEXT,
    decision_source    TEXT,
    confidence         REAL,
    reason             TEXT,
    action_status      TEXT NOT NULL DEFAULT 'planned',
    prior_status       TEXT,
    prior_attempts     INTEGER,
    prior_apply_error  TEXT,
    evidence           JSONB,
    how_to_reverse     TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_autotriage_url
    ON autotriage_actions (url, created_at);
CREATE INDEX IF NOT EXISTS idx_autotriage_status
    ON autotriage_actions (action_status, created_at);

-- apply_attempts: durable adapter submit boundary. An unresolved irreversible
-- action for a dedup key excludes every second submit path until it is verified,
-- contradicted, or quarantined by the owner-controlled workflow.
CREATE TABLE IF NOT EXISTS apply_attempts (
    attempt_id          UUID PRIMARY KEY,
    queue_name          TEXT NOT NULL,
    url                 TEXT NOT NULL,
    dedup_key           TEXT,
    worker_id           TEXT NOT NULL,
    route               TEXT NOT NULL,
    route_version       TEXT,
    state               TEXT NOT NULL CHECK (state IN (
        'prepared', 'submit_started', 'submitted_unverified', 'verified',
        'contradicted', 'quarantined', 'failed_pre_submit'
    )),
    submit_started_at   TIMESTAMPTZ,
    finalized_at        TIMESTAMPTZ,
    verification_method TEXT,
    verification_ref   TEXT,
    evidence            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_apply_attempts_unresolved_dedup
    ON apply_attempts (dedup_key)
    WHERE dedup_key IS NOT NULL
      AND state IN ('submit_started', 'submitted_unverified');
CREATE INDEX IF NOT EXISTS idx_apply_attempts_url_created
    ON apply_attempts (queue_name, url, created_at DESC);
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fleet_worker') THEN
        GRANT SELECT, INSERT, UPDATE ON apply_attempts TO fleet_worker;
        REVOKE DELETE, TRUNCATE, REFERENCES, TRIGGER ON apply_attempts FROM fleet_worker;
    END IF;
END $$;

-- apply_result_events: durable per-job terminal evidence written by the worker when
-- it closes a lease. worker_heartbeat.recent_log is only a moving tail; repair tools
-- should prefer this table for job-specific result evidence.
CREATE TABLE IF NOT EXISTS apply_result_events (
    id                  BIGSERIAL PRIMARY KEY,
    queue_name          TEXT NOT NULL DEFAULT 'apply_queue',
    url                 TEXT NOT NULL,
    worker_id           TEXT,
    machine_owner       TEXT,
    status              TEXT,
    apply_status        TEXT,
    apply_error         TEXT,
    target_host         TEXT,
    home_ip             TEXT,
    agent               TEXT,
    agent_model         TEXT,
    est_cost_usd        REAL,
    apply_duration_ms   INTEGER,
    application_tool_calls INTEGER,
    job_log_path        TEXT,
    transcript_digest   TEXT,
    final_result_source TEXT,
    result_metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_line         TEXT,
    source              TEXT NOT NULL DEFAULT 'worker',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS queue_name TEXT NOT NULL DEFAULT 'apply_queue';
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS machine_owner TEXT;
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'worker';
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS route TEXT;
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS failure_class TEXT;
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS tool_calls_total INTEGER;
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS application_tool_calls INTEGER;
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS last_tool TEXT;
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS host_policy TEXT;
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS result_metadata JSONB;
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS job_log_path TEXT;
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS transcript_digest TEXT;
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS final_result_source TEXT;
ALTER TABLE apply_result_events ADD COLUMN IF NOT EXISTS result_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
UPDATE apply_result_events SET result_metadata = '{}'::jsonb WHERE result_metadata IS NULL;
ALTER TABLE apply_result_events ALTER COLUMN result_metadata SET DEFAULT '{}'::jsonb;
ALTER TABLE apply_result_events ALTER COLUMN result_metadata SET NOT NULL;
CREATE INDEX IF NOT EXISTS idx_apply_result_events_url_created
    ON apply_result_events (queue_name, url, created_at DESC);

-- Preserve all-in spend across retries. est_cost_usd remains the latest/current
-- attempt value for diagnostics; cumulative_cost_usd is the cap/reporting ledger.
ALTER TABLE apply_queue
    ADD COLUMN IF NOT EXISTS cumulative_cost_usd NUMERIC(12,4) NOT NULL DEFAULT 0;
UPDATE apply_queue q
SET cumulative_cost_usd = GREATEST(
    COALESCE(q.cumulative_cost_usd, 0),
    COALESCE(q.est_cost_usd, 0),
    COALESCE((
        SELECT SUM(e.est_cost_usd)
        FROM apply_result_events e
        WHERE e.queue_name = 'apply_queue' AND e.url = q.url
    ), 0)
);
ALTER TABLE linkedin_queue
    ADD COLUMN IF NOT EXISTS cumulative_cost_usd NUMERIC(12,4) NOT NULL DEFAULT 0;
UPDATE linkedin_queue
SET cumulative_cost_usd = GREATEST(
    COALESCE(cumulative_cost_usd, 0), COALESCE(est_cost_usd, 0)
);

-- H19/H13 (red-team): self-contained host_skip audit + recurrence linkage + breadth evidence on
-- the diagnosis row, so a Reverse / audit can report exactly which/how-many rows were affected,
-- how many prior incidents, and how broad the block was -- without re-deriving from transient state.
ALTER TABLE fleet_diagnoses ADD COLUMN IF NOT EXISTS rows_affected        INTEGER;
ALTER TABLE fleet_diagnoses ADD COLUMN IF NOT EXISTS prior_incident_count INTEGER;
ALTER TABLE fleet_diagnoses ADD COLUMN IF NOT EXISTS distinct_hosts       INTEGER;
ALTER TABLE fleet_diagnoses ADD COLUMN IF NOT EXISTS distinct_workers     INTEGER;

-- email_reconcile_actions: audit + reversibility for the email-verification reconcile.
CREATE TABLE IF NOT EXISTS email_reconcile_actions (
    id              BIGSERIAL PRIMARY KEY,
    url             TEXT,
    message_id      TEXT,
    match_method    TEXT,
    match_score     REAL,
    stage           TEXT,
    prior_status    TEXT,
    how_to_reverse  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- dedup_repair_actions: audit + reversibility for source-specific rewrites of
-- overbroad queued dedup keys (for example aggregator/missing-company keys).
CREATE TABLE IF NOT EXISTS dedup_repair_actions (
    id              BIGSERIAL PRIMARY KEY,
    url             TEXT,
    old_dedup_key   TEXT,
    new_dedup_key   TEXT,
    reason          TEXT,
    how_to_reverse  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dedup_repair_actions_old_key
    ON dedup_repair_actions (old_dedup_key, created_at);
