-- ===========================================================================
-- Cloud apply-fleet schema (Railway Postgres) -- queue-offload design.
-- See docs/superpowers/specs/2026-06-24-cloud-apply-fleet-design.md (S3a, S5, S6).
--
-- Idempotent: safe to run on every worker/home startup (CREATE ... IF NOT EXISTS,
-- ENUM guarded by a duplicate_object catch). This is the ONLY state in the cloud --
-- a thin work queue + result mailbox. Home SQLite stays authoritative.
-- ===========================================================================

-- apply_queue_status: referenced by name in the lease/reclaim SQL, so an ENUM
-- (not a CHECK) -- but CREATE TYPE has no IF NOT EXISTS, so guard it.
DO $$ BEGIN
    CREATE TYPE apply_queue_status AS ENUM (
        'queued',             -- pushed, eligible to lease
        'leased',             -- a worker holds it (lease_expires_at in the future)
        'applied',            -- submit confirmed
        'failed',             -- terminal non-submit (expired/captcha/page_error/...)
        'blocked',            -- site/cloudflare/auth wall the offsite agent can't pass
        'crash_unconfirmed'   -- worker died mid-job, possibly post-submit: NEVER re-leased
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS fleet_decision_policies (
    policy_version TEXT PRIMARY KEY,
    lane           TEXT NOT NULL CHECK (lane IN ('ats', 'linkedin')),
    status         TEXT NOT NULL CHECK (status IN ('draft', 'validated', 'canary', 'active', 'retired')),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    activated_at   TIMESTAMPTZ,
    retired_at     TIMESTAMPTZ,
    UNIQUE (policy_version, lane)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_fleet_decision_policy_active_lane
    ON fleet_decision_policies (lane) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_fleet_decision_policy_lane_status
    ON fleet_decision_policies (lane, status);

CREATE TABLE IF NOT EXISTS apply_queue (
    -- ---- identity / job columns (pushed from home) ------------------------
    url                     TEXT        PRIMARY KEY,          -- = jobs.url (cross-system key + idempotency anchor)
    company                 TEXT,
    title                   TEXT,
    application_url         TEXT        NOT NULL,             -- offsite ATS form target
    score                   REAL        NOT NULL,             -- canonical final-score ordering copy; never independent authority
    apply_domain            TEXT,                             -- effective apply host (politeness key)

    -- ---- immutable canonical decision provenance ------------------------
    decision_id             TEXT,
    policy_version          TEXT,
    decision_action         TEXT,
    qualification_verdict   TEXT,
    qualification_score     REAL,
    qualification_floor     REAL,
    preference_score        REAL,
    outcome_score           REAL,
    final_score             REAL,
    decision_confidence     REAL,
    decision_created_at     TIMESTAMPTZ,
    decision_expires_at     TIMESTAMPTZ,
    input_hash              TEXT,

    -- ---- queue / lease state ---------------------------------------------
    status                  apply_queue_status NOT NULL DEFAULT 'queued',
    lease_owner             TEXT,
    lease_expires_at        TIMESTAMPTZ,
    last_attempted_at       TIMESTAMPTZ,                      -- set at lease time (politeness)
    attempts                INTEGER     NOT NULL DEFAULT 0,

    -- ---- result columns (written by the fleet) ---------------------------
    apply_status            TEXT,                             -- raw agent outcome: applied / failed:<reason> / expired / captcha ...
    apply_error             TEXT,
    verification_confidence TEXT,                             -- pass-through (NULL in live DB today)
    agent_model             TEXT,                             -- provider/model that ran this job (the Sonnet-vs-DeepSeek A/B)
    est_cost_usd            NUMERIC(10,4),                    -- latest/current attempt cost
    cumulative_cost_usd     NUMERIC(12,4) NOT NULL DEFAULT 0, -- all attempts; drives caps and CPA
    applied_at              TIMESTAMPTZ,
    worker_id               TEXT,
    machine_owner           TEXT,
    apply_duration_ms       INTEGER,

    -- ---- bookkeeping ------------------------------------------------------
    pushed_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    synced_to_home_at       TIMESTAMPTZ,                      -- set by PULL; NULL = not yet ingested home

    CONSTRAINT apply_queue_canonical_provenance_ck CHECK (
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
    )
);

-- Lease query: WHERE status='queued' ORDER BY score DESC LIMIT 1.
CREATE INDEX IF NOT EXISTS idx_apply_queue_lease
    ON apply_queue (score DESC)
    WHERE status = 'queued';

-- Reclaim scan: leased rows whose lease has expired.
CREATE INDEX IF NOT EXISTS idx_apply_queue_reclaim
    ON apply_queue (lease_expires_at)
    WHERE status = 'leased';

-- PULL scan: terminal rows not yet ingested back into home SQLite.
CREATE INDEX IF NOT EXISTS idx_apply_queue_unsynced
    ON apply_queue (updated_at)
    WHERE status IN ('applied','failed','blocked','crash_unconfirmed')
      AND synced_to_home_at IS NULL;

-- Politeness scan: recently-touched domains.
CREATE INDEX IF NOT EXISTS idx_apply_queue_host_recent
    ON apply_queue (apply_domain, last_attempted_at);

-- Single-row global control: spend cap + kill switch. id=1 enforced.
CREATE TABLE IF NOT EXISTS fleet_config (
    id              INTEGER       PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    spend_cap_usd   NUMERIC(10,2) NOT NULL DEFAULT 0,        -- 0 = no cap; halt on cumulative spend
    paused          BOOLEAN       NOT NULL DEFAULT FALSE,    -- global kill switch
    ats_policy_version      TEXT,
    ats_policy_lane         TEXT GENERATED ALWAYS AS ('ats'::text) STORED,
    linkedin_policy_version TEXT,
    linkedin_policy_lane    TEXT GENERATED ALWAYS AS ('linkedin'::text) STORED,
    CONSTRAINT fleet_config_ats_policy_fk FOREIGN KEY (ats_policy_version, ats_policy_lane)
        REFERENCES fleet_decision_policies(policy_version, lane),
    CONSTRAINT fleet_config_linkedin_policy_fk FOREIGN KEY (linkedin_policy_version, linkedin_policy_lane)
        REFERENCES fleet_decision_policies(policy_version, lane),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT now()
);

INSERT INTO fleet_config (id, spend_cap_usd, paused)
VALUES (1, 0, FALSE)
ON CONFLICT (id) DO NOTHING;

-- Worker assets (profile.json + resume.pdf) shipped THROUGH Postgres, so the fleet needs no
-- volume/secret juggling for PII: the home box uploads them once, each worker hydrates them to
-- disk on startup. BYTEA holds the binary resume PDF.
CREATE TABLE IF NOT EXISTS fleet_assets (
    name       TEXT        PRIMARY KEY,
    data       BYTEA       NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Durable terminal evidence is also part of the base queue schema so the
-- watchdog can record lease reclaims before the v3 fleet layer is bootstrapped.
CREATE TABLE IF NOT EXISTS apply_result_events (
    id                      BIGSERIAL PRIMARY KEY,
    queue_name              TEXT NOT NULL DEFAULT 'apply_queue',
    url                     TEXT NOT NULL,
    worker_id               TEXT,
    status                  TEXT,
    apply_status            TEXT,
    apply_error             TEXT,
    target_host             TEXT,
    home_ip                 TEXT,
    agent                   TEXT,
    agent_model             TEXT,
    est_cost_usd            REAL,
    apply_duration_ms       INTEGER,
    application_tool_calls  INTEGER,
    job_log_path            TEXT,
    transcript_digest       TEXT,
    final_result_source     TEXT,
    result_metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_line             TEXT,
    source                  TEXT NOT NULL DEFAULT 'worker',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_apply_result_events_url_created
    ON apply_result_events (queue_name, url, created_at DESC);
