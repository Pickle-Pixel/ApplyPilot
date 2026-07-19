CREATE TABLE public.brain_schema_versions (
    version INTEGER PRIMARY KEY CHECK (version > 0),
    migration_name TEXT NOT NULL UNIQUE,
    migration_checksum TEXT NOT NULL CHECK (migration_checksum ~ '^[0-9a-f]{64}$'),
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    applied_by TEXT NOT NULL DEFAULT current_user
);

CREATE TABLE public.brain_artifacts (
    request_id TEXT NOT NULL UNIQUE CHECK (btrim(request_id) <> ''),
    artifact_hash TEXT PRIMARY KEY CHECK (artifact_hash ~ '^[0-9a-f]{64}$'),
    media_type TEXT NOT NULL CHECK (btrim(media_type) <> ''),
    byte_length BIGINT NOT NULL CHECK (byte_length >= 0),
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(provenance) = 'object'),
    location TEXT NOT NULL CHECK (btrim(location) <> '')
);

CREATE TABLE public.brain_artifact_locations (
    artifact_location_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    backend TEXT NOT NULL CHECK (backend IN ('s3', 'gcs', 'azure', 'filesystem', 'other')),
    bucket_or_container TEXT,
    object_key TEXT NOT NULL CHECK (btrim(object_key) <> ''),
    provider_version_id TEXT,
    provider_checksum TEXT,
    storage_immutable BOOLEAN NOT NULL,
    encryption_mode TEXT NOT NULL CHECK (encryption_mode IN ('none', 'provider_managed', 'customer_managed')),
    encryption_key_id TEXT,
    durability TEXT NOT NULL CHECK (durability IN ('committed_unprotected', 'durable', 'verified')),
    verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_artifact_locations_identity UNIQUE NULLS NOT DISTINCT
        (artifact_hash, backend, bucket_or_container, object_key, provider_version_id),
    CONSTRAINT brain_artifact_locations_verification CHECK (
        (durability = 'verified' AND verified_at IS NOT NULL AND verified_at >= created_at
         AND provider_version_id IS NOT NULL AND btrim(provider_version_id) <> ''
         AND provider_checksum IS NOT NULL AND btrim(provider_checksum) <> ''
         AND storage_immutable AND encryption_mode <> 'none')
        OR durability <> 'verified'
    ),
    CONSTRAINT brain_artifact_locations_encryption CHECK (
        encryption_mode = 'none' OR encryption_key_id IS NOT NULL
    )
);

CREATE TABLE public.brain_jobs (
    job_id TEXT PRIMARY KEY CHECK (btrim(job_id) <> ''),
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    source_job_id TEXT NOT NULL CHECK (btrim(source_job_id) <> ''),
    canonical_url TEXT,
    title TEXT,
    company TEXT,
    current_metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(current_metadata) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_jobs_source_key UNIQUE (source_namespace, source_job_id)
);

CREATE TABLE public.brain_job_aliases (
    alias_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES public.brain_jobs(job_id),
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    source_database_fingerprint TEXT NOT NULL CHECK (source_database_fingerprint ~ '^[0-9a-f]{64}$'),
    source_item_id TEXT,
    source_url TEXT,
    alias_type TEXT NOT NULL CHECK (btrim(alias_type) <> ''),
    alias_metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(alias_metadata) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_job_aliases_endpoint CHECK (source_item_id IS NOT NULL OR source_url IS NOT NULL),
    CONSTRAINT brain_job_aliases_idempotent UNIQUE NULLS NOT DISTINCT
        (source_namespace, source_database_fingerprint, source_item_id, source_url, alias_type)
);

CREATE TABLE public.brain_job_observations (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    logical_subject_id TEXT GENERATED ALWAYS AS (source_namespace || ':job:' || job_id) STORED,
    source_observation_id TEXT NOT NULL CHECK (btrim(source_observation_id) <> ''),
    job_id TEXT NOT NULL REFERENCES public.brain_jobs(job_id),
    observed_at TIMESTAMPTZ NOT NULL,
    content_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    observation_metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(observation_metadata) = 'object'),
    supersedes_observation_id BIGINT REFERENCES public.brain_job_observations(observation_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_job_observations_source_key UNIQUE (source_namespace, source_observation_id),
    CONSTRAINT brain_job_observations_not_self CHECK (supersedes_observation_id IS NULL OR supersedes_observation_id <> observation_id)
);
CREATE UNIQUE INDEX brain_job_observations_one_successor ON public.brain_job_observations(supersedes_observation_id) WHERE supersedes_observation_id IS NOT NULL;

CREATE TABLE public.brain_label_events (
    label_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    logical_subject_id TEXT GENERATED ALWAYS AS (
        source_namespace || ':label:' || project || ':' || label_name || ':' ||
        'job:' || COALESCE(job_id, '-') || ':item:' || COALESCE(source_item_id, '-') ||
        ':url:' || COALESCE(source_item_url, '-')
    ) STORED,
    source_event_id TEXT NOT NULL CHECK (btrim(source_event_id) <> ''),
    job_id TEXT REFERENCES public.brain_jobs(job_id),
    source_item_id TEXT,
    source_item_url TEXT,
    project TEXT NOT NULL CHECK (btrim(project) <> ''),
    method TEXT NOT NULL CHECK (btrim(method) <> ''),
    confidence NUMERIC CHECK (confidence BETWEEN 0 AND 1),
    weight NUMERIC NOT NULL DEFAULT 1 CHECK (weight >= 0),
    label_name TEXT NOT NULL CHECK (btrim(label_name) <> ''),
    label_value JSONB NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    raw_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    evidence_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    event_metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(event_metadata) = 'object'),
    supersedes_label_event_id BIGINT REFERENCES public.brain_label_events(label_event_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_label_events_endpoint CHECK (job_id IS NOT NULL OR source_item_id IS NOT NULL OR source_item_url IS NOT NULL),
    CONSTRAINT brain_label_events_source_key UNIQUE (source_namespace, source_event_id),
    CONSTRAINT brain_label_events_not_self CHECK (supersedes_label_event_id IS NULL OR supersedes_label_event_id <> label_event_id)
);
CREATE UNIQUE INDEX brain_label_events_one_successor ON public.brain_label_events(supersedes_label_event_id) WHERE supersedes_label_event_id IS NOT NULL;

CREATE TABLE public.brain_pairwise_events (
    pairwise_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    logical_subject_id TEXT GENERATED ALWAYS AS (
        source_namespace || ':pair:' || project || ':' ||
        'left-job:' || COALESCE(left_job_id, '-') || ':left-item:' || COALESCE(left_source_item_id, '-') ||
        ':left-url:' || COALESCE(left_source_url, '-') || ':right-job:' || COALESCE(right_job_id, '-') ||
        ':right-item:' || COALESCE(right_source_item_id, '-') || ':right-url:' || COALESCE(right_source_url, '-')
    ) STORED,
    source_event_id TEXT NOT NULL CHECK (btrim(source_event_id) <> ''),
    left_job_id TEXT REFERENCES public.brain_jobs(job_id),
    right_job_id TEXT REFERENCES public.brain_jobs(job_id),
    left_source_item_id TEXT,
    left_source_url TEXT,
    right_source_item_id TEXT,
    right_source_url TEXT,
    project TEXT NOT NULL CHECK (btrim(project) <> ''),
    method TEXT NOT NULL CHECK (btrim(method) <> ''),
    confidence NUMERIC CHECK (confidence BETWEEN 0 AND 1),
    weight NUMERIC NOT NULL DEFAULT 1 CHECK (weight >= 0),
    preference TEXT NOT NULL CHECK (preference IN ('left', 'right', 'tie', 'unclear')),
    occurred_at TIMESTAMPTZ NOT NULL,
    raw_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    evidence_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    event_metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(event_metadata) = 'object'),
    supersedes_pairwise_event_id BIGINT REFERENCES public.brain_pairwise_events(pairwise_event_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_pairwise_events_left_endpoint CHECK (left_job_id IS NOT NULL OR left_source_item_id IS NOT NULL OR left_source_url IS NOT NULL),
    CONSTRAINT brain_pairwise_events_right_endpoint CHECK (right_job_id IS NOT NULL OR right_source_item_id IS NOT NULL OR right_source_url IS NOT NULL),
    CONSTRAINT brain_pairwise_events_distinct CHECK (left_job_id IS NULL OR right_job_id IS NULL OR left_job_id <> right_job_id),
    CONSTRAINT brain_pairwise_events_source_key UNIQUE (source_namespace, source_event_id),
    CONSTRAINT brain_pairwise_events_not_self CHECK (supersedes_pairwise_event_id IS NULL OR supersedes_pairwise_event_id <> pairwise_event_id)
);
CREATE UNIQUE INDEX brain_pairwise_events_one_successor ON public.brain_pairwise_events(supersedes_pairwise_event_id) WHERE supersedes_pairwise_event_id IS NOT NULL;

CREATE TABLE public.brain_email_events (
    email_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    logical_subject_id TEXT GENERATED ALWAYS AS (
        source_namespace || ':email:' || COALESCE(job_id, 'unresolved') || ':' || event_type
    ) STORED,
    source_event_id TEXT NOT NULL CHECK (btrim(source_event_id) <> ''),
    job_id TEXT REFERENCES public.brain_jobs(job_id),
    event_type TEXT NOT NULL CHECK (btrim(event_type) <> ''),
    occurred_at TIMESTAMPTZ NOT NULL,
    payload_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    event_metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(event_metadata) = 'object'),
    supersedes_email_event_id BIGINT REFERENCES public.brain_email_events(email_event_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_email_events_source_key UNIQUE (source_namespace, source_event_id),
    CONSTRAINT brain_email_events_job_lineage_key UNIQUE (email_event_id, job_id),
    CONSTRAINT brain_email_events_not_self CHECK (supersedes_email_event_id IS NULL OR supersedes_email_event_id <> email_event_id)
);
CREATE UNIQUE INDEX brain_email_events_one_successor ON public.brain_email_events(supersedes_email_event_id) WHERE supersedes_email_event_id IS NOT NULL;

CREATE TABLE public.brain_reviewed_outcomes (
    reviewed_outcome_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    logical_subject_id TEXT GENERATED ALWAYS AS (
        source_namespace || ':outcome:' || job_id || ':' || email_event_id::text
    ) STORED,
    source_event_id TEXT NOT NULL CHECK (btrim(source_event_id) <> ''),
    job_id TEXT NOT NULL REFERENCES public.brain_jobs(job_id),
    email_event_id BIGINT NOT NULL,
    review_status TEXT NOT NULL CHECK (btrim(review_status) <> ''),
    normalized_stage TEXT NOT NULL CHECK (btrim(normalized_stage) <> ''),
    weight NUMERIC CHECK (weight >= 0),
    reviewer TEXT NOT NULL CHECK (btrim(reviewer) <> ''),
    reason TEXT,
    evidence_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    review_metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(review_metadata) = 'object'),
    supersedes_reviewed_outcome_id BIGINT REFERENCES public.brain_reviewed_outcomes(reviewed_outcome_id),
    created_at TIMESTAMPTZ NOT NULL,
    reviewed_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT brain_reviewed_outcomes_source_key UNIQUE (source_namespace, source_event_id),
    CONSTRAINT brain_reviewed_outcomes_email_job_fk FOREIGN KEY (email_event_id, job_id)
        REFERENCES public.brain_email_events(email_event_id, job_id),
    CONSTRAINT brain_reviewed_outcomes_times CHECK (reviewed_at >= created_at AND updated_at >= reviewed_at),
    CONSTRAINT brain_reviewed_outcomes_not_self CHECK (supersedes_reviewed_outcome_id IS NULL OR supersedes_reviewed_outcome_id <> reviewed_outcome_id)
);
CREATE UNIQUE INDEX brain_reviewed_outcomes_one_successor ON public.brain_reviewed_outcomes(supersedes_reviewed_outcome_id) WHERE supersedes_reviewed_outcome_id IS NOT NULL;

CREATE TABLE public.brain_applications (
    application_id TEXT PRIMARY KEY CHECK (btrim(application_id) <> ''),
    job_id TEXT NOT NULL REFERENCES public.brain_jobs(job_id),
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    source_application_id TEXT NOT NULL CHECK (btrim(source_application_id) <> ''),
    source_channel TEXT NOT NULL CHECK (btrim(source_channel) <> ''),
    lane TEXT CHECK (lane IN ('ats', 'linkedin')),
    current_state TEXT,
    application_metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(application_metadata) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_applications_source_key UNIQUE (source_namespace, source_application_id)
);

CREATE TABLE public.brain_application_events (
    application_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    application_id TEXT NOT NULL REFERENCES public.brain_applications(application_id),
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    logical_subject_id TEXT GENERATED ALWAYS AS (
        source_namespace || ':application:' || application_id
    ) STORED,
    source_event_id TEXT NOT NULL CHECK (btrim(source_event_id) <> ''),
    event_type TEXT NOT NULL CHECK (btrim(event_type) <> ''),
    source_channel TEXT,
    occurred_at TIMESTAMPTZ NOT NULL,
    payload_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    event_metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(event_metadata) = 'object'),
    supersedes_application_event_id BIGINT REFERENCES public.brain_application_events(application_event_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_application_events_source_key UNIQUE (source_namespace, source_event_id),
    CONSTRAINT brain_application_events_not_self CHECK (supersedes_application_event_id IS NULL OR supersedes_application_event_id <> application_event_id)
);
CREATE UNIQUE INDEX brain_application_events_one_successor ON public.brain_application_events(supersedes_application_event_id) WHERE supersedes_application_event_id IS NOT NULL;

CREATE TABLE public.brain_decision_policies (
    policy_version TEXT PRIMARY KEY CHECK (btrim(policy_version) <> ''),
    lane TEXT NOT NULL CHECK (lane IN ('ats', 'linkedin')),
    gate_definition_version INTEGER NOT NULL DEFAULT 1 CHECK (gate_definition_version > 0),
    lifecycle TEXT NOT NULL CHECK (lifecycle IN ('draft', 'validated', 'canary', 'active', 'retired')),
    policy_metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(policy_metadata) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    validated_at TIMESTAMPTZ,
    canary_at TIMESTAMPTZ,
    activated_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    CONSTRAINT brain_decision_policies_version_lane_key UNIQUE (policy_version, lane),
    CONSTRAINT brain_decision_policies_times CHECK (
        (validated_at IS NULL OR validated_at >= created_at) AND
        (canary_at IS NULL OR canary_at >= created_at) AND
        (activated_at IS NULL OR activated_at >= created_at) AND
        (retired_at IS NULL OR retired_at >= created_at)
    )
);
CREATE UNIQUE INDEX brain_decision_policies_one_active_per_lane ON public.brain_decision_policies(lane) WHERE lifecycle = 'active';

CREATE TABLE public.brain_policy_artifacts (
    policy_version TEXT NOT NULL REFERENCES public.brain_decision_policies(policy_version),
    artifact_role TEXT NOT NULL CHECK (artifact_role IN (
        'qualification_model', 'preference_model', 'outcome_model', 'knowledge_graph',
        'label_snapshot', 'pairwise_snapshot', 'outcome_snapshot', 'config', 'metrics', 'replay'
    )),
    artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (policy_version, artifact_role)
);

CREATE TABLE public.brain_policy_approvals (
    approval_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_version TEXT NOT NULL REFERENCES public.brain_decision_policies(policy_version),
    approval_type TEXT NOT NULL CHECK (approval_type IN ('validated', 'canary', 'active', 'retired')),
    approved_by TEXT NOT NULL CHECK (btrim(approved_by) <> ''),
    artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    approved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    CONSTRAINT brain_policy_approvals_once UNIQUE (policy_version, approval_type)
);

CREATE TABLE public.brain_policy_gate_definitions (
    definition_version INTEGER NOT NULL CHECK (definition_version > 0),
    lane TEXT NOT NULL CHECK (lane IN ('ats', 'linkedin')),
    lifecycle TEXT NOT NULL CHECK (lifecycle IN ('validated', 'canary', 'active', 'retired')),
    gate_name TEXT NOT NULL CHECK (btrim(gate_name) <> ''),
    mandatory BOOLEAN NOT NULL DEFAULT TRUE CHECK (mandatory),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (definition_version, lane, lifecycle, gate_name)
);

INSERT INTO public.brain_policy_gate_definitions (definition_version, lane, lifecycle, gate_name)
SELECT 1, lane, lifecycle, gate_name
FROM (VALUES ('ats'), ('linkedin')) lanes(lane)
CROSS JOIN (VALUES
    ('validated', 'parity'), ('validated', 'replay'), ('validated', 'sample'),
    ('validated', 'positive_outcome'), ('validated', 'writer_freeze'),
    ('canary', 'parity'), ('canary', 'replay'), ('canary', 'sample'),
    ('canary', 'positive_outcome'), ('canary', 'writer_freeze'), ('canary', 'fleet_version'),
    ('active', 'parity'), ('active', 'replay'), ('active', 'sample'),
    ('active', 'positive_outcome'), ('active', 'writer_freeze'), ('active', 'fleet_version'),
    ('retired', 'retirement')
) gates(lifecycle, gate_name);

CREATE TABLE public.brain_policy_release_gate_events (
    gate_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_version TEXT NOT NULL REFERENCES public.brain_decision_policies(policy_version),
    lane TEXT NOT NULL CHECK (lane IN ('ats', 'linkedin')),
    lifecycle TEXT NOT NULL CHECK (lifecycle IN ('validated', 'canary', 'active', 'retired')),
    definition_version INTEGER NOT NULL DEFAULT 1,
    gate_name TEXT NOT NULL CHECK (btrim(gate_name) <> ''),
    gate_state TEXT NOT NULL CHECK (gate_state IN ('passed', 'failed', 'waived')),
    checked_by TEXT NOT NULL CHECK (btrim(checked_by) <> ''),
    report_artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    mismatch_count BIGINT NOT NULL DEFAULT 0 CHECK (mismatch_count >= 0),
    unresolved_count BIGINT NOT NULL DEFAULT 0 CHECK (unresolved_count >= 0),
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    CONSTRAINT brain_policy_release_gate_policy_fk FOREIGN KEY (policy_version, lane)
        REFERENCES public.brain_decision_policies(policy_version, lane),
    CONSTRAINT brain_policy_release_gate_definition_fk
        FOREIGN KEY (definition_version, lane, lifecycle, gate_name)
        REFERENCES public.brain_policy_gate_definitions(definition_version, lane, lifecycle, gate_name),
    CONSTRAINT brain_policy_release_gate_identity UNIQUE
        (policy_version, lane, lifecycle, definition_version, gate_name, gate_event_id, gate_state),
    CONSTRAINT brain_policy_release_gate_truth CHECK (
        gate_state <> 'passed' OR (mismatch_count = 0 AND unresolved_count = 0)
    )
);

CREATE TABLE public.brain_policy_transition_receipts (
    transition_receipt_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_version TEXT NOT NULL,
    lane TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    definition_version INTEGER NOT NULL,
    gate_name TEXT NOT NULL,
    gate_event_id BIGINT NOT NULL,
    gate_state TEXT NOT NULL DEFAULT 'passed' CHECK (gate_state = 'passed'),
    bound_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_policy_transition_receipts_once UNIQUE
        (policy_version, lifecycle, definition_version, gate_name),
    CONSTRAINT brain_policy_transition_receipts_gate_fk FOREIGN KEY
        (policy_version, lane, lifecycle, definition_version, gate_name, gate_event_id, gate_state)
        REFERENCES public.brain_policy_release_gate_events
        (policy_version, lane, lifecycle, definition_version, gate_name, gate_event_id, gate_state)
);

CREATE TABLE public.brain_policy_activation_receipts (
    activation_receipt_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_version TEXT NOT NULL UNIQUE REFERENCES public.brain_decision_policies(policy_version),
    lane TEXT NOT NULL CHECK (lane IN ('ats', 'linkedin')),
    prior_policy_version TEXT,
    pause_controls_before JSONB NOT NULL CHECK (jsonb_typeof(pause_controls_before) = 'object'),
    pause_controls_after JSONB NOT NULL CHECK (jsonb_typeof(pause_controls_after) = 'object'),
    invalidated_count BIGINT NOT NULL CHECK (invalidated_count >= 0),
    projected_count BIGINT NOT NULL CHECK (projected_count >= 0),
    activated_by TEXT NOT NULL,
    activated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_policy_activation_pause_unchanged CHECK (pause_controls_before = pause_controls_after)
);
CREATE TABLE public.brain_decision_identities (
    decision_id TEXT PRIMARY KEY CHECK (btrim(decision_id) <> ''),
    policy_version TEXT NOT NULL REFERENCES public.brain_decision_policies(policy_version),
    job_id TEXT NOT NULL REFERENCES public.brain_jobs(job_id),
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    source_decision_id TEXT NOT NULL CHECK (btrim(source_decision_id) <> ''),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_decision_identities_policy_key UNIQUE (policy_version, decision_id),
    CONSTRAINT brain_decision_identities_source_key UNIQUE (source_namespace, source_decision_id)
);

CREATE TABLE public.brain_job_decisions (
    decision_id TEXT NOT NULL CHECK (btrim(decision_id) <> ''),
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    source_decision_id TEXT NOT NULL CHECK (btrim(source_decision_id) <> ''),
    job_id TEXT NOT NULL REFERENCES public.brain_jobs(job_id),
    policy_version TEXT NOT NULL,
    lane TEXT NOT NULL CHECK (lane IN ('ats', 'linkedin')),
    qualification_score NUMERIC,
    qualification_floor NUMERIC,
    preference_score NUMERIC,
    outcome_score NUMERIC,
    final_score NUMERIC,
    qualification_verdict TEXT NOT NULL CHECK (qualification_verdict IN ('qualified', 'unqualified', 'uncertain')),
    action TEXT NOT NULL CHECK (action IN ('apply', 'review', 'reject')),
    confidence NUMERIC CHECK (confidence BETWEEN 0 AND 1),
    uncertainty JSONB NOT NULL DEFAULT '[]'::jsonb,
    blockers JSONB NOT NULL DEFAULT '[]'::jsonb,
    requirements JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_nodes JSONB NOT NULL DEFAULT '[]'::jsonb,
    title_signals JSONB NOT NULL DEFAULT '[]'::jsonb,
    explanation TEXT,
    uncertainty_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    blockers_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    requirements_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    evidence_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    decision_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    input_hash TEXT NOT NULL CHECK (input_hash ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    PRIMARY KEY (policy_version, decision_id),
    CONSTRAINT brain_job_decisions_identity_fk FOREIGN KEY (policy_version, decision_id)
        REFERENCES public.brain_decision_identities(policy_version, decision_id),
    CONSTRAINT brain_job_decisions_policy_lane_fk FOREIGN KEY (policy_version, lane)
        REFERENCES public.brain_decision_policies(policy_version, lane),
    CONSTRAINT brain_job_decisions_input_key UNIQUE (policy_version, job_id, input_hash),
    CONSTRAINT brain_job_decisions_policy_job_key UNIQUE (policy_version, job_id),
    CONSTRAINT brain_job_decisions_apply_expiry CHECK (
        (action = 'apply' AND expires_at IS NOT NULL AND expires_at > created_at) OR
        (action <> 'apply' AND (expires_at IS NULL OR expires_at > created_at))
    )
) PARTITION BY LIST (policy_version);
CREATE INDEX brain_job_decisions_job_created_idx ON public.brain_job_decisions(job_id, created_at DESC);
CREATE INDEX brain_job_decisions_policy_action_idx ON public.brain_job_decisions(policy_version, action);
CREATE UNIQUE INDEX brain_jobs_canonical_url_key ON public.brain_jobs(canonical_url) WHERE canonical_url IS NOT NULL;

CREATE TABLE public.brain_migration_sources (
    migration_source_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    source_fingerprint TEXT NOT NULL CHECK (source_fingerprint ~ '^[0-9a-f]{64}$'),
    byte_length BIGINT NOT NULL CHECK (byte_length >= 0),
    schema_metadata JSONB NOT NULL CHECK (jsonb_typeof(schema_metadata) = 'object'),
    source_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_migration_sources_fingerprint_key UNIQUE (source_namespace, source_fingerprint),
    CONSTRAINT brain_migration_sources_lineage_key UNIQUE (migration_source_id, source_namespace)
);
CREATE INDEX brain_migration_sources_created_idx ON public.brain_migration_sources(created_at);

CREATE TABLE public.brain_migration_runs (
    migration_run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    migration_source_id BIGINT NOT NULL,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    run_key TEXT NOT NULL CHECK (btrim(run_key) <> ''),
    run_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_migration_runs_source_key UNIQUE (source_namespace, migration_source_id, run_key),
    CONSTRAINT brain_migration_runs_source_fk FOREIGN KEY (migration_source_id, source_namespace)
        REFERENCES public.brain_migration_sources(migration_source_id, source_namespace),
    CONSTRAINT brain_migration_runs_lineage_key UNIQUE (migration_run_id, source_namespace)
);
CREATE INDEX brain_migration_runs_source_idx ON public.brain_migration_runs(migration_source_id, created_at);

CREATE TABLE public.brain_migration_run_events (
    migration_run_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    migration_run_id BIGINT NOT NULL,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    event_type TEXT NOT NULL CHECK (event_type IN ('planned', 'started', 'completed', 'failed', 'aborted')),
    actor_id TEXT NOT NULL CHECK (btrim(actor_id) <> ''),
    event_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    supersedes_run_event_id BIGINT REFERENCES public.brain_migration_run_events(migration_run_event_id),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_migration_run_events_one_successor UNIQUE (supersedes_run_event_id),
    CONSTRAINT brain_migration_run_events_run_fk FOREIGN KEY (migration_run_id, source_namespace)
        REFERENCES public.brain_migration_runs(migration_run_id, source_namespace),
    CONSTRAINT brain_migration_run_events_lineage_key UNIQUE
        (migration_run_id, source_namespace, migration_run_event_id, event_type)
);

CREATE TABLE public.brain_migration_batches (
    migration_batch_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    migration_run_id BIGINT NOT NULL,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    source_table TEXT NOT NULL CHECK (btrim(source_table) <> ''),
    batch_ordinal INTEGER NOT NULL CHECK (batch_ordinal > 0),
    key_start TEXT NOT NULL,
    key_end TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_migration_batches_range_key UNIQUE
        (migration_run_id, source_namespace, source_table, key_start, key_end),
    CONSTRAINT brain_migration_batches_ordinal_key UNIQUE
        (migration_run_id, source_namespace, source_table, batch_ordinal),
    CONSTRAINT brain_migration_batches_lineage_key UNIQUE
        (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal),
    CONSTRAINT brain_migration_batches_range CHECK (key_start <= key_end),
    CONSTRAINT brain_migration_batches_run_fk FOREIGN KEY (migration_run_id, source_namespace)
        REFERENCES public.brain_migration_runs(migration_run_id, source_namespace)
);
CREATE INDEX brain_migration_batches_range_idx ON public.brain_migration_batches
    (migration_run_id, source_namespace, source_table, batch_ordinal, key_start, key_end);

CREATE TABLE public.brain_migration_batch_events (
    migration_batch_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    migration_run_id BIGINT NOT NULL,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    source_table TEXT NOT NULL CHECK (btrim(source_table) <> ''),
    migration_batch_id BIGINT NOT NULL,
    batch_ordinal INTEGER NOT NULL CHECK (batch_ordinal > 0),
    event_type TEXT NOT NULL CHECK (event_type IN ('pending', 'claimed', 'completed', 'failed', 'quarantined')),
    attempt INTEGER NOT NULL CHECK (attempt >= 0),
    worker_id TEXT,
    lease_expires_at TIMESTAMPTZ,
    source_count BIGINT CHECK (source_count >= 0),
    target_count BIGINT CHECK (target_count >= 0),
    canonical_batch_hash TEXT CHECK (canonical_batch_hash ~ '^[0-9a-f]{64}$'),
    supersedes_batch_event_id BIGINT REFERENCES public.brain_migration_batch_events(migration_batch_event_id),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_migration_batch_events_batch_fk FOREIGN KEY
        (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal)
        REFERENCES public.brain_migration_batches
        (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal),
    CONSTRAINT brain_migration_batch_events_one_successor UNIQUE (supersedes_batch_event_id),
    CONSTRAINT brain_migration_batch_events_attempt_key UNIQUE (migration_batch_id, attempt, event_type),
    CONSTRAINT brain_migration_batch_events_lineage_key UNIQUE
        (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal,
         migration_batch_event_id, event_type),
    CONSTRAINT brain_migration_batch_events_claim CHECK (
        (event_type = 'claimed' AND worker_id IS NOT NULL AND lease_expires_at IS NOT NULL) OR event_type <> 'claimed'
    ),
    CONSTRAINT brain_migration_batch_events_complete CHECK (
        event_type <> 'completed' OR (
            source_count IS NOT NULL AND target_count IS NOT NULL AND source_count = target_count
            AND canonical_batch_hash IS NOT NULL
        )
    )
);
CREATE INDEX brain_migration_batch_events_claim_idx
    ON public.brain_migration_batch_events(event_type, lease_expires_at, migration_run_id, migration_batch_id);

CREATE TABLE public.brain_migration_checkpoints (
    migration_checkpoint_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    migration_run_id BIGINT NOT NULL,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    source_table TEXT NOT NULL CHECK (btrim(source_table) <> ''),
    batch_ordinal INTEGER NOT NULL CHECK (batch_ordinal > 0),
    last_key TEXT NOT NULL,
    migration_batch_id BIGINT NOT NULL,
    migration_batch_event_id BIGINT NOT NULL,
    committed_event_type TEXT NOT NULL DEFAULT 'completed' CHECK (committed_event_type = 'completed'),
    canonical_checkpoint_hash TEXT NOT NULL CHECK (canonical_checkpoint_hash ~ '^[0-9a-f]{64}$'),
    checkpointed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_migration_checkpoints_key UNIQUE
        (migration_run_id, source_namespace, source_table, batch_ordinal),
    CONSTRAINT brain_migration_checkpoints_completed_event_fk FOREIGN KEY
        (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal,
         migration_batch_event_id, committed_event_type)
        REFERENCES public.brain_migration_batch_events
        (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal,
         migration_batch_event_id, event_type)
);
CREATE INDEX brain_migration_checkpoints_run_idx ON public.brain_migration_checkpoints
    (migration_run_id, source_namespace, source_table, batch_ordinal);

CREATE TABLE public.brain_migration_quarantine (
    quarantine_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    migration_run_id BIGINT NOT NULL,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    migration_batch_id BIGINT,
    batch_ordinal INTEGER,
    source_table TEXT NOT NULL CHECK (btrim(source_table) <> ''),
    source_key TEXT NOT NULL CHECK (btrim(source_key) <> ''),
    reason_code TEXT NOT NULL CHECK (btrim(reason_code) <> ''),
    unresolved_evidence JSONB NOT NULL CHECK (jsonb_typeof(unresolved_evidence) = 'object'),
    conflict_artifact_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_migration_quarantine_key UNIQUE
        (migration_run_id, source_namespace, source_table, source_key, reason_code),
    CONSTRAINT brain_migration_quarantine_run_fk FOREIGN KEY (migration_run_id, source_namespace)
        REFERENCES public.brain_migration_runs(migration_run_id, source_namespace),
    CONSTRAINT brain_migration_quarantine_batch_fk FOREIGN KEY
        (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal)
        REFERENCES public.brain_migration_batches
        (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal)
);
CREATE INDEX brain_migration_quarantine_run_idx ON public.brain_migration_quarantine
    (migration_run_id, source_namespace, source_table);

CREATE TABLE public.brain_parity_definitions (
    definition_version INTEGER NOT NULL CHECK (definition_version > 0),
    check_key TEXT NOT NULL CHECK (btrim(check_key) <> ''),
    relation_name TEXT NOT NULL CHECK (btrim(relation_name) <> ''),
    check_type TEXT NOT NULL DEFAULT 'canonical' CHECK (check_type IN ('canonical', 'count_hash', 'membership')),
    authoritative BOOLEAN NOT NULL DEFAULT TRUE,
    mandatory BOOLEAN NOT NULL DEFAULT TRUE,
    hash_required BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (definition_version, check_key),
    CONSTRAINT brain_parity_definitions_binding_key UNIQUE
        (definition_version, check_key, relation_name, check_type),
    CONSTRAINT brain_parity_definitions_authoritative_mandatory CHECK
        (NOT authoritative OR (mandatory AND hash_required))
);

INSERT INTO public.brain_parity_definitions (definition_version, check_key, relation_name) VALUES
    (1, 'jobs', 'brain_jobs'),
    (1, 'aliases', 'brain_job_aliases'),
    (1, 'observations', 'brain_job_observations'),
    (1, 'labels', 'brain_label_events'),
    (1, 'pairwise', 'brain_pairwise_events'),
    (1, 'email', 'brain_email_events'),
    (1, 'outcomes', 'brain_reviewed_outcomes'),
    (1, 'applications', 'brain_applications'),
    (1, 'application_events', 'brain_application_events'),
    (1, 'policies', 'brain_decision_policies'),
    (1, 'decisions', 'brain_job_decisions'),
    (1, 'artifacts', 'brain_artifacts');

CREATE TABLE public.brain_parity_runs (
    parity_run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    migration_run_id BIGINT NOT NULL,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    definition_version INTEGER NOT NULL CHECK (definition_version > 0),
    completed_run_event_id BIGINT NOT NULL,
    completed_run_event_type TEXT NOT NULL DEFAULT 'completed' CHECK (completed_run_event_type = 'completed'),
    report_artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    final_delta_receipt_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    writer_freeze_receipt_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    started_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_parity_runs_run_fk FOREIGN KEY (migration_run_id, source_namespace)
        REFERENCES public.brain_migration_runs(migration_run_id, source_namespace),
    CONSTRAINT brain_parity_runs_completed_run_fk FOREIGN KEY
        (migration_run_id, source_namespace, completed_run_event_id, completed_run_event_type)
        REFERENCES public.brain_migration_run_events
        (migration_run_id, source_namespace, migration_run_event_id, event_type),
    CONSTRAINT brain_parity_runs_lineage_key UNIQUE (migration_run_id, source_namespace, parity_run_id),
    CONSTRAINT brain_parity_runs_result_binding_key UNIQUE
        (migration_run_id, source_namespace, parity_run_id, definition_version, report_artifact_hash)
);
CREATE INDEX brain_parity_runs_migration_idx ON public.brain_parity_runs(migration_run_id, started_at);

CREATE TABLE public.brain_parity_results (
    parity_result_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    migration_run_id BIGINT NOT NULL,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    parity_run_id BIGINT NOT NULL,
    definition_version INTEGER NOT NULL,
    check_key TEXT NOT NULL,
    table_name TEXT NOT NULL CHECK (btrim(table_name) <> ''),
    check_type TEXT NOT NULL CHECK (check_type IN ('canonical', 'count_hash', 'membership')),
    source_count BIGINT NOT NULL CHECK (source_count >= 0),
    target_count BIGINT NOT NULL CHECK (target_count >= 0),
    source_hash TEXT NOT NULL CHECK (source_hash ~ '^[0-9a-f]{64}$'),
    target_hash TEXT NOT NULL CHECK (target_hash ~ '^[0-9a-f]{64}$'),
    mismatch_count BIGINT NOT NULL DEFAULT 0 CHECK (mismatch_count >= 0),
    unresolved_count BIGINT NOT NULL DEFAULT 0 CHECK (unresolved_count >= 0),
    report_artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_parity_results_check_key UNIQUE (parity_run_id, definition_version, check_key),
    CONSTRAINT brain_parity_results_definition_fk
        FOREIGN KEY (definition_version, check_key, table_name, check_type)
        REFERENCES public.brain_parity_definitions(definition_version, check_key, relation_name, check_type),
    CONSTRAINT brain_parity_results_run_fk
        FOREIGN KEY (migration_run_id, source_namespace, parity_run_id, definition_version, report_artifact_hash)
        REFERENCES public.brain_parity_runs
        (migration_run_id, source_namespace, parity_run_id, definition_version, report_artifact_hash)
);
CREATE INDEX brain_parity_results_run_idx ON public.brain_parity_results(parity_run_id, table_name);

CREATE TABLE public.brain_parity_run_events (
    parity_run_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    migration_run_id BIGINT NOT NULL,
    source_namespace TEXT NOT NULL CHECK (btrim(source_namespace) <> ''),
    parity_run_id BIGINT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('passed', 'failed')),
    actor_id TEXT NOT NULL CHECK (btrim(actor_id) <> ''),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_parity_run_events_run_fk FOREIGN KEY (migration_run_id, source_namespace, parity_run_id)
        REFERENCES public.brain_parity_runs(migration_run_id, source_namespace, parity_run_id),
    CONSTRAINT brain_parity_run_events_once UNIQUE (parity_run_id, event_type)
);

CREATE SCHEMA brain_archive;
CREATE TABLE brain_archive.brain_archive_manifests (
    archive_manifest_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    retry_identity TEXT NOT NULL UNIQUE CHECK (btrim(retry_identity) <> ''),
    source_relation TEXT NOT NULL CHECK (btrim(source_relation) <> ''),
    source_partition TEXT,
    artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    row_count BIGINT NOT NULL CHECK (row_count >= 0),
    archived_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    retain_until TIMESTAMPTZ,
    manifest_metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(manifest_metadata) = 'object')
);

CREATE FUNCTION public.brain_reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is append-only', TG_TABLE_NAME USING ERRCODE = '55000';
END;
$$;

CREATE FUNCTION public.brain_register_decision() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO public.brain_decision_identities
        (decision_id, policy_version, job_id, source_namespace, source_decision_id, created_at)
    VALUES (NEW.decision_id, NEW.policy_version, NEW.job_id, NEW.source_namespace, NEW.source_decision_id, NEW.created_at);
    RETURN NEW;
END;
$$;

CREATE FUNCTION public.brain_require_controller() RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    IF current_user <> 'brain_schema_migrator'
       OR NOT EXISTS (
           WITH RECURSIVE memberships(roleid) AS (
               SELECT oid FROM pg_roles WHERE rolname = session_user
               UNION
               SELECT member.roleid FROM pg_auth_members member
               JOIN memberships prior ON prior.roleid = member.member
           )
           SELECT 1 FROM memberships
           WHERE roleid = (SELECT oid FROM pg_roles WHERE rolname = 'brain_schema_migrator')
       ) THEN
        RAISE EXCEPTION 'brain controller role is required' USING ERRCODE = '42501';
    END IF;
END;
$$;

CREATE FUNCTION public.brain_artifact_is_authoritative(candidate_hash TEXT) RETURNS BOOLEAN
LANGUAGE sql STABLE SET search_path = pg_catalog, public AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.brain_artifact_locations location
        WHERE location.artifact_hash = candidate_hash
          AND location.durability = 'verified'
          AND location.storage_immutable
          AND location.provider_version_id IS NOT NULL
          AND location.provider_checksum IS NOT NULL
          AND location.encryption_mode <> 'none'
          AND location.verified_at IS NOT NULL
    );
$$;

CREATE FUNCTION public.brain_check_controller_insert() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    PERFORM public.brain_require_controller();
    RETURN NEW;
END;
$$;

CREATE FUNCTION public.brain_check_migration_run_event() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE head public.brain_migration_run_events%ROWTYPE;
BEGIN
    PERFORM public.brain_require_controller();
    PERFORM pg_advisory_xact_lock(hashtextextended('brain-migration-run', NEW.migration_run_id));
    SELECT * INTO head FROM public.brain_migration_run_events
    WHERE migration_run_id=NEW.migration_run_id ORDER BY migration_run_event_id DESC LIMIT 1 FOR UPDATE;
    IF head.migration_run_event_id IS NULL THEN
        IF NEW.event_type NOT IN ('planned', 'started') OR NEW.supersedes_run_event_id IS NOT NULL THEN
            RAISE EXCEPTION 'illegal initial migration run event' USING ERRCODE='23514';
        END IF;
    ELSE
        IF NEW.supersedes_run_event_id IS DISTINCT FROM head.migration_run_event_id THEN
            RAISE EXCEPTION 'migration run event must supersede current head' USING ERRCODE='23514';
        END IF;
        IF NOT ((head.event_type='planned' AND NEW.event_type='started')
             OR (head.event_type='started' AND NEW.event_type IN ('completed','failed','aborted'))
             OR (head.event_type='failed' AND NEW.event_type='started')) THEN
            RAISE EXCEPTION 'illegal migration run transition % -> %', head.event_type, NEW.event_type USING ERRCODE='23514';
        END IF;
    END IF;
    IF NEW.event_type='completed' AND (
        NOT EXISTS (SELECT 1 FROM public.brain_migration_batches WHERE migration_run_id=NEW.migration_run_id)
        OR EXISTS (
            SELECT 1 FROM public.brain_migration_batches batch
            WHERE batch.migration_run_id=NEW.migration_run_id AND NOT EXISTS (
                SELECT 1 FROM public.brain_migration_batch_events event
                WHERE event.migration_batch_id=batch.migration_batch_id AND event.event_type='completed'
                  AND NOT EXISTS (SELECT 1 FROM public.brain_migration_batch_events newer
                                  WHERE newer.supersedes_batch_event_id=event.migration_batch_event_id)
            )
        ) OR EXISTS (
            SELECT 1 FROM public.brain_migration_batches batch
            WHERE batch.migration_run_id=NEW.migration_run_id AND NOT EXISTS (
                SELECT 1 FROM public.brain_migration_checkpoints checkpoint
                WHERE checkpoint.migration_batch_id=batch.migration_batch_id
            )
        )
    ) THEN
        RAISE EXCEPTION 'run completion requires all batches completed and checkpointed' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION public.brain_check_migration_batch_definition() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE prior public.brain_migration_batches%ROWTYPE;
BEGIN
    PERFORM public.brain_require_controller();
    IF NOT EXISTS (
        SELECT 1 FROM public.brain_migration_run_events event
        WHERE event.migration_run_id=NEW.migration_run_id AND event.event_type='started'
          AND NOT EXISTS (SELECT 1 FROM public.brain_migration_run_events newer
                          WHERE newer.supersedes_run_event_id=event.migration_run_event_id)
    ) THEN RAISE EXCEPTION 'batches require a started run head' USING ERRCODE='23514'; END IF;
    SELECT * INTO prior FROM public.brain_migration_batches
    WHERE migration_run_id=NEW.migration_run_id AND source_namespace=NEW.source_namespace
      AND source_table=NEW.source_table ORDER BY batch_ordinal DESC LIMIT 1 FOR UPDATE;
    IF NEW.batch_ordinal <> COALESCE(prior.batch_ordinal,0)+1
       OR (prior.migration_batch_id IS NOT NULL AND prior.key_end >= NEW.key_start) THEN
        RAISE EXCEPTION 'batch definitions must be contiguous ordered non-overlapping ranges' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION public.brain_check_migration_batch_event() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE head public.brain_migration_batch_events%ROWTYPE;
BEGIN
    PERFORM public.brain_require_controller();
    PERFORM pg_advisory_xact_lock(hashtextextended('brain-migration-batch', NEW.migration_batch_id));
    SELECT * INTO head FROM public.brain_migration_batch_events
    WHERE migration_batch_id=NEW.migration_batch_id ORDER BY migration_batch_event_id DESC LIMIT 1 FOR UPDATE;
    IF head.migration_batch_event_id IS NULL THEN
        IF NEW.event_type <> 'pending' OR NEW.attempt <> 0 OR NEW.supersedes_batch_event_id IS NOT NULL THEN
            RAISE EXCEPTION 'batch event history must start pending attempt zero' USING ERRCODE='23514';
        END IF;
    ELSE
        IF NEW.supersedes_batch_event_id IS DISTINCT FROM head.migration_batch_event_id THEN
            RAISE EXCEPTION 'batch event must supersede current head' USING ERRCODE='23514';
        END IF;
        IF NEW.event_type='claimed' THEN
            IF NOT (head.event_type IN ('pending','failed','quarantined')
                    OR (head.event_type='claimed' AND head.lease_expires_at <= now()))
               OR NEW.attempt <> (CASE WHEN head.event_type='pending' THEN 1 ELSE head.attempt+1 END) THEN
                RAISE EXCEPTION 'claim requires claimable current head and monotonic attempt' USING ERRCODE='23514';
            END IF;
        ELSIF NEW.event_type IN ('completed','failed','quarantined') THEN
            IF head.event_type <> 'claimed' OR head.lease_expires_at <= now()
               OR NEW.worker_id IS DISTINCT FROM head.worker_id OR NEW.attempt <> head.attempt THEN
                RAISE EXCEPTION 'batch terminal event requires current live claim and worker' USING ERRCODE='23514';
            END IF;
        ELSE
            RAISE EXCEPTION 'illegal batch transition' USING ERRCODE='23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION public.brain_check_migration_checkpoint() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE batch public.brain_migration_batches%ROWTYPE; prior_ordinal INTEGER;
BEGIN
    PERFORM public.brain_require_controller();
    SELECT * INTO batch FROM public.brain_migration_batches
    WHERE migration_batch_id=NEW.migration_batch_id FOR UPDATE;
    SELECT max(batch_ordinal) INTO prior_ordinal FROM public.brain_migration_checkpoints
    WHERE migration_run_id=NEW.migration_run_id AND source_namespace=NEW.source_namespace
      AND source_table=NEW.source_table;
    IF NEW.batch_ordinal <> COALESCE(prior_ordinal,0)+1
       OR NEW.batch_ordinal <> batch.batch_ordinal
       OR NEW.last_key <> batch.key_end OR NEW.last_key < batch.key_start
       OR NOT EXISTS (
           SELECT 1 FROM public.brain_migration_batch_events event
           WHERE event.migration_batch_event_id=NEW.migration_batch_event_id
             AND event.migration_batch_id=NEW.migration_batch_id AND event.event_type='completed'
             AND NOT EXISTS (SELECT 1 FROM public.brain_migration_batch_events newer
                             WHERE newer.supersedes_batch_event_id=event.migration_batch_event_id)
       ) THEN
        RAISE EXCEPTION 'checkpoint requires contiguous bounded batch completion head' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION public.brain_check_policy_lifecycle() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE required_roles CONSTANT TEXT[] := ARRAY[
    'qualification_model','preference_model','outcome_model','knowledge_graph','label_snapshot',
    'pairwise_snapshot','outcome_snapshot','config','metrics','replay'];
BEGIN
    IF TG_OP='INSERT' THEN
        IF NEW.lifecycle <> 'draft' THEN RAISE EXCEPTION 'new policy must start in draft' USING ERRCODE='23514'; END IF;
        RETURN NEW;
    END IF;
    IF OLD.lifecycle <> 'draft' AND (
        NEW.lane IS DISTINCT FROM OLD.lane
        OR NEW.policy_metadata IS DISTINCT FROM OLD.policy_metadata
        OR NEW.gate_definition_version IS DISTINCT FROM OLD.gate_definition_version
    ) THEN
        RAISE EXCEPTION 'policy lane, gate definition, and metadata are immutable after draft' USING ERRCODE='23514';
    END IF;
    IF NEW.lifecycle IS DISTINCT FROM OLD.lifecycle THEN
        PERFORM public.brain_require_controller();
        IF current_setting('applypilot.policy_transition',true) IS DISTINCT FROM 'locked' THEN
            RAISE EXCEPTION 'policy lifecycle changes require locked controller function' USING ERRCODE='42501';
        END IF;
        IF NOT ((OLD.lifecycle='draft' AND NEW.lifecycle='validated')
             OR (OLD.lifecycle='validated' AND NEW.lifecycle='canary')
             OR (OLD.lifecycle='canary' AND NEW.lifecycle='active')
             OR (OLD.lifecycle='active' AND NEW.lifecycle='retired')) THEN
            RAISE EXCEPTION 'illegal policy lifecycle transition' USING ERRCODE='23514';
        END IF;
        IF (SELECT count(*) FROM public.brain_policy_transition_receipts
            WHERE policy_version=NEW.policy_version AND lifecycle=NEW.lifecycle
              AND definition_version=NEW.gate_definition_version)
           <> (SELECT count(*) FROM public.brain_policy_gate_definitions
               WHERE definition_version=NEW.gate_definition_version
                 AND lane=NEW.lane AND lifecycle=NEW.lifecycle AND mandatory) THEN
            RAISE EXCEPTION 'policy transition requires complete mandatory locked gate receipts' USING ERRCODE='23514';
        END IF;
        IF NEW.lifecycle IN ('validated','canary','active') AND (
            (SELECT count(*) FROM public.brain_policy_artifacts
             WHERE policy_version=NEW.policy_version AND artifact_role=ANY(required_roles)) <> cardinality(required_roles)
            OR NOT EXISTS (SELECT 1 FROM public.brain_policy_approvals
                           WHERE policy_version=NEW.policy_version AND approval_type=NEW.lifecycle)
        ) THEN RAISE EXCEPTION 'policy transition requires complete artifacts and approval' USING ERRCODE='23514'; END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION public.brain_bind_policy_gates(
    candidate TEXT, candidate_lane TEXT, target TEXT, candidate_definition_version INTEGER
)
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE definition RECORD; latest public.brain_policy_release_gate_events%ROWTYPE;
BEGIN
    FOR definition IN SELECT * FROM public.brain_policy_gate_definitions
        WHERE definition_version=candidate_definition_version
          AND lane=candidate_lane AND lifecycle=target AND mandatory ORDER BY gate_name
    LOOP
        SELECT * INTO latest FROM public.brain_policy_release_gate_events
        WHERE policy_version=candidate AND lane=candidate_lane AND lifecycle=target
          AND definition_version=candidate_definition_version
          AND gate_name=definition.gate_name ORDER BY gate_event_id DESC LIMIT 1 FOR UPDATE;
        IF latest.gate_event_id IS NULL OR latest.gate_state <> 'passed'
           OR latest.mismatch_count <> 0 OR latest.unresolved_count <> 0
           OR NOT public.brain_artifact_is_authoritative(latest.report_artifact_hash) THEN
            RAISE EXCEPTION 'latest mandatory gate % is not a protected clean pass', definition.gate_name USING ERRCODE='23514';
        END IF;
        INSERT INTO public.brain_policy_transition_receipts
            (policy_version,lane,lifecycle,definition_version,gate_name,gate_event_id)
        VALUES (candidate,candidate_lane,target,candidate_definition_version,
                definition.gate_name,latest.gate_event_id);
    END LOOP;
END;
$$;

CREATE FUNCTION public.brain_transition_policy(requested_policy_version TEXT, requested_lifecycle TEXT)
RETURNS VOID LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE policy_row public.brain_decision_policies%ROWTYPE; prior_policy public.brain_decision_policies%ROWTYPE;
        fleet_active TEXT; queue_name TEXT; bound_policy TEXT; controls_before JSONB; controls_after JSONB;
        invalidated BIGINT:=0; projected BIGINT:=0;
BEGIN
    PERFORM public.brain_require_controller();
    SELECT * INTO policy_row FROM public.brain_decision_policies
    WHERE policy_version=requested_policy_version FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'unknown policy_version' USING ERRCODE='P0002'; END IF;
    PERFORM pg_advisory_xact_lock(hashtext('brain-policy-lane'),hashtext(policy_row.lane));
    PERFORM public.brain_bind_policy_gates(
        requested_policy_version,policy_row.lane,requested_lifecycle,policy_row.gate_definition_version
    );
    PERFORM set_config('applypilot.policy_transition','locked',true);
    IF requested_lifecycle NOT IN ('active','retired') THEN
        UPDATE public.brain_decision_policies SET lifecycle=requested_lifecycle,
            validated_at=CASE WHEN requested_lifecycle='validated' THEN now() ELSE validated_at END,
            canary_at=CASE WHEN requested_lifecycle='canary' THEN now() ELSE canary_at END
        WHERE policy_version=requested_policy_version;
        RETURN;
    END IF;
    IF to_regclass('public.fleet_config') IS NULL OR to_regclass('public.fleet_decision_policies') IS NULL
       OR to_regclass('public.apply_queue') IS NULL OR to_regclass('public.linkedin_queue') IS NULL THEN
        RAISE EXCEPTION 'atomic active/retired transition requires fleet config, policies, and both queues'
            USING ERRCODE='55000';
    END IF;
    queue_name:=CASE WHEN policy_row.lane='ats' THEN 'apply_queue' ELSE 'linkedin_queue' END;
    EXECUTE format('LOCK TABLE public.%I IN SHARE ROW EXCLUSIVE MODE',queue_name);
    PERFORM 1 FROM public.fleet_decision_policies WHERE lane=policy_row.lane FOR UPDATE;
    SELECT jsonb_build_object('paused',paused,'ats_paused',ats_paused,'ats_apply_mode',ats_apply_mode,
               'linkedin_apply_mode',linkedin_apply_mode,'canary_enabled',canary_enabled,
               'linkedin_canary_enabled',linkedin_canary_enabled),
           CASE WHEN policy_row.lane='ats' THEN ats_policy_version ELSE linkedin_policy_version END
    INTO controls_before,bound_policy FROM public.fleet_config WHERE id=1 FOR UPDATE;
    IF controls_before IS NULL THEN RAISE EXCEPTION 'fleet_config singleton is missing' USING ERRCODE='55000'; END IF;
    IF NOT COALESCE((controls_before->>'paused')::boolean,false)
       AND NOT (policy_row.lane='ats' AND COALESCE((controls_before->>'ats_paused')::boolean,false)) THEN
        RAISE EXCEPTION 'lane must remain paused during active/retired transition' USING ERRCODE='55000';
    END IF;
    SELECT policy_version INTO fleet_active FROM public.fleet_decision_policies
    WHERE lane=policy_row.lane AND status='active';
    IF requested_lifecycle='retired' THEN
        IF policy_row.lifecycle<>'active' OR fleet_active IS DISTINCT FROM requested_policy_version
           OR bound_policy IS DISTINCT FROM requested_policy_version THEN
            RAISE EXCEPTION 'retirement requires matching active brain, fleet, and config binding'
                USING ERRCODE='55000';
        END IF;
        EXECUTE format('UPDATE public.%I SET approved_batch=NULL,decision_id=NULL,policy_version=NULL,
            decision_action=NULL,qualification_verdict=NULL,qualification_score=NULL,qualification_floor=NULL,
            preference_score=NULL,outcome_score=NULL,final_score=NULL,decision_confidence=NULL,
            decision_created_at=NULL,decision_expires_at=NULL,input_hash=NULL
            WHERE lane=$1 AND status=''queued'' AND lease_owner IS NULL AND lease_expires_at IS NULL
              AND worker_lease_id IS NULL AND policy_version=$2',queue_name)
            USING policy_row.lane,requested_policy_version;
        GET DIAGNOSTICS invalidated=ROW_COUNT;
        UPDATE public.fleet_decision_policies SET status='retired',retired_at=now()
        WHERE policy_version=requested_policy_version AND lane=policy_row.lane AND status='active';
        IF policy_row.lane='ats' THEN
            UPDATE public.fleet_config SET ats_policy_version=NULL WHERE id=1;
        ELSE
            UPDATE public.fleet_config SET linkedin_policy_version=NULL WHERE id=1;
        END IF;
        UPDATE public.brain_decision_policies SET lifecycle='retired',retired_at=now()
        WHERE policy_version=requested_policy_version;
    ELSE
        SELECT * INTO prior_policy FROM public.brain_decision_policies
        WHERE lane=policy_row.lane AND lifecycle='active' FOR UPDATE;
        IF fleet_active IS DISTINCT FROM prior_policy.policy_version
           OR bound_policy IS DISTINCT FROM prior_policy.policy_version THEN
            RAISE EXCEPTION 'brain, fleet, and config active policy bindings disagree' USING ERRCODE='55000';
        END IF;
        IF prior_policy.policy_version IS NOT NULL THEN
            PERFORM public.brain_bind_policy_gates(
                prior_policy.policy_version,policy_row.lane,'retired',prior_policy.gate_definition_version
            );
            UPDATE public.brain_decision_policies SET lifecycle='retired',retired_at=now()
            WHERE policy_version=prior_policy.policy_version;
            UPDATE public.fleet_decision_policies SET status='retired',retired_at=now()
            WHERE policy_version=prior_policy.policy_version;
        END IF;
        UPDATE public.brain_decision_policies SET lifecycle='active',activated_at=now()
        WHERE policy_version=requested_policy_version;
        INSERT INTO public.fleet_decision_policies(policy_version,lane,status,activated_at)
        VALUES(requested_policy_version,policy_row.lane,'active',now())
        ON CONFLICT(policy_version) DO UPDATE SET status='active',activated_at=EXCLUDED.activated_at;
        IF policy_row.lane='ats' THEN
            UPDATE public.fleet_config SET ats_policy_version=requested_policy_version WHERE id=1;
        ELSE
            UPDATE public.fleet_config SET linkedin_policy_version=requested_policy_version WHERE id=1;
        END IF;
        EXECUTE format('UPDATE public.%I SET approved_batch=NULL,decision_id=NULL,policy_version=NULL,
            decision_action=NULL,qualification_verdict=NULL,qualification_score=NULL,qualification_floor=NULL,
            preference_score=NULL,outcome_score=NULL,final_score=NULL,decision_confidence=NULL,
            decision_created_at=NULL,decision_expires_at=NULL,input_hash=NULL
            WHERE lane=$1 AND status=''queued'' AND lease_owner IS NULL AND lease_expires_at IS NULL
              AND worker_lease_id IS NULL',queue_name) USING policy_row.lane;
        GET DIAGNOSTICS invalidated=ROW_COUNT;
        EXECUTE format('UPDATE public.%I q SET approved_batch=$1||'':activation'',decision_id=d.decision_id,
            policy_version=d.policy_version,decision_action=d.action,qualification_verdict=d.qualification_verdict,
            qualification_score=d.qualification_score::real,qualification_floor=d.qualification_floor::real,
            preference_score=d.preference_score::real,outcome_score=d.outcome_score::real,final_score=d.final_score::real,
            score=d.final_score::real,decision_confidence=d.confidence::real,decision_created_at=d.created_at,
            decision_expires_at=d.expires_at,input_hash=d.input_hash
            FROM public.brain_job_decisions d JOIN public.brain_jobs j ON j.job_id=d.job_id
            WHERE q.url=j.canonical_url AND q.lane=$2 AND q.status=''queued'' AND q.lease_owner IS NULL
              AND q.lease_expires_at IS NULL AND q.worker_lease_id IS NULL AND d.policy_version=$1
              AND d.action=''apply'' AND d.qualification_verdict=''qualified'' AND d.expires_at>now()',queue_name)
            USING requested_policy_version,policy_row.lane;
        GET DIAGNOSTICS projected=ROW_COUNT;
        INSERT INTO public.brain_policy_activation_receipts
            (policy_version,lane,prior_policy_version,pause_controls_before,pause_controls_after,
             invalidated_count,projected_count,activated_by)
        VALUES(requested_policy_version,policy_row.lane,prior_policy.policy_version,controls_before,controls_before,
               invalidated,projected,current_user);
    END IF;
    SELECT jsonb_build_object('paused',paused,'ats_paused',ats_paused,'ats_apply_mode',ats_apply_mode,
               'linkedin_apply_mode',linkedin_apply_mode,'canary_enabled',canary_enabled,
               'linkedin_canary_enabled',linkedin_canary_enabled)
    INTO controls_after FROM public.fleet_config WHERE id=1;
    IF controls_after IS DISTINCT FROM controls_before THEN
        RAISE EXCEPTION 'pause controls changed during policy transition' USING ERRCODE='55000';
    END IF;
END;
$$;

CREATE FUNCTION public.brain_check_parity_pass() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE run public.brain_parity_runs%ROWTYPE; mandatory_count INTEGER; result_count INTEGER;
BEGIN
    PERFORM public.brain_require_controller();
    PERFORM pg_advisory_xact_lock(hashtextextended('brain-parity-pass', NEW.parity_run_id));
    SELECT * INTO run FROM public.brain_parity_runs WHERE parity_run_id=NEW.parity_run_id FOR UPDATE;
    IF NEW.event_type='passed' THEN
        SELECT count(*) INTO mandatory_count FROM public.brain_parity_definitions
        WHERE definition_version=run.definition_version AND authoritative AND mandatory;
        SELECT count(*) INTO result_count FROM public.brain_parity_results result
        JOIN public.brain_parity_definitions definition
          ON definition.definition_version=result.definition_version AND definition.check_key=result.check_key
        WHERE result.parity_run_id=NEW.parity_run_id AND definition.authoritative AND definition.mandatory
          AND result.source_count=result.target_count AND result.source_hash=result.target_hash
          AND result.mismatch_count=0 AND result.unresolved_count=0;
        IF mandatory_count=0 OR result_count<>mandatory_count
           OR EXISTS (SELECT 1 FROM public.brain_migration_quarantine WHERE migration_run_id=run.migration_run_id)
           OR NOT public.brain_artifact_is_authoritative(run.report_artifact_hash)
           OR NOT public.brain_artifact_is_authoritative(run.final_delta_receipt_hash)
           OR NOT public.brain_artifact_is_authoritative(run.writer_freeze_receipt_hash)
           OR EXISTS (SELECT 1 FROM public.brain_migration_batches batch
                      WHERE batch.migration_run_id=run.migration_run_id AND NOT EXISTS
                        (SELECT 1 FROM public.brain_migration_checkpoints checkpoint
                         WHERE checkpoint.migration_batch_id=batch.migration_batch_id)) THEN
            RAISE EXCEPTION 'parity pass requires full authoritative clean completed coverage and protected receipts'
                USING ERRCODE='23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION public.brain_check_parity_result() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NOT public.brain_artifact_is_authoritative(NEW.report_artifact_hash) THEN
        RAISE EXCEPTION 'parity result requires protected verified report artifact' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION public.brain_check_archive_manifest() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NOT public.brain_artifact_is_authoritative(NEW.artifact_hash) THEN
        RAISE EXCEPTION 'authoritative manifest requires protected verified immutable replica' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION public.brain_check_supersession() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE predecessor JSONB; candidate JSONB := to_jsonb(NEW); predecessor_id TEXT;
BEGIN
    predecessor_id := candidate ->> TG_ARGV[1];
    IF predecessor_id IS NULL THEN RETURN NEW; END IF;
    EXECUTE format('SELECT to_jsonb(t) FROM public.%I t WHERE %I::text = $1', TG_TABLE_NAME, TG_ARGV[0])
        INTO predecessor USING predecessor_id;
    IF predecessor IS NULL THEN RETURN NEW; END IF;
    IF EXISTS (
        SELECT 1 FROM unnest(string_to_array(TG_ARGV[2], ',')) AS key
        WHERE predecessor ->> key IS DISTINCT FROM candidate ->> key
    ) THEN
        RAISE EXCEPTION 'supersession logical subject/source mismatch' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER brain_job_decisions_register BEFORE INSERT ON public.brain_job_decisions FOR EACH ROW EXECUTE FUNCTION public.brain_register_decision();
CREATE TRIGGER brain_decision_policies_lifecycle BEFORE INSERT OR UPDATE ON public.brain_decision_policies FOR EACH ROW EXECUTE FUNCTION public.brain_check_policy_lifecycle();
CREATE TRIGGER brain_parity_run_events_pass BEFORE INSERT ON public.brain_parity_run_events FOR EACH ROW EXECUTE FUNCTION public.brain_check_parity_pass();
CREATE TRIGGER brain_parity_results_guard BEFORE INSERT ON public.brain_parity_results FOR EACH ROW EXECUTE FUNCTION public.brain_check_parity_result();
CREATE TRIGGER brain_archive_manifests_guard BEFORE INSERT ON brain_archive.brain_archive_manifests FOR EACH ROW EXECUTE FUNCTION public.brain_check_archive_manifest();
CREATE TRIGGER brain_migration_sources_controller BEFORE INSERT ON public.brain_migration_sources FOR EACH ROW EXECUTE FUNCTION public.brain_check_controller_insert();
CREATE TRIGGER brain_migration_runs_controller BEFORE INSERT ON public.brain_migration_runs FOR EACH ROW EXECUTE FUNCTION public.brain_check_controller_insert();
CREATE TRIGGER brain_migration_batches_definition BEFORE INSERT ON public.brain_migration_batches FOR EACH ROW EXECUTE FUNCTION public.brain_check_migration_batch_definition();
CREATE TRIGGER brain_migration_run_events_transition BEFORE INSERT ON public.brain_migration_run_events FOR EACH ROW EXECUTE FUNCTION public.brain_check_migration_run_event();
CREATE TRIGGER brain_migration_batch_events_transition BEFORE INSERT ON public.brain_migration_batch_events FOR EACH ROW EXECUTE FUNCTION public.brain_check_migration_batch_event();
CREATE TRIGGER brain_migration_checkpoints_transition BEFORE INSERT ON public.brain_migration_checkpoints FOR EACH ROW EXECUTE FUNCTION public.brain_check_migration_checkpoint();

DO $triggers$
DECLARE table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'brain_schema_versions', 'brain_artifacts', 'brain_artifact_locations', 'brain_job_aliases', 'brain_job_observations',
        'brain_label_events', 'brain_pairwise_events', 'brain_email_events', 'brain_reviewed_outcomes',
        'brain_application_events', 'brain_policy_artifacts', 'brain_policy_approvals',
        'brain_policy_gate_definitions', 'brain_policy_release_gate_events',
        'brain_policy_transition_receipts', 'brain_policy_activation_receipts',
        'brain_decision_identities', 'brain_job_decisions',
        'brain_migration_sources', 'brain_migration_runs', 'brain_migration_run_events',
        'brain_migration_batches', 'brain_migration_batch_events', 'brain_migration_checkpoints',
        'brain_migration_quarantine', 'brain_parity_definitions', 'brain_parity_runs',
        'brain_parity_results', 'brain_parity_run_events'
    ] LOOP
        EXECUTE format('CREATE TRIGGER %I_append_only BEFORE UPDATE OR DELETE ON public.%I FOR EACH ROW EXECUTE FUNCTION public.brain_reject_mutation()', table_name, table_name);
        EXECUTE format('CREATE TRIGGER %I_append_only_truncate BEFORE TRUNCATE ON public.%I FOR EACH STATEMENT EXECUTE FUNCTION public.brain_reject_mutation()', table_name, table_name);
    END LOOP;
    CREATE TRIGGER brain_archive_manifests_append_only BEFORE UPDATE OR DELETE ON brain_archive.brain_archive_manifests FOR EACH ROW EXECUTE FUNCTION public.brain_reject_mutation();
    CREATE TRIGGER brain_archive_manifests_append_only_truncate BEFORE TRUNCATE ON brain_archive.brain_archive_manifests FOR EACH STATEMENT EXECUTE FUNCTION public.brain_reject_mutation();
    CREATE TRIGGER brain_job_observations_supersession AFTER INSERT ON public.brain_job_observations FOR EACH ROW EXECUTE FUNCTION public.brain_check_supersession('observation_id', 'supersedes_observation_id', 'source_namespace,logical_subject_id');
    CREATE TRIGGER brain_label_events_supersession AFTER INSERT ON public.brain_label_events FOR EACH ROW EXECUTE FUNCTION public.brain_check_supersession('label_event_id', 'supersedes_label_event_id', 'source_namespace,logical_subject_id');
    CREATE TRIGGER brain_pairwise_events_supersession AFTER INSERT ON public.brain_pairwise_events FOR EACH ROW EXECUTE FUNCTION public.brain_check_supersession('pairwise_event_id', 'supersedes_pairwise_event_id', 'source_namespace,logical_subject_id');
    CREATE TRIGGER brain_email_events_supersession AFTER INSERT ON public.brain_email_events FOR EACH ROW EXECUTE FUNCTION public.brain_check_supersession('email_event_id', 'supersedes_email_event_id', 'source_namespace,logical_subject_id');
    CREATE TRIGGER brain_reviewed_outcomes_supersession AFTER INSERT ON public.brain_reviewed_outcomes FOR EACH ROW EXECUTE FUNCTION public.brain_check_supersession('reviewed_outcome_id', 'supersedes_reviewed_outcome_id', 'source_namespace,logical_subject_id');
    CREATE TRIGGER brain_application_events_supersession AFTER INSERT ON public.brain_application_events FOR EACH ROW EXECUTE FUNCTION public.brain_check_supersession('application_event_id', 'supersedes_application_event_id', 'source_namespace,logical_subject_id');
    CREATE TRIGGER brain_migration_run_events_supersession AFTER INSERT ON public.brain_migration_run_events FOR EACH ROW EXECUTE FUNCTION public.brain_check_supersession('migration_run_event_id', 'supersedes_run_event_id', 'migration_run_id,source_namespace');
    CREATE TRIGGER brain_migration_batch_events_supersession AFTER INSERT ON public.brain_migration_batch_events FOR EACH ROW EXECUTE FUNCTION public.brain_check_supersession('migration_batch_event_id', 'supersedes_batch_event_id', 'migration_run_id,source_namespace,source_table,migration_batch_id,batch_ordinal');
END;
$triggers$;

REVOKE ALL ON SCHEMA brain_archive FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA brain_archive FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA brain_archive FROM PUBLIC;
DO $acl_cleanup$
DECLARE rec RECORD; object_kind TEXT;
BEGIN
    FOR rec IN
        SELECT n.nspname, c.relname, c.relkind
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND left(c.relname, 6) = 'brain_'
          AND c.relkind IN ('r', 'p', 'S')
    LOOP
        object_kind := CASE WHEN rec.relkind = 'S' THEN 'SEQUENCE' ELSE 'TABLE' END;
        EXECUTE format('REVOKE ALL PRIVILEGES ON %s %I.%I FROM PUBLIC', object_kind, rec.nspname, rec.relname);
    END LOOP;
    FOR rec IN
        SELECT n.nspname, c.relname, c.relkind, owner.rolname AS owner_name, grantee.rolname AS grantee_name
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_roles owner ON owner.oid = c.relowner
        CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl, acldefault(CASE WHEN c.relkind = 'S' THEN 'S'::"char" ELSE 'r'::"char" END, c.relowner))) acl
        LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
        WHERE ((n.nspname = 'public' AND left(c.relname, 6) = 'brain_') OR n.nspname = 'brain_archive')
          AND acl.grantee <> c.relowner AND acl.grantee <> 0
    LOOP
        object_kind := CASE WHEN rec.relkind = 'S' THEN 'SEQUENCE' ELSE 'TABLE' END;
        EXECUTE format('REVOKE ALL PRIVILEGES ON %s %I.%I FROM %I', object_kind, rec.nspname, rec.relname, rec.grantee_name);
    END LOOP;
    FOR rec IN
        SELECT n.nspname, p.proname, pg_get_function_identity_arguments(p.oid) AS arguments,
               owner.rolname AS owner_name, COALESCE(grantee.rolname, 'PUBLIC') AS grantee_name
        FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        JOIN pg_roles owner ON owner.oid = p.proowner
        CROSS JOIN LATERAL aclexplode(COALESCE(p.proacl, acldefault('f', p.proowner))) acl
        LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
        WHERE n.nspname = 'public' AND left(p.proname, 6) = 'brain_' AND acl.grantee <> p.proowner
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON FUNCTION %I.%I(%s) FROM %s',
            rec.nspname, rec.proname, rec.arguments,
            CASE WHEN rec.grantee_name = 'PUBLIC' THEN 'PUBLIC' ELSE quote_ident(rec.grantee_name) END
        );
    END LOOP;
    FOR rec IN
        SELECT DISTINCT da.defaclobjtype, COALESCE(grantee.rolname, 'PUBLIC') AS grantee_name
        FROM pg_default_acl da
        LEFT JOIN pg_namespace n ON n.oid = da.defaclnamespace
        CROSS JOIN LATERAL aclexplode(da.defaclacl) acl
        LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
        WHERE da.defaclrole = (SELECT oid FROM pg_roles WHERE rolname = current_user)
          AND n.nspname = 'public' AND acl.grantee <> da.defaclrole
    LOOP
        object_kind := CASE rec.defaclobjtype WHEN 'r' THEN 'TABLES' WHEN 'S' THEN 'SEQUENCES' WHEN 'f' THEN 'FUNCTIONS' END;
        IF object_kind IS NOT NULL THEN
            EXECUTE format(
                'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public REVOKE ALL ON %s FROM %s',
                current_user, object_kind,
                CASE WHEN rec.grantee_name = 'PUBLIC' THEN 'PUBLIC' ELSE quote_ident(rec.grantee_name) END
            );
        END IF;
    END LOOP;
END;
$acl_cleanup$;
