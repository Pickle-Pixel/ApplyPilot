-- Brain schema V8: bind artifact authority to a durable database incarnation and
-- append-only storage-retention attestations. Additive over V1-V7; every object
-- created here is owned by the migration identity or the artifact-authority owner
-- and is deeply verified by applypilot.brain.schema.verify_brain_schema_v8.

SET ROLE brain_schema_migrator;

-- 1. Durable database incarnation identity ------------------------------------
--    A server-generated singleton UUID that survives restarts and is identical
--    across independent connections. Ordinary UPDATE/DELETE/TRUNCATE are denied
--    so the incarnation can never be spoofed after the migration seeds it.
CREATE TABLE public.applypilot_database_identity (
    singleton_id smallint NOT NULL DEFAULT 1,
    database_incarnation_id uuid NOT NULL DEFAULT gen_random_uuid(),
    initialized_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT applypilot_database_identity_pkey PRIMARY KEY (singleton_id),
    CONSTRAINT applypilot_database_identity_singleton_check CHECK (singleton_id = 1)
);

REVOKE ALL ON TABLE public.applypilot_database_identity FROM PUBLIC;
GRANT SELECT ON TABLE public.applypilot_database_identity TO brain_schema_verifier;
GRANT SELECT ON TABLE public.applypilot_database_identity TO brain_artifact_authority_owner;

INSERT INTO public.applypilot_database_identity (singleton_id) VALUES (1);

CREATE TRIGGER applypilot_database_identity_append_only
BEFORE UPDATE OR DELETE ON public.applypilot_database_identity
FOR EACH ROW EXECUTE FUNCTION public.brain_reject_mutation();

CREATE TRIGGER applypilot_database_identity_append_only_truncate
BEFORE TRUNCATE ON public.applypilot_database_identity
FOR EACH STATEMENT EXECUTE FUNCTION public.brain_reject_mutation();

-- 2. Append-only storage-retention attestations -------------------------------
--    Each row binds an already registered artifact to the storage location that
--    holds it (backend/bucket/object_key), the provider version and checksum
--    observed at live HEAD verification, a COMPLIANCE retain-until deadline, the
--    verification lease, and the current database incarnation. Rows are immutable;
--    extending retention requires appending a new attestation.
CREATE TABLE public.brain_artifact_retention_attestations (
    retention_attestation_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    database_incarnation_id uuid NOT NULL,
    artifact_hash text NOT NULL,
    backend text NOT NULL CHECK (btrim(backend) <> ''),
    bucket_or_container text,
    object_key text NOT NULL CHECK (btrim(object_key) <> ''),
    provider_version_id text NOT NULL CHECK (btrim(provider_version_id) <> ''),
    provider_checksum text NOT NULL CHECK (btrim(provider_checksum) <> ''),
    retention_mode text NOT NULL CHECK (retention_mode = 'COMPLIANCE'),
    retain_until timestamptz NOT NULL,
    verified_at timestamptz NOT NULL,
    verification_lease_id text NOT NULL CHECK (btrim(verification_lease_id) <> ''),
    verification_lease_expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT brain_artifact_retention_attestations_hash_format
        CHECK (artifact_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT brain_artifact_retention_attestations_deadline
        CHECK (retain_until > verified_at),
    CONSTRAINT brain_artifact_retention_attestations_lease
        CHECK (verification_lease_expires_at >= verified_at),
    CONSTRAINT brain_artifact_retention_attestations_artifact_fk
        FOREIGN KEY (artifact_hash) REFERENCES public.brain_artifacts(artifact_hash),
    CONSTRAINT brain_artifact_retention_attestations_identity UNIQUE
        (artifact_hash, backend, bucket_or_container, object_key,
         provider_version_id, provider_checksum, retain_until)
);

REVOKE ALL ON TABLE public.brain_artifact_retention_attestations FROM PUBLIC;
GRANT SELECT ON TABLE public.brain_artifact_retention_attestations TO brain_schema_verifier;
GRANT SELECT, INSERT ON TABLE public.brain_artifact_retention_attestations
    TO brain_artifact_authority_owner;

-- Closed validation of every attestation: it must bind to an existing registered
-- artifact at a verified, immutable storage location whose provider version and
-- checksum match the live HEAD observation, and to the current incarnation. This
-- forbids guessed backfill of historical registrations.
CREATE FUNCTION public.brain_check_retention_attestation() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    current_incarnation uuid;
BEGIN
    SELECT database_incarnation_id INTO current_incarnation
    FROM public.applypilot_database_identity WHERE singleton_id = 1;
    IF current_incarnation IS NULL THEN
        RAISE EXCEPTION 'retention attestation requires an initialized database incarnation'
            USING ERRCODE = '23514';
    END IF;
    IF NEW.database_incarnation_id IS DISTINCT FROM current_incarnation THEN
        RAISE EXCEPTION 'retention attestation incarnation mismatch' USING ERRCODE = '23514';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.brain_artifact_authority_registrations registration
        JOIN public.brain_artifact_locations location
          ON location.artifact_hash = registration.artifact_hash
         AND location.backend = NEW.backend
         AND location.bucket_or_container IS NOT DISTINCT FROM NEW.bucket_or_container
         AND location.object_key = NEW.object_key
         AND location.provider_version_id = NEW.provider_version_id
         AND location.provider_checksum = NEW.provider_checksum
        WHERE registration.artifact_hash = NEW.artifact_hash
          AND registration.provider_version_id = NEW.provider_version_id
          AND registration.provider_checksum = NEW.provider_checksum
          AND location.durability = 'verified'
          AND location.storage_immutable
          AND location.verified_at IS NOT NULL
          AND location.encryption_mode <> 'none'
    ) THEN
        RAISE EXCEPTION 'retention attestation requires a verified registered artifact location'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.brain_check_retention_attestation() FROM PUBLIC;

CREATE TRIGGER brain_artifact_retention_attestations_validate
BEFORE INSERT ON public.brain_artifact_retention_attestations
FOR EACH ROW EXECUTE FUNCTION public.brain_check_retention_attestation();

CREATE TRIGGER brain_artifact_retention_attestations_append_only
BEFORE UPDATE OR DELETE ON public.brain_artifact_retention_attestations
FOR EACH ROW EXECUTE FUNCTION public.brain_reject_mutation();

CREATE TRIGGER brain_artifact_retention_attestations_append_only_truncate
BEFORE TRUNCATE ON public.brain_artifact_retention_attestations
FOR EACH STATEMENT EXECUTE FUNCTION public.brain_reject_mutation();

-- 3. Parity definition version 2 ----------------------------------------------
--    The exact 16 canonical projection keys emitted by
--    applypilot.brain.parity_adapters.compute_import_parity, in emission order.
--    research_kg_runs / research_scores are source-only projections with no
--    canonical brain target and are therefore non-authoritative membership checks.
INSERT INTO public.brain_parity_definitions
    (definition_version, check_key, relation_name, check_type, authoritative, mandatory, hash_required)
VALUES
    (2, 'jobs', 'brain_jobs', 'canonical', TRUE, TRUE, TRUE),
    (2, 'aliases', 'brain_job_aliases', 'canonical', TRUE, TRUE, TRUE),
    (2, 'observations', 'brain_job_observations', 'canonical', TRUE, TRUE, TRUE),
    (2, 'applications', 'brain_applications', 'canonical', TRUE, TRUE, TRUE),
    (2, 'application_events', 'brain_application_events', 'canonical', TRUE, TRUE, TRUE),
    (2, 'email_events', 'brain_email_events', 'canonical', TRUE, TRUE, TRUE),
    (2, 'email_event_reviews', 'brain_email_events', 'canonical', TRUE, TRUE, TRUE),
    (2, 'reviewed_outcomes', 'brain_reviewed_outcomes', 'canonical', TRUE, TRUE, TRUE),
    (2, 'research_labels', 'brain_label_events', 'canonical', TRUE, TRUE, TRUE),
    (2, 'research_label_confidence', 'brain_label_events', 'canonical', TRUE, TRUE, TRUE),
    (2, 'research_pairwise_labels', 'brain_pairwise_events', 'canonical', TRUE, TRUE, TRUE),
    (2, 'research_kg_artifacts', 'brain_artifacts', 'canonical', TRUE, TRUE, TRUE),
    (2, 'research_kg_runs', 'research_kg_runs', 'membership', FALSE, FALSE, FALSE),
    (2, 'research_scores', 'research_scores', 'membership', FALSE, FALSE, FALSE),
    (2, 'decision_policy_versions', 'brain_decision_policies', 'canonical', TRUE, TRUE, TRUE),
    (2, 'job_decisions', 'brain_job_decisions', 'canonical', TRUE, TRUE, TRUE);

-- 4. Bind artifact authority to incarnation and retention ---------------------
--    The retention-attestation writer and the replacement authority predicate run
--    as the artifact-authority owner so they can read the append-only ledgers via
--    SECURITY DEFINER without widening any login role's direct table rights.
SET ROLE brain_artifact_authority_owner;

CREATE FUNCTION public.brain_attest_artifact_retention(
    artifact_hash text, backend text, bucket_or_container text, object_key text,
    provider_version_id text, provider_checksum text, retain_until timestamptz,
    verified_at timestamptz, verification_lease_id text,
    verification_lease_expires_at timestamptz
) RETURNS bigint
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    current_incarnation uuid;
    existing_id bigint;
BEGIN
    SELECT database_incarnation_id INTO current_incarnation
    FROM public.applypilot_database_identity WHERE singleton_id = 1;
    -- Idempotent replay of an identical attestation returns the durable row and
    -- can never move an already-published retain-until deadline.
    SELECT retention_attestation_id INTO existing_id
    FROM public.brain_artifact_retention_attestations attestation
    WHERE attestation.artifact_hash = brain_attest_artifact_retention.artifact_hash
      AND attestation.backend = brain_attest_artifact_retention.backend
      AND attestation.bucket_or_container IS NOT DISTINCT FROM
          brain_attest_artifact_retention.bucket_or_container
      AND attestation.object_key = brain_attest_artifact_retention.object_key
      AND attestation.provider_version_id = brain_attest_artifact_retention.provider_version_id
      AND attestation.provider_checksum = brain_attest_artifact_retention.provider_checksum
      AND attestation.retain_until = brain_attest_artifact_retention.retain_until;
    IF existing_id IS NOT NULL THEN
        RETURN existing_id;
    END IF;
    INSERT INTO public.brain_artifact_retention_attestations (
        database_incarnation_id, artifact_hash, backend, bucket_or_container, object_key,
        provider_version_id, provider_checksum, retention_mode, retain_until, verified_at,
        verification_lease_id, verification_lease_expires_at
    ) VALUES (
        current_incarnation, artifact_hash, backend, bucket_or_container, object_key,
        provider_version_id, provider_checksum, 'COMPLIANCE', retain_until, verified_at,
        verification_lease_id, verification_lease_expires_at
    ) RETURNING retention_attestation_id INTO existing_id;
    RETURN existing_id;
END;
$$;

REVOKE ALL ON FUNCTION public.brain_attest_artifact_retention(
    text,text,text,text,text,text,timestamptz,timestamptz,text,timestamptz
) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.brain_attest_artifact_retention(
    text,text,text,text,text,text,timestamptz,timestamptz,text,timestamptz
) TO brain_artifact_authority_writer;

CREATE OR REPLACE FUNCTION public.brain_artifact_is_authoritative(candidate_hash TEXT) RETURNS BOOLEAN
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM public.brain_artifact_authority_registrations registration
        JOIN public.brain_artifact_locations location
          ON location.artifact_hash = registration.artifact_hash
        JOIN public.brain_artifact_retention_attestations attestation
          ON attestation.artifact_hash = registration.artifact_hash
         AND attestation.provider_version_id = registration.provider_version_id
         AND attestation.provider_checksum = registration.provider_checksum
        WHERE registration.artifact_hash = candidate_hash
          AND location.durability = 'verified'
          AND location.storage_immutable
          AND location.provider_version_id IS NOT NULL
          AND location.provider_checksum IS NOT NULL
          AND location.encryption_mode <> 'none'
          AND location.verified_at IS NOT NULL
          AND attestation.retention_mode = 'COMPLIANCE'
          AND attestation.retain_until > now()
          AND attestation.database_incarnation_id = (
              SELECT database_incarnation_id
              FROM public.applypilot_database_identity WHERE singleton_id = 1
          )
    );
$$;

ALTER FUNCTION public.brain_artifact_is_authoritative(text)
OWNER TO brain_artifact_authority_owner;
REVOKE ALL ON FUNCTION public.brain_artifact_is_authoritative(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.brain_artifact_is_authoritative(text) TO brain_schema_migrator;

SET ROLE brain_schema_migrator;
