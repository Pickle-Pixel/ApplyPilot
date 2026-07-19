-- Brain schema V10: independently verified retention authority.
-- Additive over V9. V1-V9 migration bytes and ledger rows remain unchanged.

SET ROLE brain_schema_migrator;

CREATE TABLE public.brain_artifact_retention_verifications (
    retention_verification_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    retention_attestation_id BIGINT NOT NULL UNIQUE
        REFERENCES public.brain_artifact_retention_attestations(retention_attestation_id),
    database_incarnation_id UUID NOT NULL,
    payload_sha256 TEXT NOT NULL CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
    signing_key_id TEXT NOT NULL CHECK (btrim(signing_key_id) <> ''),
    signature TEXT NOT NULL CHECK (signature ~ '^[A-Za-z0-9_-]{43}$'),
    verified_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TRIGGER brain_artifact_retention_verifications_append_only
BEFORE UPDATE OR DELETE ON public.brain_artifact_retention_verifications
FOR EACH ROW EXECUTE FUNCTION public.brain_reject_mutation();
CREATE TRIGGER brain_artifact_retention_verifications_append_only_truncate
BEFORE TRUNCATE ON public.brain_artifact_retention_verifications
FOR EACH STATEMENT EXECUTE FUNCTION public.brain_reject_mutation();

ALTER TABLE public.brain_artifact_retention_verifications
    OWNER TO brain_artifact_authority_owner;

SET ROLE brain_artifact_authority_owner;

REVOKE ALL ON TABLE public.brain_artifact_retention_verifications FROM PUBLIC;

CREATE FUNCTION public.brain_get_artifact_retention_for_verification(
    requested_attestation_id bigint
) RETURNS TABLE (
    retention_attestation_id bigint,
    database_incarnation_id uuid,
    artifact_hash text,
    backend text,
    bucket_or_container text,
    object_key text,
    provider_version_id text,
    provider_checksum text,
    retain_until timestamptz,
    verified_at timestamptz,
    verification_lease_id text,
    verification_lease_expires_at timestamptz,
    canonical_payload bytea,
    payload_sha256 text,
    signing_key_id text,
    signature text
)
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT
        attestation.retention_attestation_id,
        attestation.database_incarnation_id,
        attestation.artifact_hash,
        attestation.backend,
        attestation.bucket_or_container,
        attestation.object_key,
        attestation.provider_version_id,
        attestation.provider_checksum,
        attestation.retain_until,
        attestation.verified_at,
        attestation.verification_lease_id,
        attestation.verification_lease_expires_at,
        attestation.canonical_payload,
        attestation.payload_sha256,
        attestation.signing_key_id,
        attestation.signature
    FROM public.brain_artifact_retention_attestations attestation
    WHERE attestation.retention_attestation_id = requested_attestation_id
      AND attestation.canonical_payload IS NOT NULL
      AND attestation.payload_sha256 IS NOT NULL
      AND attestation.signing_key_id IS NOT NULL
      AND attestation.signature IS NOT NULL;
$$;

CREATE FUNCTION public.brain_verify_artifact_retention(
    requested_attestation_id bigint,
    expected_database_incarnation_id uuid,
    expected_payload_sha256 text,
    expected_signing_key_id text,
    expected_signature text
) RETURNS bigint
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    attestation public.brain_artifact_retention_attestations%ROWTYPE;
    existing public.brain_artifact_retention_verifications%ROWTYPE;
BEGIN
    SELECT * INTO attestation
    FROM public.brain_artifact_retention_attestations stored
    WHERE stored.retention_attestation_id = requested_attestation_id;
    IF NOT FOUND
       OR attestation.database_incarnation_id IS DISTINCT FROM expected_database_incarnation_id
       OR attestation.payload_sha256 IS DISTINCT FROM expected_payload_sha256
       OR attestation.signing_key_id IS DISTINCT FROM expected_signing_key_id
       OR attestation.signature IS DISTINCT FROM expected_signature
       OR attestation.canonical_payload IS NULL
       OR expected_payload_sha256 IS DISTINCT FROM encode(sha256(attestation.canonical_payload), 'hex')
       OR expected_database_incarnation_id IS DISTINCT FROM (
           SELECT identity.database_incarnation_id
           FROM public.applypilot_database_identity identity WHERE identity.singleton_id = 1
       ) THEN
        RAISE EXCEPTION 'verified retention evidence does not match the exact committed row'
            USING ERRCODE = '23514';
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended('brain-artifact-retention-verification:' || requested_attestation_id::text, 0)
    );
    SELECT * INTO existing
    FROM public.brain_artifact_retention_verifications verification
    WHERE verification.retention_attestation_id = requested_attestation_id;
    IF FOUND THEN
        IF existing.database_incarnation_id IS DISTINCT FROM expected_database_incarnation_id
           OR existing.payload_sha256 IS DISTINCT FROM expected_payload_sha256
           OR existing.signing_key_id IS DISTINCT FROM expected_signing_key_id
           OR existing.signature IS DISTINCT FROM expected_signature THEN
            RAISE EXCEPTION 'conflicting verified retention replay' USING ERRCODE = '23505';
        END IF;
        RETURN existing.retention_verification_id;
    END IF;

    INSERT INTO public.brain_artifact_retention_verifications (
        retention_attestation_id, database_incarnation_id, payload_sha256,
        signing_key_id, signature
    ) VALUES (
        requested_attestation_id, expected_database_incarnation_id, expected_payload_sha256,
        expected_signing_key_id, expected_signature
    ) RETURNING * INTO existing;
    RETURN existing.retention_verification_id;
END;
$$;

REVOKE ALL ON FUNCTION public.brain_get_artifact_retention_for_verification(bigint) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.brain_verify_artifact_retention(bigint,uuid,text,text,text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.brain_get_artifact_retention_for_verification(bigint)
    TO brain_artifact_authority_verifier;
GRANT EXECUTE ON FUNCTION public.brain_verify_artifact_retention(bigint,uuid,text,text,text)
    TO brain_artifact_authority_verifier;

CREATE OR REPLACE FUNCTION public.brain_artifact_is_authoritative(candidate_hash text)
RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM public.brain_artifact_authority_registrations registration
        JOIN public.brain_artifact_retention_attestations attestation
          ON attestation.artifact_hash = registration.artifact_hash
         AND attestation.provider_version_id = registration.provider_version_id
         AND attestation.provider_checksum = registration.provider_checksum
        JOIN public.brain_artifact_retention_verifications verification
          ON verification.retention_attestation_id = attestation.retention_attestation_id
         AND verification.database_incarnation_id = attestation.database_incarnation_id
         AND verification.payload_sha256 = attestation.payload_sha256
         AND verification.signing_key_id = attestation.signing_key_id
         AND verification.signature = attestation.signature
        JOIN public.brain_artifact_locations location
          ON location.artifact_hash = attestation.artifact_hash
         AND location.backend = attestation.backend
         AND location.bucket_or_container IS NOT DISTINCT FROM attestation.bucket_or_container
         AND location.object_key = attestation.object_key
         AND location.provider_version_id = attestation.provider_version_id
         AND location.provider_checksum = attestation.provider_checksum
        WHERE registration.artifact_hash = candidate_hash
          AND location.durability = 'verified'
          AND location.storage_immutable
          AND location.verified_at IS NOT NULL
          AND location.verified_at <= attestation.verified_at
          AND location.encryption_mode <> 'none'
          AND attestation.retention_mode = 'COMPLIANCE'
          AND attestation.retain_until > now()
          AND attestation.verification_lease_expires_at > now()
          AND attestation.canonical_payload IS NOT NULL
          AND attestation.payload_sha256 IS NOT NULL
          AND attestation.signing_key_id IS NOT NULL
          AND attestation.signature IS NOT NULL
          AND attestation.database_incarnation_id = (
              SELECT database_incarnation_id
              FROM public.applypilot_database_identity WHERE singleton_id = 1
          )
    );
$$;

REVOKE ALL ON FUNCTION public.brain_artifact_is_authoritative(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.brain_artifact_is_authoritative(text)
    TO brain_schema_migrator, brain_artifact_authority_writer,
       brain_artifact_authority_verifier;

SET ROLE brain_schema_migrator;
