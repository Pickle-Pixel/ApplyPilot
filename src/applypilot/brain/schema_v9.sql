-- Brain schema V9: durable signed retention evidence and renewable observations.
-- Additive over V8. V1-V8 migration bytes and ledger rows remain unchanged.

SET ROLE brain_schema_migrator;

ALTER TABLE public.brain_artifact_retention_attestations
    ADD COLUMN canonical_payload bytea,
    ADD COLUMN payload_sha256 text,
    ADD COLUMN signing_key_id text,
    ADD COLUMN signature text,
    ADD CONSTRAINT brain_artifact_retention_attestations_evidence_complete CHECK (
        (canonical_payload IS NULL AND payload_sha256 IS NULL
         AND signing_key_id IS NULL AND signature IS NULL)
        OR
        (canonical_payload IS NOT NULL
         AND payload_sha256 ~ '^[0-9a-f]{64}$'
         AND payload_sha256 = encode(sha256(canonical_payload), 'hex')
         AND btrim(signing_key_id) <> ''
         AND signature ~ '^[A-Za-z0-9_-]{43}$')
    );

-- V8 keyed replay by provider retention identity. That incorrectly made a later
-- observation of an unchanged retain-until deadline conflict with the first one.
-- V9 keys replay by the digest of the exact signed observation instead: an exact
-- ambiguous replay converges, while a later genuine observation appends.
ALTER TABLE public.brain_artifact_retention_attestations
    DROP CONSTRAINT brain_artifact_retention_attestations_identity;
CREATE UNIQUE INDEX brain_artifact_retention_attestations_evidence_identity
    ON public.brain_artifact_retention_attestations
       (database_incarnation_id, payload_sha256)
    WHERE payload_sha256 IS NOT NULL;

SET ROLE brain_artifact_authority_owner;

REVOKE EXECUTE ON FUNCTION public.brain_attest_artifact_retention(
    text,text,text,text,text,text,timestamptz,timestamptz,text,timestamptz
) FROM brain_artifact_authority_writer;

CREATE FUNCTION public.brain_attest_artifact_retention(
    artifact_hash text, backend text, bucket_or_container text, object_key text,
    provider_version_id text, provider_checksum text, retain_until timestamptz,
    verified_at timestamptz, verification_lease_id text,
    verification_lease_expires_at timestamptz, canonical_payload bytea,
    payload_sha256 text, signing_key_id text, signature text
) RETURNS bigint
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    current_incarnation uuid;
    evidence jsonb;
    evidence_request_id uuid;
    existing public.brain_artifact_retention_attestations%ROWTYPE;
BEGIN
    IF canonical_payload IS NULL OR payload_sha256 !~ '^[0-9a-f]{64}$'
       OR payload_sha256 IS DISTINCT FROM encode(sha256(canonical_payload), 'hex')
       OR signing_key_id IS NULL OR btrim(signing_key_id) = ''
       OR signature !~ '^[A-Za-z0-9_-]{43}$' THEN
        RAISE EXCEPTION 'signed retention evidence is incomplete' USING ERRCODE = '22023';
    END IF;
    BEGIN
        evidence := convert_from(canonical_payload, 'UTF8')::jsonb;
        evidence_request_id := (evidence->>'requestId')::uuid;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'signed retention evidence payload is invalid' USING ERRCODE = '22023';
    END;
    IF evidence->>'purpose' IS DISTINCT FROM 'brain-artifact-retention-attestation-v1'
       OR evidence->>'artifactSha256' IS DISTINCT FROM artifact_hash
       OR evidence->>'backend' IS DISTINCT FROM backend
       OR evidence->>'bucketOrContainer' IS DISTINCT FROM bucket_or_container
       OR evidence->>'objectKey' IS DISTINCT FROM object_key
       OR evidence->>'providerVersionId' IS DISTINCT FROM provider_version_id
       OR evidence->>'providerChecksum' IS DISTINCT FROM provider_checksum
       OR evidence->>'retentionMode' IS DISTINCT FROM 'COMPLIANCE'
       OR (evidence->>'retainUntil')::timestamptz IS DISTINCT FROM retain_until
       OR (evidence->>'verifiedAt')::timestamptz IS DISTINCT FROM verified_at
       OR evidence->>'verificationLeaseId' IS DISTINCT FROM verification_lease_id
       OR (evidence->>'verificationExpiresAt')::timestamptz IS DISTINCT FROM verification_lease_expires_at
       OR NOT EXISTS (
           SELECT 1 FROM public.brain_artifact_authority_registrations registration
           WHERE registration.request_id = evidence_request_id
             AND registration.artifact_hash = brain_attest_artifact_retention.artifact_hash
             AND registration.backend = brain_attest_artifact_retention.backend
             AND registration.bucket IS NOT DISTINCT FROM brain_attest_artifact_retention.bucket_or_container
             AND registration.object_key = brain_attest_artifact_retention.object_key
             AND registration.provider_version_id = brain_attest_artifact_retention.provider_version_id
             AND registration.provider_checksum = brain_attest_artifact_retention.provider_checksum
       ) THEN
        RAISE EXCEPTION 'signed retention evidence does not bind the registered artifact'
            USING ERRCODE = '23514';
    END IF;

    SELECT database_incarnation_id INTO current_incarnation
    FROM public.applypilot_database_identity WHERE singleton_id = 1;
    IF current_incarnation IS NULL THEN
        RAISE EXCEPTION 'retention attestation requires an initialized database incarnation'
            USING ERRCODE = '23514';
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended('brain-artifact-retention:' || current_incarnation::text || ':' || payload_sha256, 0)
    );
    SELECT * INTO existing
    FROM public.brain_artifact_retention_attestations attestation
    WHERE attestation.database_incarnation_id = current_incarnation
      AND attestation.payload_sha256 = brain_attest_artifact_retention.payload_sha256;
    IF FOUND THEN
        IF existing.artifact_hash IS DISTINCT FROM artifact_hash
           OR existing.backend IS DISTINCT FROM backend
           OR existing.bucket_or_container IS DISTINCT FROM bucket_or_container
           OR existing.object_key IS DISTINCT FROM object_key
           OR existing.provider_version_id IS DISTINCT FROM provider_version_id
           OR existing.provider_checksum IS DISTINCT FROM provider_checksum
           OR existing.retain_until IS DISTINCT FROM retain_until
           OR existing.verified_at IS DISTINCT FROM verified_at
           OR existing.verification_lease_id IS DISTINCT FROM verification_lease_id
           OR existing.verification_lease_expires_at IS DISTINCT FROM verification_lease_expires_at
           OR existing.canonical_payload IS DISTINCT FROM canonical_payload
           OR existing.signing_key_id IS DISTINCT FROM signing_key_id
           OR existing.signature IS DISTINCT FROM signature THEN
            RAISE EXCEPTION 'conflicting signed retention evidence replay'
                USING ERRCODE = '23505';
        END IF;
        RETURN existing.retention_attestation_id;
    END IF;

    INSERT INTO public.brain_artifact_retention_attestations (
        database_incarnation_id, artifact_hash, backend, bucket_or_container, object_key,
        provider_version_id, provider_checksum, retention_mode, retain_until, verified_at,
        verification_lease_id, verification_lease_expires_at, canonical_payload,
        payload_sha256, signing_key_id, signature
    ) VALUES (
        current_incarnation, artifact_hash, backend, bucket_or_container, object_key,
        provider_version_id, provider_checksum, 'COMPLIANCE', retain_until, verified_at,
        verification_lease_id, verification_lease_expires_at, canonical_payload,
        payload_sha256, signing_key_id, signature
    ) RETURNING * INTO existing;
    RETURN existing.retention_attestation_id;
END;
$$;

REVOKE ALL ON FUNCTION public.brain_attest_artifact_retention(
    text,text,text,text,text,text,timestamptz,timestamptz,text,timestamptz,bytea,text,text,text
) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.brain_attest_artifact_retention(
    text,text,text,text,text,text,timestamptz,timestamptz,text,timestamptz,bytea,text,text,text
) TO brain_artifact_authority_writer;

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
    TO brain_schema_migrator, brain_artifact_authority_writer;

SET ROLE brain_schema_migrator;
