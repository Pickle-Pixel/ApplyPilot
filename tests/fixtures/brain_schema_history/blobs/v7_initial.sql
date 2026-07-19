SET ROLE brain_artifact_authority_owner;

ALTER FUNCTION public.brain_register_authoritative_artifact_manifest(
    uuid,text,text,text,timestamptz,timestamptz,text,text,jsonb
) RENAME TO brain_register_authoritative_artifact_manifest_v6;

DO $$
DECLARE
    definition text;
BEGIN
    SELECT pg_get_functiondef(
        'public.brain_register_authoritative_artifact_manifest_v6(uuid,text,text,text,timestamptz,timestamptz,text,text,jsonb)'::regprocedure
    ) INTO definition;
    definition := replace(
        definition,
        'brain_register_authoritative_artifact_manifest.request_id',
        'brain_register_authoritative_artifact_manifest_v6.request_id'
    );
    EXECUTE definition;
END;
$$;

REVOKE ALL ON FUNCTION public.brain_register_authoritative_artifact_manifest_v6(
    uuid,text,text,text,timestamptz,timestamptz,text,text,jsonb
) FROM PUBLIC, brain_artifact_authority_writer;

CREATE FUNCTION public.brain_register_authoritative_artifact_manifest(
    request_id uuid, manifest_sha256 text, purpose text, key_id text,
    issued_at timestamptz, expires_at timestamptz,
    destination_system_id text, destination_database_name text, artifacts jsonb
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    existing public.brain_artifact_authority_requests%ROWTYPE;
    database_system_id TEXT;
BEGIN
    IF request_id IS NULL OR manifest_sha256 !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'invalid artifact authority request identity' USING ERRCODE='22023';
    END IF;

    PERFORM pg_advisory_xact_lock(hashtextextended('brain-artifact-authority:' || request_id::text, 0));
    SELECT * INTO existing
    FROM public.brain_artifact_authority_requests request
    WHERE request.request_id = brain_register_authoritative_artifact_manifest.request_id;

    IF FOUND THEN
        SELECT system_identifier::text INTO database_system_id FROM pg_control_system();
        IF existing.destination_system_id IS DISTINCT FROM database_system_id
           OR existing.destination_database_name IS DISTINCT FROM current_database()
           OR destination_system_id IS DISTINCT FROM existing.destination_system_id
           OR destination_database_name IS DISTINCT FROM existing.destination_database_name THEN
            RAISE EXCEPTION 'artifact authority replay destination mismatch' USING ERRCODE='22023';
        END IF;
        IF existing.manifest_sha256 IS DISTINCT FROM manifest_sha256 THEN
            RAISE EXCEPTION 'request_id already registered with a different manifest digest'
                USING ERRCODE='23505';
        END IF;
        RETURN existing.receipt;
    END IF;

    RETURN public.brain_register_authoritative_artifact_manifest_v6(
        request_id, manifest_sha256, purpose, key_id, issued_at, expires_at,
        destination_system_id, destination_database_name, artifacts
    );
END;
$$;

REVOKE ALL ON FUNCTION public.brain_register_authoritative_artifact_manifest(
    uuid,text,text,text,timestamptz,timestamptz,text,text,jsonb
) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.brain_register_authoritative_artifact_manifest(
    uuid,text,text,text,timestamptz,timestamptz,text,text,jsonb
) TO brain_artifact_authority_writer;

CREATE FUNCTION public.brain_snapshot_reference_provenance_matches(
    policy_source_id text, policy_version text, lane text, artifact_role text
) RETURNS boolean
LANGUAGE plpgsql IMMUTABLE
SET search_path = pg_catalog, public
AS $$
DECLARE
    provenance jsonb;
    source_sha256 text;
    canonical text;
BEGIN
    IF policy_source_id IS NULL OR lane NOT IN ('ats','linkedin')
       OR artifact_role NOT IN ('label_snapshot','pairwise_snapshot','outcome_snapshot') THEN
        RETURN false;
    END IF;
    BEGIN
        provenance := policy_source_id::jsonb;
    EXCEPTION WHEN OTHERS THEN
        RETURN false;
    END;
    source_sha256 := provenance->>'sourceSha256';
    IF jsonb_typeof(provenance) IS DISTINCT FROM 'object'
       OR provenance->>'kind' IS DISTINCT FROM 'applypilot.policy.snapshot-reference'
       OR provenance->>'lane' IS DISTINCT FROM lane
       OR provenance->>'policyVersion' IS DISTINCT FROM policy_version
       OR provenance->>'role' IS DISTINCT FROM artifact_role
       OR provenance->'schemaVersion' IS DISTINCT FROM '1'::jsonb
       OR provenance->>'sourceField' IS DISTINCT FROM artifact_role
       OR source_sha256 !~ '^[0-9a-f]{64}$' THEN
        RETURN false;
    END IF;
    canonical := format(
        '{"kind":"applypilot.policy.snapshot-reference","lane":%s,'
        '"policyVersion":%s,"role":%s,"schemaVersion":1,'
        '"sourceField":%s,"sourceSha256":"%s"}',
        to_jsonb(lane)::text, to_jsonb(policy_version)::text,
        to_jsonb(artifact_role)::text, to_jsonb(artifact_role)::text, source_sha256
    );
    RETURN policy_source_id = canonical;
END;
$$;

REVOKE ALL ON FUNCTION public.brain_snapshot_reference_provenance_matches(
    text,text,text,text
) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION public.brain_snapshot_reference_provenance_matches(
    text,text,text,text
) TO brain_schema_migrator;

CREATE FUNCTION public.brain_snapshot_binding_is_authoritative(
    policy_version text, lane text, artifact_role text, artifact_hash text
) RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT public.brain_artifact_is_authoritative($4)
       AND EXISTS (
           SELECT 1
           FROM public.brain_artifact_authority_registrations registration
           WHERE registration.artifact_hash=$4
             AND public.brain_snapshot_reference_provenance_matches(
                 registration.policy_source_id,$1,$2,$3
             )
       );
$$;

REVOKE ALL ON FUNCTION public.brain_snapshot_binding_is_authoritative(
    text,text,text,text
) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.brain_snapshot_binding_is_authoritative(
    text,text,text,text
) TO brain_schema_migrator;

SET ROLE brain_schema_migrator;

CREATE OR REPLACE FUNCTION public.brain_check_policy_lifecycle() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE required_roles CONSTANT TEXT[] := ARRAY[
    'qualification_model','preference_model','outcome_model','knowledge_graph','label_snapshot',
    'pairwise_snapshot','outcome_snapshot','config','metrics','replay'];
DECLARE snapshot_roles CONSTANT TEXT[] := ARRAY[
    'label_snapshot','pairwise_snapshot','outcome_snapshot'];
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
        IF current_setting('applypilot.policy_transition',true) IS DISTINCT FROM 'locked' THEN RAISE EXCEPTION 'policy lifecycle changes require locked controller function' USING ERRCODE='42501'; END IF;
        IF NOT ((OLD.lifecycle='draft' AND NEW.lifecycle='validated') OR (OLD.lifecycle='validated' AND NEW.lifecycle='canary') OR (OLD.lifecycle='canary' AND NEW.lifecycle='active') OR (OLD.lifecycle='active' AND NEW.lifecycle='retired')) THEN RAISE EXCEPTION 'illegal policy lifecycle transition' USING ERRCODE='23514'; END IF;
        IF (SELECT count(*) FROM public.brain_policy_transition_receipts WHERE policy_version=NEW.policy_version AND lifecycle=NEW.lifecycle AND definition_version=NEW.gate_definition_version) <> (SELECT count(*) FROM public.brain_policy_gate_definitions WHERE definition_version=NEW.gate_definition_version AND lane=NEW.lane AND lifecycle=NEW.lifecycle AND mandatory) THEN RAISE EXCEPTION 'policy transition requires complete mandatory locked gate receipts' USING ERRCODE='23514'; END IF;
        IF NEW.lifecycle IN ('validated','canary','active') AND (
            (SELECT count(*) FROM public.brain_policy_artifacts WHERE policy_version=NEW.policy_version AND artifact_role=ANY(required_roles)) <> cardinality(required_roles)
            OR EXISTS (SELECT 1 FROM public.brain_policy_artifacts p WHERE p.policy_version=NEW.policy_version AND NOT public.brain_artifact_is_authoritative(p.artifact_hash))
            OR EXISTS (
                SELECT 1 FROM public.brain_policy_artifacts p
                WHERE p.policy_version=NEW.policy_version AND p.artifact_role=ANY(snapshot_roles)
                  AND NOT public.brain_snapshot_binding_is_authoritative(
                      NEW.policy_version,NEW.lane,p.artifact_role,p.artifact_hash))
            OR NOT EXISTS (SELECT 1 FROM public.brain_policy_approvals WHERE policy_version=NEW.policy_version AND approval_type=NEW.lifecycle)) THEN
            RAISE EXCEPTION 'policy transition requires complete authoritative artifacts and approval' USING ERRCODE='23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

SET ROLE brain_schema_migrator;
