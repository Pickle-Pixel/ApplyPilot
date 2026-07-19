-- Additive V5 durable factual-graph and sequenced authority contract.
-- V1-V4 bytes are immutable. V5 supersedes V4 authorization behavior without
-- rewriting V4 history or delegating final publication authority to V4.

ALTER TABLE public.brain_graph_approval_receipts
    ADD COLUMN predecessor_deny_graph_approval_receipt_id BIGINT;

DO $$
DECLARE constraint_name TEXT;
BEGIN
    SELECT con.conname INTO constraint_name
    FROM pg_constraint con
    WHERE con.conrelid='public.brain_graph_approval_receipts'::regclass
      AND con.contype='u'
      AND ARRAY(
          SELECT attribute.attname::TEXT
          FROM unnest(con.conkey) WITH ORDINALITY key_column(attnum,position)
          JOIN pg_attribute attribute
            ON attribute.attrelid=con.conrelid AND attribute.attnum=key_column.attnum
          ORDER BY key_column.position
      ) = ARRAY['authority_scope_id','authority_epoch','database_incarnation_id','graph_snapshot_id'];
    IF constraint_name IS NULL THEN
        RAISE EXCEPTION 'V4 graph approval uniqueness contract is missing' USING ERRCODE='55000';
    END IF;
    EXECUTE format('ALTER TABLE public.brain_graph_approval_receipts DROP CONSTRAINT %I', constraint_name);

    SELECT con.conname INTO constraint_name
    FROM pg_constraint con
    WHERE con.conrelid='public.brain_graph_approval_consumptions'::regclass
      AND con.contype='u'
      AND ARRAY(
          SELECT attribute.attname::TEXT
          FROM unnest(con.conkey) WITH ORDINALITY key_column(attnum,position)
          JOIN pg_attribute attribute
            ON attribute.attrelid=con.conrelid AND attribute.attnum=key_column.attnum
          ORDER BY key_column.position
      ) = ARRAY['graph_approval_receipt_id'];
    IF constraint_name IS NULL THEN
        RAISE EXCEPTION 'V4 graph approval consumption uniqueness contract is missing' USING ERRCODE='55000';
    END IF;
    EXECUTE format('ALTER TABLE public.brain_graph_approval_consumptions DROP CONSTRAINT %I', constraint_name);
END $$;

ALTER TABLE public.brain_graph_approval_receipts
    ADD CONSTRAINT brain_graph_approval_receipts_v5_identity_key UNIQUE
        (authority_scope_id,authority_epoch,database_incarnation_id,graph_snapshot_id,approval_state),
    ADD CONSTRAINT brain_graph_approval_receipts_v5_predecessor_fk
        FOREIGN KEY (predecessor_deny_graph_approval_receipt_id)
        REFERENCES public.brain_graph_approval_receipts(graph_approval_receipt_id);

CREATE INDEX brain_graph_approval_consumptions_receipt_v5
ON public.brain_graph_approval_consumptions(graph_approval_receipt_id);

CREATE TABLE public.brain_authority_epoch_events (
    authority_epoch_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    authority_scope_id BIGINT NOT NULL
        REFERENCES public.brain_authority_scope_state(authority_scope_id),
    event_sequence BIGINT NOT NULL CHECK (event_sequence > 0),
    event_type TEXT NOT NULL CHECK (event_type IN ('granted','revoked')),
    authority_epoch BIGINT NOT NULL CHECK (authority_epoch > 0),
    database_incarnation_id UUID NOT NULL,
    predecessor_event_id BIGINT UNIQUE
        REFERENCES public.brain_authority_epoch_events(authority_epoch_event_id),
    actor_id TEXT NOT NULL CHECK (btrim(actor_id) <> ''),
    transition_receipt_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (authority_scope_id,event_sequence),
    UNIQUE (authority_scope_id,authority_epoch,database_incarnation_id,event_type),
    CHECK (
        (event_sequence=1 AND event_type='granted' AND predecessor_event_id IS NULL) OR
        (event_sequence>1 AND predecessor_event_id IS NOT NULL)
    )
);

CREATE INDEX brain_authority_epoch_events_latest_v5
ON public.brain_authority_epoch_events(authority_scope_id,event_sequence DESC);

-- Preserve the complete V4 grant/revoke state during upgrade. Future scopes
-- must enter V5 through brain_record_authority_epoch_event.
INSERT INTO public.brain_authority_epoch_events(
    authority_scope_id,event_sequence,event_type,authority_epoch,database_incarnation_id,
    predecessor_event_id,actor_id,transition_receipt_hash,occurred_at
)
SELECT scope.authority_scope_id,1,'granted',scope.authority_epoch,scope.database_incarnation_id,
       NULL,grant_event.actor_id,scope.report_artifact_hash,grant_event.occurred_at
FROM public.brain_authority_scope_state scope
JOIN LATERAL (
    SELECT event.actor_id,event.occurred_at
    FROM public.brain_authority_transition_events event
    WHERE event.authority_scope_id=scope.authority_scope_id
      AND event.authority_epoch=scope.authority_epoch
      AND event.database_incarnation_id=scope.database_incarnation_id
      AND event.event_type='granted'
    ORDER BY event.authority_transition_event_id DESC LIMIT 1
) grant_event ON TRUE;

INSERT INTO public.brain_authority_epoch_events(
    authority_scope_id,event_sequence,event_type,authority_epoch,database_incarnation_id,
    predecessor_event_id,actor_id,transition_receipt_hash,occurred_at
)
SELECT scope.authority_scope_id,2,'revoked',scope.authority_epoch,scope.database_incarnation_id,
       grant_event.authority_epoch_event_id,revoke_event.actor_id,
       scope.report_artifact_hash,revoke_event.occurred_at
FROM public.brain_authority_scope_state scope
JOIN public.brain_authority_epoch_events grant_event
  ON grant_event.authority_scope_id=scope.authority_scope_id
 AND grant_event.event_sequence=1
 AND grant_event.event_type='granted'
JOIN LATERAL (
    SELECT event.actor_id,event.occurred_at
    FROM public.brain_authority_transition_events event
    WHERE event.authority_scope_id=scope.authority_scope_id
      AND event.authority_epoch=scope.authority_epoch
      AND event.database_incarnation_id=scope.database_incarnation_id
      AND event.event_type='revoked'
    ORDER BY event.authority_transition_event_id DESC LIMIT 1
) revoke_event ON TRUE;

CREATE TABLE public.brain_factual_ontology_manifests (
    owner_id TEXT NOT NULL CHECK (btrim(owner_id) <> ''),
    ontology_version TEXT NOT NULL CHECK (btrim(ontology_version) <> ''),
    ontology_manifest_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_id,ontology_version),
    UNIQUE (owner_id,ontology_version,ontology_manifest_hash),
    UNIQUE (owner_id,ontology_manifest_hash)
);

CREATE TABLE public.brain_factual_ontology_terms (
    owner_id TEXT NOT NULL CHECK (btrim(owner_id) <> ''),
    ontology_version TEXT NOT NULL CHECK (btrim(ontology_version) <> ''),
    ontology_manifest_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    predicate TEXT NOT NULL CHECK (predicate IN (
        'has_skill','has_tool_proficiency','has_experience','has_role','has_responsibility',
        'has_industry_experience','has_domain_experience','has_transferable_experience',
        'has_certification','has_education','has_language','has_location','has_work_authorization'
    )),
    term_namespace TEXT NOT NULL CHECK (term_namespace ~ '^[a-z][a-z0-9_-]*$'),
    term_digest TEXT NOT NULL CHECK (term_digest ~ '^[0-9a-f]{64}$'),
    term_id TEXT NOT NULL CHECK (term_id = term_namespace || ':' || term_digest),
    canonical_label TEXT NOT NULL CHECK (btrim(canonical_label) <> ''),
    term_artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_id,ontology_version,predicate,term_id),
    UNIQUE (owner_id,ontology_version,term_id),
    UNIQUE (owner_id,ontology_version,predicate,term_namespace,term_digest),
    FOREIGN KEY (owner_id,ontology_version,ontology_manifest_hash)
        REFERENCES public.brain_factual_ontology_manifests(
            owner_id,ontology_version,ontology_manifest_hash
        ),
    CHECK (
        (predicate='has_skill' AND term_namespace='skill') OR
        (predicate='has_tool_proficiency' AND term_namespace='tool') OR
        (predicate='has_experience' AND term_namespace='experience') OR
        (predicate='has_role' AND term_namespace='role') OR
        (predicate='has_responsibility' AND term_namespace='responsibility') OR
        (predicate='has_industry_experience' AND term_namespace='industry') OR
        (predicate='has_domain_experience' AND term_namespace='domain') OR
        (predicate='has_transferable_experience' AND term_namespace='transfer') OR
        (predicate='has_certification' AND term_namespace='certification') OR
        (predicate='has_education' AND term_namespace='education') OR
        (predicate='has_language' AND term_namespace='language') OR
        (predicate='has_location' AND term_namespace='location') OR
        (predicate='has_work_authorization' AND term_namespace='work-authorization')
    )
);

CREATE TABLE public.brain_factual_ontology_closures (
    owner_id TEXT NOT NULL,
    ontology_version TEXT NOT NULL,
    ontology_manifest_hash TEXT NOT NULL,
    term_count BIGINT NOT NULL CHECK (term_count >= 0),
    ontology_root_hash TEXT NOT NULL CHECK (ontology_root_hash ~ '^[0-9a-f]{64}$'),
    close_receipt_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    closed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (owner_id,ontology_version),
    UNIQUE (owner_id,ontology_version,ontology_manifest_hash,ontology_root_hash),
    FOREIGN KEY (owner_id,ontology_version,ontology_manifest_hash)
        REFERENCES public.brain_factual_ontology_manifests(
            owner_id,ontology_version,ontology_manifest_hash
        )
);

CREATE TABLE public.brain_factual_generations (
    owner_id TEXT NOT NULL CHECK (btrim(owner_id) <> ''),
    generation_id TEXT NOT NULL CHECK (btrim(generation_id) <> ''),
    membership_manifest_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    ontology_version TEXT NOT NULL,
    ontology_manifest_hash TEXT NOT NULL,
    ontology_root_hash TEXT NOT NULL CHECK (ontology_root_hash ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_id,generation_id),
    FOREIGN KEY (owner_id,ontology_version,ontology_manifest_hash,ontology_root_hash)
        REFERENCES public.brain_factual_ontology_closures(
            owner_id,ontology_version,ontology_manifest_hash,ontology_root_hash
        )
);

CREATE TABLE public.brain_factual_generation_members (
    owner_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    source_span_id TEXT NOT NULL CHECK (btrim(source_span_id) <> ''),
    source_artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    source_class TEXT NOT NULL CHECK (source_class IN (
        'resume','resume_context','profile_evidence',
        'human_approved_factual_correction','human_approved_transfer_evidence'
    )),
    member_ordinal BIGINT NOT NULL CHECK (member_ordinal >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_id,generation_id,source_span_id),
    UNIQUE (owner_id,generation_id,member_ordinal),
    UNIQUE (owner_id,generation_id,source_span_id,source_artifact_hash,source_class),
    FOREIGN KEY (owner_id,generation_id)
        REFERENCES public.brain_factual_generations(owner_id,generation_id)
);

CREATE TABLE public.brain_factual_generation_closures (
    owner_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    membership_count BIGINT NOT NULL CHECK (membership_count >= 0),
    membership_root_hash TEXT NOT NULL CHECK (membership_root_hash ~ '^[0-9a-f]{64}$'),
    close_receipt_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    closed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (owner_id,generation_id),
    FOREIGN KEY (owner_id,generation_id)
        REFERENCES public.brain_factual_generations(owner_id,generation_id)
);

CREATE TABLE public.brain_factual_approval_receipts (
    owner_id TEXT NOT NULL,
    human_approval_id TEXT NOT NULL CHECK (btrim(human_approval_id) <> ''),
    generation_id TEXT NOT NULL,
    approval_receipt_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    claim_projection_hash TEXT NOT NULL CHECK (claim_projection_hash ~ '^[0-9a-f]{64}$'),
    ontology_version TEXT NOT NULL,
    predicate TEXT NOT NULL,
    term_id TEXT NOT NULL,
    source_artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    source_span_id TEXT NOT NULL,
    source_class TEXT NOT NULL CHECK (source_class IN (
        'resume','resume_context','profile_evidence',
        'human_approved_factual_correction','human_approved_transfer_evidence'
    )),
    mutation_action TEXT NOT NULL CHECK (mutation_action IN ('assert','supersede')),
    issued_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_id,human_approval_id),
    UNIQUE (owner_id,approval_receipt_hash),
    UNIQUE (owner_id,human_approval_id,approval_receipt_hash),
    UNIQUE (owner_id,human_approval_id,approval_receipt_hash,claim_projection_hash,
            ontology_version,predicate,term_id,mutation_action),
    FOREIGN KEY (owner_id,ontology_version,predicate,term_id)
        REFERENCES public.brain_factual_ontology_terms(owner_id,ontology_version,predicate,term_id),
    FOREIGN KEY (owner_id,generation_id,source_span_id,source_artifact_hash,source_class)
        REFERENCES public.brain_factual_generation_members(
            owner_id,generation_id,source_span_id,source_artifact_hash,source_class
        )
);

CREATE TABLE public.brain_graph_fact_events (
    owner_id TEXT NOT NULL,
    event_id TEXT NOT NULL CHECK (btrim(event_id) <> ''),
    generation_id TEXT NOT NULL,
    source_span_id TEXT NOT NULL,
    human_approval_id TEXT NOT NULL,
    approval_receipt_hash TEXT NOT NULL,
    claim_projection_hash TEXT NOT NULL CHECK (claim_projection_hash ~ '^[0-9a-f]{64}$'),
    ontology_version TEXT NOT NULL,
    predicate TEXT NOT NULL,
    term_id TEXT NOT NULL,
    event_artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    system_receipt_sequence BIGINT NOT NULL CHECK (system_receipt_sequence > 0),
    mutation_action TEXT NOT NULL CHECK (mutation_action IN ('assert','supersede')),
    supersedes_event_id TEXT,
    transaction_time TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (owner_id,event_id),
    UNIQUE (owner_id,generation_id,system_receipt_sequence),
    UNIQUE (owner_id,generation_id,source_span_id,event_id),
    UNIQUE (owner_id,event_id,approval_receipt_hash),
    FOREIGN KEY (owner_id,human_approval_id,approval_receipt_hash,claim_projection_hash,
                 ontology_version,predicate,term_id,mutation_action)
        REFERENCES public.brain_factual_approval_receipts(
            owner_id,human_approval_id,approval_receipt_hash,claim_projection_hash,
            ontology_version,predicate,term_id,mutation_action
        ),
    FOREIGN KEY (owner_id,generation_id,source_span_id,supersedes_event_id)
        REFERENCES public.brain_graph_fact_events(owner_id,generation_id,source_span_id,event_id),
    CHECK (
        (mutation_action='assert' AND supersedes_event_id IS NULL) OR
        (mutation_action='supersede' AND supersedes_event_id IS NOT NULL)
    )
);

CREATE TABLE public.brain_factual_approval_consumptions (
    owner_id TEXT NOT NULL,
    human_approval_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    approval_receipt_hash TEXT NOT NULL,
    consumed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (owner_id,human_approval_id),
    UNIQUE (owner_id,event_id),
    FOREIGN KEY (owner_id,human_approval_id,approval_receipt_hash)
        REFERENCES public.brain_factual_approval_receipts(owner_id,human_approval_id,approval_receipt_hash),
    FOREIGN KEY (owner_id,event_id,approval_receipt_hash)
        REFERENCES public.brain_graph_fact_events(owner_id,event_id,approval_receipt_hash)
);

CREATE TABLE public.brain_factual_generation_coverage (
    owner_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    source_span_id TEXT NOT NULL,
    disposition TEXT NOT NULL CHECK (disposition IN ('assertion','exclusion')),
    event_id TEXT,
    exclusion_reason TEXT,
    review_receipt_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    reviewed_at TIMESTAMPTZ,
    reviewer_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_id,generation_id,source_span_id),
    FOREIGN KEY (owner_id,generation_id,source_span_id)
        REFERENCES public.brain_factual_generation_members(owner_id,generation_id,source_span_id),
    FOREIGN KEY (owner_id,generation_id,source_span_id,event_id)
        REFERENCES public.brain_graph_fact_events(owner_id,generation_id,source_span_id,event_id),
    CHECK (
        (disposition='assertion' AND event_id IS NOT NULL AND exclusion_reason IS NULL
            AND review_receipt_hash IS NULL AND reviewed_at IS NULL AND reviewer_id IS NULL) OR
        (disposition='exclusion' AND event_id IS NULL AND btrim(exclusion_reason) <> ''
            AND review_receipt_hash IS NOT NULL AND reviewed_at IS NOT NULL AND btrim(reviewer_id) <> '')
    )
);

CREATE TABLE public.brain_factual_contradictions (
    owner_id TEXT NOT NULL,
    contradiction_id TEXT NOT NULL CHECK (btrim(contradiction_id) <> ''),
    generation_id TEXT NOT NULL,
    contradiction_artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_id,contradiction_id),
    FOREIGN KEY (owner_id,generation_id)
        REFERENCES public.brain_factual_generations(owner_id,generation_id)
);

CREATE TABLE public.brain_factual_contradiction_events (
    contradiction_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_id TEXT NOT NULL,
    contradiction_id TEXT NOT NULL,
    event_sequence BIGINT NOT NULL CHECK (event_sequence > 0),
    event_type TEXT NOT NULL CHECK (event_type IN ('opened','confirmed','resolved','dismissed')),
    state_after TEXT NOT NULL CHECK (state_after IN ('active','resolved','dismissed')),
    severity TEXT NOT NULL CHECK (severity IN ('noncritical','critical')),
    previous_event_id BIGINT UNIQUE REFERENCES public.brain_factual_contradiction_events(contradiction_event_id),
    review_receipt_hash TEXT REFERENCES public.brain_artifacts(artifact_hash),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (owner_id,contradiction_id,event_sequence),
    FOREIGN KEY (owner_id,contradiction_id)
        REFERENCES public.brain_factual_contradictions(owner_id,contradiction_id),
    CHECK (
        (event_type IN ('opened','confirmed') AND state_after='active') OR
        (event_type='resolved' AND state_after='resolved' AND review_receipt_hash IS NOT NULL) OR
        (event_type='dismissed' AND state_after='dismissed' AND review_receipt_hash IS NOT NULL)
    )
);

CREATE VIEW public.brain_factual_contradiction_state AS
SELECT DISTINCT ON (identity.owner_id,identity.contradiction_id)
    identity.owner_id,identity.contradiction_id,identity.generation_id,
    event.contradiction_event_id,event.event_sequence,event.state_after,event.severity,event.occurred_at
FROM public.brain_factual_contradictions identity
JOIN public.brain_factual_contradiction_events event
  ON event.owner_id=identity.owner_id AND event.contradiction_id=identity.contradiction_id
ORDER BY identity.owner_id,identity.contradiction_id,event.event_sequence DESC;

CREATE TABLE public.brain_factual_graph_snapshots (
    owner_id TEXT NOT NULL,
    graph_snapshot_id TEXT NOT NULL CHECK (btrim(graph_snapshot_id) <> ''),
    generation_id TEXT NOT NULL,
    semantic_root_hash TEXT NOT NULL CHECK (semantic_root_hash ~ '^[0-9a-f]{64}$'),
    coverage_receipt_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    membership_root_hash TEXT NOT NULL CHECK (membership_root_hash ~ '^[0-9a-f]{64}$'),
    event_high_water BIGINT NOT NULL CHECK (event_high_water >= 0),
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    transaction_time TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    snapshot_artifact_hash TEXT NOT NULL REFERENCES public.brain_artifacts(artifact_hash),
    PRIMARY KEY (owner_id,graph_snapshot_id),
    UNIQUE (owner_id,generation_id,semantic_root_hash),
    FOREIGN KEY (owner_id,generation_id)
        REFERENCES public.brain_factual_generation_closures(owner_id,generation_id),
    CHECK (valid_to IS NULL OR valid_to > valid_from)
);

CREATE TABLE public.brain_factual_snapshot_approval_bindings (
    graph_approval_receipt_id BIGINT PRIMARY KEY
        REFERENCES public.brain_graph_approval_receipts(graph_approval_receipt_id),
    owner_id TEXT NOT NULL,
    graph_snapshot_id TEXT NOT NULL,
    authority_epoch_event_id BIGINT NOT NULL
        REFERENCES public.brain_authority_epoch_events(authority_epoch_event_id),
    bound_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    FOREIGN KEY (owner_id,graph_snapshot_id)
        REFERENCES public.brain_factual_graph_snapshots(owner_id,graph_snapshot_id)
);

CREATE TABLE public.brain_v5_candidate_publication_events (
    candidate_decision_id TEXT PRIMARY KEY
        REFERENCES public.brain_v4_candidate_decisions(candidate_decision_id),
    graph_approval_receipt_id BIGINT NOT NULL
        REFERENCES public.brain_graph_approval_receipts(graph_approval_receipt_id),
    authority_scope_id BIGINT NOT NULL
        REFERENCES public.brain_authority_scope_state(authority_scope_id),
    authority_epoch_event_id BIGINT NOT NULL
        REFERENCES public.brain_authority_epoch_events(authority_epoch_event_id),
    published_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (authority_scope_id,candidate_decision_id)
);

CREATE INDEX brain_v5_candidate_publication_receipt_idx
ON public.brain_v5_candidate_publication_events(graph_approval_receipt_id);

CREATE FUNCTION public.brain_v5_sha256_text(requested_value TEXT) RETURNS TEXT
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE SET search_path=pg_catalog,public AS $$
    SELECT encode(sha256(convert_to(requested_value,'UTF8')),'hex')
$$;

CREATE FUNCTION public.brain_v5_frame(requested_value TEXT) RETURNS TEXT
LANGUAGE sql IMMUTABLE PARALLEL SAFE SET search_path=pg_catalog,public AS $$
    SELECT CASE WHEN requested_value IS NULL THEN '-1:'
        ELSE octet_length(convert_to(requested_value,'UTF8'))::TEXT || ':' || requested_value END
$$;

CREATE FUNCTION public.brain_compute_factual_ontology_root(
    requested_owner_id TEXT,requested_ontology_version TEXT
) RETURNS TEXT
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$
    SELECT public.brain_v5_sha256_text(
        'applypilot:factual-ontology:v1' || E'\n' || COALESCE(string_agg(
            jsonb_build_array(term.predicate,term.term_namespace,term.term_digest,term.term_id,
                              term.canonical_label,term.term_artifact_hash)::TEXT,
            E'\n' ORDER BY term.predicate,term.term_id
        ),'')
    )
    FROM public.brain_factual_ontology_terms term
    WHERE term.owner_id=requested_owner_id AND term.ontology_version=requested_ontology_version
$$;

CREATE FUNCTION public.brain_compute_factual_membership_root(
    requested_owner_id TEXT,requested_generation_id TEXT
) RETURNS TEXT
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$
    SELECT public.brain_v5_sha256_text(
        'applypilot:factual-membership:v1' || E'\n' || COALESCE(string_agg(
            jsonb_build_array(member.member_ordinal,member.source_span_id,
                              member.source_artifact_hash,member.source_class)::TEXT,
            E'\n' ORDER BY member.member_ordinal,member.source_span_id
        ),'')
    )
    FROM public.brain_factual_generation_members member
    WHERE member.owner_id=requested_owner_id AND member.generation_id=requested_generation_id
$$;

CREATE FUNCTION public.brain_compute_factual_semantic_root(
    requested_owner_id TEXT,requested_generation_id TEXT
) RETURNS TEXT
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE generation public.brain_factual_generations%ROWTYPE;
        generation_closure public.brain_factual_generation_closures%ROWTYPE;
        ontology_closure public.brain_factual_ontology_closures%ROWTYPE;
        active_events TEXT; coverage_state TEXT; active_contradictions TEXT;
BEGIN
    SELECT * INTO generation FROM public.brain_factual_generations
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id;
    SELECT * INTO generation_closure FROM public.brain_factual_generation_closures
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id;
    IF generation.owner_id IS NULL OR generation_closure.owner_id IS NULL THEN
        RAISE EXCEPTION 'closed factual generation is required' USING ERRCODE='55000';
    END IF;
    SELECT * INTO ontology_closure FROM public.brain_factual_ontology_closures
    WHERE owner_id=generation.owner_id AND ontology_version=generation.ontology_version;
    IF ontology_closure.owner_id IS NULL
       OR ontology_closure.ontology_manifest_hash<>generation.ontology_manifest_hash
       OR ontology_closure.ontology_root_hash<>generation.ontology_root_hash THEN
        RAISE EXCEPTION 'matching closed factual ontology is required' USING ERRCODE='55000';
    END IF;

    SELECT COALESCE(string_agg(public.brain_v5_frame(row_payload),'' ORDER BY source_span_id COLLATE "C",
        system_receipt_sequence,event_id COLLATE "C"),'') INTO active_events
    FROM (
        SELECT event.source_span_id,event.system_receipt_sequence,event.event_id,
            public.brain_v5_frame(event.source_span_id)
            || public.brain_v5_frame(event.event_id)
            || public.brain_v5_frame(event.human_approval_id)
            || public.brain_v5_frame(event.approval_receipt_hash)
            || public.brain_v5_frame(event.claim_projection_hash)
            || public.brain_v5_frame(event.ontology_version)
            || public.brain_v5_frame(event.predicate)
            || public.brain_v5_frame(event.term_id)
            || public.brain_v5_frame(event.event_artifact_hash)
            || public.brain_v5_frame(event.system_receipt_sequence::TEXT)
            || public.brain_v5_frame(event.mutation_action)
            || public.brain_v5_frame(event.supersedes_event_id)
            || public.brain_v5_frame(approval.source_artifact_hash)
            || public.brain_v5_frame(approval.source_class)
            || public.brain_v5_frame(to_char(approval.issued_at AT TIME ZONE 'UTC',
                                             'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')) AS row_payload
        FROM public.brain_graph_fact_events event
        JOIN public.brain_factual_approval_receipts approval
          ON approval.owner_id=event.owner_id AND approval.human_approval_id=event.human_approval_id
        WHERE event.owner_id=requested_owner_id AND event.generation_id=requested_generation_id
          AND NOT EXISTS (
              SELECT 1 FROM public.brain_graph_fact_events later
              WHERE later.owner_id=event.owner_id AND later.generation_id=event.generation_id
                AND later.source_span_id=event.source_span_id
                AND later.supersedes_event_id=event.event_id
          )
    ) rows;

    SELECT COALESCE(string_agg(public.brain_v5_frame(row_payload),''
        ORDER BY source_span_id COLLATE "C"),'') INTO coverage_state
    FROM (
        SELECT coverage.source_span_id,
            public.brain_v5_frame(coverage.source_span_id)
            || public.brain_v5_frame(coverage.disposition)
            || public.brain_v5_frame(coverage.event_id)
            || public.brain_v5_frame(coverage.exclusion_reason)
            || public.brain_v5_frame(coverage.review_receipt_hash)
            || public.brain_v5_frame(coverage.reviewer_id)
            || public.brain_v5_frame(CASE WHEN coverage.reviewed_at IS NULL THEN NULL ELSE
                to_char(coverage.reviewed_at AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.US"Z"') END)
                AS row_payload
        FROM public.brain_factual_generation_coverage coverage
        WHERE coverage.owner_id=requested_owner_id AND coverage.generation_id=requested_generation_id
    ) rows;

    SELECT COALESCE(string_agg(public.brain_v5_frame(row_payload),''
        ORDER BY contradiction_id COLLATE "C"),'') INTO active_contradictions
    FROM (
        SELECT contradiction.contradiction_id,
            public.brain_v5_frame(contradiction.contradiction_id)
            || public.brain_v5_frame(contradiction.contradiction_artifact_hash)
            || public.brain_v5_frame(state.event_sequence::TEXT)
            || public.brain_v5_frame(state.state_after)
            || public.brain_v5_frame(state.severity)
            || public.brain_v5_frame(event.review_receipt_hash) AS row_payload
        FROM public.brain_factual_contradiction_state state
        JOIN public.brain_factual_contradictions contradiction
          ON contradiction.owner_id=state.owner_id AND contradiction.contradiction_id=state.contradiction_id
        JOIN public.brain_factual_contradiction_events event
          ON event.contradiction_event_id=state.contradiction_event_id
        WHERE state.owner_id=requested_owner_id AND state.generation_id=requested_generation_id
          AND state.state_after='active'
    ) rows;

    RETURN public.brain_v5_sha256_text(
        public.brain_v5_frame('applypilot:factual-semantic-root:v1')
        || public.brain_v5_frame(requested_owner_id)
        || public.brain_v5_frame(requested_generation_id)
        || public.brain_v5_frame(generation.membership_manifest_hash)
        || public.brain_v5_frame(generation.ontology_version)
        || public.brain_v5_frame(ontology_closure.ontology_manifest_hash)
        || public.brain_v5_frame(ontology_closure.term_count::TEXT)
        || public.brain_v5_frame(ontology_closure.ontology_root_hash)
        || public.brain_v5_frame(generation_closure.membership_count::TEXT)
        || public.brain_v5_frame(generation_closure.membership_root_hash)
        || public.brain_v5_frame(active_events)
        || public.brain_v5_frame(coverage_state)
        || public.brain_v5_frame(active_contradictions)
    );
END;
$$;

CREATE FUNCTION public.brain_reject_closed_ontology_term() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
    PERFORM 1 FROM public.brain_factual_ontology_manifests
    WHERE owner_id=NEW.owner_id AND ontology_version=NEW.ontology_version FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'factual ontology is missing' USING ERRCODE='55000'; END IF;
    IF EXISTS (
        SELECT 1 FROM public.brain_factual_ontology_closures
        WHERE owner_id=NEW.owner_id AND ontology_version=NEW.ontology_version
    ) THEN RAISE EXCEPTION 'factual ontology is closed' USING ERRCODE='55000'; END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER brain_factual_ontology_terms_closed
BEFORE INSERT ON public.brain_factual_ontology_terms
FOR EACH ROW EXECUTE FUNCTION public.brain_reject_closed_ontology_term();

CREATE FUNCTION public.brain_reject_closed_generation_member() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
    PERFORM 1 FROM public.brain_factual_generations
    WHERE owner_id=NEW.owner_id AND generation_id=NEW.generation_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'factual generation is missing' USING ERRCODE='55000'; END IF;
    IF EXISTS (
        SELECT 1 FROM public.brain_factual_generation_closures
        WHERE owner_id=NEW.owner_id AND generation_id=NEW.generation_id
    ) THEN RAISE EXCEPTION 'factual generation is closed' USING ERRCODE='55000'; END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER brain_factual_generation_members_closed
BEFORE INSERT ON public.brain_factual_generation_members
FOR EACH ROW EXECUTE FUNCTION public.brain_reject_closed_generation_member();

CREATE FUNCTION public.brain_create_factual_ontology(
    requested_owner_id TEXT,requested_ontology_version TEXT,requested_manifest_hash TEXT
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
    INSERT INTO public.brain_factual_ontology_manifests(
        owner_id,ontology_version,ontology_manifest_hash
    ) VALUES (requested_owner_id,requested_ontology_version,requested_manifest_hash);
    RETURN requested_ontology_version;
END;
$$;

CREATE FUNCTION public.brain_add_factual_ontology_term(
    requested_owner_id TEXT,requested_ontology_version TEXT,requested_manifest_hash TEXT,
    requested_predicate TEXT,requested_term_namespace TEXT,requested_term_digest TEXT,
    requested_term_id TEXT,requested_canonical_label TEXT,requested_term_artifact_hash TEXT
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
    PERFORM 1 FROM public.brain_factual_ontology_manifests
    WHERE owner_id=requested_owner_id AND ontology_version=requested_ontology_version FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'factual ontology is missing' USING ERRCODE='55000'; END IF;
    IF EXISTS (
        SELECT 1 FROM public.brain_factual_ontology_closures
        WHERE owner_id=requested_owner_id AND ontology_version=requested_ontology_version
    ) THEN RAISE EXCEPTION 'factual ontology is closed' USING ERRCODE='55000'; END IF;
    INSERT INTO public.brain_factual_ontology_terms(
        owner_id,ontology_version,ontology_manifest_hash,predicate,term_namespace,
        term_digest,term_id,canonical_label,term_artifact_hash
    ) VALUES (requested_owner_id,requested_ontology_version,requested_manifest_hash,
        requested_predicate,requested_term_namespace,requested_term_digest,requested_term_id,
        requested_canonical_label,requested_term_artifact_hash);
    RETURN requested_term_id;
END;
$$;

CREATE FUNCTION public.brain_close_factual_ontology(
    requested_owner_id TEXT,requested_ontology_version TEXT,requested_term_count BIGINT,
    requested_ontology_root_hash TEXT,requested_close_receipt_hash TEXT
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE actual_count BIGINT; actual_root TEXT; manifest_hash TEXT;
BEGIN
    SELECT ontology_manifest_hash INTO manifest_hash
    FROM public.brain_factual_ontology_manifests
    WHERE owner_id=requested_owner_id AND ontology_version=requested_ontology_version FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'factual ontology is missing' USING ERRCODE='55000'; END IF;
    SELECT count(*) INTO actual_count FROM public.brain_factual_ontology_terms
    WHERE owner_id=requested_owner_id AND ontology_version=requested_ontology_version;
    actual_root := public.brain_compute_factual_ontology_root(
        requested_owner_id,requested_ontology_version
    );
    IF requested_term_count<>actual_count OR requested_ontology_root_hash<>actual_root THEN
        RAISE EXCEPTION 'ontology closure does not match computed term membership' USING ERRCODE='55000';
    END IF;
    INSERT INTO public.brain_factual_ontology_closures(
        owner_id,ontology_version,ontology_manifest_hash,term_count,ontology_root_hash,close_receipt_hash
    ) VALUES (requested_owner_id,requested_ontology_version,manifest_hash,actual_count,actual_root,
              requested_close_receipt_hash);
    RETURN actual_root;
END;
$$;

CREATE FUNCTION public.brain_create_factual_generation(
    requested_owner_id TEXT,requested_generation_id TEXT,requested_membership_manifest_hash TEXT,
    requested_ontology_version TEXT,requested_ontology_root_hash TEXT
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE ontology public.brain_factual_ontology_closures%ROWTYPE;
BEGIN
    SELECT * INTO ontology FROM public.brain_factual_ontology_closures
    WHERE owner_id=requested_owner_id AND ontology_version=requested_ontology_version FOR KEY SHARE;
    IF NOT FOUND OR ontology.ontology_root_hash<>requested_ontology_root_hash THEN
        RAISE EXCEPTION 'matching closed ontology is required' USING ERRCODE='55000';
    END IF;
    INSERT INTO public.brain_factual_generations(
        owner_id,generation_id,membership_manifest_hash,ontology_version,
        ontology_manifest_hash,ontology_root_hash
    ) VALUES (requested_owner_id,requested_generation_id,requested_membership_manifest_hash,
        requested_ontology_version,ontology.ontology_manifest_hash,ontology.ontology_root_hash);
    RETURN requested_generation_id;
END;
$$;

CREATE FUNCTION public.brain_add_factual_generation_member(
    requested_owner_id TEXT,requested_generation_id TEXT,requested_source_span_id TEXT,
    requested_source_artifact_hash TEXT,requested_source_class TEXT,requested_member_ordinal BIGINT
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
    PERFORM 1 FROM public.brain_factual_generations
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'factual generation is missing' USING ERRCODE='55000'; END IF;
    IF EXISTS (
        SELECT 1 FROM public.brain_factual_generation_closures
        WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id
    ) THEN RAISE EXCEPTION 'factual generation is closed' USING ERRCODE='55000'; END IF;
    INSERT INTO public.brain_factual_generation_members(
        owner_id,generation_id,source_span_id,source_artifact_hash,source_class,member_ordinal
    ) VALUES (requested_owner_id,requested_generation_id,requested_source_span_id,
              requested_source_artifact_hash,requested_source_class,requested_member_ordinal);
    RETURN requested_source_span_id;
END;
$$;

CREATE FUNCTION public.brain_close_factual_generation(
    requested_owner_id TEXT,requested_generation_id TEXT,requested_membership_count BIGINT,
    requested_membership_root_hash TEXT,requested_close_receipt_hash TEXT
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE actual_count BIGINT; actual_root TEXT;
BEGIN
    PERFORM 1 FROM public.brain_factual_generations
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'factual generation is missing' USING ERRCODE='55000'; END IF;
    SELECT count(*) INTO actual_count FROM public.brain_factual_generation_members
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id;
    actual_root := public.brain_compute_factual_membership_root(
        requested_owner_id,requested_generation_id
    );
    IF requested_membership_count<>actual_count OR requested_membership_root_hash<>actual_root THEN
        RAISE EXCEPTION 'generation closure does not match computed membership' USING ERRCODE='55000';
    END IF;
    INSERT INTO public.brain_factual_generation_closures(
        owner_id,generation_id,membership_count,membership_root_hash,close_receipt_hash
    ) VALUES (requested_owner_id,requested_generation_id,actual_count,actual_root,
              requested_close_receipt_hash);
    RETURN actual_root;
END;
$$;

CREATE FUNCTION public.brain_admit_factual_event(
    requested_owner_id TEXT,requested_generation_id TEXT,requested_source_span_id TEXT,
    requested_human_approval_id TEXT,requested_approval_receipt_hash TEXT,
    requested_claim_projection_hash TEXT,requested_source_artifact_hash TEXT,
    requested_source_class TEXT,requested_ontology_version TEXT,requested_predicate TEXT,
    requested_term_id TEXT,requested_event_id TEXT,requested_event_artifact_hash TEXT,
    requested_system_receipt_sequence BIGINT,requested_mutation_action TEXT,
    requested_supersedes_event_id TEXT,requested_issued_at TIMESTAMPTZ
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE generation public.brain_factual_generations%ROWTYPE;
        previous_event public.brain_graph_fact_events%ROWTYPE;
        current_high_water BIGINT; active_event_id TEXT;
BEGIN
    SELECT * INTO generation FROM public.brain_factual_generations
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id FOR UPDATE;
    IF NOT FOUND OR NOT EXISTS (
        SELECT 1 FROM public.brain_factual_generation_closures closure
        WHERE closure.owner_id=requested_owner_id AND closure.generation_id=requested_generation_id
    ) THEN RAISE EXCEPTION 'closed factual generation is required' USING ERRCODE='55000'; END IF;
    IF generation.ontology_version<>requested_ontology_version THEN
        RAISE EXCEPTION 'generation ontology version mismatch' USING ERRCODE='55000';
    END IF;
    IF requested_issued_at>clock_timestamp() THEN
        RAISE EXCEPTION 'approval receipt cannot be issued after admission' USING ERRCODE='55000';
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.brain_factual_approval_consumptions consumption
        WHERE consumption.owner_id=requested_owner_id
          AND consumption.human_approval_id=requested_human_approval_id
    ) THEN
        RAISE EXCEPTION 'human approval identity has already been consumed' USING ERRCODE='23505';
    END IF;
    SELECT COALESCE(max(system_receipt_sequence),0) INTO current_high_water
    FROM public.brain_graph_fact_events
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id;
    IF requested_system_receipt_sequence<>current_high_water+1 THEN
        RAISE EXCEPTION 'system receipt sequence must advance by exactly one' USING ERRCODE='55000';
    END IF;
    SELECT event.event_id INTO active_event_id
    FROM public.brain_graph_fact_events event
    WHERE event.owner_id=requested_owner_id AND event.generation_id=requested_generation_id
      AND event.source_span_id=requested_source_span_id
      AND NOT EXISTS (
          SELECT 1 FROM public.brain_graph_fact_events later
          WHERE later.owner_id=event.owner_id AND later.generation_id=event.generation_id
            AND later.source_span_id=event.source_span_id AND later.supersedes_event_id=event.event_id
      )
    ORDER BY event.system_receipt_sequence DESC LIMIT 1;
    IF requested_mutation_action='assert' AND active_event_id IS NOT NULL THEN
        RAISE EXCEPTION 'assertion cannot replace an active same-span event' USING ERRCODE='55000';
    ELSIF requested_mutation_action='supersede' THEN
        SELECT * INTO previous_event FROM public.brain_graph_fact_events
        WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id
          AND source_span_id=requested_source_span_id AND event_id=requested_supersedes_event_id;
        IF NOT FOUND OR active_event_id IS DISTINCT FROM requested_supersedes_event_id THEN
            RAISE EXCEPTION 'supersession must target the active earlier same-span event' USING ERRCODE='55000';
        END IF;
    ELSIF requested_supersedes_event_id IS NOT NULL THEN
        RAISE EXCEPTION 'assertion cannot supersede an event' USING ERRCODE='55000';
    END IF;
    INSERT INTO public.brain_factual_approval_receipts(
        owner_id,human_approval_id,generation_id,approval_receipt_hash,claim_projection_hash,
        ontology_version,predicate,term_id,source_artifact_hash,source_span_id,source_class,
        mutation_action,issued_at
    ) VALUES (requested_owner_id,requested_human_approval_id,requested_generation_id,
        requested_approval_receipt_hash,requested_claim_projection_hash,requested_ontology_version,
        requested_predicate,requested_term_id,requested_source_artifact_hash,
        requested_source_span_id,requested_source_class,requested_mutation_action,requested_issued_at);
    INSERT INTO public.brain_graph_fact_events(
        owner_id,event_id,generation_id,source_span_id,human_approval_id,approval_receipt_hash,
        claim_projection_hash,ontology_version,predicate,term_id,event_artifact_hash,
        system_receipt_sequence,mutation_action,supersedes_event_id
    ) VALUES (requested_owner_id,requested_event_id,requested_generation_id,requested_source_span_id,
        requested_human_approval_id,requested_approval_receipt_hash,requested_claim_projection_hash,
        requested_ontology_version,requested_predicate,requested_term_id,requested_event_artifact_hash,
        requested_system_receipt_sequence,requested_mutation_action,requested_supersedes_event_id);
    INSERT INTO public.brain_factual_approval_consumptions(
        owner_id,human_approval_id,event_id,approval_receipt_hash
    ) VALUES (requested_owner_id,requested_human_approval_id,requested_event_id,
              requested_approval_receipt_hash);
    RETURN requested_event_id;
END;
$$;

CREATE FUNCTION public.brain_record_factual_assertion_coverage(
    requested_owner_id TEXT,requested_generation_id TEXT,requested_source_span_id TEXT,
    requested_event_id TEXT
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
    PERFORM 1 FROM public.brain_factual_generations
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'factual generation is missing' USING ERRCODE='55000'; END IF;
    IF NOT EXISTS (
        SELECT 1 FROM public.brain_graph_fact_events event
        WHERE event.owner_id=requested_owner_id AND event.generation_id=requested_generation_id
          AND event.source_span_id=requested_source_span_id AND event.event_id=requested_event_id
          AND NOT EXISTS (
              SELECT 1 FROM public.brain_graph_fact_events later
              WHERE later.owner_id=event.owner_id AND later.generation_id=event.generation_id
                AND later.source_span_id=event.source_span_id AND later.supersedes_event_id=event.event_id
          )
    ) THEN RAISE EXCEPTION 'coverage requires the active same-span event' USING ERRCODE='55000'; END IF;
    INSERT INTO public.brain_factual_generation_coverage(
        owner_id,generation_id,source_span_id,disposition,event_id
    ) VALUES (requested_owner_id,requested_generation_id,requested_source_span_id,'assertion',requested_event_id);
    RETURN requested_event_id;
END;
$$;

CREATE FUNCTION public.brain_review_factual_exclusion(
    requested_owner_id TEXT,requested_generation_id TEXT,requested_source_span_id TEXT,
    requested_reason TEXT,requested_review_receipt_hash TEXT,requested_reviewer_id TEXT,
    requested_reviewed_at TIMESTAMPTZ
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
    PERFORM 1 FROM public.brain_factual_generations
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id FOR UPDATE;
    IF NOT FOUND OR NOT EXISTS (
        SELECT 1 FROM public.brain_factual_generation_closures
        WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id
    ) THEN RAISE EXCEPTION 'closed factual generation is required' USING ERRCODE='55000'; END IF;
    IF requested_reviewed_at>clock_timestamp() THEN
        RAISE EXCEPTION 'exclusion review cannot be future-dated' USING ERRCODE='22007';
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.brain_graph_fact_events event
        WHERE event.owner_id=requested_owner_id AND event.generation_id=requested_generation_id
          AND event.source_span_id=requested_source_span_id
          AND NOT EXISTS (
              SELECT 1 FROM public.brain_graph_fact_events later
              WHERE later.owner_id=event.owner_id AND later.generation_id=event.generation_id
                AND later.source_span_id=event.source_span_id
                AND later.supersedes_event_id=event.event_id
          )
    ) THEN RAISE EXCEPTION 'exclusion conflicts with active factual event' USING ERRCODE='55000'; END IF;
    INSERT INTO public.brain_factual_generation_coverage(
        owner_id,generation_id,source_span_id,disposition,exclusion_reason,
        review_receipt_hash,reviewed_at,reviewer_id
    ) VALUES (requested_owner_id,requested_generation_id,requested_source_span_id,'exclusion',
        requested_reason,requested_review_receipt_hash,requested_reviewed_at,requested_reviewer_id);
    RETURN requested_source_span_id;
END;
$$;

CREATE FUNCTION public.brain_create_factual_contradiction(
    requested_owner_id TEXT,requested_contradiction_id TEXT,requested_generation_id TEXT,
    requested_contradiction_artifact_hash TEXT
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
    PERFORM 1 FROM public.brain_factual_generations
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'factual generation is missing' USING ERRCODE='55000'; END IF;
    INSERT INTO public.brain_factual_contradictions(
        owner_id,contradiction_id,generation_id,contradiction_artifact_hash
    ) VALUES (requested_owner_id,requested_contradiction_id,requested_generation_id,
              requested_contradiction_artifact_hash);
    RETURN requested_contradiction_id;
END;
$$;

CREATE FUNCTION public.brain_append_factual_contradiction_event(
    requested_owner_id TEXT,requested_contradiction_id TEXT,requested_event_sequence BIGINT,
    requested_event_type TEXT,requested_state_after TEXT,requested_severity TEXT,
    requested_previous_event_id BIGINT,requested_review_receipt_hash TEXT
) RETURNS BIGINT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE contradiction public.brain_factual_contradictions%ROWTYPE;
        prior public.brain_factual_contradiction_events%ROWTYPE; inserted_id BIGINT;
BEGIN
    SELECT * INTO contradiction FROM public.brain_factual_contradictions
    WHERE owner_id=requested_owner_id AND contradiction_id=requested_contradiction_id;
    IF NOT FOUND THEN RAISE EXCEPTION 'contradiction identity is missing' USING ERRCODE='55000'; END IF;
    PERFORM 1 FROM public.brain_factual_generations
    WHERE owner_id=contradiction.owner_id AND generation_id=contradiction.generation_id FOR UPDATE;
    PERFORM 1 FROM public.brain_factual_contradictions
    WHERE owner_id=requested_owner_id AND contradiction_id=requested_contradiction_id FOR UPDATE;
    SELECT * INTO prior FROM public.brain_factual_contradiction_events
    WHERE owner_id=requested_owner_id AND contradiction_id=requested_contradiction_id
    ORDER BY event_sequence DESC LIMIT 1;
    IF prior.contradiction_event_id IS NULL THEN
        IF requested_event_sequence<>1 OR requested_previous_event_id IS NOT NULL OR requested_event_type<>'opened' THEN
            RAISE EXCEPTION 'first contradiction transition must open sequence one' USING ERRCODE='55000';
        END IF;
    ELSIF requested_event_sequence<>prior.event_sequence+1
       OR requested_previous_event_id<>prior.contradiction_event_id
       OR prior.state_after<>'active'
       OR requested_severity<>prior.severity THEN
        RAISE EXCEPTION 'invalid contradiction transition lineage' USING ERRCODE='55000';
    END IF;
    INSERT INTO public.brain_factual_contradiction_events(
        owner_id,contradiction_id,event_sequence,event_type,state_after,severity,
        previous_event_id,review_receipt_hash
    ) VALUES (requested_owner_id,requested_contradiction_id,requested_event_sequence,
        requested_event_type,requested_state_after,requested_severity,requested_previous_event_id,
        requested_review_receipt_hash) RETURNING contradiction_event_id INTO inserted_id;
    RETURN inserted_id;
END;
$$;

CREATE FUNCTION public.brain_publish_factual_snapshot(
    requested_owner_id TEXT,requested_graph_snapshot_id TEXT,requested_generation_id TEXT,
    requested_semantic_root_hash TEXT,requested_coverage_receipt_hash TEXT,
    requested_membership_root_hash TEXT,requested_event_high_water BIGINT,
    requested_valid_from TIMESTAMPTZ,requested_valid_to TIMESTAMPTZ,
    requested_snapshot_artifact_hash TEXT
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE closure public.brain_factual_generation_closures%ROWTYPE;
        coverage_count BIGINT; actual_high_water BIGINT; actual_root TEXT;
        actual_semantic_root TEXT; validation_time TIMESTAMPTZ := clock_timestamp();
BEGIN
    PERFORM 1 FROM public.brain_factual_generations
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'factual generation is missing' USING ERRCODE='55000'; END IF;
    IF requested_valid_from>validation_time
       OR (requested_valid_to IS NOT NULL AND requested_valid_to<=validation_time) THEN
        RAISE EXCEPTION 'factual snapshot must be currently valid at creation' USING ERRCODE='22007';
    END IF;
    SELECT * INTO closure FROM public.brain_factual_generation_closures
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id;
    actual_root := public.brain_compute_factual_membership_root(
        requested_owner_id,requested_generation_id
    );
    IF NOT FOUND OR closure.membership_root_hash<>actual_root
       OR requested_membership_root_hash<>actual_root THEN
        RAISE EXCEPTION 'matching computed closed factual generation is required' USING ERRCODE='55000';
    END IF;
    SELECT count(*) INTO coverage_count FROM public.brain_factual_generation_coverage
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id;
    IF coverage_count<>closure.membership_count OR EXISTS (
        SELECT 1 FROM public.brain_factual_generation_members member
        LEFT JOIN public.brain_factual_generation_coverage coverage
          ON coverage.owner_id=member.owner_id AND coverage.generation_id=member.generation_id
         AND coverage.source_span_id=member.source_span_id
        WHERE member.owner_id=requested_owner_id AND member.generation_id=requested_generation_id
          AND coverage.source_span_id IS NULL
    ) THEN RAISE EXCEPTION 'complete factual generation coverage is required' USING ERRCODE='55000'; END IF;
    IF EXISTS (
        SELECT 1 FROM public.brain_factual_generation_coverage coverage
        JOIN public.brain_graph_fact_events event
          ON event.owner_id=coverage.owner_id AND event.generation_id=coverage.generation_id
         AND event.source_span_id=coverage.source_span_id AND event.event_id=coverage.event_id
        WHERE coverage.owner_id=requested_owner_id AND coverage.generation_id=requested_generation_id
          AND coverage.disposition='assertion' AND EXISTS (
              SELECT 1 FROM public.brain_graph_fact_events later
              WHERE later.owner_id=event.owner_id AND later.generation_id=event.generation_id
                AND later.source_span_id=event.source_span_id AND later.supersedes_event_id=event.event_id
          )
    ) THEN RAISE EXCEPTION 'coverage must reference active factual events' USING ERRCODE='55000'; END IF;
    IF EXISTS (
        SELECT 1 FROM public.brain_factual_generation_coverage coverage
        WHERE coverage.owner_id=requested_owner_id AND coverage.generation_id=requested_generation_id
          AND coverage.disposition='exclusion' AND EXISTS (
              SELECT 1 FROM public.brain_graph_fact_events event
              WHERE event.owner_id=coverage.owner_id AND event.generation_id=coverage.generation_id
                AND event.source_span_id=coverage.source_span_id
                AND NOT EXISTS (
                    SELECT 1 FROM public.brain_graph_fact_events later
                    WHERE later.owner_id=event.owner_id AND later.generation_id=event.generation_id
                      AND later.source_span_id=event.source_span_id
                      AND later.supersedes_event_id=event.event_id
                )
          )
    ) THEN RAISE EXCEPTION 'exclusion conflicts with active factual event at snapshot publication'
        USING ERRCODE='55000'; END IF;
    SELECT COALESCE(max(system_receipt_sequence),0) INTO actual_high_water
    FROM public.brain_graph_fact_events
    WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id;
    IF actual_high_water<>requested_event_high_water THEN
        RAISE EXCEPTION 'factual event high-water mismatch' USING ERRCODE='55000';
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.brain_factual_contradiction_state
        WHERE owner_id=requested_owner_id AND generation_id=requested_generation_id
          AND state_after='active' AND severity='critical'
    ) THEN RAISE EXCEPTION 'active critical contradiction blocks snapshot' USING ERRCODE='55000'; END IF;
    actual_semantic_root := public.brain_compute_factual_semantic_root(
        requested_owner_id,requested_generation_id
    );
    IF requested_semantic_root_hash<>actual_semantic_root THEN
        RAISE EXCEPTION 'requested semantic root does not match database factual state' USING ERRCODE='55000';
    END IF;
    INSERT INTO public.brain_factual_graph_snapshots(
        owner_id,graph_snapshot_id,generation_id,semantic_root_hash,coverage_receipt_hash,
        membership_root_hash,event_high_water,valid_from,valid_to,snapshot_artifact_hash
    ) VALUES (requested_owner_id,requested_graph_snapshot_id,requested_generation_id,
        actual_semantic_root,requested_coverage_receipt_hash,actual_root,
        requested_event_high_water,requested_valid_from,requested_valid_to,requested_snapshot_artifact_hash);
    RETURN requested_graph_snapshot_id;
END;
$$;

CREATE FUNCTION public.brain_record_authority_epoch_event(
    requested_authority_scope_id BIGINT,requested_event_sequence BIGINT,requested_event_type TEXT,
    requested_authority_epoch BIGINT,requested_database_incarnation_id UUID,
    requested_predecessor_event_id BIGINT,requested_actor_id TEXT,
    requested_transition_receipt_hash TEXT
) RETURNS BIGINT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE scope_row public.brain_authority_scope_state%ROWTYPE;
        prior public.brain_authority_epoch_events%ROWTYPE; inserted_id BIGINT;
BEGIN
    SELECT * INTO scope_row FROM public.brain_authority_scope_state
    WHERE authority_scope_id=requested_authority_scope_id FOR UPDATE;
    IF NOT FOUND OR scope_row.state<>'active' THEN
        RAISE EXCEPTION 'active authority scope is required' USING ERRCODE='55000';
    END IF;
    SELECT * INTO prior FROM public.brain_authority_epoch_events
    WHERE authority_scope_id=requested_authority_scope_id
    ORDER BY event_sequence DESC LIMIT 1;
    IF prior.authority_epoch_event_id IS NULL THEN
        IF requested_event_sequence<>1 OR requested_event_type<>'granted'
           OR requested_predecessor_event_id IS NOT NULL
           OR requested_authority_epoch<>scope_row.authority_epoch
           OR requested_database_incarnation_id<>scope_row.database_incarnation_id THEN
            RAISE EXCEPTION 'first authority event must grant the scope epoch at sequence one' USING ERRCODE='55000';
        END IF;
    ELSE
        IF requested_event_sequence<>prior.event_sequence+1
           OR requested_predecessor_event_id<>prior.authority_epoch_event_id THEN
            RAISE EXCEPTION 'authority event must use the exact predecessor and next sequence' USING ERRCODE='55000';
        END IF;
        IF prior.event_type='granted' AND (
            requested_event_type<>'revoked'
            OR requested_authority_epoch<>prior.authority_epoch
            OR requested_database_incarnation_id<>prior.database_incarnation_id
        ) THEN
            RAISE EXCEPTION 'a grant may only transition to revoke in the same epoch' USING ERRCODE='55000';
        ELSIF prior.event_type='revoked' AND (
            requested_event_type<>'granted' OR requested_authority_epoch<=prior.authority_epoch
        ) THEN
            RAISE EXCEPTION 'a revoke may only transition to a newer-epoch grant' USING ERRCODE='55000';
        END IF;
    END IF;
    INSERT INTO public.brain_authority_epoch_events(
        authority_scope_id,event_sequence,event_type,authority_epoch,database_incarnation_id,
        predecessor_event_id,actor_id,transition_receipt_hash
    ) VALUES (requested_authority_scope_id,requested_event_sequence,requested_event_type,
        requested_authority_epoch,requested_database_incarnation_id,requested_predecessor_event_id,
        requested_actor_id,requested_transition_receipt_hash)
    RETURNING authority_epoch_event_id INTO inserted_id;
    RETURN inserted_id;
END;
$$;

CREATE FUNCTION public.brain_record_graph_approval_v5(
    requested_authority_scope_id BIGINT,requested_authority_epoch BIGINT,
    requested_database_incarnation_id UUID,requested_graph_snapshot_id TEXT,
    requested_approval_state TEXT,requested_approval_artifact_hash TEXT,
    requested_predecessor_deny_graph_approval_receipt_id BIGINT,
    requested_predecessor_deny_receipt_hash TEXT
) RETURNS BIGINT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE latest public.brain_authority_epoch_events%ROWTYPE;
        predecessor public.brain_graph_approval_receipts%ROWTYPE; inserted_id BIGINT;
BEGIN
    PERFORM 1 FROM public.brain_authority_scope_state
    WHERE authority_scope_id=requested_authority_scope_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'authority scope is missing' USING ERRCODE='55000'; END IF;
    SELECT * INTO latest FROM public.brain_authority_epoch_events
    WHERE authority_scope_id=requested_authority_scope_id
    ORDER BY event_sequence DESC LIMIT 1;
    IF latest.event_type IS DISTINCT FROM 'granted'
       OR latest.authority_epoch<>requested_authority_epoch
       OR latest.database_incarnation_id<>requested_database_incarnation_id THEN
        RAISE EXCEPTION 'latest authority event must be the matching grant' USING ERRCODE='55000';
    END IF;
    IF requested_approval_state='denied' THEN
        IF requested_predecessor_deny_graph_approval_receipt_id IS NOT NULL
           OR requested_predecessor_deny_receipt_hash IS NOT NULL THEN
            RAISE EXCEPTION 'denial cannot have a predecessor denial' USING ERRCODE='55000';
        END IF;
    ELSIF requested_approval_state='approved' THEN
        SELECT * INTO predecessor FROM public.brain_graph_approval_receipts
        WHERE graph_approval_receipt_id=requested_predecessor_deny_graph_approval_receipt_id
        FOR KEY SHARE;
        IF NOT FOUND OR predecessor.authority_scope_id<>requested_authority_scope_id
           OR predecessor.authority_epoch<>requested_authority_epoch
           OR predecessor.database_incarnation_id<>requested_database_incarnation_id
           OR predecessor.graph_snapshot_id<>requested_graph_snapshot_id
           OR predecessor.approval_state<>'denied'
           OR predecessor.approval_artifact_hash<>requested_predecessor_deny_receipt_hash THEN
            RAISE EXCEPTION 'matching predecessor denial is required' USING ERRCODE='55000';
        END IF;
    ELSE
        RAISE EXCEPTION 'unsupported graph approval state' USING ERRCODE='22023';
    END IF;
    INSERT INTO public.brain_graph_approval_receipts(
        authority_scope_id,authority_epoch,database_incarnation_id,graph_snapshot_id,
        approval_state,approval_artifact_hash,predecessor_deny_graph_approval_receipt_id,
        predecessor_deny_receipt_hash
    ) VALUES (requested_authority_scope_id,requested_authority_epoch,
        requested_database_incarnation_id,requested_graph_snapshot_id,requested_approval_state,
        requested_approval_artifact_hash,requested_predecessor_deny_graph_approval_receipt_id,
        requested_predecessor_deny_receipt_hash)
    RETURNING graph_approval_receipt_id INTO inserted_id;
    RETURN inserted_id;
END;
$$;

CREATE FUNCTION public.brain_bind_factual_snapshot_approval(
    requested_owner_id TEXT,requested_graph_snapshot_id TEXT,requested_graph_approval_receipt_id BIGINT
) RETURNS BIGINT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE approval public.brain_graph_approval_receipts%ROWTYPE;
        predecessor public.brain_graph_approval_receipts%ROWTYPE;
        latest public.brain_authority_epoch_events%ROWTYPE;
        snapshot public.brain_factual_graph_snapshots%ROWTYPE;
        validation_time TIMESTAMPTZ;
BEGIN
    SELECT * INTO approval FROM public.brain_graph_approval_receipts
    WHERE graph_approval_receipt_id=requested_graph_approval_receipt_id;
    IF NOT FOUND THEN RAISE EXCEPTION 'graph approval receipt is missing' USING ERRCODE='55000'; END IF;
    PERFORM 1 FROM public.brain_authority_scope_state
    WHERE authority_scope_id=approval.authority_scope_id AND owner_id=requested_owner_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'matching authority scope is required' USING ERRCODE='55000'; END IF;
    SELECT * INTO approval FROM public.brain_graph_approval_receipts
    WHERE graph_approval_receipt_id=requested_graph_approval_receipt_id FOR UPDATE;
    SELECT * INTO latest FROM public.brain_authority_epoch_events
    WHERE authority_scope_id=approval.authority_scope_id ORDER BY event_sequence DESC LIMIT 1;
    IF approval.approval_state<>'approved' OR approval.graph_snapshot_id<>requested_graph_snapshot_id
       OR latest.event_type IS DISTINCT FROM 'granted'
       OR latest.authority_epoch<>approval.authority_epoch
       OR latest.database_incarnation_id<>approval.database_incarnation_id THEN
        RAISE EXCEPTION 'latest authority event must be the matching grant' USING ERRCODE='55000';
    END IF;
    SELECT * INTO predecessor FROM public.brain_graph_approval_receipts
    WHERE graph_approval_receipt_id=approval.predecessor_deny_graph_approval_receipt_id FOR KEY SHARE;
    IF NOT FOUND OR predecessor.authority_scope_id<>approval.authority_scope_id
       OR predecessor.authority_epoch<>approval.authority_epoch
       OR predecessor.database_incarnation_id<>approval.database_incarnation_id
       OR predecessor.graph_snapshot_id<>approval.graph_snapshot_id
       OR predecessor.approval_state<>'denied'
       OR predecessor.approval_artifact_hash<>approval.predecessor_deny_receipt_hash THEN
        RAISE EXCEPTION 'matching predecessor denial is required' USING ERRCODE='55000';
    END IF;
    SELECT * INTO snapshot FROM public.brain_factual_graph_snapshots
    WHERE owner_id=requested_owner_id AND graph_snapshot_id=requested_graph_snapshot_id FOR KEY SHARE;
    IF NOT FOUND THEN RAISE EXCEPTION 'matching factual snapshot is required' USING ERRCODE='55000'; END IF;
    validation_time := clock_timestamp();
    IF snapshot.valid_from>validation_time
       OR (snapshot.valid_to IS NOT NULL AND snapshot.valid_to<=validation_time) THEN
        RAISE EXCEPTION 'factual snapshot must be currently valid at approval binding' USING ERRCODE='55000';
    END IF;
    INSERT INTO public.brain_factual_snapshot_approval_bindings(
        graph_approval_receipt_id,owner_id,graph_snapshot_id,authority_epoch_event_id
    ) VALUES (requested_graph_approval_receipt_id,requested_owner_id,
              requested_graph_snapshot_id,latest.authority_epoch_event_id);
    RETURN requested_graph_approval_receipt_id;
END;
$$;

CREATE FUNCTION public.brain_publish_v5_candidate(
    requested_owner_id TEXT,requested_campaign_id TEXT,requested_recommendation_lane TEXT,
    requested_execution_channel TEXT,requested_execution_scope TEXT,requested_authority_epoch BIGINT,
    requested_database_incarnation_id UUID,requested_candidate_decision_id TEXT,
    requested_semantic_content_hash TEXT,requested_candidate_artifact_hash TEXT,
    requested_envelope_id TEXT,requested_envelope_artifact_hash TEXT,
    requested_graph_approval_receipt_id BIGINT
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE scope_row public.brain_authority_scope_state%ROWTYPE;
        approval public.brain_graph_approval_receipts%ROWTYPE;
        predecessor public.brain_graph_approval_receipts%ROWTYPE;
        latest public.brain_authority_epoch_events%ROWTYPE;
        binding public.brain_factual_snapshot_approval_bindings%ROWTYPE;
        snapshot public.brain_factual_graph_snapshots%ROWTYPE;
        validation_time TIMESTAMPTZ;
BEGIN
    SELECT * INTO scope_row FROM public.brain_authority_scope_state
    WHERE owner_id=requested_owner_id AND campaign_id=requested_campaign_id
      AND recommendation_lane=requested_recommendation_lane
      AND execution_channel=requested_execution_channel
      AND execution_scope=requested_execution_scope FOR UPDATE;
    IF NOT FOUND OR scope_row.state<>'active' THEN
        RAISE EXCEPTION 'active authority scope is required' USING ERRCODE='55000';
    END IF;
    SELECT * INTO latest FROM public.brain_authority_epoch_events
    WHERE authority_scope_id=scope_row.authority_scope_id ORDER BY event_sequence DESC LIMIT 1;
    IF latest.event_type IS DISTINCT FROM 'granted'
       OR latest.authority_epoch<>requested_authority_epoch
       OR latest.database_incarnation_id<>requested_database_incarnation_id THEN
        RAISE EXCEPTION 'latest authority event must be the matching grant' USING ERRCODE='55000';
    END IF;
    SELECT * INTO approval FROM public.brain_graph_approval_receipts
    WHERE graph_approval_receipt_id=requested_graph_approval_receipt_id
      AND authority_scope_id=scope_row.authority_scope_id FOR UPDATE;
    SELECT * INTO binding FROM public.brain_factual_snapshot_approval_bindings
    WHERE graph_approval_receipt_id=requested_graph_approval_receipt_id FOR KEY SHARE;
    IF NOT FOUND OR approval.approval_state<>'approved'
       OR approval.authority_epoch<>latest.authority_epoch
       OR approval.database_incarnation_id<>latest.database_incarnation_id
       OR binding.owner_id<>requested_owner_id
       OR binding.graph_snapshot_id<>approval.graph_snapshot_id
       OR binding.authority_epoch_event_id<>latest.authority_epoch_event_id THEN
        RAISE EXCEPTION 'durable factual snapshot approval binding for the latest grant is required' USING ERRCODE='55000';
    END IF;
    SELECT * INTO predecessor FROM public.brain_graph_approval_receipts
    WHERE graph_approval_receipt_id=approval.predecessor_deny_graph_approval_receipt_id FOR KEY SHARE;
    IF NOT FOUND OR predecessor.authority_scope_id<>approval.authority_scope_id
       OR predecessor.authority_epoch<>approval.authority_epoch
       OR predecessor.database_incarnation_id<>approval.database_incarnation_id
       OR predecessor.graph_snapshot_id<>approval.graph_snapshot_id
       OR predecessor.approval_state<>'denied'
       OR predecessor.approval_artifact_hash<>approval.predecessor_deny_receipt_hash THEN
        RAISE EXCEPTION 'matching predecessor denial is required' USING ERRCODE='55000';
    END IF;
    SELECT * INTO snapshot FROM public.brain_factual_graph_snapshots
    WHERE owner_id=binding.owner_id AND graph_snapshot_id=binding.graph_snapshot_id FOR KEY SHARE;
    validation_time := clock_timestamp();
    IF NOT FOUND OR snapshot.valid_from>validation_time
       OR (snapshot.valid_to IS NOT NULL AND snapshot.valid_to<=validation_time) THEN
        RAISE EXCEPTION 'factual snapshot must be currently valid at candidate publication' USING ERRCODE='55000';
    END IF;
    INSERT INTO public.brain_v4_candidate_decisions(
        candidate_decision_id,authority_scope_id,semantic_content_hash,candidate_artifact_hash,
        graph_approval_receipt_id
    ) VALUES (requested_candidate_decision_id,scope_row.authority_scope_id,
        requested_semantic_content_hash,requested_candidate_artifact_hash,
        approval.graph_approval_receipt_id);
    INSERT INTO public.brain_v4_decision_envelopes(
        envelope_id,candidate_decision_id,envelope_artifact_hash
    ) VALUES (requested_envelope_id,requested_candidate_decision_id,requested_envelope_artifact_hash);
    INSERT INTO public.brain_immutable_artifact_references(
        artifact_hash,reference_type,subject_id
    ) VALUES (requested_candidate_artifact_hash,'candidate_payload',requested_candidate_decision_id),
             (requested_envelope_artifact_hash,'decision_envelope',requested_envelope_id),
             (approval.approval_artifact_hash,'graph_approval_receipt',approval.graph_snapshot_id),
             (predecessor.approval_artifact_hash,'predecessor_deny_receipt',predecessor.graph_snapshot_id)
    ON CONFLICT DO NOTHING;
    INSERT INTO public.brain_graph_approval_consumptions(
        graph_approval_receipt_id,candidate_decision_id,authority_scope_id
    ) VALUES (approval.graph_approval_receipt_id,requested_candidate_decision_id,
              scope_row.authority_scope_id);
    INSERT INTO public.brain_v5_candidate_publication_events(
        candidate_decision_id,graph_approval_receipt_id,authority_scope_id,authority_epoch_event_id
    ) VALUES (requested_candidate_decision_id,approval.graph_approval_receipt_id,
              scope_row.authority_scope_id,latest.authority_epoch_event_id);
    RETURN requested_candidate_decision_id;
END;
$$;

DO $$
DECLARE relation_name TEXT;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'brain_authority_epoch_events','brain_factual_ontology_manifests',
        'brain_factual_ontology_terms','brain_factual_ontology_closures',
        'brain_factual_generations','brain_factual_generation_members',
        'brain_factual_generation_closures','brain_factual_approval_receipts',
        'brain_graph_fact_events','brain_factual_approval_consumptions',
        'brain_factual_generation_coverage','brain_factual_contradictions',
        'brain_factual_contradiction_events','brain_factual_graph_snapshots',
        'brain_factual_snapshot_approval_bindings','brain_v5_candidate_publication_events'
    ] LOOP
        EXECUTE format('CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON public.%I '
            || 'FOR EACH ROW EXECUTE FUNCTION public.brain_reject_mutation()',
            relation_name || '_immutable', relation_name);
    END LOOP;
END $$;

REVOKE ALL PRIVILEGES ON TABLE
    public.brain_authority_epoch_events,
    public.brain_factual_ontology_manifests,public.brain_factual_ontology_terms,
    public.brain_factual_ontology_closures,public.brain_factual_generations,
    public.brain_factual_generation_members,public.brain_factual_generation_closures,
    public.brain_factual_approval_receipts,public.brain_graph_fact_events,
    public.brain_factual_approval_consumptions,public.brain_factual_generation_coverage,
    public.brain_factual_contradictions,public.brain_factual_contradiction_events,
    public.brain_factual_contradiction_state,public.brain_factual_graph_snapshots,
    public.brain_factual_snapshot_approval_bindings,public.brain_v5_candidate_publication_events
FROM PUBLIC,brain_policy_controller,brain_graph_authority,brain_candidate_reader,brain_candidate_writer;

REVOKE ALL PRIVILEGES ON SEQUENCE
    public.brain_authority_epoch_events_authority_epoch_event_id_seq,
    public.brain_factual_contradiction_events_contradiction_event_id_seq
FROM PUBLIC,brain_policy_controller,brain_graph_authority,brain_candidate_reader,brain_candidate_writer;

DO $$
DECLARE function_identity REGPROCEDURE;
BEGIN
    FOR function_identity IN
        SELECT function.oid::REGPROCEDURE
        FROM pg_proc function
        JOIN pg_namespace namespace ON namespace.oid=function.pronamespace
        WHERE namespace.nspname='public' AND function.proname=ANY(ARRAY[
            'brain_v5_sha256_text','brain_v5_frame','brain_compute_factual_ontology_root',
            'brain_compute_factual_membership_root','brain_compute_factual_semantic_root',
            'brain_reject_closed_ontology_term',
            'brain_reject_closed_generation_member','brain_create_factual_ontology',
            'brain_add_factual_ontology_term','brain_close_factual_ontology',
            'brain_create_factual_generation','brain_add_factual_generation_member',
            'brain_close_factual_generation','brain_admit_factual_event',
            'brain_record_factual_assertion_coverage','brain_review_factual_exclusion',
            'brain_create_factual_contradiction','brain_append_factual_contradiction_event',
            'brain_publish_factual_snapshot','brain_record_authority_epoch_event',
            'brain_record_graph_approval_v5','brain_bind_factual_snapshot_approval',
            'brain_publish_v5_candidate'
        ])
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON FUNCTION %s FROM PUBLIC,brain_policy_controller,brain_graph_authority,brain_candidate_reader,brain_candidate_writer',
            function_identity
        );
    END LOOP;
END $$;

REVOKE EXECUTE ON FUNCTION public.brain_publish_v4_candidate(
    TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT,UUID,TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT
) FROM brain_candidate_writer;

REVOKE CREATE ON SCHEMA public FROM brain_graph_authority;
GRANT USAGE ON SCHEMA public TO brain_graph_authority;

GRANT SELECT ON TABLE
    public.brain_authority_epoch_events,
    public.brain_factual_ontology_manifests,public.brain_factual_ontology_terms,
    public.brain_factual_ontology_closures,public.brain_factual_generations,
    public.brain_factual_generation_members,public.brain_factual_generation_closures,
    public.brain_factual_approval_receipts,public.brain_graph_fact_events,
    public.brain_factual_approval_consumptions,public.brain_factual_generation_coverage,
    public.brain_factual_contradictions,public.brain_factual_contradiction_events,
    public.brain_factual_contradiction_state,public.brain_factual_graph_snapshots,
    public.brain_factual_snapshot_approval_bindings,public.brain_v5_candidate_publication_events
TO brain_schema_verifier;

GRANT EXECUTE ON FUNCTION
    public.brain_v5_sha256_text(TEXT),
    public.brain_v5_frame(TEXT),
    public.brain_compute_factual_ontology_root(TEXT,TEXT),
    public.brain_compute_factual_membership_root(TEXT,TEXT),
    public.brain_compute_factual_semantic_root(TEXT,TEXT),
    public.brain_create_factual_ontology(TEXT,TEXT,TEXT),
    public.brain_add_factual_ontology_term(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT),
    public.brain_close_factual_ontology(TEXT,TEXT,BIGINT,TEXT,TEXT),
    public.brain_create_factual_generation(TEXT,TEXT,TEXT,TEXT,TEXT),
    public.brain_add_factual_generation_member(TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT),
    public.brain_close_factual_generation(TEXT,TEXT,BIGINT,TEXT,TEXT),
    public.brain_admit_factual_event(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT,TEXT,TEXT,TIMESTAMPTZ),
    public.brain_record_factual_assertion_coverage(TEXT,TEXT,TEXT,TEXT),
    public.brain_review_factual_exclusion(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TIMESTAMPTZ),
    public.brain_create_factual_contradiction(TEXT,TEXT,TEXT,TEXT),
    public.brain_append_factual_contradiction_event(TEXT,TEXT,BIGINT,TEXT,TEXT,TEXT,BIGINT,TEXT),
    public.brain_publish_factual_snapshot(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT,TIMESTAMPTZ,TIMESTAMPTZ,TEXT)
TO brain_graph_authority;

GRANT EXECUTE ON FUNCTION
    public.brain_record_authority_epoch_event(BIGINT,BIGINT,TEXT,BIGINT,UUID,BIGINT,TEXT,TEXT),
    public.brain_record_graph_approval_v5(BIGINT,BIGINT,UUID,TEXT,TEXT,TEXT,BIGINT,TEXT),
    public.brain_bind_factual_snapshot_approval(TEXT,TEXT,BIGINT)
TO brain_policy_controller;

GRANT EXECUTE ON FUNCTION public.brain_publish_v5_candidate(
    TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT,UUID,TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT
) TO brain_candidate_writer;

COMMENT ON TABLE public.brain_authority_epoch_events IS
    'Immutable sequenced authority epochs; grant, revoke, and newer-epoch regrant are serialized by scope.';
COMMENT ON TABLE public.brain_graph_approval_consumptions IS
    'Per-candidate use of a snapshot approval; receipt reuse is permitted, candidate identity reuse is not.';
