-- ApplyPilot canonical brain schema v2: least-privilege lifecycle principals.
-- Capability roles are cluster-level prerequisites created by the operator.

DO $$
DECLARE invalid_roles TEXT[];
BEGIN
    SELECT array_agg(rolname ORDER BY rolname) INTO invalid_roles
    FROM pg_roles
    WHERE rolname IN ('brain_status_reader','brain_policy_controller')
      AND (rolcanlogin OR rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication
           OR rolbypassrls OR rolinherit);
    IF (SELECT count(*) FROM pg_roles
        WHERE rolname IN ('brain_status_reader','brain_policy_controller'))<>2
       OR invalid_roles IS NOT NULL THEN
        RAISE EXCEPTION 'fixed lifecycle capability roles are missing or have unsafe attributes: %',
            COALESCE(invalid_roles,ARRAY[]::TEXT[]);
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.brain_require_controller() RETURNS VOID
LANGUAGE plpgsql AS $$
BEGIN
    IF current_user <> 'brain_schema_migrator'
       OR NOT EXISTS (
           WITH RECURSIVE memberships(roleid) AS (
               SELECT oid FROM pg_roles WHERE rolname=session_user
               UNION
               SELECT member.roleid FROM pg_auth_members member
               JOIN memberships prior ON prior.roleid=member.member
           )
           SELECT 1 FROM memberships
           WHERE roleid IN (
               SELECT oid FROM pg_roles
               WHERE rolname IN ('brain_schema_migrator','brain_policy_controller')
           )
       ) THEN
        RAISE EXCEPTION 'brain lifecycle controller role is required' USING ERRCODE='42501';
    END IF;
    IF EXISTS (
        WITH RECURSIVE memberships(roleid) AS (
            SELECT oid FROM pg_roles WHERE rolname=session_user
            UNION
            SELECT member.roleid FROM pg_auth_members member
            JOIN memberships prior ON prior.roleid=member.member
        )
        SELECT 1 FROM memberships JOIN pg_roles role ON role.oid=memberships.roleid
        WHERE role.rolname='brain_policy_controller'
    ) THEN
        IF EXISTS (
               WITH RECURSIVE memberships(roleid) AS (
                   SELECT oid FROM pg_roles WHERE rolname=session_user
                   UNION
                   SELECT member.roleid FROM pg_auth_members member
                   JOIN memberships prior ON prior.roleid=member.member
               )
               SELECT 1 FROM memberships JOIN pg_roles role ON role.oid=memberships.roleid
               WHERE role.rolname='brain_schema_migrator'
           )
           OR EXISTS (
               WITH RECURSIVE memberships(roleid) AS (
                   SELECT oid FROM pg_roles WHERE rolname=session_user
                   UNION
                   SELECT member.roleid FROM pg_auth_members member
                   JOIN memberships prior ON prior.roleid=member.member
               )
               SELECT 1 FROM memberships JOIN pg_roles role ON role.oid=memberships.roleid
               WHERE role.rolname NOT IN (session_user,'brain_policy_controller')
           )
           OR EXISTS (
               SELECT 1 FROM pg_roles WHERE rolname=session_user
                 AND (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls)
           )
           OR has_schema_privilege(session_user,'public','CREATE')
           OR EXISTS (
               SELECT 1 FROM pg_class relation JOIN pg_namespace namespace
                 ON namespace.oid=relation.relnamespace
               WHERE namespace.nspname='public' AND relation.relkind IN ('r','p','v','m','f')
                 AND (has_table_privilege(session_user,relation.oid,
                     'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')
                     OR has_any_column_privilege(session_user,relation.oid,
                        'INSERT,UPDATE,REFERENCES'))
           )
           OR EXISTS (
               SELECT 1 FROM pg_proc function JOIN pg_namespace namespace
                 ON namespace.oid=function.pronamespace
               CROSS JOIN LATERAL aclexplode(function.proacl) acl
               WHERE namespace.nspname='public'
                 AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=session_user)
           ) THEN
            RAISE EXCEPTION 'controller login has authority outside its lifecycle capability'
                USING ERRCODE='42501';
        END IF;
    END IF;
END;
$$;

CREATE TABLE public.brain_canary_lifecycle_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_version TEXT NOT NULL REFERENCES public.brain_decision_policies(policy_version),
    lane TEXT NOT NULL,
    event_type TEXT NOT NULL,
    prior_ats_pause_source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT brain_canary_lifecycle_events_lane_check
      CHECK (lane IN ('ats','linkedin')),
    CONSTRAINT brain_canary_lifecycle_events_type_check
      CHECK (event_type IN ('armed','stopped'))
);
CREATE TRIGGER brain_canary_lifecycle_events_append_only
BEFORE UPDATE OR DELETE ON public.brain_canary_lifecycle_events
FOR EACH ROW EXECUTE FUNCTION public.brain_reject_mutation();
CREATE TRIGGER brain_canary_lifecycle_events_append_only_truncate
BEFORE TRUNCATE ON public.brain_canary_lifecycle_events
FOR EACH STATEMENT EXECUTE FUNCTION public.brain_reject_mutation();
REVOKE ALL PRIVILEGES ON TABLE public.brain_canary_lifecycle_events
FROM PUBLIC,brain_status_reader,brain_policy_controller;

CREATE FUNCTION public.brain_controller_transition_policy(
    requested_policy_version TEXT,
    requested_lifecycle TEXT,
    expected_lane TEXT DEFAULT NULL
) RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE policy_row public.brain_decision_policies%ROWTYPE; cfg public.fleet_config%ROWTYPE;
BEGIN
    PERFORM public.brain_require_controller();
    SELECT * INTO policy_row FROM public.brain_decision_policies
    WHERE policy_version=requested_policy_version FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'unknown policy_version' USING ERRCODE='P0002'; END IF;
    IF expected_lane IS NOT NULL AND policy_row.lane<>expected_lane THEN
        RAISE EXCEPTION 'policy lane does not match expected lane' USING ERRCODE='23514';
    END IF;
    PERFORM pg_advisory_xact_lock(hashtext('brain-policy-lane'),hashtext(policy_row.lane));
    PERFORM 1 FROM public.fleet_decision_policies
      WHERE lane=policy_row.lane FOR UPDATE;
    SELECT * INTO STRICT cfg FROM public.fleet_config WHERE id=1 FOR UPDATE;
    IF NOT cfg.paused OR cfg.ats_apply_mode<>'stopped' OR cfg.linkedin_apply_mode<>'stopped'
       OR cfg.canary_enabled OR cfg.linkedin_canary_enabled
       OR (policy_row.lane='ats' AND NOT cfg.ats_paused) THEN
        RAISE EXCEPTION 'policy transition requires global pause and both stopped lane gates'
            USING ERRCODE='55000';
    END IF;
    PERFORM public.brain_transition_policy(requested_policy_version,requested_lifecycle);
    IF requested_lifecycle IN ('validated','canary') THEN
        INSERT INTO public.fleet_decision_policies(policy_version,lane,status)
        VALUES(requested_policy_version,policy_row.lane,requested_lifecycle)
        ON CONFLICT(policy_version) DO UPDATE SET status=EXCLUDED.status
        WHERE public.fleet_decision_policies.lane=EXCLUDED.lane
          AND public.fleet_decision_policies.status NOT IN ('active','retired');
        IF NOT FOUND THEN
            RAISE EXCEPTION 'fleet policy staging conflict' USING ERRCODE='55000';
        END IF;
    END IF;
    SELECT * INTO policy_row FROM public.brain_decision_policies
    WHERE policy_version=requested_policy_version;
    RETURN to_jsonb(policy_row);
END;
$$;

CREATE FUNCTION public.brain_controller_arm_canary(
    requested_policy_version TEXT,
    requested_lane TEXT,
    requested_capacity INTEGER,
    expected_ats_pause_source TEXT,
    expect_null_ats_pause_source BOOLEAN,
    heartbeat_max_age_seconds INTEGER DEFAULT 90
) RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE policy_row public.brain_decision_policies%ROWTYPE;
        cfg public.fleet_config%ROWTYPE; fleet_row public.fleet_decision_policies%ROWTYPE;
        worker_id TEXT; worker_ip TEXT; expected_version TEXT; candidate_url TEXT;
        brain_active TEXT; fleet_active TEXT; configured_policy TEXT;
        ats_leases BIGINT; linkedin_leases BIGINT;
BEGIN
    PERFORM public.brain_require_controller();
    IF requested_lane NOT IN ('ats','linkedin') OR requested_capacity NOT BETWEEN 1 AND 100
       OR heartbeat_max_age_seconds NOT BETWEEN 15 AND 120 THEN
        RAISE EXCEPTION 'invalid canary lane, capacity, or heartbeat window' USING ERRCODE='23514';
    END IF;
    IF requested_lane='ats' AND NOT expect_null_ats_pause_source
       AND expected_ats_pause_source IS NULL THEN
        RAISE EXCEPTION 'ATS pause source expectation is required' USING ERRCODE='23514';
    ELSIF requested_lane='linkedin'
       AND (expect_null_ats_pause_source OR expected_ats_pause_source IS NOT NULL) THEN
        RAISE EXCEPTION 'ATS pause source expectation is invalid for LinkedIn' USING ERRCODE='23514';
    END IF;

    SELECT * INTO policy_row FROM public.brain_decision_policies
    WHERE policy_version=requested_policy_version FOR UPDATE;
    IF NOT FOUND OR policy_row.lane<>requested_lane OR policy_row.lifecycle<>'canary' THEN
        RAISE EXCEPTION 'matching canary lifecycle policy is required' USING ERRCODE='55000';
    END IF;
    PERFORM pg_advisory_xact_lock(hashtext('brain-policy-lane'),hashtext(requested_lane));
    SELECT * INTO fleet_row FROM public.fleet_decision_policies
    WHERE policy_version=requested_policy_version FOR UPDATE;
    IF NOT FOUND OR fleet_row.lane<>requested_lane OR fleet_row.status<>'canary' THEN
        RAISE EXCEPTION 'matching staged fleet canary policy is required' USING ERRCODE='55000';
    END IF;
    SELECT * INTO STRICT cfg FROM public.fleet_config WHERE id=1 FOR UPDATE;
    IF NOT cfg.paused OR cfg.ats_apply_mode<>'stopped' OR cfg.linkedin_apply_mode<>'stopped'
       OR cfg.canary_enabled OR cfg.linkedin_canary_enabled
       OR (requested_lane='ats' AND NOT cfg.ats_paused) THEN
        RAISE EXCEPTION 'global pause and both stopped lane gates are required' USING ERRCODE='55000';
    END IF;
    IF requested_lane='ats' AND (
         (expect_null_ats_pause_source AND cfg.ats_pause_source IS NOT NULL)
         OR (NOT expect_null_ats_pause_source
             AND cfg.ats_pause_source IS DISTINCT FROM expected_ats_pause_source)) THEN
        RAISE EXCEPTION 'ATS pause source changed' USING ERRCODE='40001';
    END IF;

    SELECT policy_version INTO brain_active FROM public.brain_decision_policies
    WHERE lane=requested_lane AND lifecycle='active';
    SELECT policy_version INTO fleet_active FROM public.fleet_decision_policies
    WHERE lane=requested_lane AND status='active';
    configured_policy:=CASE requested_lane WHEN 'ats' THEN cfg.ats_policy_version
                       ELSE cfg.linkedin_policy_version END;
    IF brain_active IS DISTINCT FROM fleet_active
       OR configured_policy IS DISTINCT FROM brain_active THEN
        RAISE EXCEPTION 'brain, fleet, and config active bindings disagree' USING ERRCODE='55000';
    END IF;
    IF cfg.pinned_worker_version IS NULL OR cfg.canary_worker_id IS NULL THEN
        RAISE EXCEPTION 'pinned release and canary worker are required' USING ERRCODE='55000';
    END IF;
    expected_version:=COALESCE(cfg.canary_version,cfg.pinned_worker_version);
    SELECT w.worker_id,w.public_ip INTO worker_id,worker_ip
    FROM public.workers w
    JOIN public.worker_heartbeat h USING(worker_id)
    JOIN public.fleet_worker_principals principal USING(worker_id)
    JOIN public.fleet_desired_state desired ON desired.machine_owner=w.machine_owner
    WHERE w.worker_id=cfg.canary_worker_id AND w.revoked_at IS NULL AND w.validated
      AND principal.contract=CASE requested_lane WHEN 'ats' THEN 'apply' ELSE 'linkedin' END
      AND COALESCE((w.capabilities->>CASE requested_lane WHEN 'ats' THEN 'can_ats'
                                    ELSE 'can_linkedin' END)::boolean,FALSE)
      AND NULLIF(btrim(w.public_ip),'') IS NOT NULL
      AND desired.desired_workers>0 AND desired.updated_at>=now()-interval '5 minutes'
      AND h.sw_version=expected_version
      AND h.last_beat>=now()-(heartbeat_max_age_seconds*interval '1 second')
    FOR SHARE OF w,h,principal,desired;
    IF worker_id IS NULL THEN
        RAISE EXCEPTION 'fresh desired validated exact-version canary worker is required'
            USING ERRCODE='55000';
    END IF;
    IF requested_lane='linkedin' AND cfg.linkedin_owner_ip IS DISTINCT FROM worker_ip THEN
        RAISE EXCEPTION 'LinkedIn worker IP must equal owner IP' USING ERRCODE='55000';
    END IF;
    SELECT count(*) INTO ats_leases FROM public.apply_queue
    WHERE status='leased' OR lease_owner IS NOT NULL OR lease_expires_at IS NOT NULL;
    SELECT count(*) INTO linkedin_leases FROM public.linkedin_queue
    WHERE status='leased' OR lease_owner IS NOT NULL OR lease_expires_at IS NOT NULL;
    IF ats_leases<>0 OR linkedin_leases<>0 THEN
        RAISE EXCEPTION 'zero outstanding leases are required in both lanes' USING ERRCODE='55000';
    END IF;

    INSERT INTO public.brain_canary_lifecycle_events(
      policy_version,lane,event_type,prior_ats_pause_source)
    VALUES(requested_policy_version,requested_lane,'armed',cfg.ats_pause_source);
    IF requested_lane='ats' THEN
        SELECT q.url INTO candidate_url
        FROM public.apply_queue q
        JOIN public.rate_governor host
          ON host.scope_key='host:'||COALESCE(q.target_host,q.apply_domain)
        JOIN public.rate_governor home ON home.scope_key='home_ip:'||worker_ip
        JOIN public.rate_governor glob ON glob.scope_key='global'
        WHERE q.status='queued' AND q.lane='ats' AND q.approved_batch IS NOT NULL
          AND q.decision_id IS NOT NULL AND q.policy_version=requested_policy_version
          AND q.decision_action='apply' AND q.qualification_verdict='qualified'
          AND q.qualification_score>=q.qualification_floor
          AND q.decision_expires_at>now() AND q.score=q.final_score
          AND (cfg.approval_threshold IS NULL OR q.final_score>=cfg.approval_threshold)
          AND COALESCE(q.apply_error,'') NOT ILIKE 'requeued_by_%'
          AND NOT EXISTS (SELECT 1 FROM public.apply_result_events prior
              WHERE prior.queue_name='apply_queue' AND prior.url=q.url
                AND (COALESCE(prior.application_tool_calls,0)>0
                     OR COALESCE(prior.apply_error,'') ILIKE 'requeued_by_%'))
          AND NOT EXISTS (SELECT 1 FROM public.apply_attempts a WHERE a.dedup_key=q.dedup_key
              AND a.state IN ('submit_started','submitted_unverified'))
          AND NOT EXISTS (SELECT 1 FROM public.applied_set a WHERE a.dedup_key=q.dedup_key)
          AND NOT EXISTS (SELECT 1 FROM public.fleet_worker_blocklist b
              WHERE (b.kind='company' AND lower(btrim(COALESCE(q.company,'')))=b.value)
                 OR (b.kind='pattern' AND
                     (q.url ILIKE b.value OR COALESCE(q.application_url,'') ILIKE b.value)))
          AND (COALESCE(cfg.spend_cap_usd,0)<=0 OR
               (SELECT COALESCE(sum(x.cumulative_cost_usd),0) FROM public.apply_queue x)
                 < cfg.spend_cap_usd)
          AND glob.count_24h<glob.daily_cap
          AND COALESCE(glob.breaker_state,'ok') NOT IN ('paused','demoted')
          AND home.count_24h<home.daily_cap AND COALESCE(home.breaker_state,'ok')<>'demoted'
          AND NOT (COALESCE(home.breaker_state,'ok')='paused'
                   AND COALESCE(home.breaker_until,'infinity'::timestamptz)>=now())
          AND host.count_24h<host.daily_cap AND COALESCE(host.breaker_state,'ok')<>'demoted'
          AND NOT (COALESCE(host.breaker_state,'ok')='paused'
                   AND COALESCE(host.breaker_until,'infinity'::timestamptz)>=now())
          AND COALESCE(host.doctor_skip_until,'-infinity'::timestamptz)<now()
          AND (COALESCE(host.last_applied_at,host.last_attempt_at) IS NULL OR
               COALESCE(host.last_applied_at,host.last_attempt_at)<now()-make_interval(
                 secs=>GREATEST(COALESCE(host.min_gap_seconds,90),
                                COALESCE(host.doctor_min_gap_floor,0))))
          AND (NOT COALESCE(q.liveness_required,FALSE) OR
               (q.liveness_status='live' AND q.liveness_checked_at>=now()-interval '15 minutes'))
          AND (NOT COALESCE(q.eligibility_required,FALSE) OR q.eligibility_status='eligible')
          AND (NOT COALESCE(q.routing_required,FALSE) OR q.execution_route='deterministic')
        ORDER BY q.score DESC,q.url LIMIT 1 FOR UPDATE OF q,host,home,glob;
    ELSE
        SELECT q.url INTO candidate_url
        FROM public.linkedin_queue q
        JOIN public.rate_governor account ON account.scope_key='account:linkedin'
        JOIN public.rate_governor glob ON glob.scope_key='global'
        WHERE q.status='queued' AND q.lane='linkedin' AND q.approved_batch IS NOT NULL
          AND q.score>=GREATEST(COALESCE(cfg.approval_threshold,7),7)
          AND q.linkedin_resolve_status IN ('easy_apply','resolved_offsite')
          AND q.linkedin_resolved_at>=now()-interval '3 days'
          AND q.decision_id IS NOT NULL AND q.policy_version=requested_policy_version
          AND q.decision_action='apply' AND q.qualification_verdict='qualified'
          AND q.qualification_score>=q.qualification_floor
          AND q.decision_expires_at>now() AND q.score=q.final_score
          AND (account.halted_until IS NULL OR account.halted_until<now())
          AND account.count_24h<account.daily_cap
          AND COALESCE(account.breaker_state,'ok')<>'demoted'
          AND NOT (COALESCE(account.breaker_state,'ok')='paused'
                   AND COALESCE(account.breaker_until,'infinity'::timestamptz)>=now())
          AND (account.last_applied_at IS NULL OR account.last_applied_at<now()-make_interval(
               secs=>COALESCE(account.min_gap_seconds,1200)))
          AND glob.count_24h<glob.daily_cap
          AND COALESCE(glob.breaker_state,'ok') NOT IN ('paused','demoted')
          AND NOT EXISTS (SELECT 1 FROM public.applied_set d WHERE d.dedup_key=q.dedup_key)
          AND NOT EXISTS (SELECT 1 FROM public.fleet_worker_blocklist b
              WHERE (b.kind='company' AND lower(btrim(COALESCE(q.company,'')))=b.value)
                 OR (b.kind='pattern' AND
                     (q.url ILIKE b.value OR COALESCE(q.application_url,'') ILIKE b.value)))
        ORDER BY q.score DESC,q.url LIMIT 1 FOR UPDATE OF q,account,glob;
    END IF;
    IF candidate_url IS NULL THEN
        RAISE EXCEPTION 'a currently leaseable approved candidate is required' USING ERRCODE='55000';
    END IF;

    IF requested_lane='ats' THEN
        UPDATE public.fleet_config SET paused=FALSE,ats_paused=FALSE,
          ats_pause_source='canonical_canary:'||requested_policy_version,
          ats_policy_version=requested_policy_version,ats_apply_mode='canary',
          canary_enabled=TRUE,canary_remaining=requested_capacity,
          linkedin_apply_mode='stopped',linkedin_canary_enabled=FALSE,
          linkedin_canary_remaining=NULL,updated_at=now() WHERE id=1;
    ELSE
        UPDATE public.fleet_config SET paused=FALSE,
          linkedin_policy_version=requested_policy_version,linkedin_apply_mode='canary',
          linkedin_canary_enabled=TRUE,linkedin_canary_remaining=requested_capacity,
          ats_apply_mode='stopped',canary_enabled=FALSE,canary_remaining=NULL,
          updated_at=now() WHERE id=1;
    END IF;
    RETURN jsonb_build_object('policy_version',requested_policy_version,'lane',requested_lane,
      'capacity',requested_capacity,'worker_id',worker_id,'prior_active_policy',brain_active,
      'pinned_worker_version',cfg.pinned_worker_version,'expected_worker_version',expected_version,
      'candidate_url',candidate_url,'fleet_config',(SELECT jsonb_build_object(
        'paused',paused,'ats_paused',ats_paused,'ats_pause_source',ats_pause_source,
        'ats_apply_mode',ats_apply_mode,'linkedin_apply_mode',linkedin_apply_mode,
        'canary_enabled',canary_enabled,'linkedin_canary_enabled',linkedin_canary_enabled,
        'canary_remaining',canary_remaining,'linkedin_canary_remaining',linkedin_canary_remaining,
        'ats_policy_version',ats_policy_version,'linkedin_policy_version',linkedin_policy_version,
        'pinned_worker_version',pinned_worker_version,'canary_worker_id',canary_worker_id,
        'canary_version',canary_version,'linkedin_owner_ip',linkedin_owner_ip)
        FROM public.fleet_config WHERE id=1));
END;
$$;

CREATE FUNCTION public.brain_controller_stop_canary(requested_lane TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE cfg public.fleet_config%ROWTYPE; candidate public.brain_decision_policies%ROWTYPE;
        candidate_version TEXT; brain_active TEXT; fleet_active TEXT;
        prior_ats_pause_source TEXT;
BEGIN
    PERFORM public.brain_require_controller();
    IF requested_lane NOT IN ('ats','linkedin') THEN
        RAISE EXCEPTION 'invalid canary lane' USING ERRCODE='23514';
    END IF;
    SELECT CASE requested_lane WHEN 'ats' THEN ats_policy_version ELSE linkedin_policy_version END
      INTO candidate_version FROM public.fleet_config WHERE id=1;
    IF candidate_version IS NULL THEN RAISE EXCEPTION 'selected lane has no candidate' USING ERRCODE='55000'; END IF;
    SELECT * INTO candidate FROM public.brain_decision_policies
      WHERE policy_version=candidate_version FOR UPDATE;
    IF NOT FOUND OR candidate.lane<>requested_lane OR candidate.lifecycle<>'canary' THEN
        RAISE EXCEPTION 'configured candidate is not the selected canary' USING ERRCODE='55000';
    END IF;
    PERFORM pg_advisory_xact_lock(hashtext('brain-policy-lane'),hashtext(requested_lane));
    PERFORM 1 FROM public.fleet_decision_policies
      WHERE policy_version=candidate_version AND lane=requested_lane AND status='canary' FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'fleet candidate is not the selected canary' USING ERRCODE='55000'; END IF;
    SELECT * INTO STRICT cfg FROM public.fleet_config WHERE id=1 FOR UPDATE;
    IF (requested_lane='ats' AND (cfg.ats_policy_version IS DISTINCT FROM candidate_version
          OR NOT cfg.canary_enabled
          OR NOT (cfg.ats_apply_mode='canary'
                  OR (cfg.ats_apply_mode='stopped' AND cfg.canary_remaining=0))))
       OR (requested_lane='linkedin' AND (cfg.linkedin_policy_version IS DISTINCT FROM candidate_version
          OR NOT cfg.linkedin_canary_enabled
          OR NOT (cfg.linkedin_apply_mode='canary'
                  OR (cfg.linkedin_apply_mode='stopped' AND cfg.linkedin_canary_remaining=0)))) THEN
        RAISE EXCEPTION 'selected lane is not the armed canary' USING ERRCODE='55000';
    END IF;
    SELECT policy_version INTO brain_active FROM public.brain_decision_policies
      WHERE lane=requested_lane AND lifecycle='active';
    SELECT policy_version INTO fleet_active FROM public.fleet_decision_policies
      WHERE lane=requested_lane AND status='active';
    IF brain_active IS DISTINCT FROM fleet_active THEN
        RAISE EXCEPTION 'brain and fleet active bindings disagree' USING ERRCODE='55000';
    END IF;
    SELECT event.prior_ats_pause_source INTO prior_ats_pause_source
      FROM public.brain_canary_lifecycle_events event
      WHERE event.policy_version=candidate_version AND event.lane=requested_lane
        AND event.event_type='armed'
      ORDER BY event.event_id DESC LIMIT 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'armed canary provenance receipt is missing' USING ERRCODE='55000';
    END IF;
    INSERT INTO public.brain_canary_lifecycle_events(
      policy_version,lane,event_type,prior_ats_pause_source)
    VALUES(candidate_version,requested_lane,'stopped',prior_ats_pause_source);
    IF requested_lane='ats' THEN
        UPDATE public.fleet_config SET paused=TRUE,ats_paused=TRUE,
          ats_pause_source=prior_ats_pause_source,ats_policy_version=brain_active,
          ats_apply_mode='stopped',canary_enabled=FALSE,canary_remaining=NULL,
          linkedin_apply_mode='stopped',linkedin_canary_enabled=FALSE,
          linkedin_canary_remaining=NULL,updated_at=now() WHERE id=1;
    ELSE
        UPDATE public.fleet_config SET paused=TRUE,linkedin_policy_version=brain_active,
          linkedin_apply_mode='stopped',linkedin_canary_enabled=FALSE,
          linkedin_canary_remaining=NULL,ats_apply_mode='stopped',canary_enabled=FALSE,
          canary_remaining=NULL,updated_at=now() WHERE id=1;
    END IF;
    RETURN jsonb_build_object('lane',requested_lane,'candidate_policy',candidate_version,
      'restored_active_policy',brain_active,'fleet_config',(SELECT jsonb_build_object(
        'paused',paused,'ats_paused',ats_paused,'ats_pause_source',ats_pause_source,
        'ats_apply_mode',ats_apply_mode,'linkedin_apply_mode',linkedin_apply_mode,
        'canary_enabled',canary_enabled,'linkedin_canary_enabled',linkedin_canary_enabled,
        'canary_remaining',canary_remaining,'linkedin_canary_remaining',linkedin_canary_remaining,
        'ats_policy_version',ats_policy_version,'linkedin_policy_version',linkedin_policy_version)
        FROM public.fleet_config WHERE id=1));
END;
$$;

REVOKE ALL PRIVILEGES ON FUNCTION public.brain_transition_policy(TEXT,TEXT),
    public.brain_controller_transition_policy(TEXT,TEXT,TEXT),
    public.brain_controller_arm_canary(TEXT,TEXT,INTEGER,TEXT,BOOLEAN,INTEGER),
    public.brain_controller_stop_canary(TEXT)
FROM PUBLIC,brain_status_reader,brain_policy_controller;
GRANT EXECUTE ON FUNCTION
    public.brain_controller_transition_policy(TEXT,TEXT,TEXT),
    public.brain_controller_arm_canary(TEXT,TEXT,INTEGER,TEXT,BOOLEAN,INTEGER),
    public.brain_controller_stop_canary(TEXT)
TO brain_policy_controller;

REVOKE ALL PRIVILEGES ON SCHEMA public FROM brain_status_reader,brain_policy_controller;
GRANT USAGE ON SCHEMA public TO brain_status_reader,brain_policy_controller;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public
FROM brain_status_reader,brain_policy_controller;
GRANT SELECT ON TABLE public.brain_decision_policies,public.brain_policy_artifacts,
    public.brain_policy_approvals,public.brain_parity_runs,public.brain_parity_run_events
TO brain_status_reader;
GRANT SELECT (id,paused,ats_paused,ats_pause_source,ats_apply_mode,linkedin_apply_mode,
    canary_enabled,linkedin_canary_enabled,canary_remaining,linkedin_canary_remaining,
    ats_policy_version,linkedin_policy_version,pinned_worker_version,canary_worker_id,
    canary_version,approval_threshold,spend_cap_usd,linkedin_owner_ip)
ON public.fleet_config TO brain_status_reader;
GRANT SELECT (policy_version,lane,status,activated_at,retired_at)
ON public.fleet_decision_policies TO brain_status_reader;
GRANT SELECT (status,lease_owner,lease_expires_at,decision_id,policy_version,decision_action,
    qualification_verdict,qualification_score,qualification_floor,preference_score,
    outcome_score,final_score,decision_confidence,decision_created_at,decision_expires_at,input_hash)
ON public.apply_queue,public.linkedin_queue TO brain_status_reader;
