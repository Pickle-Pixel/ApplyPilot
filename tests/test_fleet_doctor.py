"""FLEET DOCTOR v1 tests. Uses the shared ``fleet_db`` fixture (disposable PG, clean v3
schema per test). Covers: clustering by reason/host; the four conservative auto-fixes; the
D1 gate rejecting every forbidden (activity-increasing / LinkedIn) action; idempotency;
the TTL sweep; and that non-auto-fixable clusters become recommendations (not auto-applied).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from applypilot.apply import pgqueue
from applypilot.fleet import doctor, queue


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------
def _seed_apply_failure(
    conn,
    *,
    url,
    host,
    worker_id="home-0",
    apply_error=None,
    apply_status=None,
    status="failed",
    approved_batch="b1",
    queued=False,
    score=8.5,
):
    """Insert one apply_queue row carrying a failure signal (recent updated_at). When
    ``queued`` is True the row is left status='queued' + approved (so host_skip can un-approve
    it); otherwise it's a terminal failure row used for clustering."""
    st = "queued" if queued else status
    with conn.cursor() as cur:
        # Columns (12 placeholders): url, application_url, apply_domain, target_host,
        # dedup_key, approved_batch, status, apply_error, apply_status, worker_id.
        # company/title/score/lane/updated_at are literals/now().
        cur.execute(
            "INSERT INTO apply_queue (url, company, title, application_url, score, "
            "apply_domain, target_host, lane, dedup_key, approved_batch, status, "
            "apply_error, apply_status, worker_id, updated_at) "
            "VALUES (%s,'Co','T',%s,%s,%s,%s,'ats',%s,%s,%s,%s,%s,%s, now()) "
            "ON CONFLICT (url) DO UPDATE SET apply_error=EXCLUDED.apply_error, "
            "apply_status=EXCLUDED.apply_status, status=EXCLUDED.status, "
            "approved_batch=EXCLUDED.approved_batch, target_host=EXCLUDED.target_host, "
            "worker_id=EXCLUDED.worker_id, updated_at=now()",
            (url, url, score, host, host, url, approved_batch, st, apply_error, apply_status, worker_id),
        )
        if queued:
            policy = "doctor-ats-policy"
            now = datetime.now(timezone.utc)
            cur.execute(
                "INSERT INTO fleet_decision_policies (policy_version,lane,status) "
                "VALUES (%s,'ats','active') ON CONFLICT (policy_version) DO UPDATE SET status='active'",
                (policy,),
            )
            cur.execute(
                "UPDATE fleet_config SET ats_policy_version=%s WHERE id=1",
                (policy,),
            )
            cur.execute(
                "UPDATE apply_queue SET decision_id=%s, policy_version=%s, decision_action='apply', "
                "qualification_verdict='qualified', qualification_score=9, qualification_floor=0, "
                "preference_score=8, outcome_score=8, final_score=score, decision_confidence=.9, "
                "decision_created_at=%s, decision_expires_at=%s, input_hash=%s WHERE url=%s",
                (f"decision-{url}", policy, now, now + timedelta(days=1), f"hash-{url}", url),
            )
    conn.commit()


def _queued_count_for_host(conn, host, *, approved_only=True):
    with conn.cursor() as cur:
        if approved_only:
            cur.execute(
                "SELECT count(*) AS n FROM apply_queue WHERE COALESCE(target_host,apply_domain)=%s "
                "AND status='queued' AND approved_batch IS NOT NULL", (host,))
        else:
            cur.execute(
                "SELECT count(*) AS n FROM apply_queue WHERE COALESCE(target_host,apply_domain)=%s "
                "AND status='queued'", (host,))
        return cur.fetchone()["n"]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def test_clustering_by_reason_and_host(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        _seed_apply_failure(conn, url="u1", host="acme.com", apply_error="blocked")
        _seed_apply_failure(conn, url="u2", host="acme.com", apply_error="captcha")
        _seed_apply_failure(conn, url="u3", host="other.com", apply_error="failed:timeout")
        clusters = doctor.analyze(conn, window_minutes=60)
    hf = clusters["host_failures"]
    assert hf["acme.com"]["hard_block"] == 2
    assert hf["other.com"]["timeout"] == 1
    # hard-block share of all apply-lane failures = 2/3
    assert clusters["lane_failures"] == 3
    assert abs(clusters["lane_hard_block_rate"] - (2 / 3)) < 1e-9


def test_rate_limited_substring_maps_to_hard_block(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        _seed_apply_failure(conn, url="u1", host="h.com", apply_error="host_rate_limited")
        clusters = doctor.analyze(conn, window_minutes=60)
    assert clusters["host_failures"]["h.com"]["hard_block"] == 1


def test_modern_failure_taxonomy_maps_into_doctor_action_groups():
    assert doctor._reason_group("failed:browser_unavailable") == "agent"
    assert doctor._reason_group("failed:no_confirmation") == "agent"
    assert doctor._reason_group("failed:email_verification_required") == "auth"
    assert doctor._reason_group("failed:not_a_salaried_position") == "location"
    assert doctor._reason_group("stale_unapproved") == "non_actionable"
    assert doctor._reason_group("dedup:already_applied") == "non_actionable"


def test_analyze_excludes_retired_and_deduplicated_rows_from_failure_backlog(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        _seed_apply_failure(conn, url="retired", host="old.example", apply_error="stale_unapproved")
        _seed_apply_failure(conn, url="duplicate", host="dup.example", apply_error="dedup:already_applied")
        _seed_apply_failure(conn, url="browser", host="broken.example", apply_error="failed:browser_unavailable")
        clusters = doctor.analyze(conn, window_minutes=60)

    assert clusters["lane_failures"] == 1
    assert {row["reason"] for row in clusters["cluster_rows"]} == {"agent"}
    assert "old.example" not in clusters["host_failures"]
    assert "dup.example" not in clusters["host_failures"]


def _seed_host_governor(conn, host, *, attempts):
    """Seed the host governor scope with enough ATTEMPTS that the H6 rate+denominator host_skip
    gate qualifies (block_24h drives both attempts and a high block rate)."""
    sk = f"host:{host}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO rate_governor (scope_key, block_24h) VALUES (%s,%s) "
            "ON CONFLICT (scope_key) DO UPDATE SET block_24h=EXCLUDED.block_24h", (sk, attempts))
    conn.commit()


# ---------------------------------------------------------------------------
# HOST_SKIP (H6/H19) -- a self-expiring lease FILTER (doctor_skip_until), NOT an un-approve.
# Vetted approved_batch is PRESERVED; nothing is deleted; it auto-reverts at TTL.
# ---------------------------------------------------------------------------
def test_host_skip_sets_filter_and_preserves_approval_never_deletes(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        # 3 hard-block failures on acme + a governor denominator so the H6 rate gate qualifies;
        # plus a QUEUED approved acme row + an unrelated queued approved row that must be untouched.
        _seed_apply_failure(conn, url="f1", host="acme.com", apply_error="blocked")
        _seed_apply_failure(conn, url="f2", host="acme.com", apply_error="captcha")
        _seed_apply_failure(conn, url="f3", host="acme.com", apply_error="cloudflare_blocked")
        _seed_apply_failure(conn, url="q-acme", host="acme.com", queued=True)
        _seed_apply_failure(conn, url="q-other", host="other.com", queued=True)
        _seed_host_governor(conn, "acme.com", attempts=5)  # 3/5 hard-block rate (>=0.3) -> qualifies

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM apply_queue")
            before_total = cur.fetchone()["n"]

        summary = doctor.run_doctor(conn, window_minutes=60)

        # H6: approval is PRESERVED on BOTH hosts (the old code NULLed acme's approved_batch).
        assert _queued_count_for_host(conn, "acme.com", approved_only=True) == 1
        assert _queued_count_for_host(conn, "other.com", approved_only=True) == 1

        # The skip is a governor FILTER, not an un-approve: doctor_skip_until is in the future.
        with conn.cursor() as cur:
            cur.execute("SELECT doctor_skip_until > now() AS skipping FROM rate_governor WHERE scope_key='host:acme.com'")
            assert cur.fetchone()["skipping"] is True
            cur.execute("SELECT count(*) AS n FROM apply_queue")
            assert cur.fetchone()["n"] == before_total  # nothing deleted

        # acme cannot lease (filtered), other.com still can.
        with conn.cursor() as cur:
            cur.execute("UPDATE rate_governor SET last_applied_at=now() - interval '1 day' "
                        "WHERE scope_key IN ('host:acme.com','host:other.com')")
        conn.commit()
        leased = queue.lease_apply(conn, "w1", home_ip="1.1.1.1")
        assert leased is not None and leased["target_host"] == "other.com"  # acme is skip-filtered

        with conn.cursor() as cur:
            # A5: host_skip knob is recorded under the canonical 'host:<h>' scope (was bare 'acme.com').
            cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE knob_type='host_skip' AND active AND scope_key='host:acme.com'")
            assert cur.fetchone()["n"] == 1
            cur.execute("SELECT how_to_reverse, status, rows_affected FROM fleet_diagnoses WHERE auto_action='host_skip'")
            d = cur.fetchone()
            assert d["status"] == "auto_applied"
            assert d["how_to_reverse"]
            assert d["rows_affected"] is not None  # H19: self-contained audit

    assert any(a["knob_type"] == "host_skip" for a in summary["auto_applied"])


def test_host_skip_does_not_trip_on_healthy_big_host(fleet_db):
    """H6: a 400-attempt host with only 3 stray hard-blocks (0.75% rate) must NOT be skipped."""
    from applypilot.fleet import queue  # noqa: F401
    with pgqueue.connect(fleet_db) as conn:
        _seed_apply_failure(conn, url="b1", host="big.com", apply_error="blocked")
        _seed_apply_failure(conn, url="b2", host="big.com", apply_error="captcha")
        _seed_apply_failure(conn, url="b3", host="big.com", apply_error="site_blocked")
        _seed_host_governor(conn, "big.com", attempts=400)  # 3/400 == 0.75% << 30%
        summary = doctor.run_doctor(conn, window_minutes=60)
        assert not any(a["knob_type"] == "host_skip" for a in summary["auto_applied"])
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE knob_type='host_skip' AND active")
            assert cur.fetchone()["n"] == 0


# ---------------------------------------------------------------------------
# TIMEOUT_BUMP -- writes a clamped override the worker would read.
# ---------------------------------------------------------------------------
def test_timeout_bump_writes_clamped_override(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        for i in range(3):
            _seed_apply_failure(conn, url=f"t{i}", host="slow.com", apply_error="failed:timeout")
        summary = doctor.run_doctor(conn, window_minutes=60)
        with conn.cursor() as cur:
            cur.execute("SELECT agent_timeout_override FROM fleet_config WHERE id=1")
            ov = cur.fetchone()["agent_timeout_override"]
    assert ov is not None
    assert doctor._DEFAULT_AGENT_TIMEOUT < ov <= doctor._TIMEOUT_CEILING
    assert any(a["knob_type"] == "timeout_bump" for a in summary["auto_applied"])


def test_worker_reads_timeout_override(fleet_db):
    from applypilot.fleet import apply_worker_main
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE fleet_config SET agent_timeout_override=600 WHERE id=1")
        conn.commit()
        eff = apply_worker_main.resolve_agent_timeout(conn, env_default=300)
    assert eff == 600


def test_worker_falls_back_to_default_when_no_override(fleet_db):
    from applypilot.fleet import apply_worker_main
    with pgqueue.connect(fleet_db) as conn:
        eff = apply_worker_main.resolve_agent_timeout(conn, env_default=321)
    assert eff == 321


# ---------------------------------------------------------------------------
# QUARANTINE -- reuses the existing poison_jobs path.
# ---------------------------------------------------------------------------
def test_quarantine_reuses_existing_poison_path(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        # H10: the poison pattern now requires a HARD reason across >=2 workers. We exercise the
        # decide() rule by injecting a poison url with count>=K and >=2 distinct workers.
        _seed_apply_failure(conn, url="poison", host="h.com", apply_error="failed:timeout")
        clusters = doctor.analyze(conn, window_minutes=60)
        clusters["url_failures"]["poison"]["count"] = 3            # simulate 3 attempts/poison
        clusters["url_workers"]["poison"] = ["home-0", "home-1"]   # H10: span 2 workers
        planned = doctor.decide(conn, clusters)
        q = [p for p in planned if p["knob_type"] == "quarantine"]
        assert q, "expected a quarantine auto action"
        res = doctor.apply_auto(conn, q[0])
        assert res["applied"]
        # the EXISTING poison_jobs row is set (quarantined_at not null), reason manual-prefixed
        with conn.cursor() as cur:
            cur.execute("SELECT quarantined_at, reason FROM poison_jobs WHERE url='poison'")
            row = cur.fetchone()
        assert row is not None and row["quarantined_at"] is not None
        assert row["reason"].startswith("manual:")
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE knob_type='quarantine' AND active")
            assert cur.fetchone()["n"] == 1


def test_quarantine_requires_two_workers_not_one_flaky_box(fleet_db):
    """H10: a poison url seen on only ONE worker is NOT auto-quarantined (could be a flaky box)."""
    with pgqueue.connect(fleet_db) as conn:
        _seed_apply_failure(conn, url="poison1", host="h.com", apply_error="failed:timeout")
        clusters = doctor.analyze(conn, window_minutes=60)
        clusters["url_failures"]["poison1"]["count"] = 5
        clusters["url_workers"]["poison1"] = ["home-0"]  # ONE worker
        planned = doctor.decide(conn, clusters)
        assert not any(p["knob_type"] == "quarantine" for p in planned)


# ---------------------------------------------------------------------------
# PACE_OR_PAUSE -- pace widens the gap; severe pauses the apply lane.
# ---------------------------------------------------------------------------
def test_pace_widens_host_gap_and_actually_throttles_leasing(fleet_db):
    """The pace tier must throttle the scope the apply lease ENFORCES (the per-host scope
    'host:'||<host>, queue.py:37,51-52), not the inert 'global' scope. We assert BOTH that the
    host scope's min_gap widened AND that a lease is actually delayed within the widened window
    (so the old no-op -- where only a DB column changed but leasing never slowed -- cannot
    regress)."""
    from applypilot.fleet import queue
    with pgqueue.connect(fleet_db) as conn:
        # Seed the offending host's governor scope at the pristine base (90s).
        with conn.cursor() as cur:
            # block_24h=12 gives a.com a host_skip RATE of 3/12=0.25 (< 0.3) so it does NOT trip the
            # H6 host_skip gate -- isolating the elevated PACE tier (host_skip would otherwise
            # supersede the pace under H12).
            cur.execute("INSERT INTO rate_governor (scope_key, min_gap_seconds, base_min_gap_seconds, block_24h) "
                        "VALUES ('host:a.com', 90, 90, 12)")
        conn.commit()
        # 5 failures, 3 hard-block (all on a.com) -> rate 0.6 in [0.5, 0.8) -> pace (not pause).
        _seed_apply_failure(conn, url="p1", host="a.com", apply_error="blocked")
        _seed_apply_failure(conn, url="p2", host="a.com", apply_error="captcha")
        _seed_apply_failure(conn, url="p3", host="a.com", apply_error="site_blocked")
        _seed_apply_failure(conn, url="p4", host="d.com", apply_error="no_result_line")
        _seed_apply_failure(conn, url="p5", host="e.com", apply_error="stuck")
        summary = doctor.run_doctor(conn, window_minutes=60)
        with conn.cursor() as cur:
            # H4: the pace writes the Doctor-OWNED floor, not min_gap_seconds (the breaker's column).
            cur.execute("SELECT min_gap_seconds, doctor_min_gap_floor FROM rate_governor WHERE scope_key='host:a.com'")
            row = cur.fetchone()
            cur.execute("SELECT paused, ats_paused FROM fleet_config WHERE id=1")
            cfg = cur.fetchone()
        floor = row["doctor_min_gap_floor"]
        assert row["min_gap_seconds"] == 90                       # breaker column UNTOUCHED
        assert floor == min(doctor._PACE_GAP_CEILING, 90 * doctor._PACE_MULTIPLIER)  # 270
        assert floor > 90
        assert cfg["paused"] is False and cfg["ats_paused"] is False  # pace, not pause
        assert any(a["knob_type"] == "pace_or_pause" for a in summary["auto_applied"])

        # Now PROVE leasing actually slows: seed one approved, leasable queued row on a.com and
        # set its host scope's last_attempt_at to 150s ago -- leaving last_applied_at NULL (A3: a
        # never-succeeded, hard-blocking host has NO confirmed apply, so last_applied_at stays NULL;
        # the lease gates off COALESCE(last_applied_at, last_attempt_at)). With the WIDENED 270s gap
        # the window is 270*[0.7,1.4] = [189,378]s, so 150s ago is INSIDE the gap -> no lease.
        _seed_apply_failure(conn, url="lease-me", host="a.com", queued=True)
        with conn.cursor() as cur:
            cur.execute("UPDATE rate_governor SET last_applied_at = NULL, "
                        "last_attempt_at = now() - interval '150 seconds' WHERE scope_key='host:a.com'")
        conn.commit()
        # Blocked at the widened gap (the pace actually throttles -- on a NEVER-SUCCEEDED host).
        assert queue.lease_apply(conn, "w1", home_ip="1.1.1.1") is None
        # Sanity: winding last_attempt_at well past the widened window lets it lease again.
        with conn.cursor() as cur:
            cur.execute("UPDATE rate_governor SET last_applied_at = NULL, "
                        "last_attempt_at = now() - interval '500 seconds' WHERE scope_key='host:a.com'")
        conn.commit()
        assert queue.lease_apply(conn, "w1", home_ip="1.1.1.1") is not None


def _seed_broad_block(conn, *, hosts, workers, per_host_attempts):
    """Seed a BROAD block: hard-block failures spanning many hosts/workers + governor block attempts
    so the H5 attempts-based lane PAUSE tier (>=3 hosts AND >=2 workers AND >=20 attempts at >=0.8
    block rate) qualifies. Each host's governor block_24h=per_host_attempts but only ONE in-window
    hard-block failure per (host,worker) -- so the per-host host_skip RATE gate (hb/attempts) stays
    BELOW 0.3 and the systemic detector does NOT trip (no host individually qualifies for host_skip),
    isolating the lane-pause debounce path under test."""
    i = 0
    for h in hosts:
        for w in workers:
            _seed_apply_failure(conn, url=f"b-{h}-{w}-{i}", host=h, worker_id=w, apply_error="blocked")
            i += 1
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rate_governor (scope_key, block_24h) VALUES (%s,%s) "
                "ON CONFLICT (scope_key) DO UPDATE SET block_24h=EXCLUDED.block_24h",
                (f"host:{h}", per_host_attempts))
        conn.commit()


def test_concentrated_block_skips_host_does_not_pause_lane(fleet_db):
    """H5: 3+ blocks on ONE host (concentrated) -> host_skip, NOT a lane pause."""
    with pgqueue.connect(fleet_db) as conn:
        for i in range(4):
            _seed_apply_failure(conn, url=f"c{i}", host="one.com", worker_id="home-0", apply_error="blocked")
        _seed_host_governor(conn, "one.com", attempts=5)
        doctor.run_doctor(conn, window_minutes=60)
        with conn.cursor() as cur:
            cur.execute("SELECT paused, ats_paused FROM fleet_config WHERE id=1")
            cfg = cur.fetchone()
            assert cfg["paused"] is False and cfg["ats_paused"] is False  # NOT paused
            cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE knob_type='host_skip' AND active")
            assert cur.fetchone()["n"] == 1  # host_skip instead


def test_broad_block_pauses_ats_only_after_debounce(fleet_db):
    """H5+H1: a BROAD block (>=3 hosts, >=2 workers, >=20 attempts) pauses ONLY after a 2-pass
    debounce, and writes ats_paused (NOT the shared fleet_config.paused)."""
    with pgqueue.connect(fleet_db) as conn:
        _seed_broad_block(conn, hosts=[f"h{i}.com" for i in range(4)],
                          workers=["home-0", "home-1", "home-2"], per_host_attempts=12)
        # 1st pass: ARMS, does NOT pause yet.
        doctor.run_doctor(conn, window_minutes=60)
        with conn.cursor() as cur:
            cur.execute("SELECT ats_paused, paused, doctor_pause_armed_at FROM fleet_config WHERE id=1")
            c1 = cur.fetchone()
        assert c1["ats_paused"] is False and c1["paused"] is False
        assert c1["doctor_pause_armed_at"] is not None  # armed
        # Force the debounce window to have elapsed, then run the 2nd pass -> PAUSE (ATS-only).
        with conn.cursor() as cur:
            cur.execute("UPDATE fleet_config SET doctor_pause_armed_at = now() - interval '10 minutes' WHERE id=1")
        conn.commit()
        doctor.run_doctor(conn, window_minutes=60)
        with conn.cursor() as cur:
            cur.execute("SELECT ats_paused, paused, ats_pause_source FROM fleet_config WHERE id=1")
            c2 = cur.fetchone()
            cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE knob_type='pace_or_pause' AND value_text='paused' AND active")
            n = cur.fetchone()["n"]
        assert c2["ats_paused"] is True            # H1: ATS-only flag set
        assert c2["paused"] is False               # H1: shared kill switch UNTOUCHED -> LinkedIn safe
        assert c2["ats_pause_source"] == "doctor"  # H8: provenance recorded
        assert n == 1


# ---------------------------------------------------------------------------
# D1 GATE -- rejects every forbidden / activity-increasing / LinkedIn action.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action", [
    {"knob_type": "resume"},                                   # not allow-listed
    {"knob_type": "re_approve"},                               # not allow-listed
    {"knob_type": "raise_cap"},                                # not allow-listed
    {"knob_type": "host_skip", "op": "resume"},                # forbidden op
    {"knob_type": "host_skip", "op": "apply"},                 # forbidden op
    {"knob_type": "pace_or_pause", "op": "raise_cap", "op_kind": "pace", "new_gap": 200},
    {"knob_type": "host_skip", "scope_key": "linkedin.com"},   # LinkedIn scope
    {"knob_type": "host_skip", "lane": "linkedin"},            # LinkedIn lane
    {"knob_type": "timeout_bump", "new_timeout": 9999, "current_default": 300},  # over ceiling
    {"knob_type": "timeout_bump", "new_timeout": 200, "current_default": 300},   # lowers timeout
    {"knob_type": "pace_or_pause", "op_kind": "pace", "new_gap": 10, "old_gap": 90},  # speeds up
    {"knob_type": "pace_or_pause", "op_kind": "pause", "lane": "linkedin"},     # pause linkedin
    # H1 EFFECT-LEVEL: a pause routing to the SHARED kill switch is refused even though its lane
    # label says 'ats' -- the actuator (set_paused/fleet_config.paused) would halt LinkedIn.
    {"knob_type": "pace_or_pause", "op_kind": "pause", "lane": "ats", "actuator": "set_paused"},
    {"knob_type": "pace_or_pause", "op_kind": "pause", "lane": "ats", "actuator": "fleet_config.paused"},
    # H1: a pause that declares NO actuator (the v1 shape) is refused -- it must explicitly route
    # to ats_paused so it can never accidentally fall back to the shared flag.
    {"knob_type": "pace_or_pause", "op_kind": "pause", "lane": "ats"},
    # H1: a pace must target a per-host scope, never the inert/global scope.
    {"knob_type": "pace_or_pause", "op_kind": "pace", "scope_key": "global", "new_gap": 270, "old_gap": 90},
    {"knob_type": "host_skip", "scope_key": "acme.com", "actuator": "account:linkedin"},  # li actuator
    {"knob_type": "frobnicate"},                               # unknown knob
    "not_a_dict",                                              # wrong type
])
def test_assert_conservative_rejects_forbidden(action):
    with pytest.raises(doctor.ConservativeViolation):
        doctor._assert_conservative(action)


@pytest.mark.parametrize("action", [
    # A10: every auto action now declares its per-knob actuator (mandatory + allow-listed).
    {"knob_type": "host_skip", "scope_key": "acme.com", "actuator": "doctor_skip_until"},
    # H7: the ceiling dropped below the watchdog kill (600 -> 540); 540 is still a valid raise.
    {"knob_type": "timeout_bump", "new_timeout": 390, "current_default": 300,
     "actuator": "agent_timeout_override"},
    {"knob_type": "quarantine", "scope_key": "u1", "actuator": "poison_jobs.quarantined_at"},
    # H1: a pace MUST target a per-host 'host:<h>' scope now (the lease never read 'global').
    {"knob_type": "pace_or_pause", "op_kind": "pace", "scope_key": "host:acme.com",
     "new_gap": 270, "old_gap": 90, "actuator": "doctor_min_gap_floor"},
    # H1: a pause MUST declare actuator='ats_paused' (ATS-only) -- never fleet_config.paused.
    {"knob_type": "pace_or_pause", "op_kind": "pause", "lane": "ats", "actuator": "ats_paused"},
])
def test_assert_conservative_allows_conservative(action):
    doctor._assert_conservative(action)  # must NOT raise


def test_apply_auto_refuses_hostile_action_before_mutating(fleet_db):
    """A hostile 'resume' shaped as an auto action must be REJECTED by the gate inside
    apply_auto BEFORE any DB mutation (no fleet_knobs/diagnoses row written)."""
    with pgqueue.connect(fleet_db) as conn:
        with pytest.raises(doctor.ConservativeViolation):
            doctor.apply_auto(conn, {"knob_type": "resume", "op": "resume"})
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_knobs")
            assert cur.fetchone()["n"] == 0
            cur.execute("SELECT count(*) AS n FROM fleet_diagnoses")
            assert cur.fetchone()["n"] == 0
            cur.execute("SELECT paused FROM fleet_config WHERE id=1")
            assert cur.fetchone()["paused"] is False


# ---------------------------------------------------------------------------
# IDEMPOTENCY -- a 2nd run applies nothing new + does not duplicate diagnoses.
# ---------------------------------------------------------------------------
def test_idempotent_second_run_is_noop(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        _seed_apply_failure(conn, url="f1", host="acme.com", apply_error="blocked")
        _seed_apply_failure(conn, url="f2", host="acme.com", apply_error="captcha")
        _seed_apply_failure(conn, url="f3", host="acme.com", apply_error="site_blocked")
        s1 = doctor.run_doctor(conn, window_minutes=60)
        assert any(a["knob_type"] == "host_skip" for a in s1["auto_applied"])

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_knobs")
            knobs_after_1 = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM fleet_diagnoses")
            diags_after_1 = cur.fetchone()["n"]

        s2 = doctor.run_doctor(conn, window_minutes=60)
        assert not s2["auto_applied"]  # nothing newly applied
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_knobs")
            assert cur.fetchone()["n"] == knobs_after_1  # no duplicate knobs
            cur.execute("SELECT count(*) AS n FROM fleet_diagnoses")
            assert cur.fetchone()["n"] == diags_after_1  # no duplicate diagnoses


# ---------------------------------------------------------------------------
# TTL SWEEP -- expires a stale knob + its open diagnosis.
# ---------------------------------------------------------------------------
def test_ttl_sweep_expires_stale_knob(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO fleet_knobs (knob_type, scope_key, value_text, reason, active, expires_at) "
                "VALUES ('host_skip','stale.com','skip','old', TRUE, now() - interval '1 hour')")
            cur.execute(
                "INSERT INTO fleet_diagnoses (cluster_key, reason, host, auto_action, status, expires_at) "
                "VALUES ('hard_block|stale.com|-|ats','hard_block','stale.com','host_skip','auto_applied', now() - interval '1 hour')")
        conn.commit()
        swept = doctor.sweep_expired(conn)
        assert swept["knobs_expired"] == 1
        assert swept["diagnoses_expired"] == 1
        with conn.cursor() as cur:
            cur.execute("SELECT active FROM fleet_knobs WHERE scope_key='stale.com'")
            assert cur.fetchone()["active"] is False
            cur.execute("SELECT status FROM fleet_diagnoses WHERE host='stale.com'")
            assert cur.fetchone()["status"] == "expired"


def test_sweep_leaves_fresh_knob_active(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO fleet_knobs (knob_type, scope_key, value_text, active, expires_at) "
                "VALUES ('host_skip','fresh.com','skip', TRUE, now() + interval '1 hour')")
        conn.commit()
        doctor.sweep_expired(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT active FROM fleet_knobs WHERE scope_key='fresh.com'")
            assert cur.fetchone()["active"] is True


# ---------------------------------------------------------------------------
# RECOMMENDATIONS -- non-auto-fixable clusters are recorded, NOT auto-applied.
# ---------------------------------------------------------------------------
def test_agent_and_auth_clusters_become_recommendations_not_auto(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        _seed_apply_failure(conn, url="a1", host="x.com", apply_error="no_result_line")  # agent
        _seed_apply_failure(conn, url="a2", host="y.com", apply_error="auth_required")   # auth
        _seed_apply_failure(conn, url="a3", host="z.com", apply_error="not_eligible_location")  # location
        summary = doctor.run_doctor(conn, window_minutes=60)
        # none of these produced an auto-fix knob
        assert not summary["auto_applied"]
        assert summary["recommendations"] >= 3
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_diagnoses WHERE status='recommended'")
            assert cur.fetchone()["n"] >= 3
            # and NO knob was written for these
            cur.execute("SELECT count(*) AS n FROM fleet_knobs")
            assert cur.fetchone()["n"] == 0


def test_parsing_drift_breach_becomes_recommendation_only(tmp_path, fleet_db, monkeypatch):
    from datetime import datetime, timezone

    from applypilot import config, database

    brain_db = tmp_path / "brain.db"
    bconn = database.init_db(brain_db)
    bconn.execute(
        "INSERT INTO desc_quality_drift "
        "(snapshot_at, board, window_days, total, null_rate, stub_rate, short_rate, html_rate, "
        "junk_rate, board_summary_rate, title_echo_rate, no_req_marker_rate) "
        "VALUES (?, 'hiringcafe', 7, 100, 0.03, 0.10, 0.10, 0.03, 0.00, 0.01, 0.02, 0.01)"
    , (datetime(2026, 7, 4, tzinfo=timezone.utc).isoformat(),))
    bconn.commit()
    monkeypatch.setattr(config, "DB_PATH", brain_db)

    with pgqueue.connect(fleet_db) as conn:
        summary = doctor.run_doctor(conn, window_minutes=60)

    assert summary["recommendations"] >= 3
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM fleet_diagnoses "
                "WHERE status='recommended' AND cluster_key LIKE 'parsing_drift|hiringcafe|%'"
            )
            assert cur.fetchone()["n"] >= 3
            cur.execute("SELECT count(*) AS n FROM fleet_knobs")
            assert cur.fetchone()["n"] == 0


def test_parsing_drift_doctor_alerts_only_null_short_html(tmp_path):
    from datetime import datetime, timezone

    from applypilot import database

    brain_db = tmp_path / "brain.db"
    bconn = database.init_db(brain_db)
    bconn.execute(
        "INSERT INTO desc_quality_drift "
        "(snapshot_at, board, window_days, total, null_rate, stub_rate, short_rate, html_rate, "
        "junk_rate, board_summary_rate, title_echo_rate, no_req_marker_rate) "
        "VALUES (?, 'hiringcafe', 7, 100, 0.00, 0.99, 0.00, 0.00, 0.00, 0.99, 0.00, 0.00)",
        (datetime(2026, 7, 4, tzinfo=timezone.utc).isoformat(),),
    )
    bconn.commit()

    assert doctor.parsing_drift_actions(str(brain_db)) == []


def test_parsing_drift_missing_brain_db_is_silent_no_recommendations(fleet_db, monkeypatch, tmp_path):
    from applypilot import config

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "missing.db")
    with pgqueue.connect(fleet_db) as conn:
        summary = doctor.run_doctor(conn, window_minutes=60)

    assert summary["recommendations"] == 0
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_diagnoses WHERE status='recommended'")
            assert cur.fetchone()["n"] == 0


def test_recommendations_are_idempotent(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        _seed_apply_failure(conn, url="a1", host="x.com", apply_error="no_result_line")
        doctor.run_doctor(conn, window_minutes=60)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_diagnoses WHERE status='recommended'")
            n1 = cur.fetchone()["n"]
        doctor.run_doctor(conn, window_minutes=60)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_diagnoses WHERE status='recommended'")
            assert cur.fetchone()["n"] == n1


# ===========================================================================
# RED-TEAM HARDENING PROVING TESTS (H1-H9, the named tests).
# ===========================================================================

def _seed_linkedin_job(conn, url="li0", dk="lidk0"):
    policy = "doctor-linkedin-policy"
    now = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO fleet_decision_policies (policy_version,lane,status) "
            "VALUES (%s,'linkedin','active')",
            (policy,),
        )
        cur.execute(
            "UPDATE fleet_config SET linkedin_policy_version=%s,linkedin_apply_mode='steady' WHERE id=1",
            (policy,),
        )
        cur.execute(
            "INSERT INTO linkedin_queue (url, application_url, score, status, lane, approved_batch, dedup_key, "
            "linkedin_resolve_status,linkedin_resolved_at,decision_id,policy_version,decision_action,qualification_verdict,qualification_score,qualification_floor,"
            "preference_score,outcome_score,final_score,decision_confidence,decision_created_at,decision_expires_at,input_hash) "
            "VALUES (%s,%s,9.0,'queued','linkedin','b1',%s,'easy_apply',now(),%s,%s,'apply','qualified',9,0,8,8,9,.9,%s,%s,%s)",
            (url, f"https://linkedin.com/jobs/{url}", dk, f"decision-{url}", policy,
             now, now + timedelta(days=1), f"hash-{url}"),
        )
    conn.commit()


def test_H1_doctor_pause_cannot_halt_linkedin_lane(fleet_db, acquisition_admitted):
    """H1 CATASTROPHE: arm a LinkedIn worker context, fire a Doctor lane-PAUSE, and assert the
    LinkedIn lane is UNAFFECTED -- it still leases AND linkedin_should_halt() == False. This is the
    single most important proof: the Doctor's pause writes ats_paused, never fleet_config.paused
    (which the LinkedIn loop reads via linkedin_should_halt)."""
    from applypilot.fleet import linkedin_worker_main as lm
    with pgqueue.connect(fleet_db) as conn:
        _seed_linkedin_job(conn)
        # Fire the Doctor's lane pause directly (the ATS-only actuator).
        pause = {"knob_type": "pace_or_pause", "op_kind": "pause", "actuator": "ats_paused",
                 "scope_key": "ats", "lane": "ats", "reason": "hard_block",
                 "cluster_key": "pp|lane|pause|ats", "diagnosis": "d", "recommendation": "r",
                 "sample_count": 20, "severity": "severe"}
        res = doctor.apply_auto(conn, pause)
        assert res["applied"]
        # The ATS lane IS halted...
        assert pgqueue.ats_should_halt(conn) is True
        # ...but the LinkedIn lane's halt gate is FALSE (it reads only the shared kill switch).
        assert pgqueue.linkedin_should_halt(conn) is False
        # And the LinkedIn lease STILL returns a job (the catastrophe lane keeps running).
        leased = queue.lease_linkedin(conn, "w1", public_ip="1.1.1.1", owner_ip="1.1.1.1",
                                      min_gap_seconds=0)
        assert leased is not None and leased["url"] == "li0"
        # Belt-and-suspenders: the shared kill switch was NEVER written.
        with conn.cursor() as cur:
            cur.execute("SELECT paused, ats_paused FROM fleet_config WHERE id=1")
            row = cur.fetchone()
        assert row["paused"] is False and row["ats_paused"] is True
        # The run_linkedin loop's pre-lease gate (should_halt) does not halt.
        counts = lm.run_linkedin(lambda: pgqueue.connect(fleet_db),
                                 _NoopLoop(), max_iterations=1, idle_sleep=0)
        assert counts["halted"] == 0


class _NoopLoop:
    role = "linkedin"
    def run_once(self):
        return {"action": "idle"}


def test_H1_gate_forbids_set_paused_actuator_even_with_ats_lane_label():
    """H1 EFFECT-LEVEL GATE: a pause whose LANE label is 'ats' but whose ACTUATOR is the shared
    set_paused / fleet_config.paused is REFUSED -- the gate inspects the mechanism, not the label."""
    import pytest
    for bad in ("set_paused", "fleet_config.paused", "paused"):
        with pytest.raises(doctor.ConservativeViolation):
            doctor._assert_conservative({"knob_type": "pace_or_pause", "op_kind": "pause",
                                         "lane": "ats", "actuator": bad})


def test_H4_multi_actor_effective_gap_is_monotone_until_both_ttls_expire(fleet_db):
    """H4 MULTI-ACTOR MONOTONE: co-run the watchdog breaker + a Doctor pace on ONE throttled host;
    assert the EFFECTIVE gap (GREATEST(min_gap_seconds, doctor_min_gap_floor)) is monotone
    non-decreasing -- the breaker recovery can never wipe the still-active Doctor pace."""
    from applypilot.fleet import governor
    with pgqueue.connect(fleet_db) as conn:
        sk = "host:slow.com"
        with conn.cursor() as cur:
            cur.execute("INSERT INTO rate_governor (scope_key, min_gap_seconds, base_min_gap_seconds) "
                        "VALUES (%s, 90, 90)", (sk,))
        conn.commit()
        # Doctor paces the host (writes its OWN floor to 270).
        pace = {"knob_type": "pace_or_pause", "actuator": "doctor_min_gap_floor", "op_kind": "pace", "scope_key": sk, "host": "slow.com",
                "lane": "ats", "reason": "hard_block", "cluster_key": "pp|slow.com|pace|ats",
                "diagnosis": "d", "recommendation": "r", "sample_count": 3, "severity": "warn",
                "old_gap": 90, "new_gap": 270}
        doctor.apply_auto(conn, pace)

        def eff_gap():
            with conn.cursor() as cur:
                cur.execute("SELECT GREATEST(COALESCE(min_gap_seconds,90), COALESCE(doctor_min_gap_floor,0)) AS g "
                            "FROM rate_governor WHERE scope_key=%s", (sk,))
                return cur.fetchone()["g"]

        g0 = eff_gap()
        assert g0 == 270
        # Now the breaker trips (throttle widens min_gap_seconds) then RECOVERS (restores it).
        with conn.cursor() as cur:
            cur.execute("UPDATE rate_governor SET block_24h=0, captcha_24h=6, success_24h=4 WHERE scope_key=%s", (sk,))
        conn.commit()
        governor.evaluate_breakers(conn, captcha_threshold=0.4, min_samples=8)
        g1 = eff_gap()
        assert g1 >= g0  # monotone non-decreasing (breaker only widened)
        # Breaker recovery (clears its throttle, restores min_gap_seconds to base 90).
        with conn.cursor() as cur:
            cur.execute("UPDATE rate_governor SET breaker_until = now() - interval '1 second' WHERE scope_key=%s", (sk,))
        conn.commit()
        governor.clear_expired_breakers(conn)
        g2 = eff_gap()
        # CRITICAL: even though the breaker restored min_gap_seconds to 90, the Doctor floor (270)
        # still holds -- the effective gap did NOT collapse. THIS is the H4 fix.
        assert g2 == 270 and g2 >= g1
        # Only after the Doctor knob's TTL expires + sweep does the floor clear.
        with conn.cursor() as cur:
            cur.execute("UPDATE fleet_knobs SET expires_at = now() - interval '1 second' "
                        "WHERE knob_type='pace_or_pause' AND active")
        conn.commit()
        doctor.sweep_expired(conn)
        assert eff_gap() == 90  # both TTLs expired -> back to base


def test_H3_sweep_mechanically_undoes_each_knob_type(fleet_db):
    """H3 SWEEP-TIME MECHANICAL UNDO: after each knob type expires + sweep, assert the EFFECT is
    really reversed (override NULL / floor cleared / skip filter cleared + recommend row written /
    ats pause cleared) -- not merely active=FALSE."""
    with pgqueue.connect(fleet_db) as conn:
        # timeout_bump (override -> restored to pre-bump NULL when no other bump remains)
        tb = {"knob_type": "timeout_bump", "actuator": "agent_timeout_override", "scope_key": "ats", "host": "s.com", "lane": "ats",
              "reason": "timeout", "new_timeout": 390, "current_default": 300,
              "cluster_key": "to|-|-|ats", "diagnosis": "d", "recommendation": "r",
              "sample_count": 3, "severity": "warn"}
        doctor.apply_auto(conn, tb)
        # pace (floor)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO rate_governor (scope_key, min_gap_seconds, base_min_gap_seconds) "
                        "VALUES ('host:p.com',90,90)")
        conn.commit()
        pc = {"knob_type": "pace_or_pause", "actuator": "doctor_min_gap_floor", "op_kind": "pace", "scope_key": "host:p.com", "host": "p.com",
              "lane": "ats", "reason": "hard_block", "cluster_key": "pp|p.com|pace|ats",
              "diagnosis": "d", "recommendation": "r", "sample_count": 3, "severity": "warn",
              "old_gap": 90, "new_gap": 270}
        doctor.apply_auto(conn, pc)
        # host_skip (doctor_skip_until)
        hs = {"knob_type": "host_skip", "actuator": "doctor_skip_until", "scope_key": "h.com", "host": "h.com", "lane": "ats",
              "reason": "hard_block", "cluster_key": "hs|h.com|-|ats", "diagnosis": "d",
              "recommendation": "r", "sample_count": 3, "severity": "warn"}
        doctor.apply_auto(conn, hs)
        # ats pause (doctor-authored)
        pa = {"knob_type": "pace_or_pause", "op_kind": "pause", "actuator": "ats_paused",
              "scope_key": "ats", "lane": "ats", "reason": "hard_block",
              "cluster_key": "pp|lane|pause|ats", "diagnosis": "d", "recommendation": "r",
              "sample_count": 20, "severity": "severe"}
        doctor.apply_auto(conn, pa)

        # Expire ALL active knobs, then sweep.
        with conn.cursor() as cur:
            cur.execute("UPDATE fleet_knobs SET expires_at = now() - interval '1 second' WHERE active")
        conn.commit()
        swept = doctor.sweep_expired(conn)

        with conn.cursor() as cur:
            cur.execute("SELECT agent_timeout_override FROM fleet_config WHERE id=1")
            assert cur.fetchone()["agent_timeout_override"] is None     # timeout restored
            cur.execute("SELECT doctor_min_gap_floor FROM rate_governor WHERE scope_key='host:p.com'")
            assert cur.fetchone()["doctor_min_gap_floor"] is None        # pace floor cleared
            cur.execute("SELECT doctor_skip_until FROM rate_governor WHERE scope_key='host:h.com'")
            assert cur.fetchone()["doctor_skip_until"] is None           # host_skip filter cleared
            cur.execute("SELECT ats_paused FROM fleet_config WHERE id=1")
            assert cur.fetchone()["ats_paused"] is False                 # doctor ats pause auto-reverted
            cur.execute("SELECT count(*) AS n FROM fleet_diagnoses "
                        "WHERE status='recommended' AND cluster_key LIKE 'host_skip_expired%'")
            assert cur.fetchone()["n"] == 1                              # re-approve recommend written
        assert swept["reversed"].get("timeout_restored") == 1
        assert swept["reversed"].get("host_skip_cleared") == 1
        assert swept["reversed"].get("ats_pause_cleared") == 1


def test_H3_sweep_leaves_operator_ats_pause_alone(fleet_db):
    """H3/H8: if an OPERATOR pause took over ats_paused, the sweep of an expired Doctor pause knob
    must NOT clear it (only the Doctor's OWN pause auto-reverts)."""
    with pgqueue.connect(fleet_db) as conn:
        pa = {"knob_type": "pace_or_pause", "op_kind": "pause", "actuator": "ats_paused",
              "scope_key": "ats", "lane": "ats", "reason": "hard_block",
              "cluster_key": "pp|lane|pause|ats", "diagnosis": "d", "recommendation": "r",
              "sample_count": 20, "severity": "severe"}
        doctor.apply_auto(conn, pa)
        with conn.cursor() as cur:
            cur.execute("UPDATE fleet_config SET ats_pause_source='operator' WHERE id=1")
            cur.execute("UPDATE fleet_knobs SET expires_at = now() - interval '1 second' WHERE active")
        conn.commit()
        doctor.sweep_expired(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT ats_paused FROM fleet_config WHERE id=1")
            assert cur.fetchone()["ats_paused"] is True  # operator pause preserved


def test_H2_systemic_event_downgrades_to_recommend_and_does_not_unapprove(fleet_db):
    """H2 AGGREGATE BREAKER: a pass with > M blocked hosts crossing threshold downgrades the
    host_skips to recommend-only (no auto un-approve / no doctor_skip_until set) and emits ONE
    SYSTEMIC recommendation."""
    with pgqueue.connect(fleet_db) as conn:
        hosts = [f"sys{i}.com" for i in range(7)]   # > _SYSTEMIC_HOST_COUNT (5)
        # A6: a SYSTEMIC event must span >= _SYSTEMIC_MIN_WORKERS distinct workers (a single bad
        # worker/IP is NOT systemic) -- spread the 3 hard-blocks per host across two workers.
        for h in hosts:
            for j, w in enumerate(("home-0", "home-1", "home-0")):
                _seed_apply_failure(conn, url=f"{h}-{j}", host=h, worker_id=w, apply_error="blocked")
            _seed_host_governor(conn, h, attempts=5)  # 3/5 -> each would individually trip
        summary = doctor.run_doctor(conn, window_minutes=60)
        # NO host_skip auto-applied (all downgraded under the systemic breaker).
        assert not any(a["knob_type"] == "host_skip" for a in summary["auto_applied"])
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE knob_type='host_skip' AND active")
            assert cur.fetchone()["n"] == 0
            # No host got a doctor_skip_until set (the queue was not gated).
            cur.execute("SELECT count(*) AS n FROM rate_governor WHERE doctor_skip_until IS NOT NULL")
            assert cur.fetchone()["n"] == 0
            # Exactly one SYSTEMIC recommendation row.
            cur.execute("SELECT count(*) AS n FROM fleet_diagnoses WHERE reason='systemic_block' AND status='recommended'")
            assert cur.fetchone()["n"] == 1


def test_H2_per_day_budget_caps_host_skips(fleet_db):
    """H2 DAILY BUDGET: once the per-day host_skip budget is exhausted, further host_skips are
    downgraded to recommendations rather than applied."""
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            # Pre-spend the daily budget right up to the cap.
            import datetime as _d
            today = _d.datetime.now(_d.timezone.utc).date()
            cur.execute("UPDATE fleet_config SET doctor_budget_day=%s, doctor_host_skips_today=%s WHERE id=1",
                        (today, doctor._MAX_HOST_SKIPS_PER_DAY))
        conn.commit()
        for j in range(3):
            _seed_apply_failure(conn, url=f"b{j}", host="cap.com", worker_id="home-0", apply_error="blocked")
        _seed_host_governor(conn, "cap.com", attempts=5)
        summary = doctor.run_doctor(conn, window_minutes=60)
        assert not any(a["knob_type"] == "host_skip" for a in summary["auto_applied"])
        assert summary["budget_downgraded"] >= 1
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE knob_type='host_skip' AND active")
            assert cur.fetchone()["n"] == 0


def test_H7_timeout_ceiling_is_below_watchdog_kill():
    """H7: the timeout ceiling MUST be strictly below the watchdog's job_max_seconds, or a bump
    pushes an agent past the watchdog kill (manufacturing job_over_max poison). Asserted directly."""
    from applypilot.fleet.watchdog import WatchdogConfig
    assert doctor._TIMEOUT_CEILING < WatchdogConfig.job_max_seconds
    doctor._assert_timeout_ceiling_below_watchdog()  # must not raise


def test_H7_timeout_bump_requires_pure_timeout_cluster(fleet_db):
    """H7: a host with timeouts AND agent failures is NOT timeout-bumped (the agent is wedged, not
    the page slow -- a longer timeout would only let it hang longer)."""
    with pgqueue.connect(fleet_db) as conn:
        for j in range(3):
            _seed_apply_failure(conn, url=f"t{j}", host="mixed.com", apply_error="failed:timeout")
        _seed_apply_failure(conn, url="ta", host="mixed.com", apply_error="no_result_line")  # agent!
        summary = doctor.run_doctor(conn, window_minutes=60)
        assert not any(a["knob_type"] == "timeout_bump" for a in summary["auto_applied"])
        with conn.cursor() as cur:
            cur.execute("SELECT agent_timeout_override FROM fleet_config WHERE id=1")
            assert cur.fetchone()["agent_timeout_override"] is None


def test_H9_singleton_second_concurrent_pass_noops(fleet_db):
    """H9 SINGLETON: a second run_doctor on a SEPARATE connection no-ops while the first holds the
    advisory lock. We hold the lock on conn A, then run a pass on conn B and assert it skipped."""
    conn_a = pgqueue.connect(fleet_db)
    try:
        assert doctor._try_singleton_lock(conn_a) is True  # A holds the lock
        with pgqueue.connect(fleet_db) as conn_b:
            # Seed a clear host_skip trigger on B's view.
            for j in range(3):
                _seed_apply_failure(conn_b, url=f"s{j}", host="lock.com", worker_id="home-0", apply_error="blocked")
            _seed_host_governor(conn_b, "lock.com", attempts=5)
            summary = doctor.run_doctor(conn_b, window_minutes=60)
            assert summary.get("skipped_singleton") is True
            with conn_b.cursor() as cur:
                cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE active")
                assert cur.fetchone()["n"] == 0  # nothing applied under the lock
    finally:
        doctor._release_singleton_lock(conn_a)
        conn_a.close()


def test_H9_partial_unique_index_blocks_duplicate_active_knob(fleet_db):
    """H9 backstop: the partial-unique index prevents two active knobs for the same
    (knob_type, scope) -- a racing duplicate INSERT is a no-op via ON CONFLICT DO NOTHING."""
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            inserted1 = doctor._record_knob(cur, knob_type="host_skip", scope_key="dup.com",
                                            value_text="skip", reason="r", ttl_seconds=3600)
            inserted2 = doctor._record_knob(cur, knob_type="host_skip", scope_key="dup.com",
                                            value_text="skip", reason="r", ttl_seconds=3600)
        conn.commit()
        assert inserted1 is True and inserted2 is False
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE active AND knob_type='host_skip' AND scope_key='dup.com'")
            assert cur.fetchone()["n"] == 1


def test_H12_host_skip_supersedes_active_pace(fleet_db):
    """H12 SEVERITY-AWARE IDEMPOTENCY: an active (weaker) pace on a host does NOT suppress a
    (stronger) host_skip escalation on the same host."""
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO rate_governor (scope_key, min_gap_seconds, base_min_gap_seconds) "
                        "VALUES ('host:esc.com',90,90)")
        conn.commit()
        pace = {"knob_type": "pace_or_pause", "actuator": "doctor_min_gap_floor", "op_kind": "pace", "scope_key": "host:esc.com",
                "host": "esc.com", "lane": "ats", "reason": "hard_block",
                "cluster_key": "pp|esc.com|pace|ats", "diagnosis": "d", "recommendation": "r",
                "sample_count": 3, "severity": "warn", "old_gap": 90, "new_gap": 270}
        doctor.apply_auto(conn, pace)
        # Now a host_skip on the SAME host must NOT be suppressed (escalation).
        hs = {"knob_type": "host_skip", "actuator": "doctor_skip_until", "scope_key": "esc.com", "host": "esc.com", "lane": "ats",
              "reason": "hard_block", "cluster_key": "hs|esc.com|-|ats", "diagnosis": "d",
              "recommendation": "r", "sample_count": 6, "severity": "severe"}
        res = doctor.apply_auto(conn, hs)
        assert res["applied"] is True
        with conn.cursor() as cur:
            # A5: host_skip now records under the canonical 'host:<h>' scope (was bare 'esc.com').
            cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE active AND knob_type='host_skip' AND scope_key='host:esc.com'")
            assert cur.fetchone()["n"] == 1


def test_H11_quarantine_revert_clears_poison_state(fleet_db):
    """H11: the quarantine Reverse ACTUALLY clears poison_jobs.quarantined_at (the v1 Reverse was a
    no-op that left the url invisibly quarantined)."""
    from applypilot.fleet import console_app, heartbeat
    with pgqueue.connect(fleet_db) as conn:
        q = {"knob_type": "quarantine", "actuator": "poison_jobs.quarantined_at", "scope_key": "purl", "url": "purl", "host": "h.com",
             "lane": "ats", "reason": "poison", "cluster_key": "q|h.com|purl|ats",
             "diagnosis": "d", "recommendation": "r", "sample_count": 3, "severity": "warn"}
        doctor.apply_auto(conn, q)
        assert heartbeat.is_quarantined(conn, "purl") is True
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM fleet_knobs WHERE knob_type='quarantine' AND active")
            knob_id = cur.fetchone()["id"]
        console_app._do_doctor_revert(conn, {"knob_id": knob_id})
        # H11 assertion: after revert, is_quarantined is False.
        assert heartbeat.is_quarantined(conn, "purl") is False


def test_H18_doctor_signal_in_status(fleet_db):
    """H18: the fast /api/status build exposes a Doctor signal (last_pass_at + active auto-fix
    count + provenance) without the heavy diagnostics blob."""
    from applypilot.fleet import console_app
    with pgqueue.connect(fleet_db) as conn:
        for j in range(3):
            _seed_apply_failure(conn, url=f"d{j}", host="sig.com", worker_id="home-0", apply_error="blocked")
        _seed_host_governor(conn, "sig.com", attempts=5)
        doctor.run_doctor(conn, window_minutes=60)
        sig = console_app._doctor_signal(conn)
    assert sig["last_pass_at"] is not None
    assert sig["active_auto_fix_count"] >= 1
    assert "ats_paused" in sig


# ===========================================================================
# SUPPLEMENTAL RED-TEAM (A1-A12): fixes to the fixes.
# ===========================================================================

def test_A3_pace_floor_throttles_never_succeeded_host(fleet_db):
    """A3 PROVING TEST: a host that NEVER succeeded (last_applied_at IS NULL) is still spaced by the
    Doctor's doctor_min_gap_floor, via the lease's COALESCE(last_applied_at, last_attempt_at) gate."""
    from applypilot.fleet import queue
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rate_governor (scope_key, min_gap_seconds, base_min_gap_seconds, "
                "doctor_min_gap_floor, last_applied_at, last_attempt_at, block_24h) "
                "VALUES ('host:nv.com', 90, 90, 270, NULL, now() - interval '120 seconds', 5)")
        conn.commit()
        _seed_apply_failure(conn, url="nv-1", host="nv.com", queued=True)
        assert queue.lease_apply(conn, "w1", home_ip="1.1.1.1") is None
        with conn.cursor() as cur:
            cur.execute("UPDATE rate_governor SET last_attempt_at = now() - interval '500 seconds' "
                        "WHERE scope_key='host:nv.com'")
        conn.commit()
        assert queue.lease_apply(conn, "w1", home_ip="1.1.1.1") is not None


def test_A3_record_outcome_stamps_last_attempt_on_block(fleet_db):
    """A3: a block/captcha outcome stamps last_attempt_at (not just confirmed applies)."""
    from applypilot.fleet import governor
    with pgqueue.connect(fleet_db) as conn:
        governor.ensure_scope(conn, "host:b.com")
        governor.record_outcome(conn, ["host:b.com"], "block")
        with conn.cursor() as cur:
            cur.execute("SELECT last_applied_at, last_attempt_at FROM rate_governor WHERE scope_key='host:b.com'")
            row = cur.fetchone()
        assert row["last_applied_at"] is None
        assert row["last_attempt_at"] is not None


def test_A4_pause_scope_cannot_reach_linkedin(fleet_db):
    """A4 PROVING TEST: monitor.pause_scope CANNOT halt the LinkedIn lane; forbidden scopes rejected
    before any write; legitimate host pause is auto-expiring (breaker_until set, not NULL)."""
    from applypilot.fleet import monitor, governor
    with pgqueue.connect(fleet_db) as conn:
        governor.ensure_scope(conn, governor.LINKEDIN_ACCOUNT)
        acts = monitor.MonitorActions(conn)
        for forbidden in (governor.LINKEDIN_ACCOUNT, "global", "home_ip:1.2.3.4", "random"):
            with pytest.raises(monitor.ScopeNotPausable):
                acts.pause_scope(forbidden)
            conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT breaker_state, breaker_until FROM rate_governor WHERE scope_key=%s",
                        (governor.LINKEDIN_ACCOUNT,))
            row = cur.fetchone()
        assert row["breaker_state"] == "ok"
        governor.ensure_scope(conn, "host:ok.com")
        acts.pause_scope("host:ok.com")
        with conn.cursor() as cur:
            cur.execute("SELECT breaker_state, breaker_until FROM rate_governor WHERE scope_key='host:ok.com'")
            row = cur.fetchone()
        assert row["breaker_state"] == "paused"
        assert row["breaker_until"] is not None


def test_A4_codex_bridge_pause_scope_structured_rejection(fleet_db, fleet_control_db, monkeypatch):
    """A4: the codex_bridge MCP tool surfaces a structured rejection for a forbidden scope."""
    from applypilot.fleet import codex_bridge, governor
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    with pgqueue.connect(fleet_db) as conn:
        governor.ensure_scope(conn, governor.LINKEDIN_ACCOUNT)
    fn = getattr(codex_bridge.pause_scope, "fn", codex_bridge.pause_scope)
    res = fn("account:linkedin")
    assert res.get("rejected") is True and "error" in res


def test_A2_systemic_never_sticky_and_streak(fleet_db):
    """A2/A6: a SYSTEMIC pass never trips a sticky breaker (stays recommend-only); streak advances."""
    from applypilot.fleet import governor
    with pgqueue.connect(fleet_db) as conn:
        for i in range(6):
            h = f"sys{i}.com"
            for j, w in enumerate(("w0", "w1", "w0")):  # 3 hard-blocks/host across 2 workers
                _seed_apply_failure(conn, url=f"s{i}-{j}", host=h, worker_id=w, apply_error="blocked")
            with conn.cursor() as cur:
                cur.execute("INSERT INTO rate_governor (scope_key, block_24h) VALUES (%s,5)",
                            (governor.host_scope(h),))
            conn.commit()
        clusters = doctor.analyze(conn, window_minutes=60)
        planned = doctor.decide(conn, clusters)
        assert any(p.get("reason") == "systemic_block" for p in planned)
        assert not any(p.get("mode") == "auto" and p.get("knob_type") == "host_skip" for p in planned)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM rate_governor WHERE breaker_state IN ('paused','demoted')")
            assert cur.fetchone()["n"] == 0
            cur.execute("SELECT doctor_systemic_streak FROM fleet_config WHERE id=1")
            assert cur.fetchone()["doctor_systemic_streak"] == 1


def test_A6_single_bad_homeip_not_systemic(fleet_db):
    """A6: correlated blocks concentrated on ONE worker are NOT systemic (per-host fixes proceed)."""
    from applypilot.fleet import governor
    with pgqueue.connect(fleet_db) as conn:
        for i in range(6):
            h = f"ip{i}.com"
            for j in range(3):  # 3 hard-blocks/host but ALL on one worker -> NOT systemic (A6)
                _seed_apply_failure(conn, url=f"ip{i}-{j}", host=h, worker_id="only-worker",
                                    apply_error="blocked")
            with conn.cursor() as cur:
                cur.execute("INSERT INTO rate_governor (scope_key, block_24h) VALUES (%s,5)",
                            (governor.host_scope(h),))
            conn.commit()
        clusters = doctor.analyze(conn, window_minutes=60)
        planned = doctor.decide(conn, clusters)
        assert not any(p.get("reason") == "systemic_block" for p in planned)
        assert any(p.get("mode") == "auto" and p.get("knob_type") == "host_skip" for p in planned)


def test_A7_two_timeout_clusters_one_knob(fleet_db):
    """A7: two slow hosts in one pass -> exactly one timeout_bump knob + override changed once."""
    with pgqueue.connect(fleet_db) as conn:
        for h in ("slow1.com", "slow2.com"):
            for j in range(3):
                _seed_apply_failure(conn, url=f"{h}-{j}", host=h, apply_error="failed:timeout")
        doctor.run_doctor(conn, window_minutes=60)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE knob_type='timeout_bump' AND active")
            assert cur.fetchone()["n"] == 1
            cur.execute("SELECT agent_timeout_override FROM fleet_config WHERE id=1")
            ov = cur.fetchone()["agent_timeout_override"]
        assert ov == min(doctor._TIMEOUT_CEILING, doctor._DEFAULT_AGENT_TIMEOUT + doctor._TIMEOUT_BUMP_STEP)


def test_A5_host_skip_and_pace_share_canonical_scope(fleet_db):
    """A5: host_skip records under the CANONICAL 'host:<h>' scope so a pace on the same host is
    suppressed by an active host_skip."""
    from applypilot.fleet import governor
    with pgqueue.connect(fleet_db) as conn:
        for j in range(4):
            _seed_apply_failure(conn, url=f"hs{j}", host="canon.com", apply_error="blocked")
        _seed_host_governor(conn, "canon.com", attempts=5)
        res = doctor.apply_auto(conn, {
            "mode": "auto", "knob_type": "host_skip", "actuator": "doctor_skip_until",
            "scope_key": "canon.com", "host": "canon.com", "lane": "ats", "reason": "hard_block",
            "sample_count": 4, "severity": "warn",
            "cluster_key": "hard_block|canon.com|-|ats", "rows_affected": 0,
        })
        assert res["applied"] and res["scope_key"] == governor.host_scope("canon.com")
        assert doctor._suppressed_by_active(conn, {
            "knob_type": "pace_or_pause", "op_kind": "pace", "host": "canon.com",
            "scope_key": governor.host_scope("canon.com")})


def test_A10_actuator_mandatory_and_allowlisted():
    """A10: actuator mandatory + validated against the closed per-knob allow-set."""
    with pytest.raises(doctor.ConservativeViolation):
        doctor._assert_conservative({"knob_type": "host_skip", "scope_key": "h.com"})
    with pytest.raises(doctor.ConservativeViolation):
        doctor._assert_conservative({"knob_type": "host_skip", "actuator": "agent_timeout_override",
                                     "scope_key": "h.com"})
    doctor._assert_conservative({"knob_type": "host_skip", "actuator": "doctor_skip_until",
                                 "scope_key": "h.com"})


def test_A10_no_forbidden_mutation_outside_dispatcher():
    """A10 static test: no forbidden SQL mutation (set_paused / fleet_config paused / breaker /
    account:linkedin / halted_until / kill_linkedin) outside _execute_actuator + the gate constants."""
    import ast
    import re
    src = open("src/applypilot/fleet/doctor.py", encoding="utf-8").read()
    tree = ast.parse(src)
    # Only inspect strings that are SQL UPDATE *statements* (UPDATE <table> ... SET ...), not prose
    # docstrings/comments that merely mention these tokens.
    is_update_stmt = re.compile(r"\bupdate\s+\w+\b.*\bset\b", re.IGNORECASE | re.DOTALL)
    forbidden_in_update = ("breaker_state", "breaker_until", "halted_until")
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_actuator":
            continue  # the single dispatcher is allowed to write the allow-set columns
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                low = sub.value.lower()
                if not is_update_stmt.search(low):
                    continue
                # Forbidden: UPDATE fleet_config ... SET paused (the shared kill switch) -- but the
                # ATS-only ats_paused / provenance / debounce columns ARE allowed outside the dispatcher.
                if "update fleet_config" in low and re.search(r"\bset\b[^;]*\bpaused\b", low) \
                        and "ats_paused" not in low and "pause_source" not in low \
                        and "pause_armed" not in low:
                    offenders.append(sub.value)
                for tok in forbidden_in_update:
                    if tok in low:
                        offenders.append(sub.value)
    assert not offenders, f"forbidden UPDATE statements outside the dispatcher: {offenders}"


def test_A1_sweep_and_breaker_no_deadlock(fleet_db):
    """A1: co-run evaluate_breakers + sweep_expired in two connections on a shared throttled+expiring
    multi-host set; assert no deadlock (40P01)."""
    import threading
    from applypilot.fleet import governor
    hosts = [f"dl{i}.com" for i in range(8)]
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            for h in hosts:
                sk = governor.host_scope(h)
                cur.execute("INSERT INTO rate_governor (scope_key, min_gap_seconds, base_min_gap_seconds, "
                            "doctor_min_gap_floor, block_24h, breaker_state, breaker_until) "
                            "VALUES (%s, 90, 90, 270, 5, 'throttled', now() - interval '1 second')", (sk,))
                cur.execute("INSERT INTO fleet_knobs (knob_type, scope_key, value_text, reason, active, expires_at) "
                            "VALUES ('pace_or_pause', %s, '270', 'x', TRUE, now() - interval '1 second')", (sk,))
        conn.commit()

    errors = []
    def run(fn):
        try:
            with pgqueue.connect(fleet_db) as c:
                for _ in range(15):
                    fn(c)
        except Exception as e:
            errors.append(repr(e))

    t1 = threading.Thread(target=run, args=(lambda c: governor.evaluate_breakers(c, min_samples=1),))
    t2 = threading.Thread(target=run, args=(lambda c: doctor.sweep_expired(c),))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not any("40P01" in e or "deadlock" in e.lower() for e in errors), errors
