"""Postgres work-queue layer for the cloud apply fleet (queue-offload design).

See docs/superpowers/specs/2026-06-24-cloud-apply-fleet-design.md (S3a / S5 / S6).

This is the ONLY state that lives in the cloud: a thin offsite-apply work queue + result
mailbox on Railway Postgres. Home SQLite stays authoritative -- if Postgres is lost, home
re-pushes and nothing important is gone.

Every function takes an OPEN psycopg connection so callers control transaction scope and the
whole layer is trivially testable against a throwaway local Postgres. The production DSN comes
only from FLEET_PG_DSN; generic database variables may identify an unrelated database.

Concurrency contract (the part that must never break):
  * lease_one  -> FOR UPDATE SKIP LOCKED: N workers each grab a DISTINCT row, never blocking.
  * a crashed worker's lease expires -> reclaim parks it; a job that may have hit submit is
    pinned crash_unconfirmed and NEVER re-leased (no double-submit under Jonathan's name).
  * write_result is lease_owner-guarded: only the holder may close a row.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

_SCHEMA_SQL = (Path(__file__).with_name("fleet_schema.sql")).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

_CONNECTION_DEFAULTS = {
    "application_name": "ApplyPilot",
    "connect_timeout": "10",
    "keepalives": "1",
    "keepalives_idle": "30",
    "keepalives_interval": "10",
    "keepalives_count": "5",
}
_INDIRECT_CONNINFO_OPTIONS = {"passfile", "service"}
LIBPQ_CONNECTION_ENV_VARS = frozenset(
    {
        "PGAPPNAME",
        "PGCHANNELBINDING",
        "PGCLIENTENCODING",
        "PGCONNECT_TIMEOUT",
        "PGDATABASE",
        "PGDATESTYLE",
        "PGGEQO",
        "PGGSSDELEGATION",
        "PGGSSENCMODE",
        "PGGSSLIB",
        "PGHOST",
        "PGHOSTADDR",
        "PGKRBSRVNAME",
        "PGLOADBALANCEHOSTS",
        "PGLOCALEDIR",
        "PGMAXPROTOCOLVERSION",
        "PGMINPROTOCOLVERSION",
        "PGOPTIONS",
        "PGPASSFILE",
        "PGPASSWORD",
        "PGPORT",
        "PGREQUIREAUTH",
        "PGREQUIREPEER",
        "PGREQUIRESSL",
        "PGSERVICE",
        "PGSERVICEFILE",
        "PGSYSCONFDIR",
        "PGSSLCERT",
        "PGSSLCERTMODE",
        "PGSSLCOMPRESSION",
        "PGSSLCRL",
        "PGSSLCRLDIR",
        "PGSSLKEY",
        "PGSSLMAXPROTOCOLVERSION",
        "PGSSLMINPROTOCOLVERSION",
        "PGSSLMODE",
        "PGSSLNEGOTIATION",
        "PGSSLROOTCERT",
        "PGSSLSNI",
        "PGTARGETSESSIONATTRS",
        "PGTZ",
        "PGUSER",
    }
)
_LOCAL_PG_HOSTS = {"127.0.0.1", "::1", "localhost"}
_DNS_HOST_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*")


def _is_authenticated_private_host(host: str) -> bool:
    normalized = host.strip().rstrip(".").lower()
    if normalized in _LOCAL_PG_HOSTS:
        return True

    try:
        ipaddress.ip_address(normalized)
    except ValueError:
        if normalized.startswith(("/", "@")) or _DNS_HOST_RE.fullmatch(normalized) is None:
            raise RuntimeError("Fleet Postgres host has invalid topology syntax; connection details were not logged.")
    else:
        return False

    labels = normalized.split(".")
    if normalized.endswith(".railway.internal"):
        return len(labels) >= 3
    if normalized.endswith(".ts.net"):
        if len(labels) < 4:
            raise RuntimeError(
                "Fleet Postgres Tailscale topology requires a full MagicDNS hostname; "
                "connection details were not logged."
            )
        return True
    return False


def _explicit_targets(params: dict[str, str]) -> list[str]:
    if "hostaddr" in params:
        raise RuntimeError("Fleet Postgres DSN must not contain hostaddr; connection details were not logged.")

    host = params.get("host")
    if not host:
        raise RuntimeError("Fleet Postgres DSN must contain an explicit host; connection details were not logged.")

    hosts = host.split(",")
    if any(not item.strip() for item in hosts):
        raise RuntimeError("Fleet Postgres host entries must be non-empty; connection details were not logged.")
    return hosts


@dataclass(frozen=True)
class RuntimeConninfo:
    conninfo: str
    principal: str


def _validate_explicit_conninfo(dsn: str) -> tuple[dict[str, str], list[str]]:
    """Parse one DSN once and enforce the policy shared by every connection class."""
    ambient = sorted(name for name in LIBPQ_CONNECTION_ENV_VARS if name in os.environ)
    if ambient:
        names = ", ".join(ambient)
        raise RuntimeError(
            f"Ambient libpq variables are not allowed for fleet Postgres: {names}; connection details were not logged."
        )

    try:
        params = conninfo_to_dict(dsn)
    except (psycopg.Error, ValueError) as exc:
        raise RuntimeError("Invalid fleet Postgres DSN; connection details were not logged.") from exc

    indirect = sorted(_INDIRECT_CONNINFO_OPTIONS.intersection(params))
    if indirect:
        names = ", ".join(indirect)
        raise RuntimeError(
            f"Fleet Postgres DSN indirection is not allowed: {names}; connection details were not logged."
        )

    targets = _explicit_targets(params)
    if params.get("password") in (None, ""):
        raise RuntimeError("Fleet Postgres DSN must contain an explicit password; connection details were not logged.")

    return params, targets


def _validate_transport(params: dict[str, str], targets: list[str]) -> None:
    uses_public_tcp = any(not _is_authenticated_private_host(target) for target in targets)
    if uses_public_tcp:
        if params.get("sslmode", "").lower() != "verify-full":
            raise RuntimeError(
                "Public fleet Postgres connections must specify sslmode=verify-full; "
                "connection details were not logged."
            )
        if not params.get("sslrootcert", "").strip():
            raise RuntimeError(
                "Public fleet Postgres connections must specify sslrootcert explicitly; "
                "connection details were not logged."
            )


def _render_conninfo(dsn: str, params: dict[str, str]) -> str:
    defaults = {key: value for key, value in _CONNECTION_DEFAULTS.items() if key not in params}
    try:
        return make_conninfo(dsn, **defaults)
    except (psycopg.Error, ValueError) as exc:
        raise RuntimeError("Invalid fleet Postgres DSN; connection details were not logged.") from exc


def validate_runtime_conninfo(
    dsn: str,
    *,
    expected_principal: str | None = None,
    require_expected_principal: bool = False,
) -> RuntimeConninfo:
    """Validate one effective runtime DSN without connecting or logging secrets."""
    params, targets = _validate_explicit_conninfo(dsn)

    if params.get("options"):
        raise RuntimeError(
            "Fleet Postgres DSN options are not allowed because they can change the runtime role; "
            "connection details were not logged."
        )

    user = params.get("user")
    if not user or not user.strip():
        raise RuntimeError(
            "Fleet Postgres DSN does not name an explicit mapped principal; connection details were not logged."
        )
    principal = user.strip()
    if principal.casefold() in FORBIDDEN_LAUNCH_PRINCIPALS:
        raise RuntimeError(
            "Fleet Postgres DSN selects a forbidden admin/shared principal (postgres/fleet_worker); "
            "connection details were not logged."
        )
    approved = expected_principal or os.environ.get("APPLYPILOT_PG_PRINCIPAL")
    if require_expected_principal and not approved:
        raise RuntimeError(
            "Approved mapped Postgres principal is not configured; set APPLYPILOT_PG_PRINCIPAL."
        )
    if approved and principal != approved:
        raise RuntimeError(
            "Fleet Postgres DSN principal does not match the approved mapped principal; "
            "connection details were not logged."
        )
    _validate_transport(params, targets)
    sslmode = params.get("sslmode", "").casefold()
    if sslmode not in {"require", "verify-ca", "verify-full"}:
        raise RuntimeError(
            "Fleet Postgres DSN must explicitly require TLS; connection details were not logged."
        )

    return RuntimeConninfo(conninfo=_render_conninfo(dsn, params), principal=principal)


def _prepare_conninfo(dsn: str) -> str:
    return validate_runtime_conninfo(dsn, require_expected_principal=True).conninfo


def _prepare_admin_conninfo(dsn: str) -> str:
    """Validate an explicitly privileged controller/migrator connection.

    This deliberately separate path never backs a runtime launcher. It retains the
    network, password, ambient-environment, and indirection defenses while permitting
    the database owner/admin identities required for schema and disposable-test setup.
    """
    params, targets = _validate_explicit_conninfo(dsn)
    if params.get("options"):
        raise RuntimeError(
            "Fleet Postgres DSN options are not allowed because they can change the session role; "
            "connection details were not logged."
        )
    _validate_transport(params, targets)
    return _render_conninfo(dsn, params)


def get_dsn(dsn: str | None = None) -> str:
    if dsn:
        return dsn
    dsn = os.environ.get("FLEET_PG_DSN")
    if not dsn:
        raise RuntimeError("No fleet Postgres DSN: set FLEET_PG_DSN explicitly.")
    return dsn


# Admin / shared launch principals that MUST NOT back a runtime fleet DSN. The mapped
# topology gives every node a unique login; ``postgres`` is the ambiguous local admin and
# ``fleet_worker`` is the retired shared role.
FORBIDDEN_LAUNCH_PRINCIPALS = frozenset({"postgres", "fleet_worker"})


def assert_mapped_principal(
    dsn: str,
    *,
    expected_principal: str | None = None,
) -> str:
    """Reuse the hardened conninfo parser to reject admin/shared launch principals.

    This is the single, hardened principal parser -- there is deliberately no second,
    weaker regex parser. It rejects the postgres admin / shared fleet_worker principal
    selected via a libpq keyword DSN, a PostgreSQL URL DSN (including percent-encoded and
    query-string identities), mixed case / quoted spellings, the ambient PGUSER form, and
    the indirect service / passfile forms that hide the principal. It returns the
    validated principal and never logs the DSN value.
    """
    return validate_runtime_conninfo(
        dsn,
        expected_principal=expected_principal,
        require_expected_principal=True,
    ).principal


def connect(dsn: str | None = None, *, autocommit: bool = False) -> psycopg.Connection:
    """Open a runtime-principal connection to the fleet Postgres."""
    return psycopg.connect(_prepare_conninfo(get_dsn(dsn)), autocommit=autocommit, row_factory=dict_row)


def connect_admin(dsn: str | None = None, *, autocommit: bool = False) -> psycopg.Connection:
    """Open an explicit privileged controller/migrator connection.

    Runtime launchers and workers must never call this API.
    """
    return psycopg.connect(
        _prepare_admin_conninfo(get_dsn(dsn)),
        autocommit=autocommit,
        row_factory=dict_row,
    )


def connect_runtime(
    dsn: str | None,
    *,
    worker_id: str,
    contract: str,
    autocommit: bool = False,
) -> psycopg.Connection:
    """Connect and authenticate the exact database-mapped worker before use."""
    from applypilot.fleet.pg_roles import validate_runtime_principal

    conn = connect(dsn, autocommit=autocommit)
    try:
        validate_runtime_principal(conn, worker_id=worker_id, contract=contract)
    except BaseException:
        conn.close()
        raise
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    """Idempotently create apply_queue + fleet_config (+ indexes, seed config row)."""
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Lease  (worker pulls the top host-polite queued job, atomically)
# ---------------------------------------------------------------------------

_LEASE_SQL = """
WITH host_recent AS (
    SELECT apply_domain, MAX(last_attempted_at) AS last_at
    FROM apply_queue
    WHERE last_attempted_at > now() - make_interval(secs => %(politeness)s)
    GROUP BY apply_domain
),
next_job AS (
    SELECT q.url
    FROM apply_queue q
    LEFT JOIN host_recent hr ON hr.apply_domain = q.apply_domain
    WHERE q.status = 'queued'
      AND q.approved_batch IS NOT NULL
      AND NOT (
          LOWER(TRIM(COALESCE(q.company,''))) = ANY(%(blocked_names)s)
          OR q.url ILIKE ANY(%(blocked_pats)s)
          OR COALESCE(q.application_url,'') ILIKE ANY(%(blocked_pats)s)
      )
      AND (hr.last_at IS NULL
           OR hr.last_at < now() - make_interval(
                  secs => (%(politeness)s * (0.7 + random() * 0.7))::int))
    ORDER BY q.score DESC, q.url            -- url tie-break = deterministic, mirrors home
    LIMIT 1
    FOR UPDATE OF q SKIP LOCKED             -- N workers each grab a distinct row, no blocking
)
UPDATE apply_queue q
SET status            = 'leased',
    lease_owner       = %(worker)s,
    lease_expires_at  = now() + make_interval(secs => %(ttl)s),
    last_attempted_at = now(),
    attempts          = q.attempts + 1,
    updated_at        = now()
FROM next_job
WHERE q.url = next_job.url
RETURNING q.url, q.company, q.title, q.application_url, q.score, q.apply_domain, q.attempts;
"""


def lease_one(
    conn: psycopg.Connection,
    worker_id: str,
    *,
    ttl_seconds: int = 1200,
    politeness_seconds: int = 90,
) -> dict[str, Any] | None:
    """Retired score-only lease API; use ``fleet.queue.lease_apply``."""
    del conn, worker_id, ttl_seconds, politeness_seconds
    raise RuntimeError(
        "legacy score-only lease_one is disabled; use fleet.queue.lease_apply "
        "with an active canonical ATS policy"
    )


# ---------------------------------------------------------------------------
# Reclaim  (startup + periodic crash sweep)
# ---------------------------------------------------------------------------

_RECLAIM_SQL = """
WITH stale AS (
    SELECT url
    FROM apply_queue
    WHERE status = 'leased'
      AND lease_expires_at < now() - make_interval(secs => %(grace)s)
    FOR UPDATE SKIP LOCKED
)
UPDATE apply_queue q
SET status           = 'crash_unconfirmed'::apply_queue_status,
    apply_error      = 'crash_unconfirmed',
    attempts         = 99,
    lease_owner      = NULL,
    lease_expires_at = NULL,
    updated_at       = now()
FROM stale s
WHERE q.url = s.url
RETURNING q.url, q.status, q.worker_id, q.apply_domain;
"""


def reclaim_stale_leases(conn: psycopg.Connection, *, grace_seconds: int = 30) -> list[dict[str, Any]]:
    """Sweep leased rows whose lease expired (a clean finish always writes terminal status, so
    an expired lease == a hard crash) and PARK every one as crash_unconfirmed (attempts=99),
    NEVER re-leased. Mirrors launcher.reclaim_stale_leases.

    Why park unconditionally (not requeue the 'pre-launch' ones): `attempts` is bumped ONCE,
    at lease time, so attempts=1 covers BOTH "leased but never launched" AND "leased, launched,
    crashed at the submit step before any terminal write" -- they are INDISTINGUISHABLE here.
    A live incident (2026-06-29) hit exactly the second case: a worker died at "Now submitting
    the application" with the row still at attempts=1, apply_error=NULL. Requeuing on attempts<=1
    would have applied a SECOND time under Jonathan's name. The owner's hard rule (NEVER
    double-apply) outranks the cost of re-reviewing the rare job killed genuinely pre-launch:
    crash_unconfirmed rows surface in apply-failures for a manual decision. Returns the reclaimed
    (url, status)."""
    with conn.cursor() as cur:
        cur.execute(_RECLAIM_SQL, {"grace": grace_seconds})
        rows = cur.fetchall()
        # A watchdog reclaim used to leave no result event behind, making an old
        # hard-crash indistinguishable from an uninstrumented row. Record the
        # conservative outcome and its missing-evidence state for later triage.
        for row in rows:
            cur.execute(
                "INSERT INTO apply_result_events ("
                "queue_name, url, worker_id, status, apply_status, apply_error, "
                "target_host, application_tool_calls, result_metadata, result_line, source"
                ") VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    "apply_queue",
                    row["url"],
                    row["worker_id"],
                    "crash_unconfirmed",
                    "crash_unconfirmed",
                    "crash_unconfirmed",
                    row["apply_domain"],
                    None,
                    Jsonb({"reclaim_reason": "lease_expired", "execution_evidence": "missing"}),
                    "RESULT:crash_unconfirmed",
                    "watchdog",
                ),
            )
    conn.commit()
    return rows


# ---------------------------------------------------------------------------
# Spend cap / kill switch
# ---------------------------------------------------------------------------

def should_halt(conn: psycopg.Connection) -> bool:
    """True if the fleet must stop leasing: globally paused OR cumulative spend >= cap.
    A soft pre-lease gate (spec S6) -- up to ~N in-flight jobs may overshoot, which is fine
    against the POC budget.

    NOTE (H1): this reads ONLY the shared kill switch (fleet_config.paused) + spend cap. It
    deliberately does NOT read ats_paused. The ATS lane uses ats_should_halt() instead
    (which OR-s in ats_paused); the LinkedIn lane uses linkedin_should_halt() to avoid
    reading apply_queue from the catastrophe lane's pre-flight tick."""
    with conn.cursor() as cur:
        cur.execute("SELECT public.fleet_worker_admission_snapshot() AS state")
        row = cur.fetchone()["state"] or {}
    conn.rollback()  # read-only
    return bool(row.get("global_should_halt"))


def linkedin_should_halt(conn: psycopg.Connection) -> bool:
    """LinkedIn-lane pre-lease halt gate.

    This intentionally reads only the shared operator kill switch. LinkedIn's lane-specific
    blast-radius controls (account halt, canary, daily cap, min-gap, and mutex) are enforced
    atomically by queue.lease_linkedin(). The pre-flight tick must not read apply_queue:
    that couples LinkedIn to ATS spend state and can deadlock with schema maintenance.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT public.fleet_worker_admission_snapshot() AS state")
        row = cur.fetchone()["state"] or {}
    conn.rollback()  # read-only
    return bool(row.get("linkedin_should_halt"))


def ats_should_halt(conn: psycopg.Connection) -> bool:
    """ATS-lane pre-lease halt gate: the shared kill switch (paused/spend cap) OR the
    Fleet Doctor's ATS-only ats_paused (H1). The LinkedIn worker must NEVER call this -- it
    uses linkedin_should_halt() so a Doctor ATS pause can never stop the LinkedIn lane."""
    with conn.cursor() as cur:
        cur.execute("SELECT public.fleet_worker_admission_snapshot() AS state")
        row = cur.fetchone()["state"] or {}
    conn.rollback()  # read-only
    return bool(row.get("ats_should_halt"))


def set_ats_paused(conn: psycopg.Connection, paused: bool, *, source: str | None = None) -> None:
    """Set the ATS-only pause flag (H1). The Fleet Doctor routes its lane-pause HERE, never to
    set_paused(). source ('doctor'|NULL) records provenance so the console can label it and the
    Doctor only ever auto-reverts its OWN pause. Clearing the pause also clears the source."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE fleet_config SET ats_paused = %s, "
            "ats_pause_source = CASE WHEN %s THEN %s ELSE NULL END, updated_at = now() WHERE id = 1",
            (paused, paused, source),
        )
    conn.commit()


def set_spend_cap(conn: psycopg.Connection, cap_usd: float) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE fleet_config SET spend_cap_usd = %s, updated_at = now() WHERE id = 1",
            (cap_usd,),
        )
    conn.commit()


def set_paused(conn: psycopg.Connection, paused: bool) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE fleet_config SET paused = %s, updated_at = now() WHERE id = 1",
            (paused,),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Result write  (releases the lease; lease_owner-guarded)
# ---------------------------------------------------------------------------

_RESULT_SQL = """
UPDATE apply_queue
SET status                  = %(status)s::apply_queue_status,
    apply_status            = %(apply_status)s,
    apply_error             = %(apply_error)s,
    verification_confidence = %(verification_confidence)s,
    agent_model             = %(agent_model)s,
    est_cost_usd            = %(est_cost_usd)s,
    cumulative_cost_usd     = COALESCE(cumulative_cost_usd, 0) + %(est_cost_usd)s,
    applied_at              = CASE WHEN %(status)s = 'applied' THEN now() ELSE applied_at END,
    worker_id               = %(worker)s,
    apply_duration_ms       = %(apply_duration_ms)s,
    lease_owner             = NULL,
    lease_expires_at        = NULL,
    updated_at              = now()
WHERE url = %(url)s
  AND lease_owner = %(worker)s          -- only the lease holder may close it
RETURNING url;
"""


def write_result(
    conn: psycopg.Connection,
    worker_id: str,
    url: str,
    *,
    status: str,
    apply_status: str | None = None,
    apply_error: str | None = None,
    verification_confidence: str | None = None,
    agent_model: str | None = None,
    est_cost_usd: float | None = 0,
    apply_duration_ms: int | None = None,
) -> bool:
    """Write a terminal result and release the lease. est_cost_usd is written UNCONDITIONALLY
    (0 when the CLI reports none) so the cap SUM stays consistent. Returns True only if this
    worker held the lease (a reclaimed row, now owned by no one, is never clobbered)."""
    if est_cost_usd is None:
        est_cost_usd = 0
    with conn.cursor() as cur:
        cur.execute(
            _RESULT_SQL,
            {
                "status": status,
                "apply_status": apply_status,
                "apply_error": apply_error,
                "verification_confidence": verification_confidence,
                "agent_model": agent_model,
                "est_cost_usd": est_cost_usd,
                "worker": worker_id,
                "apply_duration_ms": apply_duration_ms,
                "url": url,
            },
        )
        landed = cur.fetchone() is not None
    conn.commit()
    return landed


# ---------------------------------------------------------------------------
# Re-queue  (a RETRYABLE non-submit: back to 'queued', lease released; lease_owner-guarded)
# ---------------------------------------------------------------------------

_REQUEUE_SQL = """
UPDATE apply_queue
SET status           = 'queued'::apply_queue_status,
    apply_error      = %(apply_error)s,
    -- Undo the lease-time attempt bump: a usage/quota wall hit on turn 1 PROVABLY never
    -- touched the page, so this lease did nothing -- don't penalize the job. NOT pinned to
    -- 99 (that's crash_unconfirmed / reclaim). GREATEST guards against underflow.
    attempts         = GREATEST(attempts - 1, 0),
    lease_owner      = NULL,
    lease_expires_at = NULL,
    updated_at       = now()
WHERE url = %(url)s
  AND lease_owner = %(worker)s          -- only the lease holder may re-queue it
RETURNING url;
"""


def requeue_job(
    conn: psycopg.Connection,
    worker_id: str,
    url: str,
    *,
    apply_error: str | None = None,
) -> bool:
    """Return a leased job to 'queued' so it can be re-leased -- the RETRYABLE counterpart
    to write_result. Use ONLY when the run provably never touched the application form
    (e.g. an agent usage/quota wall hit before any browser tool call): this re-queue can
    NEVER cause a double-submit. attempts is NOT pinned (the lease bump is undone). Mirrors
    write_result's lease_owner guard, so a reclaimed/foreign row is never clobbered. Returns
    True only if this worker held the lease."""
    with conn.cursor() as cur:
        cur.execute(
            _REQUEUE_SQL,
            {"apply_error": apply_error, "worker": worker_id, "url": url},
        )
        landed = cur.fetchone() is not None
    conn.commit()
    return landed


# ---------------------------------------------------------------------------
# PUSH / PULL sync surface (home box drives these; see fleet_sync.py)
# ---------------------------------------------------------------------------

_PUSH_SQL = """
INSERT INTO apply_queue (url, company, title, application_url, score, apply_domain, status)
VALUES (%(url)s, %(company)s, %(title)s, %(application_url)s, %(score)s, %(apply_domain)s, 'queued')
ON CONFLICT (url) DO UPDATE
SET company         = EXCLUDED.company,
    title           = EXCLUDED.title,
    application_url = EXCLUDED.application_url,
    score           = EXCLUDED.score,
    apply_domain    = EXCLUDED.apply_domain,
    updated_at      = now()
WHERE apply_queue.status = 'queued';   -- only refresh+requeue still-pending rows
"""


def push_jobs(conn: psycopg.Connection, rows: Iterable[dict[str, Any]]) -> int:
    """Retired score-only push API; use ``fleet.queue.push_apply_jobs``."""
    del conn, rows
    raise RuntimeError(
        "legacy score-only push_jobs is disabled; use fleet.queue.push_apply_jobs "
        "with canonical decision provenance"
    )


# PULL works for both lanes: linkedin_queue is schema-identical to apply_queue
# (LIKE apply_queue). Table names come ONLY from this allowlist -- never from input.
_PULL_TABLES = ("apply_queue", "linkedin_queue")

_PULL_SQL = """
SELECT url, status, apply_status, apply_error, verification_confidence,
       agent_model, est_cost_usd, applied_at, worker_id, apply_duration_ms
FROM {table}
WHERE status IN ('applied','failed','blocked','crash_unconfirmed')
  AND synced_to_home_at IS NULL
ORDER BY updated_at
LIMIT %(limit)s;
"""


def _check_pull_table(table: str) -> str:
    if table not in _PULL_TABLES:
        raise ValueError(f"unknown pull table {table!r} (allowed: {_PULL_TABLES})")
    return table


def fetch_pending_results(conn: psycopg.Connection, *, limit: int = 500,
                          table: str = "apply_queue") -> list[dict[str, Any]]:
    """PULL step A: terminal rows not yet ingested into home SQLite."""
    with conn.cursor() as cur:
        cur.execute(_PULL_SQL.format(table=_check_pull_table(table)), {"limit": limit})
        rows = cur.fetchall()
    conn.rollback()
    return rows


def mark_synced(conn: psycopg.Connection, url: str, *, table: str = "apply_queue") -> None:
    """Stamp a row ingested-home so PULL is idempotent."""
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {_check_pull_table(table)} SET synced_to_home_at = now() WHERE url = %s",
                    (url,))
    conn.commit()


# ---------------------------------------------------------------------------
# Observability  (POC metrics, spec S7)
# ---------------------------------------------------------------------------

_STATS_SQL = """
SELECT
    COUNT(*) FILTER (WHERE status='applied')                                              AS applied,
    COUNT(*) FILTER (WHERE status IN ('applied','failed','blocked','crash_unconfirmed'))  AS attempted,
    COUNT(*) FILTER (WHERE status='queued')                                               AS queued,
    COUNT(*) FILTER (WHERE status='leased')                                               AS leased,
    COUNT(*) FILTER (WHERE status='blocked' OR apply_error ILIKE '%%captcha%%')           AS blocked_or_captcha,
    COUNT(*) FILTER (WHERE status='crash_unconfirmed')                                    AS crash_unconfirmed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status='applied')
          / NULLIF(COUNT(*) FILTER (WHERE status IN ('applied','failed','blocked','crash_unconfirmed')),0), 1) AS success_pct,
    COALESCE(SUM(cumulative_cost_usd),0)                                                   AS total_cost,
    ROUND(COALESCE(SUM(cumulative_cost_usd),0)
          / NULLIF(COUNT(*) FILTER (WHERE status='applied'),0), 4)                         AS cost_per_apply
FROM apply_queue
{group};
"""


def put_asset(conn: psycopg.Connection, name: str, data: bytes) -> None:
    """Store a worker asset (profile.json / resume.pdf) in Postgres for the fleet to hydrate."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO fleet_assets (name, data) VALUES (%s, %s) "
            "ON CONFLICT (name) DO UPDATE SET data = EXCLUDED.data, updated_at = now()",
            (name, data),
        )
    conn.commit()


def get_asset(conn: psycopg.Connection, name: str) -> bytes | None:
    with conn.cursor() as cur:
        cur.execute("SELECT data FROM fleet_assets WHERE name = %s", (name,))
        row = cur.fetchone()
    conn.rollback()
    return bytes(row["data"]) if row else None


def queue_stats(conn: psycopg.Connection, *, by_model: bool = False) -> list[dict[str, Any]]:
    """POC metrics (spec S7). by_model=True breaks the numbers down by agent_model for the
    Sonnet-vs-DeepSeek A/B."""
    if by_model:
        sql = _STATS_SQL.replace("FROM apply_queue\n{group}", "FROM apply_queue\nGROUP BY agent_model")
        sql = sql.replace("SELECT\n", "SELECT\n    agent_model,\n")
    else:
        sql = _STATS_SQL.format(group="")
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    conn.rollback()
    return rows
