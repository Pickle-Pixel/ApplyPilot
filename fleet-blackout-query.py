"""fleet-blackout-query.py <machine_label> [role]

PowerShell-friendly guard for launchers. Prints:
  OK|label|role|||
  BLOCKED|label|role|policy|expires_at|reason

Any configuration or database error is reported as BLOCKED so the fleet agent
and direct launchers fail closed.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from urllib.parse import parse_qsl, unquote, urlsplit

# Endpoint, database/role, credentials, TLS/auth, and routing define authority.
# Client tuning such as timeouts, application names, and keepalives is excluded.
_AUTHORITY_IDENTITY_FIELDS = frozenset(
    {
        "service",
        "servicefile",
        "host",
        "hostaddr",
        "port",
        "dbname",
        "user",
        "password",
        "passfile",
        "options",
        "channel_binding",
        "sslmode",
        "sslnegotiation",
        "sslcompression",
        "sslcert",
        "sslkey",
        "sslkeylogfile",
        "sslcertmode",
        "sslpassword",
        "sslrootcert",
        "sslcrl",
        "sslcrldir",
        "sslsni",
        "requirepeer",
        "require_auth",
        "min_protocol_version",
        "max_protocol_version",
        "ssl_min_protocol_version",
        "ssl_max_protocol_version",
        "gssencmode",
        "krbsrvname",
        "gsslib",
        "gssdelegation",
        "replication",
        "target_session_attrs",
        "load_balance_hosts",
        "scram_client_key",
        "scram_server_key",
        "oauth_issuer",
        "oauth_client_id",
        "oauth_client_secret",
        "oauth_scope",
    }
)
# libpq treats an empty value as the effective environment/built-in default for
# these connection identity fields; focused tests cover both default sources.
_EMPTY_MEANS_DEFAULT_FIELDS = frozenset(
    {"host", "hostaddr", "port", "user", "password", "passfile", "dbname"}
)
_SENSITIVE_AUTHORITY_FIELDS = frozenset(
    {
        "password",
        "sslpassword",
        "passfile",
        "sslkey",
        "sslkeylogfile",
        "oauth_client_secret",
        "scram_client_key",
        "scram_server_key",
    }
)
_SENSITIVE_KEY_PATTERN = "|".join(
    re.escape(key) for key in sorted(_SENSITIVE_AUTHORITY_FIELDS, key=len, reverse=True)
)
_CONNINFO_VALUE_PATTERN = r'''(?:'(?:\\.|[^'])*'|"(?:\\.|[^"])*"|(?:\\.|[^\s])+)'''
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    rf"(?i)\b({_SENSITIVE_KEY_PATTERN})\s*=\s*({_CONNINFO_VALUE_PATTERN})"
)
_POSTGRES_USERINFO_RE = re.compile(r"(?i)\b(postgres(?:ql)?://)([^@\s/]+)@")


def _normalized_dsn(dsn: str, *, pq, conninfo_to_dict) -> dict[str, str]:
    defaults = {
        option.keyword.decode(): option.val.decode()
        for option in pq.Conninfo.get_defaults()
        if option.val is not None
    }
    explicit = conninfo_to_dict(dsn)
    normalized = dict(defaults)
    normalized.update(explicit)
    for key, value in explicit.items():
        if value != "" or key not in _EMPTY_MEANS_DEFAULT_FIELDS:
            continue
        if defaults.get(key):
            normalized[key] = defaults[key]
        else:
            normalized.pop(key, None)
    if not normalized.get("dbname") and normalized.get("user"):
        normalized["dbname"] = normalized["user"]
    normalized.setdefault("sslcertmode", "allow")
    return {key: value for key, value in normalized.items() if key in _AUTHORITY_IDENTITY_FIELDS}


def _safe_field(value: object) -> str:
    return " ".join(str(value).replace("|", "/").split())[:300]


def _safe_expiry(value: object) -> str:
    expires = _safe_field(value)
    if expires:
        try:
            datetime.fromisoformat(expires.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RuntimeError("Invalid machine blackout expiry protocol") from exc
    return expires


def _unquote_conninfo_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return re.sub(r"\\(.)", r"\1", value)


def _sensitive_values_from_dsn(dsn: str, *, conninfo_to_dict=None) -> set[str]:
    if conninfo_to_dict is not None:
        try:
            parsed = conninfo_to_dict(dsn)
            return {
                value
                for key, value in parsed.items()
                if key in _SENSITIVE_AUTHORITY_FIELDS and value
            }
        except Exception:
            pass
    values = {
        _unquote_conninfo_value(match.group(2))
        for match in _SENSITIVE_ASSIGNMENT_RE.finditer(dsn)
    }
    try:
        parsed = urlsplit(dsn)
        if parsed.scheme.lower() in {"postgres", "postgresql"}:
            if parsed.password:
                values.add(unquote(parsed.password))
            values.update(
                value
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                if key.lower() in _SENSITIVE_AUTHORITY_FIELDS and value
            )
    except ValueError:
        pass
    return {value for value in values if value}


def _sanitize_text(value: object, *dsns: str, conninfo_to_dict=None) -> str:
    message = str(value)
    for dsn in dsns:
        secrets = _sensitive_values_from_dsn(
            dsn,
            conninfo_to_dict=conninfo_to_dict,
        )
        for secret in sorted(secrets, key=len, reverse=True):
            message = message.replace(secret, "***")
        if dsn:
            message = message.replace(dsn, "***")
    message = _POSTGRES_USERINFO_RE.sub(r"\1***@", message)
    message = _SENSITIVE_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}=***", message
    )
    return _safe_field(message)


def _sanitize_status_line(line: object, *dsns: str, conninfo_to_dict=None) -> str:
    parts = str(line).split("|", 5)
    if len(parts) != 6 or parts[0] not in {"OK", "BLOCKED"}:
        raise RuntimeError("Invalid machine blackout status protocol")
    if parts[0] == "OK" and parts[3:] != ["", "", ""]:
        raise RuntimeError("Invalid machine blackout OK status protocol")
    status, machine, role, policy, expires, reason = parts
    safe_machine = _safe_field(machine)
    safe_role = _safe_field(role)
    safe_policy = _sanitize_text(
        policy,
        *dsns,
        conninfo_to_dict=conninfo_to_dict,
    )
    safe_expires = _safe_expiry(expires)
    safe_reason = _sanitize_text(
        reason,
        *dsns,
        conninfo_to_dict=conninfo_to_dict,
    )
    return f"{status}|{safe_machine}|{safe_role}|{safe_policy}|{safe_expires}|{safe_reason}"

label = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("APPLYPILOT_FLEET_LABEL", "home")
role = sys.argv[2] if len(sys.argv) > 2 else "fleet"
fleet_dsn = os.environ.get("FLEET_PG_DSN", "").strip()
# APPLYPILOT_FLEET_DSN is NOT a connection source anymore (no generic fallback); it is read
# only so a stale value can be redacted from diagnostics and rejected if it disagrees.
applypilot_fleet_dsn = os.environ.get("APPLYPILOT_FLEET_DSN", "").strip()
conninfo_parser = None

try:
    from psycopg import pq
    from psycopg.conninfo import conninfo_to_dict

    conninfo_parser = conninfo_to_dict

    from applypilot.apply import pgqueue
    from applypilot.fleet import machine_blackout

    if (
        fleet_dsn
        and applypilot_fleet_dsn
        and _normalized_dsn(fleet_dsn, pq=pq, conninfo_to_dict=conninfo_to_dict)
        != _normalized_dsn(
            applypilot_fleet_dsn,
            pq=pq,
            conninfo_to_dict=conninfo_to_dict,
        )
    ):
        raise RuntimeError("Inconsistent fleet Postgres DSN references")
    dsn = fleet_dsn
    if not dsn:
        raise RuntimeError("No fleet Postgres DSN: set FLEET_PG_DSN")
    conn = pgqueue.connect(dsn)
    conn.read_only = True
    print(
        _sanitize_status_line(
            machine_blackout.status_line(conn, label, role=role),
            fleet_dsn,
            applypilot_fleet_dsn,
            conninfo_to_dict=conninfo_parser,
        )
    )
except Exception as exc:
    diagnostic = _sanitize_text(
        f"{type(exc).__name__}: {exc}",
        os.environ.get("FLEET_PG_DSN", ""),
        os.environ.get("APPLYPILOT_FLEET_DSN", ""),
        conninfo_to_dict=conninfo_parser,
    )
    safe_label = _safe_field(label).lower()
    safe_role = _safe_field(role).lower()
    print(f"fleet-blackout-query: {diagnostic}", file=sys.stderr)
    print(f"BLOCKED|{safe_label}|{safe_role}|blackout-query-error||{diagnostic}")
