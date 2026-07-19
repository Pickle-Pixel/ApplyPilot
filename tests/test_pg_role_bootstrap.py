"""Live disposable PostgreSQL coverage for one-time role bootstrap."""

from __future__ import annotations

import importlib.util
import inspect
from itertools import count
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from applypilot.apply import pgqueue
from applypilot.brain import schema as brain_schema
from applypilot.fleet import pg_roles
from conftest import require_disposable_postgres


RETIRED_ADMIN = "bootstrap_retired_admin_test"
OWNER = "bootstrap_database_owner_test"
CONTROLLER = "bootstrap_controller_test"
CONTROLLER_PASSWORD = "bootstrap-controller-password"
BACKUP_PASSWORD = "bootstrap-backup-password"
VERIFIER = "bootstrap_verifier_test"
MIGRATOR = "bootstrap_migrator_test"
WORKER_ROLE = "bootstrap_worker_test"
WORKER_ID = "bootstrap-worker-node"
WORKER_PASSWORD = "bootstrap-worker-password"
ATOMIC_OWNER = "atomic_bootstrap_owner_test"
ATOMIC_CONTROLLER = "atomic_bootstrap_controller_test"
ATOMIC_RETIRED_ADMIN = "atomic_bootstrap_retired_admin_test"
ATOMIC_LIFECYCLE_ROLES = ("brain_status_reader", "brain_policy_controller")
CROSS_DATABASE = "applypilot_stale_default_test"
ACTIVE_DATABASE = "applypilot_active_session_test"
ACTIVE_DATABASE_ROLE = "applypilot_active_other_service_test"
RACE_DATABASE_ROLE = "applypilot_fence_race_test"
_EVIDENCE_TEMP = tempfile.TemporaryDirectory(prefix="applypilot-bootstrap-evidence-")
_EVIDENCE_KEY = b"applypilot-test-rollback-hmac-key-v1"
_EVIDENCE_KEY_ID = "pytest-v1"
_EVIDENCE_COUNTER = count()


def _test_evidence_paths() -> pg_roles.DurableEvidencePaths:
    sequence = next(_EVIDENCE_COUNTER)
    root = Path(_EVIDENCE_TEMP.name).resolve()
    return pg_roles.DurableEvidencePaths(
        preparation_receipt_path=root / f"prepared-{sequence}.json",
        rollback_sql_path=root / f"rollback-{sequence}.sql",
        authentication_key=_EVIDENCE_KEY,
        authentication_key_id=_EVIDENCE_KEY_ID,
    )


def _bootstrap_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap-fleet-pg-roles.py"
    spec = importlib.util.spec_from_file_location("applypilot_bootstrap_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rollback_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rollback-fleet-pg-role.py"
    spec = importlib.util.spec_from_file_location("applypilot_rollback_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("option", "value"),
    (("--verifier-role", "custom_verifier"), ("--migrator-role", "custom_migrator")),
)
def test_atomic_bootstrap_cli_rejects_schema_incompatible_role_names(tmp_path: Path, option: str, value: str) -> None:
    module = _bootstrap_script_module()
    arguments = [
        "--database-owner-role",
        "owner",
        "--controller-role",
        "controller",
        "--verifier-role",
        "brain_schema_verifier",
        "--migrator-role",
        "brain_schema_migrator",
        "--retired-admin-role",
        "retired",
        "--receipt-path",
        str(tmp_path / "receipt.json"),
        "--rollback-sql",
        str(tmp_path / "rollback.sql"),
    ]
    arguments[arguments.index(option) + 1] = value
    with pytest.raises(SystemExit):
        module._parser().parse_args(arguments)


def test_bootstrap_api_has_no_caller_supplied_sql_callback() -> None:
    bootstrap_parameters = inspect.signature(pg_roles.bootstrap_database_roles).parameters
    worker_parameters = inspect.signature(pg_roles.ensure_fleet_worker_role).parameters
    assert "post_bootstrap_callback" not in bootstrap_parameters
    assert "evidence_writer" not in bootstrap_parameters
    assert "evidence_writer" not in worker_parameters
    assert "install_brain_authority" in bootstrap_parameters


def test_evidence_api_rejects_hostile_callable_without_invoking_it(monkeypatch) -> None:
    invoked = False
    captured_arguments: list[object] = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class CapturableConnection:
        autocommit = False
        commit_calls = 0
        info = type("Info", (), {"transaction_status": type("Status", (), {"name": "IDLE"})()})()

        def commit(self):
            self.commit_calls += 1

        def cursor(self):
            return FakeCursor()

    class HostileCallable:
        def __call__(self, *args, **_kwargs):
            nonlocal invoked
            invoked = True
            captured_arguments.extend(args)
            captured_arguments.extend(_kwargs.values())

    hostile = HostileCallable()
    connection = CapturableConnection()
    monkeypatch.setattr(pg_roles, "_require_pg18_authority_catalog", lambda _cursor: None)
    with pytest.raises(TypeError, match="DurableEvidencePaths"):
        pg_roles.bootstrap_database_roles(
            connection,
            "password",
            "backup-password",
            topology=pg_roles.BootstrapTopology(
                database_owner_role="owner",
                controller_role="controller",
                verifier_role="verifier",
                migrator_role="migrator",
                retired_admin_roles=("retired",),
            ),
            evidence_paths=hostile,
        )
    with pytest.raises(TypeError, match="DurableEvidencePaths"):
        pg_roles.ensure_fleet_worker_role(
            connection,
            "password",
            regrant_manifest=pg_roles.RegrantManifest(
                database_owner_role="owner",
                controller_roles=("controller",),
                verifier_roles=("verifier",),
                retired_admin_roles=("retired",),
            ),
            evidence_paths=hostile,
        )
    assert invoked is False
    assert captured_arguments == []
    assert connection.commit_calls == 0


def test_bootstrap_crash_after_commit_leaves_precommit_receipt_in_doubt(tmp_path: Path, monkeypatch) -> None:
    module = _bootstrap_script_module()
    receipt_path = tmp_path / "receipt.json"
    rollback_path = tmp_path / "rollback.sql"
    committed = False
    authority_requested = False

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_bootstrap(_conn, _password, _backup_password, *, topology, evidence_paths, install_brain_authority):
        nonlocal authority_requested, committed
        inventory = {
            "prepared_at": "2001-02-03T04:05:06+00:00",
            "database_name": "postgres",
            "atomic_bootstrap": True,
            "automatic_rollback_supported": False,
            "commit_outcome_on_interruption": "unknown",
            "legacy_rollback_sql_recovers_v1_v4": False,
            "legacy_rollback_sql_recovers_v1_v5": False,
            "rollback_mode": "forward_v7_containment",
            "infrastructure_superuser_roles": topology.infrastructure_superuser_roles,
        }
        rollback_sql = "SELECT 1;\n"
        pg_roles._write_preparation_evidence(
            evidence_paths,
            inventory=inventory,
            rollback_sql=rollback_sql,
        )
        authority_requested = install_brain_authority
        committed = True
        return pg_roles.BootstrapReceipt(
            database_name="postgres",
            session_user="postgres",
            topology={"infrastructure_superuser_roles": ("postgres",)},
            inventory=inventory,
            effective_connect_grantees=(),
            rollback_sql=rollback_sql,
            escalation_required=False,
            bootstrapped_at="2001-02-03T04:05:06+00:00",
        )

    monkeypatch.setattr(module.psycopg, "connect", lambda *_args, **_kwargs: FakeConnection())
    monkeypatch.setattr(module, "bootstrap_database_roles", fake_bootstrap)
    monkeypatch.setattr(module, "_replace_durable", lambda *_args: (_ for _ in ()).throw(OSError("crash")))
    monkeypatch.setenv("APPLYPILOT_ADMIN_PG_DSN", "secret-admin-dsn")
    monkeypatch.setenv("APPLYPILOT_CONTROLLER_PG_PASSWORD", "secret-controller-password")
    monkeypatch.setenv("APPLYPILOT_BACKUP_PG_PASSWORD", "secret-backup-password")
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_HEX", _EVIDENCE_KEY.hex())
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_ID", _EVIDENCE_KEY_ID)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bootstrap-fleet-pg-roles.py",
            "--database-owner-role",
            "owner",
            "--controller-role",
            "controller",
            "--verifier-role",
            "brain_schema_verifier",
            "--migrator-role",
            "brain_schema_migrator",
            "--retired-admin-role",
            "legacy",
            "--infrastructure-superuser-role",
            "postgres",
            "--receipt-path",
            str(receipt_path),
            "--rollback-sql",
            str(rollback_path),
        ],
    )
    with pytest.raises(OSError, match="crash"):
        module.main()
    assert authority_requested is True
    assert committed is True
    prepared = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert prepared["atomic_bootstrap"] is True
    assert prepared["automatic_rollback_supported"] is False
    assert prepared["commit_outcome_on_interruption"] == "unknown"
    assert prepared["legacy_rollback_sql_recovers_v1_v4"] is False
    assert prepared["legacy_rollback_sql_recovers_v1_v5"] is False
    assert prepared["status"] == "prepared_before_database_mutation"
    assert prepared["escalation_required"] is True
    assert prepared["in_doubt"] is True
    assert "secret" not in receipt_path.read_text(encoding="utf-8")


def _signed_rollback_receipt(rollback_path: Path, rollback_bytes: bytes, **extra) -> dict:
    return pg_roles.authenticate_evidence_receipt(
        {
            "rollback_mode": "topology_exact",
            "rollback_sql_path": str(rollback_path.resolve()),
            "rollback_sql_sha256": hashlib.sha256(rollback_bytes).hexdigest(),
            **extra,
        },
        authentication_key=_EVIDENCE_KEY,
        authentication_key_id=_EVIDENCE_KEY_ID,
    )


def test_rollback_refuses_unsigned_receipt(tmp_path: Path) -> None:
    module = _rollback_script_module()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text('{"atomic_bootstrap":true}', encoding="utf-8")

    with pytest.raises(SystemExit, match="unsigned"):
        module._verified_inputs(
            receipt_path,
            tmp_path / "rollback.sql",
            authentication_key=_EVIDENCE_KEY,
            expected_key_id=_EVIDENCE_KEY_ID,
        )


def test_rollback_receipt_binds_canonical_sql_path(tmp_path: Path) -> None:
    module = _rollback_script_module()
    rollback_path = tmp_path / "rollback.sql"
    alternate_path = tmp_path / "alternate.sql"
    rollback_bytes = b"SELECT 1;\n"
    rollback_path.write_bytes(rollback_bytes)
    alternate_path.write_bytes(rollback_bytes)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(_signed_rollback_receipt(rollback_path, rollback_bytes)),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="path does not match"):
        module._verified_inputs(
            receipt_path,
            alternate_path,
            authentication_key=_EVIDENCE_KEY,
            expected_key_id=_EVIDENCE_KEY_ID,
        )


def test_hba_restore_is_bound_to_show_path_digest_owner_and_mode(tmp_path: Path) -> None:
    module = _rollback_script_module()
    live_path = (tmp_path / "pg_hba.conf").resolve()
    backup_path = (tmp_path / "pg_hba.conf.applypilot-backup").resolve()
    live_bytes = b"host all all 127.0.0.1/32 reject\n"
    backup_bytes = b"local all all trust\n"
    live_path.write_bytes(live_bytes)
    backup_path.write_bytes(backup_bytes)
    os.chmod(backup_path, 0o600)
    live_stat = os.stat(live_path)
    backup_stat = os.stat(backup_path)

    class FakeResult:
        def fetchone(self):
            return {"hba_file": str(live_path)}

    class FakeConnection:
        def execute(self, statement):
            assert statement == "SHOW hba_file"
            return FakeResult()

    receipt = {
        "hba_restore": {
            "format": "applypilot-hba-restore-v2",
            "live_hba_path": str(live_path),
            "backup_path": str(backup_path),
            "expected_target_sha256": hashlib.sha256(live_bytes).hexdigest(),
            "backup_sha256": hashlib.sha256(backup_bytes).hexdigest(),
            "backup_size": len(backup_bytes),
            "backup_mode": backup_stat.st_mode & 0o777,
            "owner": {"uid": live_stat.st_uid, "gid": live_stat.st_gid},
        }
    }

    validated = module._validated_hba_restore(FakeConnection(), receipt)
    assert validated[0] == live_path
    assert validated[1:3] == (live_bytes, backup_bytes)

    receipt["hba_restore"]["backup_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="backup_sha256"):
        module._validated_hba_restore(FakeConnection(), receipt)


def test_database_rollback_commits_before_hba_restore(tmp_path: Path, monkeypatch) -> None:
    module = _rollback_script_module()
    events: list[str] = []

    class FakeResult:
        def fetchone(self):
            return {
                "session_user": "postgres",
                "current_user": "postgres",
                "rolsuper": True,
                "database_name": "postgres",
                "system_identifier": "123456789",
                "database_oid": 12345,
            }

    class FakeTransaction:
        def __enter__(self):
            events.append("database_transaction_enter")

        def __exit__(self, exc_type, *_args):
            if exc_type is None:
                events.append("database_transaction_commit")
            return False

    class FakeConnection:
        info = type("Info", (), {"transaction_status": type("Status", (), {"name": "IDLE"})()})()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement):
            if statement == "ROLLBACK SQL":
                events.append("database_rollback_sql")
            return FakeResult()

        def commit(self):
            events.append("identity_commit")

        def transaction(self):
            return FakeTransaction()

    fake_receipt = {"topology": {"infrastructure_superuser_roles": ["postgres"]}}
    (tmp_path / "receipt.json").write_text(json.dumps(fake_receipt), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_verified_inputs",
        lambda *_args, **_kwargs: (fake_receipt, "ROLLBACK SQL"),
    )
    monkeypatch.setattr(module.psycopg, "connect", lambda *_args, **_kwargs: FakeConnection())
    monkeypatch.setattr(module, "_pin_rollback_search_path", lambda _conn: None)

    def commit_containment(conn, rollback_sql, *, marker, **_kwargs):
        with conn.transaction():
            conn.execute(rollback_sql)
        return {
            **marker,
            "affected_role_names": [],
            "relation": "public.applypilot_rollback_commit_markers",
            "visible": True,
        }

    monkeypatch.setattr(module, "_commit_database_containment", commit_containment)
    monkeypatch.setattr(module, "_terminate_affected_sessions", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        module,
        "_assert_candidate_artifact_containment",
        lambda *_args, **_kwargs: {"candidate_execute_routes": 0, "artifact_execute_routes": 0, "set_role_routes": 0},
    )
    monkeypatch.setattr(
        module,
        "_validated_hba_restore",
        lambda *_args: events.append("hba_preflight") or ("validated",),
    )
    monkeypatch.setattr(
        module,
        "_restore_hba_and_reload",
        lambda *_args, **_kwargs: events.append("hba_restore"),
    )
    monkeypatch.setattr(module, "_hba_restore_supported", lambda: True)
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_HEX", _EVIDENCE_KEY.hex())
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_ID", _EVIDENCE_KEY_ID)
    monkeypatch.setenv("APPLYPILOT_ADMIN_PG_DSN", "postgresql://trusted-break-glass")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollback-fleet-pg-role.py",
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--rollback-sql",
            str(tmp_path / "rollback.sql"),
            "--restore-hba",
        ],
    )

    assert module.main() == 0
    assert events.index("hba_preflight") < events.index("database_transaction_enter")
    assert events.index("database_rollback_sql") < events.index("database_transaction_commit")
    assert events.index("database_transaction_commit") < events.index("hba_restore")


def _authenticated_forward_v7_receipt(tmp_path: Path) -> tuple[dict, str]:
    rollback_sql = "ROLLBACK SQL"
    rollback_path = (tmp_path / "rollback.sql").resolve()
    return (
        pg_roles.authenticate_evidence_receipt(
            {
                "status": "atomic_existing_topology_upgrade_committed",
                "rollback_mode": "forward_v7_containment",
                "database_name": "postgres",
                "inventory": {"system_identifier": "123456789", "database_oid": 12345},
                "topology": {"infrastructure_superuser_roles": ["postgres"]},
                "rollback_sql_path": str(rollback_path),
                "rollback_sql_sha256": hashlib.sha256(rollback_sql.encode()).hexdigest(),
                "hba_restore": {
                    "format": "applypilot-hba-restore-v2",
                    "live_hba_path": "/protected/pg_hba.conf",
                    "backup_path": "/protected/pg_hba.conf.applypilot-backup",
                    "expected_target_sha256": "a" * 64,
                    "backup_sha256": "b" * 64,
                    "backup_size": 17,
                    "backup_mode": 0o600,
                    "owner": {"uid": 100, "gid": 101},
                },
            },
            authentication_key=_EVIDENCE_KEY,
            authentication_key_id=_EVIDENCE_KEY_ID,
        ),
        rollback_sql,
    )


class _RollbackMainFakeConnection:
    info = type("Info", (), {"transaction_status": type("Status", (), {"name": "IDLE"})()})()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, *_args):
        assert "session_user" in statement
        return type(
            "Result",
            (),
            {
                "fetchone": lambda _self: {
                    "session_user": "postgres",
                    "current_user": "postgres",
                    "rolsuper": True,
                    "database_name": "postgres",
                    "system_identifier": "123456789",
                    "database_oid": 12345,
                }
            },
        )()

    def commit(self):
        return None

    def rollback(self):
        return None


def _configure_rollback_main(
    module,
    tmp_path: Path,
    monkeypatch,
    *,
    partial_state_path: Path,
) -> tuple[dict, str]:
    receipt, rollback_sql = _authenticated_forward_v7_receipt(tmp_path)
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(module, "_verified_inputs", lambda *_args, **_kwargs: (receipt, rollback_sql))
    monkeypatch.setattr(module, "_pin_rollback_search_path", lambda _conn: None)
    monkeypatch.setattr(
        module.psycopg,
        "connect",
        lambda *_args, **_kwargs: _RollbackMainFakeConnection(),
    )
    monkeypatch.setattr(module, "_hba_restore_supported", lambda: True)
    monkeypatch.setattr(module, "_validated_hba_restore", lambda *_args: ("validated",))
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_HEX", _EVIDENCE_KEY.hex())
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_ID", _EVIDENCE_KEY_ID)
    monkeypatch.setenv("APPLYPILOT_ADMIN_PG_DSN", "postgresql://trusted-break-glass")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollback-fleet-pg-role.py",
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--rollback-sql",
            str(tmp_path / "rollback.sql"),
            "--restore-hba",
            "--partial-state-receipt",
            str(partial_state_path),
        ],
    )
    return receipt, rollback_sql


@pytest.mark.parametrize(
    "fault_seam",
    ("sql_execution", "marker_assertion", "post_containment_assertion"),
)
def test_rollback_fault_before_hba_never_starts_hba(
    tmp_path: Path, monkeypatch, fault_seam: str
) -> None:
    module = _rollback_script_module()
    partial = (tmp_path / "partial.json").resolve()
    _configure_rollback_main(module, tmp_path, monkeypatch, partial_state_path=partial)
    events: list[str] = []
    monkeypatch.setattr(module, "_affected_authority_roles", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(module, "_terminate_affected_sessions", lambda *_args, **_kwargs: 0)

    monkeypatch.setattr(
        module,
        "_execute_database_containment_sql",
        lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(RuntimeError("injected SQL execution failure"))
            if fault_seam == "sql_execution"
            else events.append("sql_executed")
        ),
    )
    monkeypatch.setattr(module, "_write_database_containment_marker", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_commit_database_containment_transaction",
        lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(RuntimeError("injected transaction commit failure"))
            if fault_seam == "transaction_commit"
            else events.append("transaction_committed")
        ),
    )
    def assert_marker(_conn, *, marker, expected_phase=None, **_kwargs):
        if fault_seam == "marker_assertion":
            raise RuntimeError("injected marker assertion failure")
        completed_phases = ["phase1"]
        if expected_phase == "phase2":
            completed_phases.append("phase2")
        return {
            **marker,
            "completed_phases": completed_phases,
            "relation": "public.applypilot_rollback_commit_markers",
            "visible": True,
        }

    monkeypatch.setattr(module, "_assert_database_containment_marker", assert_marker)
    monkeypatch.setattr(
        module,
        "_assert_candidate_artifact_containment",
        lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(RuntimeError("injected post-containment assertion failure"))
            if fault_seam == "post_containment_assertion"
            else {"candidate_execute_routes": 0, "artifact_execute_routes": 0, "set_role_routes": 0}
        ),
    )
    monkeypatch.setattr(
        module,
        "_restore_hba_and_reload",
        lambda *_args, **_kwargs: events.append("hba_restore_started"),
    )

    expected_message = (
        "authenticated checkpoint" if fault_seam == "post_containment_assertion" else "injected"
    )
    with pytest.raises(RuntimeError, match=expected_message):
        module.main()
    assert "hba_restore_started" not in events
    assert partial.exists() is (fault_seam == "post_containment_assertion")


def test_precommit_failure_preserves_operation_and_rollback_errors(monkeypatch) -> None:
    module = _rollback_script_module()

    class RollbackFailureConnection:
        def rollback(self):
            raise OSError("rollback transport failed")

    monkeypatch.setattr(
        module,
        "_execute_database_containment_sql",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("containment failed")),
    )
    with pytest.raises(RuntimeError, match="rollback reporting also failed") as failure:
        module._commit_database_containment(
            RollbackFailureConnection(),
            "ROLLBACK SQL",
            marker={},
        )
    assert isinstance(failure.value.__cause__, BaseExceptionGroup)
    assert [type(error) for error in failure.value.__cause__.exceptions] == [ValueError, OSError]


def test_session_termination_uses_one_global_deadline(monkeypatch) -> None:
    module = _rollback_script_module()
    clock = iter((100.0, 105.1))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))

    class Result:
        def __init__(self, row=None, rows=None):
            self.row = row
            self.rows = rows

        def fetchone(self):
            return self.row

        def fetchall(self):
            return self.rows

    class BusySessionsConnection:
        commits = 0

        def execute(self, statement, *_args):
            if "AS signaled" in statement:
                return Result(row={"signaled": 100})
            return Result(rows=[{"pid": 4242, "usename": "provider"}])

        def commit(self):
            self.commits += 1

    conn = BusySessionsConnection()
    with pytest.raises(RuntimeError, match="survived termination"):
        module._terminate_affected_sessions(
            conn,
            affected_role_names=["provider"],
            affected_role_oids=[4242],
            protected_roles=set(),
        )
    assert conn.commits == 2


def test_hba_failure_emits_authenticated_target_bound_partial_state_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    module = _rollback_script_module()
    partial = (tmp_path / "partial.json").resolve()
    receipt, rollback_sql = _configure_rollback_main(
        module, tmp_path, monkeypatch, partial_state_path=partial
    )
    marker = {
        "relation": "public.applypilot_rollback_commit_markers",
        "marker_id": "marker-1",
        "affected_role_names": [],
        "affected_role_oids": [],
        "visible": True,
        "committed_at": "2026-07-19T14:00:00+00:00",
    }
    monkeypatch.setattr(
        module,
        "_commit_with_authenticated_reconciliation",
        lambda *_args, **_kwargs: marker,
    )
    monkeypatch.setattr(
        module,
        "_assert_candidate_artifact_containment",
        lambda *_args, **_kwargs: {"candidate_execute_routes": 0, "artifact_execute_routes": 0, "set_role_routes": 0},
    )
    monkeypatch.setattr(
        module,
        "_observe_hba_state",
        lambda *_args, **_kwargs: {
            "live_hba_path": "/protected/pg_hba.conf",
            "sha256": "c" * 64,
            "size": 23,
            "mode": 0o600,
            "owner": {"uid": 100, "gid": 101},
        },
    )
    monkeypatch.setattr(
        module,
        "_restore_hba_and_reload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected HBA reload failure")),
    )

    with pytest.raises(RuntimeError, match="partial-state receipt"):
        module.main()

    partial_receipt = json.loads(partial.read_text(encoding="utf-8"))
    pg_roles.verify_evidence_receipt(
        partial_receipt,
        authentication_key=_EVIDENCE_KEY,
        expected_key_id=_EVIDENCE_KEY_ID,
    )
    assert partial_receipt["status"] == "database_containment_committed_hba_restore_failed"
    assert partial_receipt["target_identity"] == {
        "database_name": "postgres",
        "system_identifier": "123456789",
        "database_oid": 12345,
        "session_user": "postgres",
    }
    assert partial_receipt["database_containment"] == marker
    assert partial_receipt["rollback_sql"] == {
        "path": receipt["rollback_sql_path"],
        "sha256": hashlib.sha256(rollback_sql.encode()).hexdigest(),
    }
    assert partial_receipt["hba"]["expected"] == receipt["hba_restore"]
    assert partial_receipt["hba"]["observed"]["sha256"] == "c" * 64
    assert partial_receipt["failure"] == {
        "type": "RuntimeError",
        "message": "injected HBA reload failure",
    }
    assert partial_receipt["evidence"]["source_receipt_authentication"] == receipt[
        "receipt_authentication"
    ]
    assert partial_receipt["evidence"]["predecessor_partial_state_receipt"] is None


@pytest.mark.parametrize(
    ("fault_seam", "attempted_phase", "expected_completed_phases"),
    (
        ("session_termination", "session_termination", ["phase1"]),
        ("phase2", "phase2", ["phase1"]),
        ("postcondition", "postcondition", ["phase1", "phase2"]),
    ),
)
def test_post_phase_one_ordinary_failure_emits_authenticated_phase_checkpoint(
    tmp_path: Path,
    monkeypatch,
    fault_seam: str,
    attempted_phase: str,
    expected_completed_phases: list[str],
) -> None:
    module = _rollback_script_module()
    partial = (tmp_path / "phase-partial.json").resolve()
    _configure_rollback_main(module, tmp_path, monkeypatch, partial_state_path=partial)

    def commit_phase(*_args, marker, phase, **_kwargs):
        if phase == "phase2" and fault_seam == "phase2":
            raise RuntimeError("injected ordinary phase2 failure")
        completed = ["phase1"] if phase == "phase1" else ["phase1", "phase2"]
        return {
            **marker,
            "affected_role_names": ["provider"],
            "affected_role_oids": [4242],
            "completed_phases": completed,
            "relation": "public.applypilot_rollback_commit_markers",
            "visible": True,
            "committed_at": "2026-07-19T14:00:00+00:00",
        }

    monkeypatch.setattr(module, "_commit_with_authenticated_reconciliation", commit_phase)
    monkeypatch.setattr(
        module,
        "_terminate_affected_sessions",
        lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(RuntimeError("injected termination failure"))
            if fault_seam == "session_termination"
            else 1
        ),
    )
    monkeypatch.setattr(
        module,
        "_assert_candidate_artifact_containment",
        lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(RuntimeError("injected postcondition failure"))
            if fault_seam == "postcondition"
            else {"candidate_execute_routes": 0}
        ),
    )
    monkeypatch.setattr(
        module,
        "_restore_hba_and_reload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("HBA must not start after a database-phase failure")
        ),
    )

    with pytest.raises(RuntimeError, match="authenticated checkpoint"):
        module.main()

    checkpoint = json.loads(partial.read_text(encoding="utf-8"))
    pg_roles.verify_evidence_receipt(
        checkpoint,
        authentication_key=_EVIDENCE_KEY,
        expected_key_id=_EVIDENCE_KEY_ID,
    )
    assert checkpoint["status"] == "database_containment_phase_failed"
    assert checkpoint["attempted_phase"] == attempted_phase
    assert checkpoint["database_containment"]["completed_phases"] == expected_completed_phases
    assert checkpoint["database_containment"]["visible"] is True
    assert fault_seam.split("_")[-1] in checkpoint["failure"]["message"]


def test_phase_two_commit_unknown_checkpoint_preserves_known_phase_one_marker(
    tmp_path: Path, monkeypatch
) -> None:
    module = _rollback_script_module()
    partial = (tmp_path / "commit-unknown.json").resolve()
    _configure_rollback_main(module, tmp_path, monkeypatch, partial_state_path=partial)

    def commit_phase(*_args, marker, phase, **_kwargs):
        if phase == "phase2":
            raise module.ContainmentCommitUnreconciled(
                "injected ambiguous phase2 commit",
                evidence_errors=BaseExceptionGroup(
                    "ambiguous phase2", [ConnectionError("commit acknowledgement lost")]
                ),
            )
        return {
            **marker,
            "affected_role_names": ["provider"],
            "affected_role_oids": [4242],
            "completed_phases": ["phase1"],
            "relation": "public.applypilot_rollback_commit_markers",
            "visible": True,
            "committed_at": "2026-07-19T14:00:00+00:00",
        }

    monkeypatch.setattr(module, "_commit_with_authenticated_reconciliation", commit_phase)
    monkeypatch.setattr(module, "_terminate_affected_sessions", lambda *_args, **_kwargs: 1)

    with pytest.raises(RuntimeError, match="COMMIT outcome is unknown"):
        module.main()

    checkpoint = json.loads(partial.read_text(encoding="utf-8"))
    pg_roles.verify_evidence_receipt(
        checkpoint,
        authentication_key=_EVIDENCE_KEY,
        expected_key_id=_EVIDENCE_KEY_ID,
    )
    assert checkpoint["attempted_phase"] == "phase2"
    assert checkpoint["database_containment"]["completed_phases"] == ["phase1"]
    assert checkpoint["database_containment"]["commit_outcome"] == "unknown"


def test_phase_two_commit_evidence_failure_is_not_downgraded_to_ordinary_failure(
    tmp_path: Path, monkeypatch
) -> None:
    module = _rollback_script_module()
    partial = (tmp_path / "commit-evidence-failed.json").resolve()
    _configure_rollback_main(module, tmp_path, monkeypatch, partial_state_path=partial)

    def commit_phase(*_args, marker, phase, **_kwargs):
        if phase == "phase2":
            raise module.ContainmentCommitUnreconciled(
                "injected ambiguous phase2 commit",
                evidence_errors=BaseExceptionGroup(
                    "ambiguous phase2", [ConnectionError("commit acknowledgement lost")]
                ),
            )
        return {
            **marker,
            "affected_role_names": [],
            "affected_role_oids": [],
            "completed_phases": ["phase1"],
            "visible": True,
        }

    monkeypatch.setattr(module, "_commit_with_authenticated_reconciliation", commit_phase)
    monkeypatch.setattr(module, "_terminate_affected_sessions", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        module,
        "_emit_commit_unknown_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("checkpoint storage failed")),
    )
    monkeypatch.setattr(
        module,
        "_emit_database_phase_failure_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous commit must not be relabeled as an ordinary failure")
        ),
    )

    with pytest.raises(module.AuthenticatedCheckpointFailure) as failure:
        module.main()
    assert "checkpoint creation also failed" in str(failure.value)
    assert isinstance(failure.value.__cause__, BaseExceptionGroup)


def test_hba_and_partial_receipt_failure_preserves_both_errors_and_machine_state(
    tmp_path: Path, monkeypatch
) -> None:
    module = _rollback_script_module()
    partial = (tmp_path / "partial.json").resolve()
    _configure_rollback_main(module, tmp_path, monkeypatch, partial_state_path=partial)
    marker = {
        "marker_id": "marker-receipt-failure",
        "relation": "public.applypilot_rollback_commit_markers",
        "visible": True,
    }
    monkeypatch.setattr(
        module,
        "_commit_with_authenticated_reconciliation",
        lambda *_args, **_kwargs: {
            **marker,
            "affected_role_names": [],
            "affected_role_oids": [],
        },
    )
    monkeypatch.setattr(
        module,
        "_assert_candidate_artifact_containment",
        lambda *_args, **_kwargs: {"candidate_execute_routes": 0, "artifact_execute_routes": 0, "set_role_routes": 0},
    )
    monkeypatch.setattr(module, "_observe_hba_state", lambda *_args, **_kwargs: {"sha256": "e" * 64})
    monkeypatch.setattr(
        module,
        "_restore_hba_and_reload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("original HBA failure")),
    )
    monkeypatch.setattr(
        module,
        "_emit_partial_state_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("durable receipt write failure")),
    )

    with pytest.raises(RuntimeError) as failure:
        module.main()
    message = str(failure.value)
    assert "database_containment_committed=true" in message
    assert "hba_state=uncertain_not_restored" in message
    assert "partial_receipt_created=false" in message
    assert "recovery=manual_containment_required" in message
    assert isinstance(failure.value.__cause__, ExceptionGroup)
    assert [str(error) for error in failure.value.__cause__.exceptions] == [
        "original HBA failure",
        "durable receipt write failure",
    ]


def test_forward_v7_target_rejects_same_name_recreated_database_oid() -> None:
    module = _rollback_script_module()
    receipt, _rollback_sql = _authenticated_forward_v7_receipt(Path.cwd())

    class FakeConnection:
        def execute(self, _statement):
            return type(
                "Result",
                (),
                {
                    "fetchone": lambda _self: {
                        "session_user": "postgres",
                        "current_user": "postgres",
                        "rolsuper": True,
                        "database_name": "postgres",
                        "system_identifier": "123456789",
                        "database_oid": 54321,
                    }
                },
            )()

    with pytest.raises(SystemExit, match="database OID"):
        module._validated_target_identity(FakeConnection(), receipt)


def test_target_identity_rejects_hostile_public_catalog_oid_shadow(fleet_db: str) -> None:
    module = _rollback_script_module()
    with pgqueue.connect(fleet_db) as root:
        require_disposable_postgres(root)
        identity = root.execute(
            "SELECT pg_catalog.current_database() AS database_name,"
            "(pg_catalog.pg_control_system()).system_identifier::text AS system_identifier,"
            "(SELECT oid FROM pg_catalog.pg_database "
            "WHERE datname=pg_catalog.current_database()) AS database_oid,session_user"
        ).fetchone()
        try:
            root.execute("CREATE TABLE public.pg_roles(rolname name,rolsuper boolean)")
            root.execute("CREATE TABLE public.pg_database(oid oid,datname name)")
            root.execute(
                "INSERT INTO public.pg_roles VALUES(%s,TRUE)",
                (identity["session_user"],),
            )
            root.execute(
                "INSERT INTO public.pg_database VALUES(424242,%s)",
                (identity["database_name"],),
            )
            root.execute("SET search_path=public,pg_catalog")
            receipt = {
                "rollback_mode": "forward_v7_containment",
                "database_name": identity["database_name"],
                "inventory": {
                    "system_identifier": identity["system_identifier"],
                    "database_oid": 424242,
                },
                "topology": {
                    "infrastructure_superuser_roles": [identity["session_user"]]
                },
            }

            with pytest.raises(SystemExit, match="database OID"):
                module._validated_target_identity(root, receipt)
        finally:
            root.rollback()
            root.execute("DROP TABLE IF EXISTS public.pg_database")
            root.execute("DROP TABLE IF EXISTS public.pg_roles")
            root.commit()


def test_containment_rejects_stale_breakglass_name_takeover(fleet_db: str) -> None:
    victim = "rollback_stale_breakglass_test"
    controller = "rollback_incarnation_controller_test"
    provider = "rollback_incarnation_provider_test"
    probe = "brain_incarnation_owner_probe"
    with pgqueue.connect(fleet_db) as root:
        require_disposable_postgres(root)
        if int(root.execute("SHOW server_version_num").fetchone()["server_version_num"]) // 10000 != 18:
            pytest.skip("protected-incarnation rollback coverage requires PostgreSQL 18")
        database_name = conninfo_to_dict(root.info.dsn)["dbname"]
        try:
            for role_name in (victim, controller, provider):
                root.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name)))
            root.execute(
                sql.SQL("CREATE ROLE {} LOGIN SUPERUSER PASSWORD {}").format(
                    sql.Identifier(victim), sql.Literal("stale-breakglass-password")
                )
            )
            root.execute(
                sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER PASSWORD {}").format(
                    sql.Identifier(controller), sql.Literal("incarnation-controller-password")
                )
            )
            root.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(provider)))
            root.execute(sql.SQL("CREATE TABLE public.{}(id bigint)").format(sql.Identifier(probe)))
            root.execute(
                sql.SQL("ALTER TABLE public.{} OWNER TO {}").format(
                    sql.Identifier(probe), sql.Identifier(provider)
                )
            )
            root.commit()
            containment_sql, assertion_sql = pg_roles._catalog_derived_authority_containment_sql(
                root.cursor(),
                database_name=database_name,
                infrastructure_superuser_roles=(victim, "postgres"),
                controller_role=controller,
                managed_role_names=(provider,),
                future_owner_role_names=(),
            )
            assert containment_sql.startswith(
                "SET LOCAL search_path=pg_catalog,pg_temp;\n"
            )
            root.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(victim)))
            root.execute(
                sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER PASSWORD {}").format(
                    sql.Identifier(victim), sql.Literal("taken-over-breakglass-password")
                )
            )
            root.execute("CREATE TABLE public.pg_roles(oid oid,rolname name,rolsuper boolean)")
            root.execute("SET search_path=public,pg_catalog")
            root.commit()

            with pytest.raises(psycopg.Error, match="protected role incarnation changed"):
                root.execute(containment_sql + "\n" + assertion_sql)
            root.rollback()
            assert root.execute(
                "SELECT owner.rolname AS owner_name FROM pg_catalog.pg_class relation "
                "JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace "
                "JOIN pg_catalog.pg_roles owner ON owner.oid=relation.relowner "
                "WHERE namespace.nspname='public' AND relation.relname=%s",
                (probe,),
            ).fetchone()["owner_name"] == provider
            assert root.execute(
                "SELECT owner.rolname AS owner_name FROM pg_catalog.pg_database database "
                "JOIN pg_catalog.pg_roles owner ON owner.oid=database.datdba "
                "WHERE database.datname=pg_catalog.current_database()"
            ).fetchone()["owner_name"] == "postgres"
        finally:
            root.rollback()
            root.execute(sql.SQL("ALTER DATABASE {} OWNER TO postgres").format(sql.Identifier(database_name)))
            root.execute(sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE").format(sql.Identifier(probe)))
            root.execute("DROP TABLE IF EXISTS public.pg_roles")
            for role_name in (provider, controller, victim):
                root.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role_name)))
                root.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name)))
            root.commit()


def test_rollback_session_pins_search_path_before_hostile_text_cast(fleet_db: str) -> None:
    module = _rollback_script_module()
    with pgqueue.connect(fleet_db) as root:
        require_disposable_postgres(root)
        identity = root.execute(
            "SELECT pg_catalog.current_database() AS database_name,"
            "(pg_catalog.pg_control_system()).system_identifier::pg_catalog.text "
            "AS system_identifier,"
            "(SELECT database.oid FROM pg_catalog.pg_database database "
            "WHERE database.datname=pg_catalog.current_database()) AS database_oid"
        ).fetchone()
        receipt = {
            "rollback_mode": "forward_v7_containment",
            "database_name": identity["database_name"],
            "inventory": {
                "system_identifier": identity["system_identifier"],
                "database_oid": identity["database_oid"],
            },
            "topology": {"infrastructure_superuser_roles": ["postgres"]},
        }
        try:
            root.execute(
                "CREATE FUNCTION public.reject_hostile_text(pg_catalog.text) "
                "RETURNS pg_catalog.bool LANGUAGE plpgsql AS $$ BEGIN "
                "RAISE EXCEPTION 'hostile text domain executed'; END $$"
            )
            root.execute(
                "CREATE DOMAIN public.text AS pg_catalog.text "
                "CHECK(public.reject_hostile_text(VALUE))"
            )
            root.commit()
            root.execute("SET search_path=public,pg_catalog")
            root.commit()
            with pytest.raises(psycopg.Error, match="hostile text domain executed"):
                root.execute("SELECT 'probe'::text")
            root.rollback()
            root.execute("SET search_path=public,pg_catalog")
            root.commit()

            module._pin_rollback_search_path(root)
            assert module._validated_target_identity(root, receipt)["database_oid"] == identity[
                "database_oid"
            ]
        finally:
            root.rollback()
            root.execute("SET search_path=pg_catalog")
            root.execute("DROP DOMAIN IF EXISTS public.text")
            root.execute("DROP FUNCTION IF EXISTS public.reject_hostile_text(pg_catalog.text)")
            root.commit()


def test_partial_state_write_rejects_parent_identity_drift(tmp_path: Path, monkeypatch) -> None:
    module = _rollback_script_module()
    destination = (tmp_path / "partial.json").resolve()
    expected = module._validate_partial_state_parent(destination)
    observed = (expected[0], expected[1], expected[2] + 1)
    monkeypatch.setattr(module, "_partial_state_parent_identity", lambda _parent: observed)

    with pytest.raises(RuntimeError, match="parent directory changed"):
        module._write_exclusive_durable(
            destination,
            b"{}\n",
            expected_parent_identity=expected,
        )
    assert not destination.exists()


def test_hba_replacement_rejects_same_inode_content_edit(tmp_path: Path, monkeypatch) -> None:
    module = _rollback_script_module()
    if not module._hba_restore_supported():
        pytest.skip("same-inode HBA replacement coverage requires Linux renameat2")
    if not hasattr(module.os, "fchmod"):
        monkeypatch.setattr(module.os, "fchmod", lambda *_args: None, raising=False)
    live = tmp_path / "pg_hba.conf"
    authenticated = b"host all all 127.0.0.1/32 scram-sha-256\n"
    edited = b"host all all 127.0.0.1/32 trust         \n"
    backup = b"local all all peer\n"
    live.write_bytes(authenticated)
    before = live.stat()
    live.write_bytes(edited)
    after = live.stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)

    with pytest.raises(RuntimeError, match="content changed"):
        module._replace_atomic(
            live,
            backup,
            expected_identity=(before.st_dev, before.st_ino),
            expected_content=authenticated,
            expected_sha256=hashlib.sha256(authenticated).hexdigest(),
            replacement_mode=stat.S_IMODE(after.st_mode),
            replacement_owner=(after.st_uid, after.st_gid),
        )
    assert live.read_bytes() == edited


def test_posix_hba_exchange_restores_same_inode_write_injected_at_mutation_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    module = _rollback_script_module()
    if not module._hba_restore_supported():
        pytest.skip("atomic HBA exchange regression requires Linux renameat2")
    live = tmp_path / "pg_hba.conf"
    authenticated = b"host all all 127.0.0.1/32 scram-sha-256\n"
    edited = b"host all all 127.0.0.1/32 trust         \n"
    replacement = b"local all all peer\n"
    live.write_bytes(authenticated)
    before = live.stat()
    real_exchange = module._rename_exchange
    injected = False

    def inject_then_exchange(parent_descriptor, first_name, second_name):
        nonlocal injected
        if not injected:
            injected = True
            live.write_bytes(edited)
            assert (live.stat().st_dev, live.stat().st_ino) == (
                before.st_dev,
                before.st_ino,
            )
        return real_exchange(parent_descriptor, first_name, second_name)

    monkeypatch.setattr(module, "_rename_exchange", inject_then_exchange)
    with pytest.raises(RuntimeError, match="immediately before atomic exchange"):
        module._replace_atomic(
            live,
            replacement,
            expected_identity=(before.st_dev, before.st_ino),
            expected_content=authenticated,
            expected_sha256=hashlib.sha256(authenticated).hexdigest(),
            replacement_mode=stat.S_IMODE(before.st_mode),
            replacement_owner=(before.st_uid, before.st_gid),
        )
    assert injected is True
    assert live.read_bytes() == edited


def test_hba_target_descriptor_stays_open_through_replace(tmp_path: Path, monkeypatch) -> None:
    module = _rollback_script_module()
    if module.os.name == "nt":
        pytest.skip("descriptor-held replacement is exercised on supported POSIX HBA hosts")
    if not hasattr(module.os, "fchmod"):
        monkeypatch.setattr(module.os, "fchmod", lambda *_args: None, raising=False)
    live = tmp_path / "pg_hba.conf"
    authenticated = b"host all all 127.0.0.1/32 scram-sha-256\n"
    replacement = b"local all all peer\n"
    live.write_bytes(authenticated)
    before = live.stat()
    real_open = module.os.open
    real_close = module.os.close
    real_exchange = module._rename_exchange
    target_descriptor = -1
    closed_descriptors: set[int] = set()

    def tracking_open(file, flags, *args, **kwargs):
        nonlocal target_descriptor
        descriptor = real_open(file, flags, *args, **kwargs)
        if Path(file).name == live.name and not flags & module.os.O_WRONLY:
            target_descriptor = descriptor
        return descriptor

    def tracking_close(descriptor):
        closed_descriptors.add(descriptor)
        return real_close(descriptor)

    def tracking_exchange(*args, **kwargs):
        assert target_descriptor >= 0
        assert target_descriptor not in closed_descriptors
        return real_exchange(*args, **kwargs)

    monkeypatch.setattr(module.os, "open", tracking_open)
    monkeypatch.setattr(module.os, "close", tracking_close)
    monkeypatch.setattr(module, "_rename_exchange", tracking_exchange)
    module._replace_atomic(
        live,
        replacement,
        expected_identity=(before.st_dev, before.st_ino),
        expected_content=authenticated,
        expected_sha256=hashlib.sha256(authenticated).hexdigest(),
        replacement_mode=stat.S_IMODE(before.st_mode),
        replacement_owner=(before.st_uid, before.st_gid),
    )
    assert live.read_bytes() == replacement


def test_rollback_retry_from_partial_state_reuses_idempotent_database_marker(
    tmp_path: Path, monkeypatch
) -> None:
    module = _rollback_script_module()
    partial = (tmp_path / "partial.json").resolve()
    receipt, rollback_sql = _authenticated_forward_v7_receipt(tmp_path)
    commits: list[str] = []
    restore_attempts = 0

    def commit(*_args, marker, **_kwargs):
        commits.append(marker["marker_id"])
        return {**marker, "relation": "public.applypilot_rollback_commit_markers", "visible": True}

    def restore(*_args, **_kwargs):
        nonlocal restore_attempts
        restore_attempts += 1
        if restore_attempts <= 2:
            raise RuntimeError(f"HBA attempt {restore_attempts} failed")

    monkeypatch.setattr(module, "_commit_with_authenticated_reconciliation", commit)
    monkeypatch.setattr(
        module,
        "_assert_candidate_artifact_containment",
        lambda *_args, **_kwargs: {"candidate_execute_routes": 0, "artifact_execute_routes": 0, "set_role_routes": 0},
    )
    monkeypatch.setattr(module, "_observe_hba_state", lambda *_args, **_kwargs: {"sha256": "d" * 64})
    monkeypatch.setattr(module, "_restore_hba_and_reload", restore)

    for attempt in range(3):
        _configure_rollback_main(module, tmp_path, monkeypatch, partial_state_path=partial)
        if attempt < 2:
            with pytest.raises(RuntimeError, match="partial-state receipt"):
                module.main()
        else:
            assert module.main() == 0
    assert len(commits) == 6
    assert len(set(commits)) == 1
    assert restore_attempts == 3
    assert partial.exists()
    successor = partial.with_name(f"{partial.name}.2")
    successor_receipt = json.loads(successor.read_text(encoding="utf-8"))
    pg_roles.verify_evidence_receipt(
        successor_receipt,
        authentication_key=_EVIDENCE_KEY,
        expected_key_id=_EVIDENCE_KEY_ID,
    )
    predecessor = successor_receipt["evidence"]["predecessor_partial_state_receipt"]
    assert predecessor["path"] == str(partial)
    assert predecessor["marker_id"] == commits[0]
    assert predecessor["target_identity"] == successor_receipt["target_identity"]


def test_reconciled_commit_checkpoint_is_authenticated_and_executor_independent(
    tmp_path: Path,
) -> None:
    module = _rollback_script_module()
    receipt, rollback_sql = _authenticated_forward_v7_receipt(tmp_path)
    receipt_path = (tmp_path / "receipt.json").resolve()
    rollback_path = (tmp_path / "rollback.sql").resolve()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    rollback_path.write_text(rollback_sql, encoding="utf-8")
    target = {
        "database_name": "postgres",
        "system_identifier": "123456789",
        "database_oid": 12345,
        "session_user": "postgres",
    }
    marker = {
        **module._containment_marker(receipt, rollback_sql, target),
        "affected_role_names": ["provider"],
        "affected_role_oids": [4242],
        "visible": True,
        "relation": "public.applypilot_rollback_commit_markers",
    }
    failure = module.ContainmentCommitReconciledConnectionLost(
        marker,
        evidence_errors=BaseExceptionGroup(
            "commit evidence",
            [ConnectionError("ack lost"), OSError("rollback report failed")],
        ),
    )
    destination = (tmp_path / "reconciled.json").resolve()
    written = module._emit_commit_reconciled_receipt(
        destination,
        source_receipt=receipt,
        source_receipt_evidence=module._source_receipt_evidence(receipt_path, receipt),
        rollback_sql_path=rollback_path,
        rollback_sql=rollback_sql,
        target_identity=target,
        failure=failure,
        authentication_key=_EVIDENCE_KEY,
        authentication_key_id=_EVIDENCE_KEY_ID,
        partial_parent_identity=module._validate_partial_state_parent(destination),
    )
    checkpoint = json.loads(written.read_text(encoding="utf-8"))
    pg_roles.verify_evidence_receipt(
        checkpoint,
        authentication_key=_EVIDENCE_KEY,
        expected_key_id=_EVIDENCE_KEY_ID,
    )
    assert checkpoint["status"] == "database_containment_committed_executor_connection_lost"
    predecessor = {
        "marker_id": marker["marker_id"],
        "target_identity": checkpoint["target_identity"],
        "source_receipt_authentication": receipt["receipt_authentication"],
    }
    module._require_matching_partial_state_predecessor(
        predecessor,
        marker_id=marker["marker_id"],
        target_identity={**target, "session_user": "alternate_breakglass"},
        source_receipt_authentication=receipt["receipt_authentication"],
    )
def test_hba_restore_observer_sees_real_rollback_commit(
    fleet_db: str, tmp_path: Path, monkeypatch
) -> None:
    module = _rollback_script_module()
    marker = "applypilot_real_rollback_commit_probe"
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cleanup:
        cleanup.execute(sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE").format(sql.Identifier(marker)))

    fake_receipt = {"topology": {"infrastructure_superuser_roles": ["postgres"]}}
    (tmp_path / "receipt.json").write_text(json.dumps(fake_receipt), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_verified_inputs",
        lambda *_args, **_kwargs: (
            fake_receipt,
            sql.SQL("CREATE TABLE IF NOT EXISTS public.{} (observed boolean NOT NULL)")
            .format(sql.Identifier(marker))
            .as_string(None),
        ),
    )

    def preflight(conn, _receipt):
        conn.execute("SHOW hba_file").fetchone()
        assert conn.info.transaction_status.name == "INTRANS"
        return ("validated",)

    rollback_visible_during_hba_restore = False

    def observe_rollback_visibility(_conn, _receipt, *, validated):
        nonlocal rollback_visible_during_hba_restore
        assert validated == ("validated",)
        with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as observer:
            rollback_visible_during_hba_restore = observer.execute(
                "SELECT to_regclass(%s) IS NOT NULL AS committed",
                (f"public.{marker}",),
            ).fetchone()["committed"]

    monkeypatch.setattr(module, "_validated_hba_restore", preflight)
    monkeypatch.setattr(module, "_restore_hba_and_reload", observe_rollback_visibility)
    monkeypatch.setattr(module, "_hba_restore_supported", lambda: True)
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_HEX", _EVIDENCE_KEY.hex())
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_ID", _EVIDENCE_KEY_ID)
    monkeypatch.setenv("APPLYPILOT_ADMIN_PG_DSN", fleet_db)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollback-fleet-pg-role.py",
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--rollback-sql",
            str(tmp_path / "rollback.sql"),
            "--restore-hba",
        ],
    )
    try:
        assert module.main() == 0
        assert rollback_visible_during_hba_restore is True
        with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as observer:
            assert observer.execute(
                "SELECT to_regclass(%s) IS NOT NULL AS committed",
                (f"public.{marker}",),
            ).fetchone()["committed"] is True
    finally:
        with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cleanup:
            cleanup.execute(sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE").format(sql.Identifier(marker)))


def test_real_hba_failure_keeps_database_marker_visible_and_emits_receipt(
    fleet_db: str, tmp_path: Path, monkeypatch
) -> None:
    module = _rollback_script_module()
    probe = "applypilot_real_hba_failure_probe"
    rollback_sql = sql.SQL("CREATE TABLE IF NOT EXISTS public.{} (observed boolean NOT NULL)").format(
        sql.Identifier(probe)
    ).as_string(None)
    rollback_path = (tmp_path / "rollback.sql").resolve()
    rollback_path.write_text(rollback_sql, encoding="utf-8")
    receipt_path = (tmp_path / "receipt.json").resolve()
    partial_path = (tmp_path / "partial.json").resolve()
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as identity_conn:
        identity = identity_conn.execute(
            "SELECT current_database() AS database_name,"
            "(SELECT oid FROM pg_database WHERE datname=current_database()) AS database_oid,"
            "(pg_control_system()).system_identifier::text AS system_identifier"
        ).fetchone()
    receipt = pg_roles.authenticate_evidence_receipt(
        {
            "status": "atomic_existing_topology_upgrade_committed",
            "rollback_mode": "forward_v7_containment",
            "database_name": identity["database_name"],
            "inventory": {
                "system_identifier": identity["system_identifier"],
                "database_oid": identity["database_oid"],
            },
            "topology": {"infrastructure_superuser_roles": ["postgres"]},
            "rollback_sql_path": str(rollback_path),
            "rollback_sql_sha256": hashlib.sha256(rollback_sql.encode()).hexdigest(),
            "hba_restore": {
                "format": "applypilot-hba-restore-v2",
                "live_hba_path": str((tmp_path / "pg_hba.conf").resolve()),
                "backup_path": str((tmp_path / "pg_hba.conf.backup").resolve()),
                "expected_target_sha256": "a" * 64,
                "backup_sha256": "b" * 64,
                "backup_size": 1,
                "backup_mode": 0o600,
                "owner": {"uid": 0, "gid": 0},
            },
        },
        authentication_key=_EVIDENCE_KEY,
        authentication_key_id=_EVIDENCE_KEY_ID,
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(module, "_hba_restore_supported", lambda: True)
    monkeypatch.setattr(module, "_validated_hba_restore", lambda *_args: ("validated",))
    monkeypatch.setattr(module, "_observe_hba_state", lambda *_args, **_kwargs: {"sha256": "c" * 64})
    monkeypatch.setattr(
        module,
        "_restore_hba_and_reload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("real injected HBA failure")),
    )
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_HEX", _EVIDENCE_KEY.hex())
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_ID", _EVIDENCE_KEY_ID)
    monkeypatch.setenv("APPLYPILOT_ADMIN_PG_DSN", fleet_db)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollback-fleet-pg-role.py",
            "--receipt",
            str(receipt_path),
            "--rollback-sql",
            str(rollback_path),
            "--restore-hba",
            "--partial-state-receipt",
            str(partial_path),
        ],
    )
    try:
        with pytest.raises(RuntimeError, match="authenticated partial-state receipt"):
            module.main()
        with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as observer:
            assert observer.execute(
                "SELECT to_regclass(%s) IS NOT NULL AS visible", (f"public.{probe}",)
            ).fetchone()["visible"] is True
            marker = observer.execute(
                "SELECT database_name,system_identifier,database_oid,executed_by,rollback_sql_sha256 "
                "FROM public.applypilot_rollback_commit_markers WHERE rollback_sql_sha256=%s",
                (hashlib.sha256(rollback_sql.encode()).hexdigest(),),
            ).fetchone()
            assert marker == {
                "database_name": identity["database_name"],
                "system_identifier": identity["system_identifier"],
                "database_oid": identity["database_oid"],
                "executed_by": "postgres",
                "rollback_sql_sha256": hashlib.sha256(rollback_sql.encode()).hexdigest(),
            }
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        pg_roles.verify_evidence_receipt(
            partial,
            authentication_key=_EVIDENCE_KEY,
            expected_key_id=_EVIDENCE_KEY_ID,
        )
        assert partial["database_containment"]["visible"] is True
        assert partial["database_containment"]["rollback_sql_sha256"] == marker[
            "rollback_sql_sha256"
        ]
    finally:
        with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cleanup:
            cleanup.execute(sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE").format(sql.Identifier(probe)))
            cleanup.execute("DROP TABLE IF EXISTS public.applypilot_rollback_commit_markers")


def test_real_pg18_session_fence_and_alternate_breakglass_retry(fleet_db: str) -> None:
    module = _rollback_script_module()
    provider = "rollback_session_fence_provider_test"
    alternate = "rollback_alternate_breakglass_test"
    provider_password = "rollback-provider-password"
    alternate_password = "rollback-alternate-password"
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as setup:
        require_disposable_postgres(setup)
        if int(setup.execute("SHOW server_version_num").fetchone()["server_version_num"]) // 10000 != 18:
            pytest.skip("session-fence rollback coverage requires PostgreSQL 18")
        setup.execute("DROP TABLE IF EXISTS public.applypilot_rollback_commit_markers")
        for role_name in (provider, alternate):
            setup.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name)))
        setup.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(provider), sql.Literal(provider_password)
            )
        )
        setup.execute(
            sql.SQL("CREATE ROLE {} LOGIN SUPERUSER PASSWORD {}").format(
                sql.Identifier(alternate), sql.Literal(alternate_password)
            )
        )
        identity = setup.execute(
            "SELECT pg_catalog.current_database() AS database_name,"
            "(SELECT database.oid FROM pg_catalog.pg_database database "
            "WHERE database.datname=pg_catalog.current_database()) AS database_oid,"
            "(pg_catalog.pg_control_system()).system_identifier::text AS system_identifier"
        ).fetchone()

    rollback_sql = (
        "DROP TABLE IF EXISTS pg_temp.applypilot_authority_containment_roles;"
        "CREATE TEMP TABLE applypilot_authority_containment_roles ON COMMIT DROP AS "
        f"SELECT oid AS role_oid FROM pg_catalog.pg_roles WHERE rolname='{provider}';"
        f'ALTER ROLE "{provider}" NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB '
        "NOCREATEROLE NOREPLICATION NOBYPASSRLS;"
        f'REVOKE CONNECT ON DATABASE "{identity["database_name"]}" FROM "{provider}";'
    )
    receipt = {
        "inventory": {"infrastructure_superuser_roles": ["postgres", alternate]},
        "receipt_authentication": {"digest": "f" * 64},
    }
    postgres_identity = {**identity, "session_user": "postgres"}
    alternate_identity = {**identity, "session_user": alternate}
    postgres_marker = module._containment_marker(receipt, rollback_sql, postgres_identity)
    alternate_marker = module._containment_marker(receipt, rollback_sql, alternate_identity)
    assert alternate_marker["marker_id"] == postgres_marker["marker_id"]

    active_provider = psycopg.connect(
        _dsn(fleet_db, user=provider, password=provider_password), row_factory=dict_row
    )
    try:
        with psycopg.connect(fleet_db, row_factory=dict_row) as root:
            phase_one = module._commit_database_containment(
                root,
                rollback_sql,
                marker=postgres_marker,
                protected_roles={"postgres", alternate},
            )
            assert phase_one["affected_role_names"] == [provider]
            assert active_provider.execute("SELECT 1 AS alive").fetchone()["alive"] == 1
            assert module._terminate_affected_sessions(
                root,
                affected_role_names=phase_one["affected_role_names"],
                affected_role_oids=phase_one["affected_role_oids"],
                protected_roles={"postgres", alternate},
            ) == 1
            with pytest.raises(psycopg.Error):
                active_provider.execute("SELECT 1")
            with pytest.raises(psycopg.OperationalError):
                psycopg.connect(
                    _dsn(fleet_db, user=provider, password=provider_password),
                    connect_timeout=2,
                )

        with psycopg.connect(
            _dsn(fleet_db, user=alternate, password=alternate_password), row_factory=dict_row
        ) as alternate_conn:
            class LostCommitAcknowledgement:
                def __init__(self, connection):
                    self.connection = connection

                def execute(self, *args, **kwargs):
                    return self.connection.execute(*args, **kwargs)

                def commit(self):
                    self.connection.commit()
                    raise ConnectionError("injected commit acknowledgement loss")

                def rollback(self):
                    self.connection.rollback()
                    raise OSError("injected rollback reporting failure")

            with pytest.raises(module.ContainmentCommitReconciledConnectionLost) as lost:
                module._commit_with_authenticated_reconciliation(
                    LostCommitAcknowledgement(alternate_conn),
                    rollback_sql,
                    marker=alternate_marker,
                    protected_roles={"postgres", alternate},
                    admin_dsn=fleet_db,
                    receipt=receipt,
                    phase="phase2",
                )
            assert lost.value.marker["marker_id"] == postgres_marker["marker_id"]
            assert [type(error) for error in lost.value.evidence_errors.exceptions] == [
                ConnectionError,
                OSError,
            ]

            phase_two = module._commit_database_containment(
                alternate_conn,
                rollback_sql,
                marker=alternate_marker,
                protected_roles={"postgres", alternate},
                approved_break_glass_roles={"postgres", alternate},
                phase="phase2",
            )
            assert phase_two["marker_id"] == postgres_marker["marker_id"]
            assert phase_two["executed_by"] == "postgres"
            assert phase_two["affected_role_names"] == [provider]
            assert len(phase_two["affected_role_oids"]) == 1
            assert phase_two["completed_phases"] == ["phase1", "phase2"]
            alternate_conn.execute(
                sql.SQL("ALTER TABLE public.applypilot_rollback_commit_markers OWNER TO {}").format(
                    sql.Identifier(provider)
                )
            )
            with pytest.raises(RuntimeError, match="marker relation is unsafe"):
                module._write_database_containment_marker(
                    alternate_conn,
                    marker=alternate_marker,
                    approved_break_glass_roles={"postgres", alternate},
                )
            alternate_conn.rollback()
            alternate_conn.execute(
                "ALTER TABLE public.applypilot_rollback_commit_markers OWNER TO postgres"
            )
            alternate_conn.commit()
    finally:
        active_provider.close()
        with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cleanup:
            cleanup.execute("DROP TABLE IF EXISTS public.applypilot_rollback_commit_markers")
            for role_name in (provider, alternate):
                cleanup.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name)))


def test_pg18_containment_distinguishes_quoted_public_role_from_pseudo_public(
    fleet_db: str,
) -> None:
    quoted_public = "PUBLIC"
    default_owner = "rollback_public_identity_owner_test"
    controller = "rollback_public_identity_controller_test"
    object_names = (
        "brain_public_identity_table",
        "brain_public_identity_sequence",
        "brain_public_identity_function",
        "brain_public_identity_type",
    )
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as root:
        require_disposable_postgres(root)
        if int(root.execute("SHOW server_version_num").fetchone()["server_version_num"]) // 10000 != 18:
            pytest.skip("quoted PUBLIC containment coverage requires PostgreSQL 18")
        root.execute("DROP SCHEMA IF EXISTS brain_public_identity_schema CASCADE")
        root.execute("DROP TABLE IF EXISTS public.brain_public_identity_table CASCADE")
        root.execute("DROP SEQUENCE IF EXISTS public.brain_public_identity_sequence CASCADE")
        root.execute("DROP FUNCTION IF EXISTS public.brain_public_identity_function() CASCADE")
        root.execute("DROP TYPE IF EXISTS public.brain_public_identity_type CASCADE")
        for role_name in (quoted_public, default_owner, controller):
            root.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name)))
        root.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(quoted_public)))
        root.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(default_owner)))
        root.execute(
            sql.SQL("CREATE ROLE {} LOGIN NOINHERIT").format(sql.Identifier(controller))
        )
        root.execute("CREATE TABLE public.brain_public_identity_table(id bigint, payload text)")
        root.execute("CREATE SEQUENCE public.brain_public_identity_sequence")
        root.execute(
            "CREATE FUNCTION public.brain_public_identity_function() RETURNS bigint "
            "LANGUAGE sql AS 'SELECT 1::bigint'"
        )
        root.execute("CREATE TYPE public.brain_public_identity_type AS ENUM ('value')")
        root.execute("CREATE SCHEMA brain_public_identity_schema")
        for grantee in (sql.SQL("PUBLIC"), sql.Identifier(quoted_public)):
            root.execute(
                sql.SQL("GRANT SELECT ON public.brain_public_identity_table TO {}").format(grantee)
            )
            root.execute(
                sql.SQL(
                    "GRANT UPDATE(payload) ON public.brain_public_identity_table TO {}"
                ).format(grantee)
            )
            root.execute(
                sql.SQL("GRANT USAGE ON SEQUENCE public.brain_public_identity_sequence TO {}").format(
                    grantee
                )
            )
            root.execute(
                sql.SQL(
                    "GRANT EXECUTE ON FUNCTION public.brain_public_identity_function() TO {}"
                ).format(grantee)
            )
            root.execute(
                sql.SQL("GRANT USAGE ON TYPE public.brain_public_identity_type TO {}").format(
                    grantee
                )
            )
            root.execute(
                sql.SQL("GRANT USAGE ON SCHEMA brain_public_identity_schema TO {}").format(grantee)
            )
        root.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES FOR ROLE {} GRANT SELECT ON TABLES TO PUBLIC"
            ).format(sql.Identifier(quoted_public))
        )
        root.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES FOR ROLE {} GRANT SELECT ON TABLES TO {}"
            ).format(sql.Identifier(default_owner), sql.Identifier(quoted_public))
        )
        try:
            root.autocommit = False
            with root.cursor() as cur:
                containment_sql, assertion_sql = pg_roles._catalog_derived_authority_containment_sql(
                    cur,
                    database_name=root.execute("SELECT current_database() AS name").fetchone()["name"],
                    infrastructure_superuser_roles=("postgres",),
                    controller_role=controller,
                    managed_role_names=(quoted_public,),
                    future_owner_role_names=(quoted_public, default_owner),
                )
            root.execute(containment_sql)
            root.execute(assertion_sql)
            acl_rows = root.execute(
                "SELECT acl.grantee AS grantee_oid,acl.grantee=0 AS is_public,"
                "grantee.rolname AS grantee_name FROM pg_catalog.pg_class relation "
                "JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(relation.relacl) acl "
                "LEFT JOIN pg_catalog.pg_roles grantee ON grantee.oid=acl.grantee "
                "WHERE namespace.nspname='public' "
                "AND relation.relname='brain_public_identity_table' "
                "AND (acl.grantee=0 OR grantee.rolname=%s)",
                (quoted_public,),
            ).fetchall()
            default_acl_rows = root.execute(
                "SELECT acl.grantee AS grantee_oid,acl.grantee=0 AS is_public,"
                "grantee.rolname AS grantee_name FROM pg_catalog.pg_default_acl defaults "
                "JOIN pg_catalog.pg_roles owner ON owner.oid=defaults.defaclrole "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(defaults.defaclacl) acl "
                "LEFT JOIN pg_catalog.pg_roles grantee ON grantee.oid=acl.grantee "
                    "WHERE owner.rolname=ANY(%s) AND acl.grantee<>defaults.defaclrole "
                    "AND (acl.grantee=0 OR grantee.rolname=%s)",
                ([quoted_public, default_owner], quoted_public),
            ).fetchall()
            assert acl_rows == []
            assert default_acl_rows == []
        finally:
            root.rollback()
            root.autocommit = True
            root.execute("DROP SCHEMA IF EXISTS brain_public_identity_schema CASCADE")
            for object_kind, object_name in (
                ("TABLE", object_names[0]),
                ("SEQUENCE", object_names[1]),
                ("FUNCTION", object_names[2] + "()"),
                ("TYPE", object_names[3]),
            ):
                root.execute(f"DROP {object_kind} IF EXISTS public.{object_name} CASCADE")
            for role_name in (quoted_public, default_owner, controller):
                root.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role_name)))
                root.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name)))


def test_pg18_legacy_containment_marker_upgrades_and_resumes_phase_two(fleet_db: str) -> None:
    module = _rollback_script_module()
    with psycopg.connect(fleet_db, row_factory=dict_row) as conn:
        require_disposable_postgres(conn)
        if int(conn.execute("SHOW server_version_num").fetchone()["server_version_num"]) // 10000 != 18:
            pytest.skip("legacy marker upgrade coverage requires PostgreSQL 18")
        conn.execute("DROP TABLE IF EXISTS public.applypilot_rollback_commit_markers")
        conn.execute(
            "CREATE TABLE public.applypilot_rollback_commit_markers("
            "marker_id text PRIMARY KEY,database_name text NOT NULL,system_identifier text NOT NULL,"
            "database_oid oid NOT NULL,executed_by text NOT NULL,affected_role_names text[] NOT NULL,"
            "affected_role_oids oid[] NOT NULL,rollback_sql_sha256 text NOT NULL,"
            "source_receipt_digest text NOT NULL,committed_at timestamptz NOT NULL "
            "DEFAULT pg_catalog.clock_timestamp())"
        )
        target = module._validated_target_identity(conn, {
            "inventory": {"infrastructure_superuser_roles": ["postgres"]},
        })
        marker = module._containment_marker(
            {"receipt_authentication": {"digest": "a" * 64}}, "SELECT 1", target
        )
        conn.execute(
            "INSERT INTO public.applypilot_rollback_commit_markers("
            "marker_id,database_name,system_identifier,database_oid,executed_by,affected_role_names,"
            "affected_role_oids,rollback_sql_sha256,source_receipt_digest) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                marker["marker_id"], marker["database_name"], marker["system_identifier"],
                marker["database_oid"], marker["executed_by"], ["legacy-role"], [12345],
                marker["rollback_sql_sha256"], marker["source_receipt_digest"],
            ),
        )
        module._write_database_containment_marker(
            conn, marker=marker, phase="phase2", approved_break_glass_roles={"postgres"}
        )
        conn.commit()
        observed = module._assert_database_containment_marker(
            conn, marker=marker, expected_phase="phase2"
        )
        assert observed["completed_phases"] == ["phase1", "phase2"]
        assert observed["affected_role_names"] == ["legacy-role"]
        assert observed["affected_role_oids"] == [12345]
        conn.execute("DROP TABLE public.applypilot_rollback_commit_markers")
        conn.commit()


def test_pg18_marker_creation_revokes_inherited_default_acl(fleet_db: str) -> None:
    module = _rollback_script_module()
    grantee = "rollback_marker_default_acl_test"
    with psycopg.connect(fleet_db, row_factory=dict_row) as conn:
        require_disposable_postgres(conn)
        if int(conn.execute("SHOW server_version_num").fetchone()["server_version_num"]) // 10000 != 18:
            pytest.skip("marker ACL coverage requires PostgreSQL 18")
        conn.execute("DROP TABLE IF EXISTS public.applypilot_rollback_commit_markers")
        conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(grantee)))
        conn.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(grantee)))
        conn.execute(
            sql.SQL("ALTER DEFAULT PRIVILEGES GRANT SELECT ON TABLES TO {}").format(
                sql.Identifier(grantee)
            )
        )
        conn.commit()
        target = module._validated_target_identity(conn, {
            "inventory": {"infrastructure_superuser_roles": ["postgres"]},
        })
        marker = module._containment_marker(
            {"receipt_authentication": {"digest": "b" * 64}}, "SELECT 1", target
        )
        try:
            module._write_database_containment_marker(conn, marker=marker)
            conn.commit()
            assert conn.execute(
                "SELECT count(*) AS count FROM pg_catalog.pg_class relation "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(relation.relacl) acl "
                "JOIN pg_catalog.pg_roles grantee ON grantee.oid=acl.grantee "
                "WHERE relation.oid='public.applypilot_rollback_commit_markers'::pg_catalog.regclass "
                "AND grantee.rolname=%s",
                (grantee,),
            ).fetchone()["count"] == 0
        finally:
            conn.rollback()
            conn.execute("DROP TABLE IF EXISTS public.applypilot_rollback_commit_markers")
            conn.execute(
                sql.SQL("ALTER DEFAULT PRIVILEGES REVOKE ALL ON TABLES FROM {}").format(
                    sql.Identifier(grantee)
                )
            )
            conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(grantee)))
            conn.commit()


def test_pg18_wrong_marker_shape_fails_before_schema_mutation(fleet_db: str) -> None:
    module = _rollback_script_module()
    with psycopg.connect(fleet_db, row_factory=dict_row) as conn:
        require_disposable_postgres(conn)
        conn.execute("DROP TABLE IF EXISTS public.applypilot_rollback_commit_markers")
        conn.execute(
            "CREATE TABLE public.applypilot_rollback_commit_markers("
            "marker_id text PRIMARY KEY,database_name text NOT NULL,system_identifier text NOT NULL,"
            "database_oid oid NOT NULL,executed_by text NOT NULL,affected_role_names text[] NOT NULL,"
            "affected_role_oids oid[] NOT NULL,rollback_sql_sha256 text NOT NULL,"
            "source_receipt_digest text NOT NULL,committed_at timestamptz NOT NULL,hostile text)"
        )
        conn.commit()
        target = module._validated_target_identity(conn, {
            "inventory": {"infrastructure_superuser_roles": ["postgres"]},
        })
        marker = module._containment_marker(
            {"receipt_authentication": {"digest": "c" * 64}}, "SELECT 1", target
        )
        with pytest.raises(RuntimeError, match="column contract mismatch"):
            module._write_database_containment_marker(conn, marker=marker)
        conn.rollback()
        columns = conn.execute(
            "SELECT attribute.attname FROM pg_catalog.pg_attribute attribute "
            "WHERE attribute.attrelid='public.applypilot_rollback_commit_markers'::pg_catalog.regclass "
            "AND attribute.attnum>0 AND NOT attribute.attisdropped ORDER BY attribute.attnum"
        ).fetchall()
        assert "completed_phases" not in [row["attname"] for row in columns]
        conn.execute("DROP TABLE public.applypilot_rollback_commit_markers")
        conn.commit()


def test_rollback_rejects_hba_restore_on_windows_before_database_access(
    tmp_path: Path, monkeypatch
) -> None:
    module = _rollback_script_module()
    database_accessed = False

    def unexpected_inputs(*_args, **_kwargs):
        nonlocal database_accessed
        database_accessed = True
        raise AssertionError("database inputs must not be opened")

    monkeypatch.setattr(module, "_verified_inputs", unexpected_inputs)
    assert module._hba_restore_supported(platform_name="nt") is False
    monkeypatch.setattr(module, "_hba_restore_supported", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollback-fleet-pg-role.py",
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--rollback-sql",
            str(tmp_path / "rollback.sql"),
            "--restore-hba",
        ],
    )
    with pytest.raises(SystemExit, match="secure HBA restoration is unsupported"):
        module.main()
    assert database_accessed is False


def test_partial_evidence_failure_removes_created_rollback(tmp_path: Path, monkeypatch) -> None:
    paths = pg_roles.DurableEvidencePaths(
        preparation_receipt_path=(tmp_path / "receipt.json").resolve(),
        rollback_sql_path=(tmp_path / "rollback.sql").resolve(),
        authentication_key=_EVIDENCE_KEY,
        authentication_key_id=_EVIDENCE_KEY_ID,
    )
    original_write = pg_roles._write_exclusive_fsync
    calls = 0

    def fail_receipt(path, payload, *, expected_parent_identity):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected receipt write failure")
        return original_write(
            path,
            payload,
            expected_parent_identity=expected_parent_identity,
        )

    monkeypatch.setattr(pg_roles, "_write_exclusive_fsync", fail_receipt)
    with pytest.raises(OSError, match="receipt write failure"):
        pg_roles._write_preparation_evidence(
            paths,
            inventory={"prepared_at": "now", "database_name": "postgres"},
            rollback_sql="SELECT 1;\n",
        )
    assert not paths.rollback_sql_path.exists()
    assert not paths.preparation_receipt_path.exists()


def test_evidence_write_fails_closed_after_parent_identity_substitution(tmp_path: Path, monkeypatch) -> None:
    target = (tmp_path / "receipt.json").resolve()
    expected = pg_roles._evidence_parent_identity(target.parent)
    monkeypatch.setattr(
        pg_roles,
        "_evidence_parent_identity",
        lambda _parent: (expected[0], expected[1], expected[2] + 1),
    )
    with pytest.raises(RuntimeError, match="parent directory changed"):
        pg_roles._write_exclusive_fsync(
            target,
            b"evidence",
            expected_parent_identity=expected,
        )
    assert not target.exists()


def test_hba_second_reload_failure_reports_active_rule_uncertainty(monkeypatch) -> None:
    module = _rollback_script_module()
    replacements: list[bytes] = []

    class FakeResult:
        def fetchone(self):
            return {"reloaded": False}

    class FakeTransaction:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    class FakeConnection:
        def execute(self, _statement):
            return FakeResult()

        def transaction(self):
            return FakeTransaction()

    monkeypatch.setattr(
        module,
        "_replace_atomic",
        lambda _path, payload, **_kwargs: replacements.append(payload),
    )
    monkeypatch.setattr(
        module.os,
        "lstat",
        lambda _path: type("Stat", (), {"st_dev": 1, "st_ino": 2})(),
    )
    validated = (Path("C:/pg_hba.conf"), b"current", b"backup", (1, 1), 0o600, 0o600, (0, 0))
    with pytest.raises(RuntimeError, match="active rules are uncertain"):
        module._restore_hba_and_reload(FakeConnection(), {}, validated=validated)
    assert replacements == [b"backup", b"current"]


def test_bootstrap_refuses_autocommit_before_database_mutation(fleet_db: str) -> None:
    topology = pg_roles.BootstrapTopology(
        database_owner_role="autocommit_owner_test",
        controller_role="autocommit_controller_test",
        verifier_role="autocommit_verifier_test",
        migrator_role="autocommit_migrator_test",
        retired_admin_roles=("autocommit_retired_admin_test",),
        infrastructure_superuser_roles=("postgres",),
    )
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as conn:
        with pytest.raises(RuntimeError, match="autocommit.*disabled"):
            pg_roles.bootstrap_database_roles(
                conn,
                CONTROLLER_PASSWORD,
                BACKUP_PASSWORD,
                topology=topology,
                evidence_paths=_test_evidence_paths(),
            )
        assert (
            conn.execute(
                "SELECT count(*) AS count FROM pg_roles WHERE rolname=ANY(%s)",
                (
                    list(
                        (
                            topology.database_owner_role,
                            topology.controller_role,
                            topology.verifier_role,
                            topology.migrator_role,
                        )
                    ),
                ),
            ).fetchone()["count"]
            == 0
        )


def _dsn(base: str, *, user: str, password: str) -> str:
    params = conninfo_to_dict(base)
    params.update(user=user, password=password)
    return make_conninfo(**params)


def _atomic_topology() -> pg_roles.BootstrapTopology:
    return pg_roles.BootstrapTopology(
        database_owner_role=ATOMIC_OWNER,
        controller_role=ATOMIC_CONTROLLER,
        verifier_role="brain_schema_verifier",
        migrator_role="brain_schema_migrator",
        retired_admin_roles=(ATOMIC_RETIRED_ADMIN,),
        infrastructure_superuser_roles=("postgres",),
    )


def test_existing_topology_upgrade_closes_legacy_admin_graph_and_installs_v7(
    fleet_db: str,
    monkeypatch,
) -> None:
    owner = "existing_authority_owner_test"
    controller = "existing_authority_controller_test"
    retired = "existing_authority_retired_test"
    legacy_controller = "existing_authority_legacy_controller_test"
    worker = "existing_authority_worker_test"
    provider_middle = "existing_authority_provider_middle_test"
    provider_login = "existing_authority_provider_login_test"
    provider_default_owner = "existing_authority_provider_default_owner_test"
    provider_parent = "existing_authority_provider_parent_test"
    topology = pg_roles.BootstrapTopology(
        database_owner_role=owner,
        controller_role=controller,
        verifier_role="brain_schema_verifier",
        migrator_role="brain_schema_migrator",
        retired_admin_roles=(retired, legacy_controller),
        infrastructure_superuser_roles=("postgres",),
    )
    role_names = (
        pg_roles.FLEET_BACKUP_ROLE,
        "brain_artifact_authority_writer",
        "brain_artifact_authority_owner",
        "brain_candidate_writer",
        "brain_candidate_reader",
        "brain_graph_authority",
        "brain_policy_controller",
        "brain_status_reader",
        "brain_schema_verifier",
        "brain_schema_migrator",
        worker,
        provider_login,
        provider_middle,
        provider_default_owner,
        provider_parent,
        legacy_controller,
        retired,
        controller,
        owner,
    )
    with pgqueue.connect(fleet_db) as root:
        require_disposable_postgres(root)
        version = int(root.execute("SHOW server_version_num").fetchone()["server_version_num"])
        if version // 10000 != 18:
            pytest.skip("existing topology authority upgrade requires PostgreSQL 18")
        database_name = conninfo_to_dict(root.info.dsn)["dbname"]
        try:
            root.execute(sql.SQL("CREATE ROLE {} NOLOGIN INHERIT").format(sql.Identifier(owner)))
            root.execute(sql.SQL("CREATE ROLE {} LOGIN INHERIT PASSWORD {}").format(
                sql.Identifier(controller), sql.Literal("existing-controller-password")
            ))
            root.execute(sql.SQL("CREATE ROLE {} NOLOGIN INHERIT").format(sql.Identifier(retired)))
            root.execute(sql.SQL("CREATE ROLE {} LOGIN INHERIT CREATEROLE PASSWORD {}").format(
                sql.Identifier(legacy_controller), sql.Literal("legacy-controller-password")
            ))
            root.execute(sql.SQL("CREATE ROLE {} LOGIN NOINHERIT PASSWORD {}").format(
                sql.Identifier(worker), sql.Literal("worker-password")
            ))
            root.execute(sql.SQL(
                "ALTER DEFAULT PRIVILEGES FOR ROLE {} REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC"
            ).format(sql.Identifier(legacy_controller)))
            for role_name in ("brain_schema_migrator", "brain_schema_verifier"):
                root.execute(sql.SQL("CREATE ROLE {} NOLOGIN INHERIT").format(sql.Identifier(role_name)))
                root.execute(sql.SQL(
                    "GRANT {} TO postgres WITH ADMIN FALSE, INHERIT TRUE, SET TRUE"
                ).format(sql.Identifier(role_name)))
            for role_name in ("brain_status_reader", "brain_policy_controller"):
                root.execute(sql.SQL("CREATE ROLE {} NOLOGIN NOINHERIT").format(sql.Identifier(role_name)))
            root.execute(sql.SQL(
                "GRANT brain_policy_controller TO {} WITH ADMIN FALSE, INHERIT TRUE, SET TRUE"
            ).format(sql.Identifier(controller)))
            root.execute(sql.SQL("GRANT {} TO {} WITH INHERIT TRUE, SET TRUE").format(
                sql.Identifier(owner), sql.Identifier(legacy_controller)
            ))
            root.execute(sql.SQL(
                "GRANT {} TO {} WITH ADMIN TRUE, INHERIT FALSE, SET FALSE"
            ).format(sql.Identifier(worker), sql.Identifier(legacy_controller)))
            root.execute(sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                sql.Identifier(database_name), sql.Identifier(owner)
            ))
            root.execute(sql.SQL(
                "REVOKE CONNECT,CREATE,TEMPORARY ON DATABASE {} FROM PUBLIC"
            ).format(sql.Identifier(database_name)))
            root.execute(sql.SQL("GRANT CREATE ON DATABASE {} TO brain_schema_migrator").format(
                sql.Identifier(database_name)
            ))
            root.execute(
                "GRANT USAGE,CREATE ON SCHEMA public TO brain_schema_migrator WITH GRANT OPTION"
            )
            root.execute(
                "GRANT SELECT ON ALL TABLES IN SCHEMA public TO brain_schema_migrator WITH GRANT OPTION"
            )
            root.execute(
                "GRANT SELECT,INSERT,UPDATE ON public.fleet_config,public.fleet_decision_policies,"
                "public.apply_queue,public.linkedin_queue,public.workers,public.worker_heartbeat,"
                "public.fleet_worker_principals,public.fleet_desired_state,public.rate_governor "
                "TO brain_schema_migrator WITH GRANT OPTION"
            )
            for column in (
                "ats_canary_worker_id",
                "ats_canary_version",
                "linkedin_canary_worker_id",
                "linkedin_canary_version",
            ):
                root.execute(f"ALTER TABLE public.fleet_config ADD COLUMN IF NOT EXISTS {column} text")
            root.execute(
                "UPDATE public.fleet_config SET paused=TRUE,ats_paused=TRUE,"
                "ats_apply_mode='stopped',linkedin_apply_mode='stopped',"
                "canary_enabled=FALSE,linkedin_canary_enabled=FALSE WHERE id=1"
            )
            for table_name in ("apply_queue", "compute_queue", "linkedin_queue"):
                root.execute(sql.SQL("DELETE FROM public.{}").format(sql.Identifier(table_name)))
            root.execute(
                "INSERT INTO public.compute_queue(url,task,status,lease_owner,lease_expires_at) "
                "VALUES('https://historical.example/job','score','leased','old-worker',"
                "clock_timestamp()+interval '1 hour')"
            )
            root.execute(
                "INSERT INTO public.workers(worker_id,machine_owner,capabilities) "
                "VALUES('existing-worker','existing-owner','{\"can_ats\":true}'::jsonb)"
            )
            root.execute(
                "INSERT INTO public.fleet_worker_principals(role_name,worker_id,contract) "
                "VALUES(%s,'existing-worker','apply')",
                (worker,),
            )
            root.execute(
                "CREATE TABLE IF NOT EXISTS public.applypilot_fleet_schema_migrations("
                "migration_id text PRIMARY KEY,migration_sha256 text NOT NULL)"
            )
            root.execute("DELETE FROM public.applypilot_fleet_schema_migrations")
            with root.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO public.applypilot_fleet_schema_migrations(migration_id,migration_sha256) VALUES(%s,%s)",
                    [
                        ("20260715_001_application_authority", "65d10d1a16e9829a7995f48776697e3200daf7a5074147a9a3ed8c0293f56bea"),
                        ("20260717_002_lane_specific_canary_pins", "1ee176c419508cd5a9c7004460798d14fd6749addbf293892fd2d859a1b6dd1d"),
                    ],
                )
            current_v1_checksum = brain_schema._schema_checksum
            monkeypatch.setattr(
                brain_schema,
                "_schema_checksum",
                lambda: "1976fdc4dbeac4cf853ce6e445efb071651696ae47ee67b61e58aab98ff4e2a5",
            )
            with root.cursor() as cur:
                cur.execute("SET LOCAL ROLE brain_schema_migrator")
                brain_schema.ensure_brain_schema_v1_in_transaction(cur)
                cur.execute("RESET ROLE")
            monkeypatch.setattr(brain_schema, "_schema_checksum", current_v1_checksum)
            root.commit()
            identity = root.execute(
                "SELECT (pg_control_system()).system_identifier::text AS system_identifier,"
                "(SELECT oid FROM pg_database WHERE datname=current_database()) AS database_oid"
            ).fetchone()
            root.commit()

            blocked_evidence = _test_evidence_paths()
            with pytest.raises(RuntimeError, match="zero active unexpired leases"):
                pg_roles.upgrade_brain_authority_existing_topology(
                    root,
                    BACKUP_PASSWORD,
                    topology=topology,
                    evidence_paths=blocked_evidence,
                    expected_database_name=database_name,
                    expected_system_identifier=identity["system_identifier"],
                )
            assert not blocked_evidence.preparation_receipt_path.exists()
            assert not blocked_evidence.rollback_sql_path.exists()
            root.execute(
                "UPDATE public.compute_queue SET status='done',lease_owner=NULL,lease_expires_at=NULL "
                "WHERE url='https://historical.example/job' AND task='score'"
            )
            root.commit()

            original_transition = pg_roles._transition_existing_authority_topology
            blocked_statements: list[str] = []

            def transition_with_lock_probe(cur, *, topology, backup_password_verifier):
                probes = (
                    "UPDATE public.fleet_config SET paused=FALSE WHERE id=1",
                    "UPDATE public.apply_queue SET status=status WHERE FALSE",
                    "UPDATE public.compute_queue SET status=status WHERE FALSE",
                    "UPDATE public.linkedin_queue SET status=status WHERE FALSE",
                )
                for statement in probes:
                    with psycopg.connect(fleet_db, row_factory=dict_row) as contender:
                        contender.execute("SET lock_timeout='100ms'")
                        with pytest.raises(psycopg.errors.LockNotAvailable):
                            contender.execute(statement)
                        contender.rollback()
                    blocked_statements.append(statement)
                return original_transition(
                    cur,
                    topology=topology,
                    backup_password_verifier=backup_password_verifier,
                )

            monkeypatch.setattr(
                pg_roles,
                "_transition_existing_authority_topology",
                transition_with_lock_probe,
            )

            receipt = pg_roles.upgrade_brain_authority_existing_topology(
                root,
                BACKUP_PASSWORD,
                topology=topology,
                evidence_paths=_test_evidence_paths(),
                expected_database_name=database_name,
                expected_system_identifier=identity["system_identifier"],
            )
            default_acl_dependencies = receipt.inventory["retired_admin_default_acl_dependencies"]
            assert receipt.inventory["database_oid"] == identity["database_oid"]
            assert len(default_acl_dependencies) == 1
            assert default_acl_dependencies[0]["role_name"] == legacy_controller
            assert default_acl_dependencies[0]["database_name"] == database_name
            assert default_acl_dependencies[0]["objsubid"] == 0

            assert root.execute(
                "SELECT array_agg(version ORDER BY version) AS versions FROM public.brain_schema_versions"
            ).fetchone()["versions"] == [1, 2, 3, 4, 5, 6, 7]
            assert root.execute(
                "SELECT migration_checksum FROM public.brain_schema_versions WHERE version=1"
            ).fetchone()["migration_checksum"] == (
                "1976fdc4dbeac4cf853ce6e445efb071651696ae47ee67b61e58aab98ff4e2a5"
            )
            assert root.execute(
                "SELECT rolcanlogin,rolinherit,rolcreatedb FROM pg_roles WHERE rolname=%s",
                (legacy_controller,),
            ).fetchone() == {"rolcanlogin": False, "rolinherit": False, "rolcreatedb": False}
            assert root.execute(
                "SELECT has_schema_privilege('brain_schema_migrator','public','USAGE') AS allowed"
            ).fetchone()["allowed"] is True
            assert root.execute(
                "SELECT count(*) AS count FROM pg_auth_members membership "
                "JOIN pg_roles parent ON parent.oid=membership.roleid "
                "JOIN pg_roles member ON member.oid=membership.member "
                "WHERE parent.rolname=%s OR member.rolname=%s",
                (legacy_controller, legacy_controller),
            ).fetchone()["count"] == 0
            assert root.execute(
                "SELECT count(*) AS count FROM pg_auth_members membership "
                "JOIN pg_roles parent ON parent.oid=membership.roleid "
                "JOIN pg_roles member ON member.oid=membership.member "
                "WHERE parent.rolname IN ('brain_schema_migrator','brain_schema_verifier') "
                "AND member.rolname='postgres'"
            ).fetchone()["count"] == 0
            assert root.execute(
                "SELECT rolcanlogin,rolinherit,rolpassword IS NOT NULL AS has_password "
                "FROM pg_authid WHERE rolname=%s",
                (owner,),
            ).fetchone() == {"rolcanlogin": True, "rolinherit": False, "has_password": False}
            assert (
                f'GRANT "{owner}" TO "{legacy_controller}"' not in receipt.rollback_sql
            )
            assert len(blocked_statements) == 4
            assert root.execute(
                "SELECT paused,ats_paused,ats_apply_mode,linkedin_apply_mode FROM public.fleet_config WHERE id=1"
            ).fetchone() == {
                "paused": True,
                "ats_paused": True,
                "ats_apply_mode": "stopped",
                "linkedin_apply_mode": "stopped",
            }
            assert root.execute(
                "SELECT status FROM public.compute_queue "
                "WHERE url='https://historical.example/job' AND task='score'"
            ).fetchone()["status"] == "done"
            assert root.execute(
                "SELECT pg_has_role(%s,'brain_schema_migrator','SET') "
                "AND pg_has_role(%s,'brain_artifact_authority_owner','SET') AS allowed",
                (owner, owner),
            ).fetchone()["allowed"] is True
            root.execute(sql.SQL("CREATE ROLE {} NOLOGIN NOINHERIT").format(sql.Identifier(provider_middle)))
            root.execute(
                sql.SQL("CREATE ROLE {} LOGIN NOINHERIT PASSWORD {}").format(
                    sql.Identifier(provider_login), sql.Literal("existing-provider-password")
                )
            )
            root.execute(sql.SQL("CREATE ROLE {} NOLOGIN NOINHERIT").format(sql.Identifier(provider_parent)))
            root.execute(
                sql.SQL("CREATE ROLE {} LOGIN NOINHERIT PASSWORD {}").format(
                    sql.Identifier(provider_default_owner),
                    sql.Literal("existing-default-owner-password"),
                )
            )
            for parent_role in (owner, "brain_candidate_writer", "brain_artifact_authority_owner"):
                root.execute(
                    sql.SQL("GRANT {} TO {} WITH ADMIN FALSE, INHERIT FALSE, SET TRUE").format(
                        sql.Identifier(parent_role), sql.Identifier(provider_middle)
                    )
                )
            root.execute(
                sql.SQL("GRANT {} TO {} WITH ADMIN FALSE, INHERIT FALSE, SET TRUE").format(
                    sql.Identifier(provider_middle), sql.Identifier(provider_login)
                )
            )
            root.execute(
                sql.SQL("GRANT {} TO {} WITH ADMIN FALSE, INHERIT FALSE, SET TRUE").format(
                    sql.Identifier(provider_parent), sql.Identifier(provider_login)
                )
            )
            root.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(database_name), sql.Identifier(provider_login)
                )
            )
            root.execute(
                "GRANT EXECUTE ON FUNCTION public.brain_publish_v4_candidate("
                "text,text,text,text,text,bigint,uuid,text,text,text,text,text,bigint) "
                "TO brain_candidate_writer"
            )
            root.execute(
                "GRANT EXECUTE ON FUNCTION public.brain_register_authoritative_artifact_manifest_v6("
                "uuid,text,text,text,timestamptz,timestamptz,text,text,jsonb) "
                "TO brain_artifact_authority_writer"
            )
            root.execute(
                "GRANT INSERT ON TABLE public.brain_v4_decision_envelopes TO PUBLIC"
            )
            root.execute(
                "GRANT UPDATE(candidate_decision_id) "
                "ON TABLE public.brain_v4_candidate_decisions TO PUBLIC"
            )
            root.execute("GRANT CREATE ON SCHEMA public TO PUBLIC")
            root.execute(
                sql.SQL(
                    "GRANT SELECT,UPDATE ON TABLE public.brain_v4_candidate_decisions TO {}"
                ).format(sql.Identifier(provider_middle))
            )
            root.execute(
                sql.SQL(
                    "GRANT USAGE,SELECT ON SEQUENCE "
                    "public.brain_authority_scope_state_authority_scope_id_seq TO {}"
                ).format(sql.Identifier(provider_middle))
            )
            root.execute(
                sql.SQL(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE brain_schema_migrator IN SCHEMA public "
                    "GRANT INSERT,UPDATE ON TABLES TO {}"
                ).format(sql.Identifier(provider_login))
            )
            root.execute(
                sql.SQL(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public "
                    "GRANT INSERT ON TABLES TO {}"
                ).format(sql.Identifier(legacy_controller), sql.Identifier(provider_login))
            )
            root.execute(
                sql.SQL(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public "
                    "GRANT UPDATE ON TABLES TO {}"
                ).format(sql.Identifier(provider_default_owner), sql.Identifier(provider_login))
            )
            if root.execute(
                "SELECT pg_catalog.to_regnamespace('brain_archive') AS schema_oid"
            ).fetchone()["schema_oid"] is None:
                root.execute(
                    sql.SQL("CREATE SCHEMA brain_archive AUTHORIZATION {}").format(
                        sql.Identifier(provider_default_owner)
                    )
                )
            else:
                root.execute(
                    sql.SQL("ALTER SCHEMA brain_archive OWNER TO {}").format(
                        sql.Identifier(provider_default_owner)
                    )
                )
            root.execute(
                "CREATE TABLE brain_archive.brain_archive_escape_probe(id bigint)"
            )
            root.execute(
                sql.SQL(
                    "ALTER TABLE brain_archive.brain_archive_escape_probe OWNER TO {}"
                ).format(sql.Identifier(provider_default_owner))
            )
            root.execute(
                "GRANT SELECT ON brain_archive.brain_archive_escape_probe TO PUBLIC"
            )
            root.execute(
                "CREATE PROCEDURE public.brain_escape_procedure() "
                "LANGUAGE sql AS 'SELECT 1'"
            )
            root.execute(
                sql.SQL("ALTER PROCEDURE public.brain_escape_procedure() OWNER TO {}").format(
                    sql.Identifier(provider_default_owner)
                )
            )
            root.execute("GRANT EXECUTE ON PROCEDURE public.brain_escape_procedure() TO PUBLIC")
            root.execute("CREATE DOMAIN brain_archive.brain_archive_escape_domain AS text")
            root.execute(
                sql.SQL(
                    "ALTER DOMAIN brain_archive.brain_archive_escape_domain OWNER TO {}"
                ).format(sql.Identifier(provider_default_owner))
            )
            root.execute(
                "CREATE TYPE brain_archive.brain_archive_escape_enum AS ENUM ('probe')"
            )
            root.execute(
                sql.SQL(
                    "ALTER TYPE brain_archive.brain_archive_escape_enum OWNER TO {}"
                ).format(sql.Identifier(provider_default_owner))
            )
            root.execute(
                "CREATE TYPE brain_archive.brain_archive_escape_composite AS (id bigint)"
            )
            root.execute(
                sql.SQL(
                    "ALTER TYPE brain_archive.brain_archive_escape_composite OWNER TO {}"
                ).format(sql.Identifier(provider_default_owner))
            )
            root.execute(
                sql.SQL(
                    "GRANT USAGE ON TYPE brain_archive.brain_archive_escape_domain,"
                    "brain_archive.brain_archive_escape_enum TO PUBLIC,{}"
                ).format(sql.Identifier(provider_login))
            )
            root.execute(
                "CREATE AGGREGATE public.brain_escape_aggregate(bigint) "
                "(SFUNC=pg_catalog.int8pl, STYPE=bigint, INITCOND='0')"
            )
            root.execute(
                sql.SQL("ALTER AGGREGATE public.brain_escape_aggregate(bigint) OWNER TO {}").format(
                    sql.Identifier(provider_default_owner)
                )
            )
            root.commit()
            with psycopg.connect(
                _dsn(fleet_db, user=provider_login, password="existing-provider-password"),
                row_factory=dict_row,
            ) as active_provider:
                active_provider.execute("SET ROLE brain_candidate_writer")
                active_provider.execute("RESET ROLE")
                active_provider.execute("SET ROLE brain_artifact_authority_owner")
                active_provider.execute("RESET ROLE")
                active_provider.execute(
                    "UPDATE public.brain_v4_candidate_decisions "
                    "SET candidate_decision_id='decision-acl-probe' WHERE FALSE"
                )
                active_provider.execute("CREATE TABLE public.brain_acl_escape_probe(id bigint)")
                active_provider.execute("DROP TABLE public.brain_acl_escape_probe")
                active_provider.execute("CALL public.brain_escape_procedure()")
                active_provider.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(provider_parent)))
                active_provider.execute("RESET ROLE")
                active_provider.commit()

                for run in range(2):
                    root.execute(receipt.rollback_sql)
                    root.commit()
                    assert root.execute(
                        "SELECT has_function_privilege('brain_candidate_writer',"
                        "'public.brain_publish_v4_candidate(text,text,text,text,text,bigint,uuid,text,text,text,text,text,bigint)',"
                        "'EXECUTE') AS v4_allowed,"
                        "has_function_privilege('brain_candidate_writer',"
                        "'public.brain_publish_v5_candidate(text,text,text,text,text,bigint,uuid,text,text,text,text,text,bigint)',"
                        "'EXECUTE') AS v5_allowed,"
                        "has_function_privilege('brain_artifact_authority_writer',"
                        "'public.brain_register_authoritative_artifact_manifest(uuid,text,text,text,timestamptz,timestamptz,text,text,jsonb)',"
                        "'EXECUTE') AS v7_allowed,"
                        "has_function_privilege('brain_artifact_authority_writer',"
                        "'public.brain_register_authoritative_artifact_manifest_v6(uuid,text,text,text,timestamptz,timestamptz,text,text,jsonb)',"
                        "'EXECUTE') AS v6_allowed,"
                        "has_function_privilege('brain_graph_authority',"
                        "'public.brain_create_factual_generation(text,text,text,text,text)',"
                        "'EXECUTE') AS graph_allowed"
                    ).fetchone() == {
                        "v4_allowed": False,
                        "v5_allowed": False,
                        "v7_allowed": False,
                        "v6_allowed": False,
                        "graph_allowed": False,
                    }
                    assert root.execute(
                        "SELECT pg_has_role(%s,'brain_schema_migrator','SET') AS migrator_allowed,"
                        "pg_has_role(%s,'brain_artifact_authority_owner','SET') AS artifact_allowed,"
                        "pg_has_role(%s,'brain_candidate_writer','SET') AS candidate_allowed,"
                        "(SELECT rolcanlogin FROM pg_roles WHERE rolname=%s) AS login_allowed,"
                        "(SELECT rolcanlogin FROM pg_roles WHERE rolname=%s) "
                        "AS default_owner_login_allowed",
                        (owner, owner, provider_login, provider_login, provider_default_owner),
                    ).fetchone() == {
                        "migrator_allowed": False,
                        "artifact_allowed": False,
                        "candidate_allowed": False,
                        "login_allowed": False,
                        "default_owner_login_allowed": False,
                    }
                    assert root.execute(
                        "SELECT pg_catalog.pg_has_role(%s,%s,'SET') AS parent_set_allowed",
                        (provider_login, provider_parent),
                    ).fetchone()["parent_set_allowed"] is False
                    assert root.execute(
                        "SELECT has_table_privilege('public','public.brain_v4_decision_envelopes','INSERT') "
                        "AS public_table_insert,"
                        "has_column_privilege('public','public.brain_v4_candidate_decisions',"
                        "'candidate_decision_id','UPDATE') AS public_column_update,"
                        "has_schema_privilege('public','public','CREATE') AS public_schema_create,"
                        "has_table_privilege(%s,'public.brain_v4_candidate_decisions','UPDATE') "
                        "AS provider_table_update,"
                        "has_sequence_privilege(%s,"
                        "'public.brain_authority_scope_state_authority_scope_id_seq','USAGE') "
                        "AS provider_sequence_usage",
                        (provider_middle, provider_middle),
                    ).fetchone() == {
                        "public_table_insert": False,
                        "public_column_update": False,
                        "public_schema_create": False,
                        "provider_table_update": False,
                        "provider_sequence_usage": False,
                    }
                    assert root.execute(
                        "SELECT count(*) AS count FROM pg_default_acl default_acl "
                        "CROSS JOIN LATERAL aclexplode(default_acl.defaclacl) acl "
                        "LEFT JOIN pg_roles owner ON owner.oid=default_acl.defaclrole "
                        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
                        "WHERE (owner.rolname=ANY(%s) OR grantee.rolname=ANY(%s)) "
                        "AND acl.grantee<>default_acl.defaclrole",
                        (
                            [
                                "brain_schema_migrator",
                                legacy_controller,
                                provider_default_owner,
                            ],
                            [provider_login],
                        ),
                    ).fetchone()["count"] == 0
                    assert root.execute(
                        "SELECT procedure_owner.rolname AS procedure_owner,"
                        "schema_owner.rolname AS archive_schema_owner,"
                        "relation_owner.rolname AS archive_relation_owner,"
                        "(SELECT count(*) FROM pg_catalog.pg_proc executable "
                        "JOIN pg_catalog.pg_namespace executable_namespace "
                        "ON executable_namespace.oid=executable.pronamespace "
                        "CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE("
                        "executable.proacl,pg_catalog.acldefault('f',executable.proowner))) acl "
                        "WHERE executable_namespace.nspname='public' "
                        "AND executable.proname='brain_escape_procedure' "
                        "AND acl.grantee<>executable.proowner) AS procedure_nonowner_routes "
                        "FROM pg_catalog.pg_proc procedure "
                        "JOIN pg_catalog.pg_namespace procedure_namespace "
                        "ON procedure_namespace.oid=procedure.pronamespace "
                        "JOIN pg_catalog.pg_roles procedure_owner "
                        "ON procedure_owner.oid=procedure.proowner "
                        "JOIN pg_catalog.pg_namespace archive_schema "
                        "ON archive_schema.nspname='brain_archive' "
                        "JOIN pg_catalog.pg_roles schema_owner "
                        "ON schema_owner.oid=archive_schema.nspowner "
                        "JOIN pg_catalog.pg_class archive_relation "
                        "ON archive_relation.relnamespace=archive_schema.oid "
                        "AND archive_relation.relname='brain_archive_escape_probe' "
                        "JOIN pg_catalog.pg_roles relation_owner "
                        "ON relation_owner.oid=archive_relation.relowner "
                        "WHERE procedure_namespace.nspname='public' "
                        "AND procedure.proname='brain_escape_procedure'"
                    ).fetchone() == {
                        "procedure_owner": "postgres",
                        "archive_schema_owner": "postgres",
                        "archive_relation_owner": "postgres",
                        "procedure_nonowner_routes": 0,
                    }
                    assert root.execute(
                        "SELECT aggregate_owner.rolname AS aggregate_owner,"
                        "domain_owner.rolname AS domain_owner FROM pg_catalog.pg_proc aggregate "
                        "JOIN pg_catalog.pg_namespace aggregate_namespace "
                        "ON aggregate_namespace.oid=aggregate.pronamespace "
                        "JOIN pg_catalog.pg_roles aggregate_owner "
                        "ON aggregate_owner.oid=aggregate.proowner "
                        "JOIN pg_catalog.pg_type domain ON domain.typname='brain_archive_escape_domain' "
                        "JOIN pg_catalog.pg_namespace domain_namespace "
                        "ON domain_namespace.oid=domain.typnamespace "
                        "AND domain_namespace.nspname='brain_archive' "
                        "JOIN pg_catalog.pg_roles domain_owner ON domain_owner.oid=domain.typowner "
                        "WHERE aggregate_namespace.nspname='public' "
                        "AND aggregate.proname='brain_escape_aggregate'"
                    ).fetchone() == {
                        "aggregate_owner": "postgres",
                        "domain_owner": "postgres",
                    }
                    assert root.execute(
                        "SELECT enum_owner.rolname AS enum_owner,"
                        "composite_owner.rolname AS composite_owner,"
                        "pg_catalog.has_type_privilege('public',"
                        "'brain_archive.brain_archive_escape_domain','USAGE') "
                        "AS public_domain_usage,"
                        "pg_catalog.has_type_privilege(%s,"
                        "'brain_archive.brain_archive_escape_enum','USAGE') "
                        "AS provider_enum_usage FROM pg_catalog.pg_type enum_type "
                        "JOIN pg_catalog.pg_namespace enum_namespace "
                        "ON enum_namespace.oid=enum_type.typnamespace "
                        "JOIN pg_catalog.pg_roles enum_owner ON enum_owner.oid=enum_type.typowner "
                        "JOIN pg_catalog.pg_type composite "
                        "ON composite.typname='brain_archive_escape_composite' "
                        "JOIN pg_catalog.pg_namespace composite_namespace "
                        "ON composite_namespace.oid=composite.typnamespace "
                        "AND composite_namespace.nspname='brain_archive' "
                        "JOIN pg_catalog.pg_roles composite_owner "
                        "ON composite_owner.oid=composite.typowner "
                        "WHERE enum_namespace.nspname='brain_archive' "
                        "AND enum_type.typname='brain_archive_escape_enum'",
                        (provider_login,),
                    ).fetchone() == {
                        "enum_owner": "postgres",
                        "composite_owner": "postgres",
                        "public_domain_usage": False,
                        "provider_enum_usage": False,
                    }
                    if run == 0:
                        for role_name in (
                            "brain_candidate_writer",
                            "brain_artifact_authority_owner",
                        ):
                            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                                active_provider.execute(
                                    sql.SQL("SET ROLE {}").format(sql.Identifier(role_name))
                                )
                            active_provider.rollback()
                        for statement in (
                            "SELECT public.brain_publish_v4_candidate("
                            "NULL::text,NULL::text,NULL::text,NULL::text,NULL::text,NULL::bigint,"
                            "NULL::uuid,NULL::text,NULL::text,NULL::text,NULL::text,NULL::text,NULL::bigint)",
                            "SELECT public.brain_publish_v5_candidate("
                            "NULL::text,NULL::text,NULL::text,NULL::text,NULL::text,NULL::bigint,"
                            "NULL::uuid,NULL::text,NULL::text,NULL::text,NULL::text,NULL::text,NULL::bigint)",
                            "SELECT public.brain_register_authoritative_artifact_manifest_v6("
                            "NULL::uuid,NULL::text,NULL::text,NULL::text,NULL::timestamptz,"
                            "NULL::timestamptz,NULL::text,NULL::text,NULL::jsonb)",
                        ):
                            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                                active_provider.execute(statement)
                            active_provider.rollback()
                        for statement in (
                            "INSERT INTO public.brain_v4_decision_envelopes("
                            "envelope_id,candidate_decision_id,envelope_artifact_hash) "
                            "SELECT 'x','y','z' WHERE FALSE",
                            "UPDATE public.brain_v4_candidate_decisions "
                            "SET candidate_decision_id='decision-acl-probe' WHERE FALSE",
                            "CREATE TABLE public.brain_acl_escape_probe(id bigint)",
                        ):
                            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                                active_provider.execute(statement)
                            active_provider.rollback()

                        root.execute(
                            "GRANT USAGE,CREATE ON SCHEMA public TO brain_schema_migrator"
                        )
                        root.execute("SET ROLE brain_schema_migrator")
                        root.execute("CREATE TABLE public.brain_future_acl_probe(id bigint)")
                        root.execute(
                            "CREATE FUNCTION public.brain_future_acl_probe_function() "
                            "RETURNS bigint LANGUAGE sql AS 'SELECT 1::bigint'"
                        )
                        root.execute("RESET ROLE")
                        root.execute(
                            "REVOKE USAGE,CREATE ON SCHEMA public FROM brain_schema_migrator"
                        )
                        root.commit()
                        assert root.execute(
                            "SELECT has_table_privilege(%s,'public.brain_future_acl_probe','INSERT') "
                            "AS provider_future_insert,"
                            "has_function_privilege('public',"
                            "'public.brain_future_acl_probe_function()','EXECUTE') "
                            "AS public_future_execute",
                            (provider_login,),
                        ).fetchone() == {
                            "provider_future_insert": False,
                            "public_future_execute": False,
                        }
            rollback_module = _rollback_script_module()
            provider_rows = root.execute(
                "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=ANY(%s) ORDER BY oid",
                ([provider_login, provider_middle, provider_default_owner],),
            ).fetchall()
            affected_oids = [int(row["oid"]) for row in provider_rows]
            clean = rollback_module._assert_candidate_artifact_containment(
                root,
                break_glass_roles={"postgres", controller},
                affected_role_names=[provider_login, provider_middle, provider_default_owner],
                affected_role_oids=affected_oids,
            )
            assert not any(clean.values())
            root.execute(
                "GRANT SELECT ON brain_archive.brain_archive_escape_probe TO PUBLIC"
            )
            root.commit()
            with pytest.raises(RuntimeError, match="postcondition failed"):
                rollback_module._assert_candidate_artifact_containment(
                    root,
                    break_glass_roles={"postgres", controller},
                    affected_role_names=[provider_login, provider_middle, provider_default_owner],
                    affected_role_oids=affected_oids,
                )
            root.execute(receipt.rollback_sql)
            root.commit()
        finally:
            root.rollback()
            require_disposable_postgres(root)
            root.execute(sql.SQL("ALTER DATABASE {} OWNER TO postgres").format(sql.Identifier(database_name)))
            root.execute("DROP SCHEMA IF EXISTS brain_archive CASCADE")
            root.execute("DROP SCHEMA public CASCADE")
            for role_name in role_names:
                if root.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    root.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role_name)))
                    root.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
            root.execute("CREATE SCHEMA public AUTHORIZATION pg_database_owner")
            root.execute("GRANT ALL ON SCHEMA public TO pg_database_owner")
            root.execute("GRANT USAGE ON SCHEMA public TO PUBLIC")
            root.commit()


def test_pg18_bootstrap_validates_connection_then_database_then_evidence(monkeypatch) -> None:
    events: list[str] = []

    class StopAfterOrdering(RuntimeError):
        pass

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeConnection:
        autocommit = False
        info = type("Info", (), {"transaction_status": type("Status", (), {"name": "IDLE"})()})()

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(
        pg_roles,
        "_require_pg18_authority_catalog",
        lambda _cursor: events.append("pg18_database_preflight"),
    )
    monkeypatch.setattr(
        pg_roles,
        "_validate_evidence_paths",
        lambda _paths: events.append("evidence_paths"),
    )

    def stop_before_scram(*_args, **_kwargs):
        raise StopAfterOrdering("ordering observed")

    monkeypatch.setattr(pg_roles, "_client_scram_verifier", stop_before_scram)
    with pytest.raises(StopAfterOrdering, match="ordering observed"):
        pg_roles.bootstrap_database_roles(
            FakeConnection(),
            CONTROLLER_PASSWORD,
            BACKUP_PASSWORD,
            topology=_atomic_topology(),
            evidence_paths=object(),
            install_brain_authority=True,
        )
    assert events == ["pg18_database_preflight", "evidence_paths"]


def test_backup_source_policy_fails_closed_for_rls_and_large_objects(fleet_db: str) -> None:
    with pgqueue.connect(fleet_db) as root:
        require_disposable_postgres(root)
        root.execute("CREATE TABLE public.backup_rls_policy_probe(id bigint)")
        root.execute("ALTER TABLE public.backup_rls_policy_probe ENABLE ROW LEVEL SECURITY")
        with pytest.raises(RuntimeError, match="RLS-bearing database"):
            pg_roles._reject_unsupported_backup_objects(root.cursor())
        root.execute("DROP TABLE public.backup_rls_policy_probe")
        large_object_oid = root.execute("SELECT lo_create(0) AS oid").fetchone()["oid"]
        with pytest.raises(RuntimeError, match="containing large objects"):
            pg_roles._reject_unsupported_backup_objects(root.cursor())
        root.execute("SELECT lo_unlink(%s)", (large_object_oid,))
        root.rollback()


def test_unsupported_major_pg16_bootstrap_rejects_before_lock_evidence_or_database_drift(
    fleet_db: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence = pg_roles.DurableEvidencePaths(
        preparation_receipt_path=tmp_path / "prepared.json",
        rollback_sql_path=tmp_path / "rollback.sql",
        authentication_key=_EVIDENCE_KEY,
        authentication_key_id=_EVIDENCE_KEY_ID,
    )
    evidence_reached = False

    def reject_evidence(*_args, **_kwargs):
        nonlocal evidence_reached
        evidence_reached = True
        raise AssertionError("bootstrap reached evidence write before PG18 preflight")

    monkeypatch.setattr(pg_roles, "_write_preparation_evidence", reject_evidence)
    with pgqueue.connect(fleet_db) as connection:
        version = int(connection.execute("SHOW server_version_num").fetchone()["server_version_num"])
        connection.commit()
        if version // 10000 == 18:
            pytest.skip("unsupported-major rejection requires the fixed PostgreSQL 16 fixture")
        roles_before = connection.execute(
            "SELECT rolname,rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,"
            "rolreplication,rolbypassrls FROM pg_roles ORDER BY rolname"
        ).fetchall()
        relations_before = connection.execute(
            "SELECT n.nspname,c.relname,c.relkind FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname IN ('public','brain_archive') "
            "ORDER BY n.nspname,c.relname,c.relkind"
        ).fetchall()
        connection.commit()
        with pytest.raises(
            RuntimeError,
            match="PostgreSQL 18 authority catalog contract required",
        ):
            pg_roles.bootstrap_database_roles(
                connection,
                CONTROLLER_PASSWORD,
                BACKUP_PASSWORD,
                topology=_atomic_topology(),
                evidence_paths=evidence,
                install_brain_authority=True,
            )
        connection.rollback()
        assert evidence_reached is False
        assert not evidence.preparation_receipt_path.exists()
        assert not evidence.rollback_sql_path.exists()
        assert connection.execute(
            "SELECT rolname,rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,"
            "rolreplication,rolbypassrls FROM pg_roles ORDER BY rolname"
        ).fetchall() == roles_before
        assert connection.execute(
            "SELECT n.nspname,c.relname,c.relkind FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname IN ('public','brain_archive') "
            "ORDER BY n.nspname,c.relname,c.relkind"
        ).fetchall() == relations_before
        connection.commit()
        acquired = connection.execute(
            "SELECT pg_try_advisory_lock(pg_catalog.hashtext(%s)) AS acquired",
            ("applypilot:fleet-role-bootstrap:v1",),
        ).fetchone()["acquired"]
        assert acquired is True
        connection.execute(
            "SELECT pg_advisory_unlock(pg_catalog.hashtext(%s))",
            ("applypilot:fleet-role-bootstrap:v1",),
        )
        connection.commit()


def _create_atomic_lifecycle_roles(conn) -> None:
    for column in (
        "ats_canary_worker_id",
        "ats_canary_version",
        "linkedin_canary_worker_id",
        "linkedin_canary_version",
    ):
        conn.execute(f"ALTER TABLE public.fleet_config ADD COLUMN IF NOT EXISTS {column} text")
    for role_name in ATOMIC_LIFECYCLE_ROLES:
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
            ).format(sql.Identifier(role_name))
        )


def _drop_atomic_bootstrap_roles(conn) -> None:
    conn.rollback()
    require_disposable_postgres(conn)
    database_name = conninfo_to_dict(conn.info.dsn)["dbname"]
    conn.execute(sql.SQL("ALTER DATABASE {} OWNER TO postgres").format(sql.Identifier(database_name)))
    conn.execute(sql.SQL("GRANT CONNECT, TEMPORARY ON DATABASE {} TO PUBLIC").format(sql.Identifier(database_name)))
    conn.execute("DROP SCHEMA IF EXISTS brain_archive CASCADE")
    conn.execute("DROP SCHEMA public CASCADE")
    for role_name in (
        pg_roles.FLEET_BACKUP_ROLE,
        "brain_candidate_reader",
        "brain_candidate_writer",
        *ATOMIC_LIFECYCLE_ROLES,
        ATOMIC_CONTROLLER,
        "brain_schema_verifier",
        "brain_schema_migrator",
        ATOMIC_OWNER,
        ATOMIC_RETIRED_ADMIN,
    ):
        if conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
            conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role_name)))
            conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
    conn.execute("CREATE SCHEMA public AUTHORIZATION pg_database_owner")
    conn.execute("GRANT ALL ON SCHEMA public TO pg_database_owner")
    conn.execute("GRANT USAGE ON SCHEMA public TO PUBLIC")
    restored = conn.execute(
        "SELECT n.nspowner::regrole::text AS owner, "
        "has_schema_privilege('public', 'public', 'USAGE') AS public_usage, "
        "has_database_privilege('public', current_database(), 'CONNECT') AS public_connect, "
        "has_database_privilege('public', current_database(), 'TEMPORARY') AS public_temporary, "
        "EXISTS (SELECT 1 FROM aclexplode(n.nspacl) acl "
        "WHERE acl.grantee='pg_database_owner'::regrole "
        "AND acl.privilege_type='CREATE') AS owner_create_grant "
        "FROM pg_namespace n WHERE n.nspname='public'"
    ).fetchone()
    assert restored == {
        "owner": "pg_database_owner",
        "public_usage": True,
        "public_connect": True,
        "public_temporary": True,
        "owner_create_grant": True,
    }
    conn.commit()


def _seed_v5_rollback_preservation_rows(conn) -> dict[str, object]:
    def artifact(label: str) -> str:
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT INTO public.brain_artifacts("
            "request_id,artifact_hash,media_type,byte_length,schema_version,location) "
            "VALUES(%s,%s,'application/json',1,1,'rollback-preservation') ON CONFLICT DO NOTHING",
            (f"rollback-preservation-{label}", digest),
        )
        return digest

    owner = "rollback-preservation-owner"
    ontology = "rollback-preservation-ontology-v1"
    generation = "rollback-preservation-generation-v1"
    span = "rollback-preservation-span-v1"
    event = "rollback-preservation-event-v1"
    snapshot = "rollback-preservation-snapshot-v1"
    manifest = artifact("ontology-manifest")
    term_digest = hashlib.sha256(b"rollback-preservation-term").hexdigest()
    term_id = f"skill:{term_digest}"
    conn.execute(
        "SELECT public.brain_create_factual_ontology(%s,%s,%s)",
        (owner, ontology, manifest),
    )
    conn.execute(
        "SELECT public.brain_add_factual_ontology_term("
        "%s,%s,%s,'has_skill','skill',%s,%s,'Rollback preservation skill',%s)",
        (owner, ontology, manifest, term_digest, term_id, artifact("ontology-term")),
    )
    ontology_root = conn.execute(
        "SELECT public.brain_compute_factual_ontology_root(%s,%s) AS root",
        (owner, ontology),
    ).fetchone()["root"]
    conn.execute(
        "SELECT public.brain_close_factual_ontology(%s,%s,1,%s,%s)",
        (owner, ontology, ontology_root, artifact("ontology-close")),
    )
    generation_manifest = artifact("generation-manifest")
    conn.execute(
        "SELECT public.brain_create_factual_generation(%s,%s,%s,%s,%s)",
        (owner, generation, generation_manifest, ontology, ontology_root),
    )
    source = artifact("source")
    conn.execute(
        "SELECT public.brain_add_factual_generation_member(%s,%s,%s,%s,'resume',0)",
        (owner, generation, span, source),
    )
    membership_root = conn.execute(
        "SELECT public.brain_compute_factual_membership_root(%s,%s) AS root",
        (owner, generation),
    ).fetchone()["root"]
    conn.execute(
        "SELECT public.brain_close_factual_generation(%s,%s,1,%s,%s)",
        (owner, generation, membership_root, artifact("generation-close")),
    )
    approval = "rollback-preservation-approval-v1"
    conn.execute(
        "SELECT public.brain_admit_factual_event("
        "%s,%s,%s,%s,%s,%s,%s,'resume',%s,'has_skill',%s,%s,%s,1,'assert',NULL,%s)",
        (
            owner,
            generation,
            span,
            approval,
            artifact("approval-receipt"),
            hashlib.sha256(b"rollback-preservation-claim").hexdigest(),
            source,
            ontology,
            term_id,
            event,
            artifact("event"),
            datetime.now(UTC) - timedelta(seconds=1),
        ),
    )
    conn.execute(
        "SELECT public.brain_record_factual_assertion_coverage(%s,%s,%s,%s)",
        (owner, generation, span, event),
    )
    semantic_root = conn.execute(
        "SELECT public.brain_compute_factual_semantic_root(%s,%s) AS root",
        (owner, generation),
    ).fetchone()["root"]
    conn.execute(
        "SELECT public.brain_publish_factual_snapshot("
        "%s,%s,%s,%s,%s,%s,1,clock_timestamp(),NULL,%s)",
        (
            owner,
            snapshot,
            generation,
            semantic_root,
            artifact("coverage"),
            membership_root,
            artifact("snapshot"),
        ),
    )
    immutable_artifact = artifact("immutable-reference")
    conn.execute(
        "INSERT INTO public.brain_immutable_artifact_references(artifact_hash,reference_type,subject_id) "
        "VALUES(%s,'candidate_payload','rollback-preservation-candidate-v1')",
        (immutable_artifact,),
    )
    conn.commit()
    return {
        "generation": conn.execute(
            "SELECT owner_id,generation_id,membership_manifest_hash,ontology_version,ontology_root_hash "
            "FROM public.brain_factual_generations WHERE owner_id=%s AND generation_id=%s",
            (owner, generation),
        ).fetchone(),
        "fact": conn.execute(
            "SELECT owner_id,generation_id,source_span_id,event_id,mutation_action "
            "FROM public.brain_graph_fact_events WHERE owner_id=%s AND event_id=%s",
            (owner, event),
        ).fetchone(),
        "coverage": conn.execute(
            "SELECT owner_id,generation_id,source_span_id,disposition,event_id "
            "FROM public.brain_factual_generation_coverage WHERE owner_id=%s AND generation_id=%s",
            (owner, generation),
        ).fetchone(),
        "snapshot": conn.execute(
            "SELECT owner_id,graph_snapshot_id,generation_id,semantic_root_hash,membership_root_hash,event_high_water "
            "FROM public.brain_factual_graph_snapshots WHERE owner_id=%s AND graph_snapshot_id=%s",
            (owner, snapshot),
        ).fetchone(),
        "immutable": conn.execute(
            "SELECT artifact_hash,reference_type,subject_id FROM public.brain_immutable_artifact_references "
            "WHERE artifact_hash=%s",
            (immutable_artifact,),
        ).fetchone(),
    }


def _read_v5_rollback_preservation_rows(conn, expected: dict[str, object]) -> dict[str, object]:
    generation = expected["generation"]
    fact = expected["fact"]
    snapshot = expected["snapshot"]
    immutable = expected["immutable"]
    return {
        "generation": conn.execute(
            "SELECT owner_id,generation_id,membership_manifest_hash,ontology_version,ontology_root_hash "
            "FROM public.brain_factual_generations WHERE owner_id=%s AND generation_id=%s",
            (generation["owner_id"], generation["generation_id"]),
        ).fetchone(),
        "fact": conn.execute(
            "SELECT owner_id,generation_id,source_span_id,event_id,mutation_action "
            "FROM public.brain_graph_fact_events WHERE owner_id=%s AND event_id=%s",
            (fact["owner_id"], fact["event_id"]),
        ).fetchone(),
        "coverage": conn.execute(
            "SELECT owner_id,generation_id,source_span_id,disposition,event_id "
            "FROM public.brain_factual_generation_coverage WHERE owner_id=%s AND generation_id=%s",
            (generation["owner_id"], generation["generation_id"]),
        ).fetchone(),
        "snapshot": conn.execute(
            "SELECT owner_id,graph_snapshot_id,generation_id,semantic_root_hash,membership_root_hash,event_high_water "
            "FROM public.brain_factual_graph_snapshots WHERE owner_id=%s AND graph_snapshot_id=%s",
            (snapshot["owner_id"], snapshot["graph_snapshot_id"]),
        ).fetchone(),
        "immutable": conn.execute(
            "SELECT artifact_hash,reference_type,subject_id FROM public.brain_immutable_artifact_references "
            "WHERE artifact_hash=%s",
            (immutable["artifact_hash"],),
        ).fetchone(),
    }


def test_fixed_atomic_authority_install_failure_rolls_back_everything(fleet_db: str, monkeypatch) -> None:
    topology = _atomic_topology()
    with pgqueue.connect(fleet_db) as root:
        root.execute(
            sql.SQL("CREATE ROLE {} LOGIN SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS").format(
                sql.Identifier(ATOMIC_RETIRED_ADMIN)
            )
        )
        _create_atomic_lifecycle_roles(root)
        root.commit()

        original_installer = pg_roles._install_brain_authority_in_transaction

        def fail_after_v5(cur, *, topology) -> None:
            original_installer(cur, topology=topology)
            raise RuntimeError("forced failure after fixed authority install")

        monkeypatch.setattr(pg_roles, "_install_brain_authority_in_transaction", fail_after_v5)

        try:
            with pytest.raises(RuntimeError, match="forced failure"):
                pg_roles.bootstrap_database_roles(
                    root,
                    CONTROLLER_PASSWORD,
                    BACKUP_PASSWORD,
                    topology=topology,
                    evidence_paths=_test_evidence_paths(),
                    install_brain_authority=True,
                )

            assert (
                root.execute(
                    "SELECT array_agg(rolname ORDER BY rolname) AS roles FROM pg_roles WHERE rolname=ANY(%s)",
                    (
                        [
                            ATOMIC_OWNER,
                            ATOMIC_CONTROLLER,
                            "brain_schema_verifier",
                            "brain_schema_migrator",
                            "brain_candidate_reader",
                            "brain_candidate_writer",
                        ],
                    ),
                ).fetchone()["roles"]
                is None
            )
            assert (
                root.execute("SELECT to_regclass('public.brain_schema_versions') AS relation").fetchone()["relation"]
                is None
            )
        finally:
            root.rollback()
            require_disposable_postgres(root)
            for role_name in (*ATOMIC_LIFECYCLE_ROLES, ATOMIC_RETIRED_ADMIN):
                if root.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    root.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
            root.commit()


def test_atomic_bootstrap_removes_provider_admin_membership_after_v5_install(fleet_db: str) -> None:
    topology = _atomic_topology()
    with pgqueue.connect(fleet_db) as root:
        root.execute(
            sql.SQL("CREATE ROLE {} LOGIN SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS").format(
                sql.Identifier(ATOMIC_RETIRED_ADMIN)
            )
        )
        _create_atomic_lifecycle_roles(root)
        root.commit()
        try:
            receipt = pg_roles.bootstrap_database_roles(
                root,
                CONTROLLER_PASSWORD,
                BACKUP_PASSWORD,
                topology=topology,
                evidence_paths=_test_evidence_paths(),
                install_brain_authority=True,
            )
            assert receipt.escalation_required is True
            assert receipt.inventory["database_oid"] > 0
            assert receipt.inventory["system_identifier"].isdigit()
            assert receipt.inventory["backup_role_boundary"] == {
                "role_name": pg_roles.FLEET_BACKUP_ROLE,
                "connection_limit": 1,
                "memberships": 0,
                "owned_objects": 0,
                "rls_policy": "fail_closed_if_present",
                "large_object_policy": "fail_closed_if_present",
                "database_identity_readable": False,
                "pg_control_system_execute": True,
                "role_settings": (
                    "default_transaction_read_only=on",
                    "search_path=pg_catalog, pg_temp",
                ),
                "default_acl_entries": (
                    ("*", "S", "SELECT", False),
                    ("*", "n", "USAGE", False),
                    ("*", "r", "SELECT", False),
                ),
            }

            root.execute("GRANT USAGE,CREATE ON SCHEMA public TO brain_schema_migrator")
            root.execute("SET ROLE brain_schema_migrator")
            root.execute("CREATE TABLE public.backup_future_acl_probe(id bigint)")
            root.execute("INSERT INTO public.backup_future_acl_probe VALUES (1)")
            root.execute("RESET ROLE")
            root.execute("REVOKE USAGE,CREATE ON SCHEMA public FROM brain_schema_migrator")
            root.commit()
            backup_dsn = _dsn(fleet_db, user=pg_roles.FLEET_BACKUP_ROLE, password=BACKUP_PASSWORD)
            with psycopg.connect(backup_dsn, row_factory=dict_row) as backup:
                assert backup.execute(
                    "SELECT id FROM public.backup_future_acl_probe"
                ).fetchone()["id"] == 1
                assert backup.execute(
                    "SELECT (pg_catalog.pg_control_system()).system_identifier::text AS system_identifier"
                ).fetchone()["system_identifier"] == receipt.inventory["system_identifier"]
                assert backup.execute("SHOW default_transaction_read_only").fetchone()[
                    "default_transaction_read_only"
                ] == "on"
                assert backup.execute("SHOW search_path").fetchone()["search_path"] == "pg_catalog, pg_temp"
                with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                    backup.execute("INSERT INTO public.backup_future_acl_probe VALUES (2)")
                backup.rollback()
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    backup.execute("SELECT public.brain_v5_sha256_text('denied')")
                backup.rollback()
            controller_dsn = _dsn(
                fleet_db,
                user=ATOMIC_CONTROLLER,
                password=CONTROLLER_PASSWORD,
            )
            with psycopg.connect(controller_dsn, row_factory=dict_row) as controller:
                controller.execute("SET ROLE brain_policy_controller")
                with pytest.raises(psycopg.errors.NoDataFound, match="unknown policy_version"):
                    controller.execute(
                        "SELECT public.brain_controller_transition_policy('missing-policy','validated',NULL)"
                    )
                controller.rollback()

            assert root.execute(
                "SELECT array_agg(version ORDER BY version) AS versions FROM public.brain_schema_versions"
            ).fetchone()["versions"] == [1, 2, 3, 4, 5, 6, 7]
            assert (
                root.execute(
                    "SELECT 1 FROM pg_auth_members membership "
                    "JOIN pg_roles parent ON parent.oid=membership.roleid "
                    "JOIN pg_roles member ON member.oid=membership.member "
                    "WHERE parent.rolname='brain_schema_migrator' AND member.rolname=session_user"
                ).fetchone()
                is None
            )
            assert (
                root.execute(
                    "SELECT has_database_privilege('brain_schema_migrator',current_database(),'CREATE') AS allowed"
                ).fetchone()["allowed"]
                is False
            )
            assert root.execute(
                "SELECT rolcanlogin,rolinherit,rolcreaterole FROM pg_roles WHERE rolname=%s",
                (ATOMIC_CONTROLLER,),
            ).fetchone() == {
                "rolcanlogin": True,
                "rolinherit": False,
                "rolcreaterole": False,
            }
            assert (
                root.execute(
                    "SELECT count(*) AS count FROM pg_auth_members membership "
                    "JOIN pg_roles parent ON parent.oid=membership.roleid "
                    "JOIN pg_roles member ON member.oid=membership.member "
                    "WHERE member.rolname=%s AND parent.rolname=ANY(%s)",
                    (ATOMIC_CONTROLLER, [ATOMIC_OWNER, "brain_schema_migrator"]),
                ).fetchone()["count"]
                == 0
            )
            preserved_v5_rows = _seed_v5_rollback_preservation_rows(root)
            assert "DROP OWNED" not in receipt.rollback_sql.upper()
            root.execute(receipt.rollback_sql)
            root.commit()
            assert _read_v5_rollback_preservation_rows(root, preserved_v5_rows) == preserved_v5_rows
            root.execute(receipt.rollback_sql)
            root.commit()
            assert _read_v5_rollback_preservation_rows(root, preserved_v5_rows) == preserved_v5_rows
            assert root.execute(
                "SELECT array_agg(version ORDER BY version) AS versions FROM public.brain_schema_versions"
            ).fetchone()["versions"] == [1, 2, 3, 4, 5, 6, 7]
            assert root.execute(
                "SELECT to_regclass('public.brain_immutable_artifact_references') AS relation"
            ).fetchone()["relation"] == "brain_immutable_artifact_references"
            assert root.execute(
                "SELECT rolcanlogin FROM pg_roles WHERE rolname=%s",
                (ATOMIC_CONTROLLER,),
            ).fetchone()["rolcanlogin"] is False
            assert root.execute(
                "SELECT has_database_privilege('public',current_database(),'CONNECT,CREATE,TEMPORARY') AS allowed"
            ).fetchone()["allowed"] is False
            assert root.execute(
                "SELECT has_function_privilege('brain_candidate_writer',"
                "'public.brain_publish_v5_candidate(text,text,text,text,text,bigint,uuid,text,text,text,text,text,bigint)',"
                "'EXECUTE') AS allowed"
            ).fetchone()["allowed"] is False
        finally:
            _drop_atomic_bootstrap_roles(root)


def test_prepared_forward_rollback_is_idempotent_after_fence_only_commit(fleet_db: str) -> None:
    topology = _atomic_topology()
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cluster_admin:
        require_disposable_postgres(cluster_admin)
        cluster_admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(CROSS_DATABASE)))
    try:
        with pgqueue.connect(fleet_db) as root:
            root.execute(
                sql.SQL("CREATE ROLE {} LOGIN SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS").format(
                    sql.Identifier(ATOMIC_RETIRED_ADMIN)
                )
            )
            _create_atomic_lifecycle_roles(root)
            root.commit()
            inventory = pg_roles._bootstrap_inventory(root.cursor(), topology=topology)
            rollback_sql = pg_roles._forward_v5_deactivation_sql(
                root.cursor(),
                topology=topology,
                database_name=inventory["database_name"],
                database_owner_before=inventory["database_owner_before"],
                ownership=inventory["application_ownership"],
                other_databases=inventory["other_connectable_databases"],
                retired_memberships=inventory["retired_admin_memberships_before"],
            )
            pg_roles._set_other_database_admission_fence(
                root.cursor(),
                databases=inventory["other_connectable_databases"],
            )
            root.commit()
            assert root.execute(
                "SELECT datconnlimit FROM pg_database WHERE datname=%s",
                (CROSS_DATABASE,),
            ).fetchone()["datconnlimit"] == 0

            root.execute(rollback_sql)
            root.commit()
            root.execute(rollback_sql)
            root.commit()
            assert root.execute(
                "SELECT datconnlimit FROM pg_database WHERE datname=%s",
                (CROSS_DATABASE,),
            ).fetchone()["datconnlimit"] == -1
            assert root.execute(
                "SELECT count(*) AS count FROM pg_roles WHERE rolname=ANY(%s)",
                ([ATOMIC_OWNER, ATOMIC_CONTROLLER, "brain_schema_verifier", "brain_schema_migrator"],),
            ).fetchone()["count"] == 0
    finally:
        with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cluster_admin:
            cluster_admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(CROSS_DATABASE)))
            for role_name in (*ATOMIC_LIFECYCLE_ROLES, ATOMIC_RETIRED_ADMIN):
                if cluster_admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    cluster_admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))


def test_post_fence_inventory_rejects_new_database_and_acl_drift(fleet_db: str) -> None:
    topology = _atomic_topology()
    drift_database = "applypilot_post_fence_drift_test"
    drift_role = "applypilot_post_fence_grantee_test"
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cluster_admin:
        require_disposable_postgres(cluster_admin)
        cluster_admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(CROSS_DATABASE)))
        cluster_admin.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(drift_role)))
    try:
        with pgqueue.connect(fleet_db) as root:
            root.execute(
                sql.SQL("CREATE ROLE {} LOGIN SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS").format(
                    sql.Identifier(ATOMIC_RETIRED_ADMIN)
                )
            )
            root.commit()
            inventory = pg_roles._bootstrap_inventory(root.cursor(), topology=topology)
            pg_roles._set_other_database_admission_fence(
                root.cursor(), databases=inventory["other_connectable_databases"]
            )
            root.commit()
            with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as concurrent:
                concurrent.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(drift_database)))
                concurrent.execute(
                    sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                        sql.Identifier(CROSS_DATABASE), sql.Identifier(drift_role)
                    )
                )
            pg_roles._lock_cluster_security_catalogs(root.cursor())
            with pytest.raises(pg_roles.CrossDatabaseInventoryDriftError, match="fence remains closed"):
                pg_roles._validate_other_database_inventory_unchanged(
                    root.cursor(),
                    baseline=inventory["other_connectable_databases"],
                    baseline_database_names=inventory["cluster_database_names_before"],
                    infrastructure_superuser_roles=topology.infrastructure_superuser_roles,
                )
            root.commit()
            fenced = root.execute(
                "SELECT datname,datallowconn,datconnlimit FROM pg_database WHERE datname=ANY(%s) ORDER BY datname",
                ([CROSS_DATABASE, drift_database],),
            ).fetchall()
            assert fenced == [
                {"datname": drift_database, "datallowconn": False, "datconnlimit": 0},
                {"datname": CROSS_DATABASE, "datallowconn": True, "datconnlimit": 0},
            ]
            assert root.execute(
                "SELECT count(*) AS count FROM pg_database database "
                "CROSS JOIN LATERAL aclexplode(database.datacl) acl "
                "WHERE database.datname=%s AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s) "
                "AND acl.privilege_type='CONNECT'",
                (CROSS_DATABASE, drift_role),
            ).fetchone()["count"] == 0
            pg_roles._restore_other_database_admission_fence(
                root.cursor(), databases=inventory["other_connectable_databases"]
            )
            root.commit()
    finally:
        with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cluster_admin:
            for database_name in (drift_database, CROSS_DATABASE):
                cluster_admin.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(database_name))
                )
            for role_name in (ATOMIC_RETIRED_ADMIN, drift_role):
                if cluster_admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    cluster_admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))


def test_post_fence_new_database_session_is_terminated_before_closed_fence_commit(fleet_db: str) -> None:
    drift_database = "applypilot_post_fence_live_drift_test"
    drift_role = "applypilot_post_fence_live_role_test"
    password = "applypilot-post-fence-live-password"
    topology = _atomic_topology()
    active_connection = None
    baseline_connection = None
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cluster_admin:
        require_disposable_postgres(cluster_admin)
        cluster_admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(CROSS_DATABASE)))
        cluster_admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(drift_role), sql.Literal(password)
            )
        )
    try:
        with pgqueue.connect(fleet_db) as root:
            root.execute(
                sql.SQL("CREATE ROLE {} LOGIN SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS").format(
                    sql.Identifier(ATOMIC_RETIRED_ADMIN)
                )
            )
            root.commit()
            inventory = pg_roles._bootstrap_inventory(root.cursor(), topology=topology)
            baseline_parameters = conninfo_to_dict(fleet_db)
            baseline_parameters.update(dbname=CROSS_DATABASE, user=drift_role, password=password)
            baseline_connection = psycopg.connect(make_conninfo(**baseline_parameters), row_factory=dict_row)
            assert baseline_connection.execute("SELECT 1 AS alive").fetchone()["alive"] == 1
            pg_roles._set_other_database_admission_fence(
                root.cursor(), databases=inventory["other_connectable_databases"]
            )
            root.commit()
            with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as concurrent:
                concurrent.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(drift_database)))
                concurrent.execute(
                    sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                        sql.Identifier(drift_database), sql.Identifier(drift_role)
                    )
                )
            active_parameters = conninfo_to_dict(fleet_db)
            active_parameters.update(dbname=drift_database, user=drift_role, password=password)
            active_connection = psycopg.connect(make_conninfo(**active_parameters), row_factory=dict_row)
            assert active_connection.execute("SELECT 1 AS alive").fetchone()["alive"] == 1
            with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as concurrent:
                concurrent.execute(
                    sql.SQL("ALTER DATABASE {} ALLOW_CONNECTIONS false").format(
                        sql.Identifier(drift_database)
                    )
                )

            pg_roles._lock_cluster_security_catalogs(root.cursor())
            with pytest.raises(pg_roles.CrossDatabaseInventoryDriftError, match="fence remains closed"):
                pg_roles._validate_other_database_inventory_unchanged(
                    root.cursor(),
                    baseline=inventory["other_connectable_databases"],
                    baseline_database_names=inventory["cluster_database_names_before"],
                    infrastructure_superuser_roles=topology.infrastructure_superuser_roles,
                )
            root.commit()
            closed = root.execute(
                "SELECT datallowconn,datconnlimit FROM pg_database WHERE datname=%s",
                (drift_database,),
            ).fetchone()
            assert closed == {"datallowconn": False, "datconnlimit": 0}
            with pytest.raises(psycopg.OperationalError):
                active_connection.execute("SELECT 1")
            with pytest.raises(psycopg.OperationalError):
                baseline_connection.execute("SELECT 1")
            assert root.execute(
                "SELECT count(*) AS count FROM pg_stat_activity WHERE datname=ANY(%s) AND usename=%s",
                ([drift_database, CROSS_DATABASE], drift_role),
            ).fetchone()["count"] == 0
    finally:
        if active_connection is not None:
            active_connection.close()
        if baseline_connection is not None:
            baseline_connection.close()
        with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cluster_admin:
            for database_name in (drift_database, CROSS_DATABASE):
                cluster_admin.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(database_name))
                )
            for role_name in (ATOMIC_RETIRED_ADMIN, drift_role):
                if cluster_admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    cluster_admin.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role_name)))
                    cluster_admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))


def test_bootstrap_isolates_other_database_and_rollback_restores_public_connect(
    fleet_db: str, monkeypatch
) -> None:
    topology = _atomic_topology()
    race_password = "applypilot-fence-race-password"
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cluster_admin:
        require_disposable_postgres(cluster_admin)
        cluster_admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(CROSS_DATABASE)))
        cluster_admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(RACE_DATABASE_ROLE), sql.Literal(race_password)
            )
        )
        cluster_admin.execute(
            sql.SQL("GRANT CONNECT, CREATE, TEMPORARY ON DATABASE {} TO {}").format(
                sql.Identifier(CROSS_DATABASE), sql.Identifier(RACE_DATABASE_ROLE)
            )
        )
    try:
        with pgqueue.connect(fleet_db) as root:
            root.execute(
                sql.SQL("CREATE ROLE {} LOGIN SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS").format(
                    sql.Identifier(ATOMIC_RETIRED_ADMIN)
                )
            )
            _create_atomic_lifecycle_roles(root)
            root.commit()
            original_terminate = pg_roles._terminate_and_validate_other_database_sessions
            fence_probed = False

            def probe_committed_fence(cur, *, databases, infrastructure_superuser_roles):
                nonlocal fence_probed
                if not fence_probed:
                    parameters = conninfo_to_dict(fleet_db)
                    parameters.update(
                        dbname=CROSS_DATABASE,
                        user=RACE_DATABASE_ROLE,
                        password=race_password,
                        connect_timeout="2",
                    )
                    with pytest.raises(psycopg.OperationalError):
                        psycopg.connect(make_conninfo(**parameters))
                    fence_probed = True
                return original_terminate(
                    cur,
                    databases=databases,
                    infrastructure_superuser_roles=infrastructure_superuser_roles,
                )

            monkeypatch.setattr(
                pg_roles,
                "_terminate_and_validate_other_database_sessions",
                probe_committed_fence,
            )
            receipt = pg_roles.bootstrap_database_roles(
                root,
                CONTROLLER_PASSWORD,
                BACKUP_PASSWORD,
                topology=topology,
                evidence_paths=_test_evidence_paths(),
            )
            assert fence_probed is True
            assert [row["database_name"] for row in receipt.inventory["other_connectable_databases"]] == [
                CROSS_DATABASE
            ]
            for role_name in (
                ATOMIC_CONTROLLER,
                "brain_schema_verifier",
                "brain_schema_migrator",
                *ATOMIC_LIFECYCLE_ROLES,
            ):
                assert (
                    root.execute(
                        "SELECT has_database_privilege(%s,%s,'CONNECT') AS allowed",
                        (role_name, CROSS_DATABASE),
                    ).fetchone()["allowed"]
                    is False
                )
            assert (
                root.execute(
                    "SELECT has_database_privilege('public',%s,'CONNECT') AS allowed",
                    (CROSS_DATABASE,),
                ).fetchone()["allowed"]
                is False
            )
            assert f'GRANT CONNECT ON DATABASE "{CROSS_DATABASE}" TO PUBLIC;' in receipt.rollback_sql
            assert f'GRANT CREATE ON DATABASE "{CROSS_DATABASE}" TO "{RACE_DATABASE_ROLE}";' in receipt.rollback_sql
            assert f'GRANT TEMPORARY ON DATABASE "{CROSS_DATABASE}" TO "{RACE_DATABASE_ROLE}";' in receipt.rollback_sql

            root.execute(receipt.rollback_sql)
            root.commit()
            assert (
                root.execute(
                    "SELECT has_database_privilege('public',%s,'CONNECT') AS allowed",
                    (CROSS_DATABASE,),
                ).fetchone()["allowed"]
                is True
            )
            assert root.execute(
                "SELECT has_database_privilege(%s,%s,'CONNECT,CREATE,TEMPORARY') AS allowed",
                (RACE_DATABASE_ROLE, CROSS_DATABASE),
            ).fetchone()["allowed"] is True
    finally:
        with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cluster_admin:
            cluster_admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(CROSS_DATABASE))
            )
            database_name = conninfo_to_dict(fleet_db)["dbname"]
            cluster_admin.execute(sql.SQL("ALTER DATABASE {} OWNER TO postgres").format(sql.Identifier(database_name)))
            pg_roles._transfer_application_ownership(cluster_admin.cursor(), new_owner_role="postgres")
            require_disposable_postgres(cluster_admin)
            for role_name in (
                *ATOMIC_LIFECYCLE_ROLES,
                ATOMIC_CONTROLLER,
                "brain_schema_verifier",
                "brain_schema_migrator",
                ATOMIC_OWNER,
                ATOMIC_RETIRED_ADMIN,
                RACE_DATABASE_ROLE,
            ):
                if cluster_admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    cluster_admin.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role_name)))
                    cluster_admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
            require_disposable_postgres(cluster_admin)


def test_pg18_other_database_acl_keeps_quoted_public_distinct_from_pseudo_public(
    fleet_db: str,
) -> None:
    database_name = "applypilot_public_identity_other_test"
    quoted_public = "PUBLIC"
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as root:
        require_disposable_postgres(root)
        if int(root.execute("SHOW server_version_num").fetchone()["server_version_num"]) // 10000 != 18:
            pytest.skip("cross-database PUBLIC identity coverage requires PostgreSQL 18")
        root.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                sql.Identifier(database_name)
            )
        )
        root.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(quoted_public)))
        root.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(quoted_public)))
        root.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
        root.execute(
            sql.SQL("GRANT CONNECT,TEMPORARY ON DATABASE {} TO PUBLIC").format(
                sql.Identifier(database_name)
            )
        )
        root.execute(
            sql.SQL("GRANT CONNECT,TEMPORARY ON DATABASE {} TO {}").format(
                sql.Identifier(database_name), sql.Identifier(quoted_public)
            )
        )
        try:
            inventory = pg_roles._other_database_inventory(
                root.cursor(), infrastructure_superuser_roles=("postgres",)
            )
            database = next(
                item for item in inventory if item["database_name"] == database_name
            )
            connect_grants = [
                grant
                for grant in database["acl_grants_before"]
                if grant["privilege"] == "CONNECT"
                and (grant["is_public"] or grant["grantee_name"] == quoted_public)
            ]
            quoted_public_oid = root.execute(
                "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s",
                (quoted_public,),
            ).fetchone()["oid"]
            assert {
                (grant["grantee_oid"], grant["is_public"], grant["grantee_name"])
                for grant in connect_grants
            } == {
                (0, True, None),
                (quoted_public_oid, False, quoted_public),
            }
            rollback_sql = pg_roles._other_database_connect_rollback_sql(
                root.cursor(),
                databases=(database,),
                infrastructure_superuser_roles=("postgres",),
            )
            assert " TO PUBLIC;" in rollback_sql
            assert f' TO "{quoted_public}";' in rollback_sql
            pg_roles._isolate_other_databases(
                root.cursor(),
                databases=(database,),
                infrastructure_superuser_roles=("postgres",),
            )
            remaining = root.execute(
                "SELECT acl.grantee AS grantee_oid,acl.grantee=0 AS is_public,"
                "grantee.rolname AS grantee_name FROM pg_catalog.pg_database database "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(database.datacl) acl "
                "LEFT JOIN pg_catalog.pg_roles grantee ON grantee.oid=acl.grantee "
                "WHERE database.datname=%s AND (acl.grantee=0 OR grantee.rolname=%s)",
                (database_name, quoted_public),
            ).fetchall()
            assert remaining == []
            root.execute(rollback_sql)
            restored = root.execute(
                "SELECT acl.grantee AS grantee_oid,acl.grantee=0 AS is_public,"
                "grantee.rolname AS grantee_name,acl.privilege_type "
                "FROM pg_catalog.pg_database database "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(database.datacl) acl "
                "LEFT JOIN pg_catalog.pg_roles grantee ON grantee.oid=acl.grantee "
                "WHERE database.datname=%s AND acl.privilege_type='CONNECT' "
                "AND (acl.grantee=0 OR grantee.rolname=%s)",
                (database_name, quoted_public),
            ).fetchall()
            assert {(row["is_public"], row["grantee_name"]) for row in restored} == {
                (True, None),
                (False, quoted_public),
            }
        finally:
            root.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(database_name)
                )
            )
            root.execute(
                sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(quoted_public))
            )
            root.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(quoted_public)))


def test_bootstrap_terminates_active_nonbreakglass_session_on_other_database(fleet_db: str) -> None:
    password = "active-other-database-password"
    topology = _atomic_topology()
    evidence_paths = _test_evidence_paths()
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cluster_admin:
        require_disposable_postgres(cluster_admin)
        cluster_admin.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS PASSWORD {}"
            ).format(sql.Identifier(ACTIVE_DATABASE_ROLE), sql.Literal(password))
        )
        cluster_admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(ACTIVE_DATABASE)))
        cluster_admin.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(ACTIVE_DATABASE), sql.Identifier(ACTIVE_DATABASE_ROLE)
            )
        )
    active_parameters = conninfo_to_dict(fleet_db)
    active_parameters.update(dbname=ACTIVE_DATABASE, user=ACTIVE_DATABASE_ROLE, password=password)
    active_connection = psycopg.connect(make_conninfo(**active_parameters), row_factory=dict_row)
    try:
        with pgqueue.connect(fleet_db) as root:
            root.execute(
                sql.SQL("CREATE ROLE {} LOGIN SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS").format(
                    sql.Identifier(ATOMIC_RETIRED_ADMIN)
                )
            )
            root.commit()
            receipt = pg_roles.bootstrap_database_roles(
                root,
                CONTROLLER_PASSWORD,
                BACKUP_PASSWORD,
                topology=topology,
                evidence_paths=evidence_paths,
            )
            with pytest.raises(psycopg.OperationalError):
                active_connection.execute("SELECT 1")
            assert root.execute(
                "SELECT count(*) AS count FROM pg_stat_activity WHERE datname=%s AND usename=%s",
                (ACTIVE_DATABASE, ACTIVE_DATABASE_ROLE),
            ).fetchone()["count"] == 0
            assert evidence_paths.preparation_receipt_path.exists()
            assert evidence_paths.rollback_sql_path.exists()
            prepared = json.loads(evidence_paths.preparation_receipt_path.read_text(encoding="utf-8"))
            pg_roles.verify_evidence_receipt(
                prepared,
                authentication_key=_EVIDENCE_KEY,
                expected_key_id=_EVIDENCE_KEY_ID,
            )
            assert prepared["status"] == "prepared_before_database_mutation"
            assert prepared["in_doubt"] is True
            assert prepared["rollback_sql_sha256"] == hashlib.sha256(
                evidence_paths.rollback_sql_path.read_bytes()
            ).hexdigest()
            root.execute(receipt.rollback_sql)
            root.commit()
    finally:
        active_connection.close()
        with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as cluster_admin:
            cluster_admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(ACTIVE_DATABASE))
            )
            require_disposable_postgres(cluster_admin)
            for role_name in (ACTIVE_DATABASE_ROLE, ATOMIC_RETIRED_ADMIN):
                if cluster_admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    cluster_admin.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role_name)))
                    cluster_admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
            require_disposable_postgres(cluster_admin)


def test_other_database_validation_catches_inherited_connect_for_arbitrary_login(fleet_db: str) -> None:
    parent_role = "applypilot_inherited_connect_parent_test"
    login_role = "applypilot_inherited_connect_login_test"
    topology = _atomic_topology()
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as root:
        require_disposable_postgres(root)
        root.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(CROSS_DATABASE)))
        root.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(parent_role)))
        root.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(login_role)))
        root.execute(sql.SQL("GRANT {} TO {}").format(sql.Identifier(parent_role), sql.Identifier(login_role)))
        root.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(CROSS_DATABASE), sql.Identifier(parent_role)
            )
        )
        baseline = pg_roles._other_database_inventory(
            root.cursor(), infrastructure_superuser_roles=("postgres",)
        )
        with pytest.raises(RuntimeError, match="effective CONNECT"):
            pg_roles._validate_other_database_isolation(
                root.cursor(), databases=baseline, topology=topology
            )
        root.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(CROSS_DATABASE)))
        root.execute(sql.SQL("REVOKE {} FROM {}").format(sql.Identifier(parent_role), sql.Identifier(login_role)))
        root.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(login_role)))
        root.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(parent_role)))


def test_bootstrap_closes_and_rollback_restores_retired_admin_membership_edges(fleet_db: str) -> None:
    parent_role = "applypilot_retired_parent_test"
    member_role = "applypilot_retired_member_test"
    topology = _atomic_topology()
    with pgqueue.connect(fleet_db) as root:
        require_disposable_postgres(root)
        root.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(parent_role)))
        root.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(member_role)))
        root.execute(
            sql.SQL("CREATE ROLE {} LOGIN SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS").format(
                sql.Identifier(ATOMIC_RETIRED_ADMIN)
            )
        )
        root.execute(
            sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(ATOMIC_RETIRED_ADMIN), sql.Identifier(member_role)
            )
        )
        root.execute(
            sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(parent_role), sql.Identifier(ATOMIC_RETIRED_ADMIN)
            )
        )
        root.commit()
        before = pg_roles._retired_membership_inventory(
            root.cursor(), retired_admin_roles=(ATOMIC_RETIRED_ADMIN,)
        )
        root.commit()
        try:
            receipt = pg_roles.bootstrap_database_roles(
                root,
                CONTROLLER_PASSWORD,
                BACKUP_PASSWORD,
                topology=topology,
                evidence_paths=_test_evidence_paths(),
            )
            assert pg_roles._retired_membership_inventory(
                root.cursor(), retired_admin_roles=(ATOMIC_RETIRED_ADMIN,)
            ) == ()
            root.execute(receipt.rollback_sql)
            root.commit()
            assert pg_roles._retired_membership_inventory(
                root.cursor(), retired_admin_roles=(ATOMIC_RETIRED_ADMIN,)
            ) == before
        finally:
            root.rollback()
            require_disposable_postgres(root)
            for parent, member in (
                (ATOMIC_RETIRED_ADMIN, member_role),
                (parent_role, ATOMIC_RETIRED_ADMIN),
            ):
                root.execute(
                    sql.SQL("REVOKE {} FROM {}").format(sql.Identifier(parent), sql.Identifier(member))
                )
            for role_name in (
                ATOMIC_CONTROLLER,
                "brain_schema_verifier",
                "brain_schema_migrator",
                ATOMIC_OWNER,
                ATOMIC_RETIRED_ADMIN,
                member_role,
                parent_role,
            ):
                if root.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    root.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
            root.commit()


def test_retired_membership_rollback_preserves_grantor_and_supported_options(fleet_db: str) -> None:
    parent_role = "applypilot_retired_option_parent_test"
    grantor_role = "applypilot_retired_option_grantor_test"
    topology = _atomic_topology()
    with pgqueue.connect(fleet_db) as root:
        require_disposable_postgres(root)
        root.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(parent_role)))
        root.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(grantor_role)))
        root.execute(
            sql.SQL("GRANT {} TO {} WITH ADMIN OPTION").format(
                sql.Identifier(parent_role), sql.Identifier(grantor_role)
            )
        )
        root.execute(
            sql.SQL("CREATE ROLE {} LOGIN SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS").format(
                sql.Identifier(ATOMIC_RETIRED_ADMIN)
            )
        )
        root.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(grantor_role)))
        root.execute(
            sql.SQL("GRANT {} TO {} WITH ADMIN TRUE").format(
                sql.Identifier(parent_role), sql.Identifier(ATOMIC_RETIRED_ADMIN)
            )
        )
        supports_membership_options = root.execute(
            "SELECT count(*)=2 AS supported FROM pg_attribute "
            "WHERE attrelid='pg_catalog.pg_auth_members'::regclass "
            "AND attname=ANY(%s) AND NOT attisdropped",
            (["inherit_option", "set_option"],),
        ).fetchone()["supported"]
        if supports_membership_options:
            root.execute(
                sql.SQL("GRANT {} TO {} WITH INHERIT FALSE").format(
                    sql.Identifier(parent_role), sql.Identifier(ATOMIC_RETIRED_ADMIN)
                )
            )
            root.execute(
                sql.SQL("GRANT {} TO {} WITH SET FALSE").format(
                    sql.Identifier(parent_role), sql.Identifier(ATOMIC_RETIRED_ADMIN)
                )
            )
        root.execute("RESET ROLE")
        root.commit()
        before = pg_roles._retired_membership_inventory(
            root.cursor(), retired_admin_roles=(ATOMIC_RETIRED_ADMIN,)
        )
        root.commit()
        try:
            receipt = pg_roles.bootstrap_database_roles(
                root,
                CONTROLLER_PASSWORD,
                BACKUP_PASSWORD,
                topology=topology,
                evidence_paths=_test_evidence_paths(),
            )
            assert pg_roles._retired_membership_inventory(
                root.cursor(), retired_admin_roles=(ATOMIC_RETIRED_ADMIN,)
            ) == ()
            root.execute(receipt.rollback_sql)
            root.commit()
            assert pg_roles._retired_membership_inventory(
                root.cursor(), retired_admin_roles=(ATOMIC_RETIRED_ADMIN,)
            ) == before
            edge = next(item for item in before if item["member_role"] == ATOMIC_RETIRED_ADMIN)
            assert edge["grantor_role"] == grantor_role
            assert edge["admin_option"] is True
            if supports_membership_options:
                assert edge["inherit_option"] is False
                assert edge["set_option"] is False
        finally:
            root.rollback()
            require_disposable_postgres(root)
            database_name = conninfo_to_dict(fleet_db)["dbname"]
            root.execute(sql.SQL("ALTER DATABASE {} OWNER TO postgres").format(sql.Identifier(database_name)))
            pg_roles._transfer_application_ownership(root.cursor(), new_owner_role="postgres")
            for role_name in (
                ATOMIC_CONTROLLER,
                "brain_schema_verifier",
                "brain_schema_migrator",
                ATOMIC_OWNER,
                ATOMIC_RETIRED_ADMIN,
                grantor_role,
                parent_role,
            ):
                if root.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    root.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role_name)))
                    root.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
            root.commit()


def test_retired_membership_closure_fails_without_cascading_dependent_grants(fleet_db: str) -> None:
    parent_role = "applypilot_retired_dependency_parent_test"
    downstream_role = "applypilot_retired_dependency_downstream_test"
    topology = _atomic_topology()
    with pgqueue.connect(fleet_db) as root:
        require_disposable_postgres(root)
        root.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(parent_role)))
        root.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(downstream_role)))
        root.execute(
            sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB CREATEROLE NOREPLICATION NOBYPASSRLS").format(
                sql.Identifier(ATOMIC_RETIRED_ADMIN)
            )
        )
        root.execute(
            sql.SQL("GRANT {} TO {} WITH ADMIN OPTION").format(
                sql.Identifier(parent_role), sql.Identifier(ATOMIC_RETIRED_ADMIN)
            )
        )
        root.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(ATOMIC_RETIRED_ADMIN)))
        root.execute(
            sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(parent_role), sql.Identifier(downstream_role)
            )
        )
        root.execute("RESET ROLE")
        root.commit()
        try:
            with pytest.raises(RuntimeError, match="dependencies prevent exact non-cascading closure"):
                pg_roles.bootstrap_database_roles(
                    root,
                    CONTROLLER_PASSWORD,
                    BACKUP_PASSWORD,
                    topology=topology,
                    evidence_paths=_test_evidence_paths(),
                )
            edges = root.execute(
                "SELECT parent.rolname AS parent_role,member.rolname AS member_role,grantor.rolname AS grantor_role "
                "FROM pg_auth_members membership "
                "JOIN pg_roles parent ON parent.oid=membership.roleid "
                "JOIN pg_roles member ON member.oid=membership.member "
                "JOIN pg_roles grantor ON grantor.oid=membership.grantor "
                "WHERE parent.rolname=%s ORDER BY member.rolname",
                (parent_role,),
            ).fetchall()
            assert edges == [
                {
                    "parent_role": parent_role,
                    "member_role": downstream_role,
                    "grantor_role": ATOMIC_RETIRED_ADMIN,
                },
                {
                    "parent_role": parent_role,
                    "member_role": ATOMIC_RETIRED_ADMIN,
                    "grantor_role": "postgres",
                },
            ]
        finally:
            root.rollback()
            require_disposable_postgres(root)
            root.execute(
                sql.SQL("REVOKE {} FROM {} GRANTED BY {}").format(
                    sql.Identifier(parent_role),
                    sql.Identifier(downstream_role),
                    sql.Identifier(ATOMIC_RETIRED_ADMIN),
                )
            )
            root.execute(
                sql.SQL("REVOKE {} FROM {} GRANTED BY postgres").format(
                    sql.Identifier(parent_role),
                    sql.Identifier(ATOMIC_RETIRED_ADMIN),
                )
            )
            for role_name in (ATOMIC_RETIRED_ADMIN, downstream_role, parent_role):
                if root.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    root.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
            root.commit()


def test_live_bootstrap_hands_off_to_non_superuser_controller_and_reconciles(fleet_db: str, tmp_path: Path) -> None:
    rollback_path = tmp_path / "bootstrap-rollback.sql"
    receipt_path = tmp_path / "bootstrap-receipt.json"
    with pgqueue.connect(fleet_db) as root:
        root.execute(
            sql.SQL("CREATE ROLE {} LOGIN SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS").format(
                sql.Identifier(RETIRED_ADMIN)
            )
        )
        root.execute(
            "INSERT INTO public.workers(worker_id,machine_owner,public_ip,validated,capabilities) "
            "VALUES(%s,'bootstrap','1.1.1.1',TRUE,'{}'::jsonb)",
            (WORKER_ID,),
        )
        root.execute(
            "INSERT INTO public.worker_heartbeat(worker_id,machine_owner,home_ip,role,state,sw_version,last_beat) "
            "VALUES(%s,'bootstrap','1.1.1.1','compute','idle','v1',now())",
            (WORKER_ID,),
        )
        root.commit()

        try:
            receipt = pg_roles.bootstrap_database_roles(
                root,
                CONTROLLER_PASSWORD,
                BACKUP_PASSWORD,
                topology=pg_roles.BootstrapTopology(
                    database_owner_role=OWNER,
                    controller_role=CONTROLLER,
                    verifier_role=VERIFIER,
                    migrator_role=MIGRATOR,
                    retired_admin_roles=(RETIRED_ADMIN,),
                    infrastructure_superuser_roles=("postgres",),
                ),
                evidence_paths=pg_roles.DurableEvidencePaths(
                    preparation_receipt_path=receipt_path.resolve(),
                    rollback_sql_path=rollback_path.resolve(),
                    authentication_key=_EVIDENCE_KEY,
                    authentication_key_id=_EVIDENCE_KEY_ID,
                ),
            )
            assert receipt.escalation_required is False
            assert rollback_path.stat().st_size > 0
            assert receipt_path.stat().st_size > 0
            postgres_effective = next(
                row for row in receipt.effective_connect_grantees if row["role_name"] == "postgres"
            )
            assert postgres_effective["reconnect_capable"] is True
            assert postgres_effective["superuser"] is True

            controller_dsn = _dsn(fleet_db, user=CONTROLLER, password=CONTROLLER_PASSWORD)
            with psycopg.connect(controller_dsn, row_factory=dict_row) as controller:
                attributes = controller.execute(
                    "SELECT rolinherit,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
                    "FROM pg_roles WHERE rolname=%s",
                    (CONTROLLER,),
                ).fetchone()
                assert attributes == {
                    "rolinherit": False,
                    "rolsuper": False,
                    "rolcreatedb": False,
                    "rolcreaterole": False,
                    "rolreplication": False,
                    "rolbypassrls": False,
                }
                memberships = controller.execute(
                    "SELECT parent.rolname FROM pg_auth_members membership "
                    "JOIN pg_roles parent ON parent.oid=membership.roleid "
                    "JOIN pg_roles member ON member.oid=membership.member "
                    "WHERE member.rolname=%s ORDER BY parent.rolname",
                    (CONTROLLER,),
                ).fetchall()
                assert {row["rolname"] for row in memberships}.isdisjoint({OWNER, MIGRATOR})
                retired = controller.execute(
                    "SELECT rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
                    "FROM pg_roles WHERE rolname=%s",
                    (RETIRED_ADMIN,),
                ).fetchone()
                assert not any(retired.values())

            worker_receipt = pg_roles.ensure_fleet_worker_role(
                root,
                WORKER_PASSWORD,
                role=WORKER_ROLE,
                worker_id=WORKER_ID,
                contract="compute",
                regrant_manifest=pg_roles.RegrantManifest(
                    database_owner_role=OWNER,
                    controller_roles=(CONTROLLER,),
                    verifier_roles=(VERIFIER,),
                    retired_admin_roles=(RETIRED_ADMIN,),
                    migrator_role=MIGRATOR,
                    infrastructure_superuser_roles=("postgres",),
                    expected_service_roles=(),
                ),
                evidence_paths=_test_evidence_paths(),
            )
            assert worker_receipt.contract == "compute"
            worker_dsn = _dsn(fleet_db, user=WORKER_ROLE, password=WORKER_PASSWORD)
            with psycopg.connect(worker_dsn, row_factory=dict_row) as worker:
                identity = worker.execute(
                    "SELECT public.fleet_worker_identity() AS identity"
                ).fetchone()["identity"]
                assert identity["worker_id"] == WORKER_ID
                assert identity["contract"] == "compute"

            root.execute(worker_receipt.rollback_sql)
            root.commit()
            root.execute(receipt.rollback_sql)
            root.commit()
            root.execute(receipt.rollback_sql)
            root.commit()
            database_owner = root.execute(
                "SELECT owner.rolname FROM pg_database d JOIN pg_roles owner ON owner.oid=d.datdba "
                "WHERE d.datname=current_database()"
            ).fetchone()["rolname"]
            assert database_owner == "postgres"
            for role_name in (CONTROLLER, VERIFIER, MIGRATOR, OWNER):
                assert root.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone() is None
            restored = root.execute(
                "SELECT rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
                "FROM pg_roles WHERE rolname=%s",
                (RETIRED_ADMIN,),
            ).fetchone()
            assert all(restored.values())
        finally:
            root.rollback()
            require_disposable_postgres(root)
            root.execute(
                sql.SQL("ALTER DATABASE {} OWNER TO postgres").format(
                    sql.Identifier(conninfo_to_dict(fleet_db)["dbname"])
                )
            )
            pg_roles._transfer_application_ownership(root.cursor(), new_owner_role="postgres")
            root.execute("DELETE FROM public.fleet_worker_principals WHERE role_name=%s", (WORKER_ROLE,))
            for role_name in (
                WORKER_ROLE,
                pg_roles.FLEET_BACKUP_ROLE,
                CONTROLLER,
                VERIFIER,
                MIGRATOR,
                OWNER,
                RETIRED_ADMIN,
            ):
                if root.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    root.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role_name)))
                    root.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
            root.commit()
