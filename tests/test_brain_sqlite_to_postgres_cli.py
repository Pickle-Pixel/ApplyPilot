from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pytest
from psycopg import sql as pg_sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from applypilot.brain.importer import SOURCE_TABLES, SealedSnapshotReceipt, SourceFileAudit
from applypilot.brain.sqlite_to_postgres import ImportSummary


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "brain-sqlite-to-postgres.py"
RUNBOOK = REPO / "docs" / "canonical-decision-policy-runbook.md"
NONCE = "n" * 32
DSN = "host=database.invalid port=5432 dbname=brain user=migrator password=top-secret-password"
FREEZE_KEY = b"f" * 32
IMPORT_KEY = b"i" * 32
FREEZE_KEY_ID = "writer-freeze-key-2026"
IMPORT_KEY_ID = "brain-import-key-2026"
SYSTEM_IDENTIFIER = "7460123456789012345"
DATABASE_OID = "16384"
INCARNATION = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"


def _load_cli():
    specification = importlib.util.spec_from_file_location("brain_sqlite_to_postgres_cli", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class _Result:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


_LOGIN_ROLE = "canonical_import_login"
# The durable identity read is a pure SELECT, so it runs under the least-privileged role V8
# grants that SELECT to -- the read-only ``brain_schema_verifier`` -- never the table owner.
_IDENTITY_ROLE = "brain_schema_verifier"
_DEFAULT_SEARCH_PATH = "\"$user\", public"


def test_runbook_provisions_canonical_import_login_for_read_only_identity_role() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    assert "GRANT brain_schema_verifier TO canonical_import_login" in runbook
    assert "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE" in runbook
    assert "SET ROLE brain_schema_verifier;" in runbook
    assert "SET ROLE brain_schema_migrator;" not in runbook


class _Connection:
    def __init__(
        self,
        *,
        database: str = "brain",
        system_identifier: str = SYSTEM_IDENTIFIER,
        database_oid: str = DATABASE_OID,
        database_incarnation_id: str | None = INCARNATION,
        identity_present: bool = True,
        identity_row_count: int = 1,
        incarnation_high: str | None = None,
    ) -> None:
        self.identity = {
            "database": database,
            "system_identifier": system_identifier,
            "database_oid": database_oid,
            "database_incarnation_id": database_incarnation_id,
            "server_version_num": "170005",
        }
        # The durable singleton is read directly from applypilot_database_identity; these knobs
        # model its presence, row count, and (for the ambiguous case) a divergent high value.
        self.identity_present = identity_present
        self.identity_row_count = identity_row_count
        self.incarnation_high = incarnation_high
        self.statements: list[str] = []
        # Session role/search_path state so tests can prove the singleton read is scoped to the
        # least-privilege role and that neither the role nor the search_path leaks afterwards.
        self.role = _LOGIN_ROLE
        self.search_path = _DEFAULT_SEARCH_PATH
        self.rollbacks = 0
        self.commits = 0
        # The privileged singleton SELECT is only visible when the verifier role is active.
        self.identity_readable_only_as_role: str | None = _IDENTITY_ROLE

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def rollback(self):
        self.rollbacks += 1

    def commit(self):
        self.commits += 1

    def execute(self, query: str, params=()):
        normalized = " ".join(query.split())
        self.statements.append(normalized)
        upper = normalized.upper()
        if upper.startswith("SET ROLE "):
            self.role = normalized.split(None, 2)[2].strip().strip(";").strip('"')
            return _Result(None)
        if upper.startswith("RESET ROLE"):
            self.role = _LOGIN_ROLE
            return _Result(None)
        if upper.startswith("SET SEARCH_PATH"):
            self.search_path = normalized.split("TO", 1)[-1].strip() if "TO" in upper else normalized
            return _Result(None)
        if upper.startswith("RESET SEARCH_PATH"):
            self.search_path = _DEFAULT_SEARCH_PATH
            return _Result(None)
        if "AS CURRENT_USER" in upper:
            return _Result({"current_user": self.role})
        if "identity_row_count" in normalized:
            if (
                self.identity_readable_only_as_role is not None
                and self.role != self.identity_readable_only_as_role
            ):
                raise psycopg.errors.InsufficientPrivilege(
                    "permission denied for table applypilot_database_identity"
                )
            low = self.identity["database_incarnation_id"]
            high = self.incarnation_high if self.incarnation_high is not None else low
            return _Result(
                {
                    "identity_row_count": self.identity_row_count,
                    "incarnation_low": low,
                    "incarnation_high": high,
                }
            )
        return _Result(
            {
                "database": self.identity["database"],
                "system_identifier": self.identity["system_identifier"],
                "database_oid": self.identity["database_oid"],
                "identity_present": self.identity_present,
                "server_version_num": self.identity["server_version_num"],
            }
        )


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _file_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "sha256": None, "size": None, "identity": None}
    return {
        "exists": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
        "identity": [str(item) for item in _file_identity(path)],
    }


def _audit(path: Path, *, ephemeral: bool = False) -> SourceFileAudit:
    state = _file_state(path)
    identity = None if state["identity"] is None else tuple(int(item) for item in state["identity"])
    return SourceFileAudit(
        path=str(path),
        before_exists=bool(state["exists"]),
        after_exists=bool(state["exists"]),
        before_sha256=state["sha256"],
        after_sha256=state["sha256"],
        before_size=state["size"],
        after_size=state["size"],
        before_mtime_ns=None if identity is None else identity[3],
        after_mtime_ns=None if identity is None else identity[3],
        before_stat_identity=identity,
        after_stat_identity=identity,
        changed=False,
        observation_complete=True,
        ephemeral=ephemeral,
    )


def _sealed_receipt(source: Path, snapshot: Path) -> SealedSnapshotReceipt:
    if not snapshot.exists():
        snapshot.write_bytes(b"sealed-sqlite")
    return SealedSnapshotReceipt(
        version=1,
        path=str(snapshot),
        source_path=str(source),
        sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        size=snapshot.stat().st_size,
        quick_check="ok",
        source_mode="online_backup",
        source_db_audit=_audit(source),
        source_wal_audit=_audit(Path(f"{source}-wal")),
        source_shm_audit=_audit(Path(f"{source}-shm"), ephemeral=True),
        source_changed_during_backup=False,
    )


def _sign(document: dict[str, object], key: bytes, key_id: str) -> dict[str, object]:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    signature = base64.b64encode(hmac.digest(key, payload, hashlib.sha256)).decode("ascii")
    return {
        **document,
        "authentication": {"algorithm": "HMAC-SHA256", "keyId": key_id, "signature": signature},
    }


def _write_freeze_marker(
    path: Path,
    source: Path,
    *,
    key: bytes = FREEZE_KEY,
    key_id: str = FREEZE_KEY_ID,
    frozen_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> Path:
    now = datetime.now(timezone.utc)
    document = {
        "schema": "applypilot.writer-freeze.v2",
        "purpose": "writer-freeze",
        "releaseId": "release-2026-07-18",
        "releaseNonce": NONCE,
        "sourcePath": str(source),
        "sourceState": {
            "database": _file_state(source),
            "wal": _file_state(Path(f"{source}-wal")),
            "shm": _file_state(Path(f"{source}-shm")),
        },
        "writerProcessCount": 0,
        "activeWriterLeaseCount": 0,
        "frozenAt": (frozen_at or now - timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
        "expiresAt": (expires_at or now + timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
    }
    path.write_text(json.dumps(_sign(document, key, key_id)), encoding="utf-8")
    return path


def _parity(count: int = 0) -> dict[str, dict[str, object]]:
    return {
        table.name: {
            "source_count": count if table.name == "jobs" else 0,
            "target_count": count if table.name == "jobs" else 0,
            "source_hash": "a" * 64,
            "target_hash": "a" * 64,
            "mismatch_count": 0,
            "unresolved_count": 0,
            "passed": True,
        }
        for table in SOURCE_TABLES
    }


def _arguments(
    tmp_path: Path,
    *,
    mode: str = "bulk-import",
    output: str = "receipt.json",
    expected_database: str = "brain",
    expected_system_identifier: str = SYSTEM_IDENTIFIER,
    expected_database_oid: str = DATABASE_OID,
    expected_database_incarnation_id: str | None = INCARNATION,
) -> list[str]:
    arguments = [
        "--source",
        str(tmp_path / "canonical.sqlite"),
        "--sealed-snapshot",
        str(tmp_path / "sealed.sqlite"),
        "--postgres-dsn-env",
        "BRAIN_IMPORT_DSN",
        "--expected-database",
        expected_database,
        "--expected-system-identifier",
        expected_system_identifier,
        "--expected-database-oid",
        expected_database_oid,
        "--mode",
        mode,
        "--release-id",
        "release-2026-07-18",
        "--release-nonce",
        NONCE,
        "--output",
        str(tmp_path / output),
    ]
    if expected_database_incarnation_id is not None:
        arguments += ["--expected-database-incarnation-id", expected_database_incarnation_id]
    return arguments


def _prepare(monkeypatch, tmp_path: Path, *, connection: _Connection | None = None):
    cli = _load_cli()
    source = tmp_path / "canonical.sqlite"
    with closing(sqlite3.connect(source)) as sqlite:
        sqlite.execute("CREATE TABLE state(value TEXT NOT NULL)")
        sqlite.execute("INSERT INTO state VALUES ('canonical')")
        sqlite.commit()
    snapshot = tmp_path / "sealed.sqlite"
    sealed = _sealed_receipt(source, snapshot)
    marker = _write_freeze_marker(tmp_path / "writer-freeze.json", source)
    monkeypatch.setenv("BRAIN_IMPORT_DSN", DSN)
    monkeypatch.setenv("APPLYPILOT_WRITER_FREEZE_ATTESTATION_KEY_B64", base64.b64encode(FREEZE_KEY).decode())
    monkeypatch.setenv("APPLYPILOT_WRITER_FREEZE_ATTESTATION_KEY_ID", FREEZE_KEY_ID)
    monkeypatch.setenv("APPLYPILOT_BRAIN_IMPORT_ATTESTATION_KEY_B64", base64.b64encode(IMPORT_KEY).decode())
    monkeypatch.setenv("APPLYPILOT_BRAIN_IMPORT_ATTESTATION_KEY_ID", IMPORT_KEY_ID)
    monkeypatch.setattr(cli, "seal_sqlite_snapshot", lambda *_args, **_kwargs: sealed)
    monkeypatch.setattr(cli, "_connect_postgres", lambda *_args, **_kwargs: connection or _Connection())
    return cli, source, snapshot, sealed, marker


def _final_arguments(tmp_path: Path, marker: Path, *extra: str) -> list[str]:
    return [
        *_arguments(tmp_path, mode="final-delta-finalize"),
        "--writer-freeze-marker",
        str(marker),
        *extra,
    ]


def test_bulk_import_writes_authenticated_secret_free_receipt(monkeypatch, tmp_path: Path) -> None:
    cli, _source, _snapshot, sealed, _marker = _prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli,
        "import_sqlite_to_postgres",
        lambda *_args, **_kwargs: ImportSummary(41, sealed.sha256, {"jobs": 12}, {"labels": 2}),
    )

    receipt_path = cli.execute(_arguments(tmp_path))

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    cli.verify_receipt(receipt, key=IMPORT_KEY, expected_key_id=IMPORT_KEY_ID, label="brain import receipt")
    assert receipt["purpose"] == "brain-import"
    assert receipt["destination"]["systemIdentifier"] == SYSTEM_IDENTIFIER
    assert receipt["result"]["importedCounts"] == {"jobs": 12}
    assert receipt["promotable"] is False
    encoded = receipt_path.read_text(encoding="utf-8")
    assert DSN not in encoded
    assert "top-secret-password" not in encoded


def test_final_receipt_contains_durable_per_table_parity_hashes(monkeypatch, tmp_path: Path) -> None:
    cli, _source, _snapshot, sealed, marker = _prepare(monkeypatch, tmp_path)
    parity = _parity(13)
    monkeypatch.setattr(
        cli,
        "import_sqlite_to_postgres",
        lambda *_args, **_kwargs: ImportSummary(52, sealed.sha256, {"jobs": 1}, {}),
    )
    monkeypatch.setattr(
        cli,
        "finalize_sqlite_to_postgres_import",
        lambda *_args, **_kwargs: ImportSummary(
            52,
            sealed.sha256,
            {name: int(record["target_count"]) for name, record in parity.items()},
            {},
            finalized=True,
            parity=parity,
            terminal_event_id=152,
        ),
    )

    receipt_path = cli.execute(_final_arguments(tmp_path, marker))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["promotable"] is False
    assert receipt["promotionEligible"] is True
    assert receipt["result"]["terminalEventId"] == 152
    assert receipt["result"]["parity"]["jobs"]["source_hash"] == "a" * 64
    assert receipt["result"]["parity"]["jobs"]["target_hash"] == "a" * 64
    cli.verify_receipt(receipt, key=IMPORT_KEY, expected_key_id=IMPORT_KEY_ID, label="brain import receipt")
    cli.verify_consumable_import_receipt(receipt_path)
    commit = json.loads(cli._publication_commit_path(receipt_path).read_text(encoding="utf-8"))
    assert commit["promotable"] is True


def test_forged_and_expired_freeze_receipts_are_rejected(monkeypatch, tmp_path: Path) -> None:
    cli, source, _snapshot, _sealed, marker = _prepare(monkeypatch, tmp_path)
    forged = json.loads(marker.read_text(encoding="utf-8"))
    forged["writerProcessCount"] = 1
    marker.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(cli.CLIError, match="authentication signature is invalid"):
        cli.execute(_final_arguments(tmp_path, marker))

    _write_freeze_marker(
        marker,
        source,
        frozen_at=datetime.now(timezone.utc) - timedelta(minutes=20),
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    with pytest.raises(cli.CLIError, match="expired|freshness"):
        cli.execute(_final_arguments(tmp_path, marker))


def test_freeze_rejects_old_but_unexpired_and_overlong_windows(monkeypatch, tmp_path: Path) -> None:
    cli, source, _snapshot, _sealed, marker = _prepare(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    _write_freeze_marker(
        marker,
        source,
        frozen_at=now - timedelta(seconds=cli.MAX_FREEZE_AGE_SECONDS + 5),
        expires_at=now + timedelta(minutes=1),
    )
    with pytest.raises(cli.CLIError, match="maximum age"):
        cli.execute(_final_arguments(tmp_path, marker))

    _write_freeze_marker(
        marker,
        source,
        frozen_at=now - timedelta(seconds=1),
        expires_at=now + timedelta(seconds=cli.MAX_FREEZE_VALIDITY_SECONDS + 5),
    )
    with pytest.raises(cli.CLIError, match="validity duration"):
        cli.execute(_final_arguments(tmp_path, marker))


@pytest.mark.parametrize("advance_stage", ["before-finalization", "before-promotion-commit"])
def test_freeze_expiry_mid_run_never_publishes_consumable_evidence(
    monkeypatch,
    tmp_path: Path,
    advance_stage: str,
) -> None:
    cli, source, _snapshot, sealed, marker = _prepare(monkeypatch, tmp_path)
    started = datetime.now(timezone.utc)
    expires = started + timedelta(seconds=30)
    _write_freeze_marker(marker, source, frozen_at=started - timedelta(seconds=1), expires_at=expires)
    clock = {"now": started}
    monkeypatch.setattr(cli, "_utc_now", lambda: clock["now"])
    parity = _parity()

    def bulk(*_args, **_kwargs):
        if advance_stage == "before-finalization":
            clock["now"] = expires + timedelta(seconds=1)
        return ImportSummary(53, sealed.sha256, {}, {})

    def finalize(*_args, **_kwargs):
        if advance_stage == "before-promotion-commit":
            clock["now"] = expires + timedelta(seconds=1)
        return ImportSummary(53, sealed.sha256, {name: 0 for name in parity}, {}, True, parity, 153)

    monkeypatch.setattr(cli, "import_sqlite_to_postgres", bulk)
    monkeypatch.setattr(cli, "finalize_sqlite_to_postgres_import", finalize)

    with pytest.raises(cli.CLIError, match="expired|freshness"):
        cli.execute(_final_arguments(tmp_path, marker))
    assert not cli._publication_commit_path(tmp_path / "receipt.json").exists()


def test_freeze_requires_zero_writers_and_distinct_purpose_key(monkeypatch, tmp_path: Path) -> None:
    cli, source, _snapshot, _sealed, marker = _prepare(monkeypatch, tmp_path)
    document = json.loads(marker.read_text(encoding="utf-8"))
    document.pop("authentication")
    document["activeWriterLeaseCount"] = 1
    marker.write_text(json.dumps(_sign(document, FREEZE_KEY, FREEZE_KEY_ID)), encoding="utf-8")
    with pytest.raises(cli.CLIError, match="writer and lease counts must both be zero"):
        cli.execute(_final_arguments(tmp_path, marker))

    _write_freeze_marker(marker, source)
    monkeypatch.setenv("APPLYPILOT_BRAIN_IMPORT_ATTESTATION_KEY_B64", base64.b64encode(FREEZE_KEY).decode())
    monkeypatch.setenv("APPLYPILOT_BRAIN_IMPORT_ATTESTATION_KEY_ID", FREEZE_KEY_ID)
    with pytest.raises(cli.CLIError, match="distinct keys"):
        cli.execute(_final_arguments(tmp_path, marker))


@pytest.mark.parametrize("mutated", ["database", "wal", "shm"])
def test_last_publication_window_source_mutation_removes_receipt(monkeypatch, tmp_path: Path, mutated: str) -> None:
    cli, source, _snapshot, sealed, marker = _prepare(monkeypatch, tmp_path)
    parity = _parity()
    monkeypatch.setattr(
        cli,
        "import_sqlite_to_postgres",
        lambda *_args, **_kwargs: ImportSummary(62, sealed.sha256, {}, {}),
    )
    monkeypatch.setattr(
        cli,
        "finalize_sqlite_to_postgres_import",
        lambda *_args, **_kwargs: ImportSummary(
            62, sealed.sha256, {name: 0 for name in parity}, {}, True, parity, 162
        ),
    )
    original_publish = cli.atomic_write_no_overwrite

    def mutate_then_publish(path, content, **kwargs):
        target = source if mutated == "database" else Path(f"{source}-{mutated}")
        with target.open("ab") as handle:
            handle.write(b"late-mutation")
        original_publish(path, content, **kwargs)

    monkeypatch.setattr(cli, "atomic_write_no_overwrite", mutate_then_publish)

    with pytest.raises(cli.CLIError, match="changed across the final delta"):
        cli.execute(_final_arguments(tmp_path, marker))

    receipt_path = tmp_path / "receipt.json"
    assert not cli._publication_commit_path(receipt_path).exists()
    if receipt_path.exists():
        with pytest.raises(cli.CLIError, match="publication commit"):
            cli.verify_consumable_import_receipt(receipt_path)


def test_output_snapshot_source_and_marker_must_be_pairwise_distinct(monkeypatch, tmp_path: Path) -> None:
    cli, _source, snapshot, _sealed, marker = _prepare(monkeypatch, tmp_path)
    arguments = _final_arguments(tmp_path, marker)
    arguments[arguments.index("--output") + 1] = str(snapshot)

    with pytest.raises(cli.CLIError, match="pairwise distinct"):
        cli.execute(arguments)


@pytest.mark.parametrize(
    ("database", "system_identifier", "message"),
    [("wrong", SYSTEM_IDENTIFIER, "database name"), ("brain", "999", "system identifier")],
)
def test_wrong_destination_binding_fails_before_import(
    monkeypatch,
    tmp_path: Path,
    database: str,
    system_identifier: str,
    message: str,
) -> None:
    connection = _Connection(database=database, system_identifier=system_identifier)
    cli, _source, _snapshot, _sealed, _marker = _prepare(monkeypatch, tmp_path, connection=connection)
    monkeypatch.setattr(
        cli,
        "import_sqlite_to_postgres",
        lambda *_args, **_kwargs: pytest.fail("wrong destination must fail before import"),
    )

    with pytest.raises(cli.CLIError, match=message):
        cli.execute(_arguments(tmp_path))


def test_database_oid_and_incarnation_are_enforced(monkeypatch, tmp_path: Path) -> None:
    connection = _Connection(database_oid="999")
    cli, _source, _snapshot, _sealed, _marker = _prepare(monkeypatch, tmp_path, connection=connection)

    # A wrong OID fails before any incarnation binding is considered.
    with pytest.raises(cli.CLIError, match="database OID"):
        cli.execute(_arguments(tmp_path))

    # A durable singleton incarnation that differs from the expected UUID fails closed.
    connection.identity["database_oid"] = DATABASE_OID
    connection.identity["database_incarnation_id"] = "00000000-0000-0000-0000-000000000123"
    with pytest.raises(cli.CLIError, match="incarnation does not match"):
        cli.execute(_arguments(tmp_path))

    # A malformed singleton incarnation is rejected even against a valid expected UUID.
    connection.identity["database_incarnation_id"] = "not-a-uuid"
    with pytest.raises(cli.CLIError, match="incarnation is malformed"):
        cli.execute(_arguments(tmp_path))


def test_write_capable_import_requires_expected_incarnation_flag(monkeypatch, tmp_path: Path) -> None:
    cli, _source, _snapshot, sealed, _marker = _prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli, "import_sqlite_to_postgres", lambda *_a, **_k: pytest.fail("must not import without incarnation")
    )
    # argparse rejects the write-capable invocation when the flag is absent.
    with pytest.raises(SystemExit):
        cli.execute(_arguments(tmp_path, expected_database_incarnation_id=None))


def test_incarnation_is_read_from_singleton_not_a_spoofable_guc(monkeypatch, tmp_path: Path) -> None:
    connection = _Connection()
    cli, _source, _snapshot, sealed, _marker = _prepare(monkeypatch, tmp_path, connection=connection)
    monkeypatch.setattr(
        cli, "import_sqlite_to_postgres", lambda *_a, **_k: ImportSummary(1, sealed.sha256, {"jobs": 1}, {})
    )

    receipt = json.loads(cli.execute(_arguments(tmp_path)).read_text(encoding="utf-8"))

    assert receipt["destination"]["databaseIncarnationId"] == INCARNATION
    joined = " ".join(connection.statements)
    assert "applypilot_database_identity" in joined
    # The removed ambient/custom-GUC fallback is never consulted, so a spoofed setting is inert.
    assert "current_setting('applypilot.database_incarnation_id'" not in joined
    assert "applypilot.database_incarnation_id" not in joined


def test_durable_identity_read_is_scoped_to_least_privilege_role_without_leak(
    monkeypatch, tmp_path: Path
) -> None:
    # The fake singleton refuses to answer unless the verifier role is active, so a green run
    # proves the CLI genuinely wraps every durable read (pre- and post-operation) in the scoped
    # SET ROLE rather than relying on ambient superuser privilege.
    connection = _Connection()
    cli, _source, _snapshot, sealed, _marker = _prepare(monkeypatch, tmp_path, connection=connection)
    monkeypatch.setattr(
        cli, "import_sqlite_to_postgres", lambda *_a, **_k: ImportSummary(1, sealed.sha256, {"jobs": 1}, {})
    )

    cli.execute(_arguments(tmp_path))

    statements = [statement.upper() for statement in connection.statements]
    read_positions = [i for i, s in enumerate(statements) if "IDENTITY_ROW_COUNT" in s]
    set_role_positions = [i for i, s in enumerate(statements) if s.startswith("SET ROLE BRAIN_SCHEMA_VERIFIER")]
    search_path_positions = [i for i, s in enumerate(statements) if s.startswith("SET SEARCH_PATH TO PG_CATALOG, PUBLIC")]
    reset_role_positions = [i for i, s in enumerate(statements) if s.startswith("RESET ROLE")]
    # Both the pre-operation and the post-operation durable reads are exercised on success.
    assert len(read_positions) == 2
    for read_at in read_positions:
        set_before = max((p for p in set_role_positions if p < read_at), default=None)
        path_before = max((p for p in search_path_positions if p < read_at), default=None)
        reset_after = min((p for p in reset_role_positions if p > read_at), default=None)
        assert set_before is not None and path_before is not None and reset_after is not None
        # No other statement runs between elevating the role and the guarded read.
        assert set_before < path_before < read_at
    # After the run neither the role nor the hardened search_path leaks onto the connection, and
    # each scoped read was torn down with a rollback.
    assert connection.role == _LOGIN_ROLE
    assert connection.search_path == _DEFAULT_SEARCH_PATH
    assert connection.rollbacks >= 2


@pytest.mark.parametrize(
    ("connection_kwargs", "message"),
    [
        ({"identity_present": False}, "missing the durable database identity singleton"),
        ({"identity_row_count": 0}, "missing or ambiguous"),
        (
            {"identity_row_count": 2, "incarnation_high": "11111111-1111-1111-1111-111111111111"},
            "missing or ambiguous",
        ),
    ],
)
def test_absent_missing_or_ambiguous_singleton_is_rejected(
    monkeypatch, tmp_path: Path, connection_kwargs: dict, message: str
) -> None:
    connection = _Connection(**connection_kwargs)
    cli, _source, _snapshot, _sealed, _marker = _prepare(monkeypatch, tmp_path, connection=connection)
    monkeypatch.setattr(
        cli, "import_sqlite_to_postgres", lambda *_a, **_k: pytest.fail("must not import without a singleton")
    )
    with pytest.raises(cli.CLIError, match=message):
        cli.execute(_arguments(tmp_path))


def test_path_anchor_revalidated_before_publication(monkeypatch, tmp_path: Path) -> None:
    cli, _source, _snapshot, sealed, marker = _prepare(monkeypatch, tmp_path)
    parity = _parity()
    monkeypatch.setattr(cli, "import_sqlite_to_postgres", lambda *_a, **_k: ImportSummary(73, sealed.sha256, {}, {}))
    monkeypatch.setattr(
        cli,
        "finalize_sqlite_to_postgres_import",
        lambda *_a, **_k: ImportSummary(73, sealed.sha256, {name: 0 for name in parity}, {}, True, parity, 173),
    )
    original = cli._revalidate_path_anchor
    calls = 0

    def replaced(anchor, label):
        nonlocal calls
        calls += 1
        if calls > 8:
            raise cli.CLIError(f"{label} parent path was replaced")
        return original(anchor, label)

    monkeypatch.setattr(cli, "_revalidate_path_anchor", replaced)
    with pytest.raises(cli.CLIError, match="parent path was replaced"):
        cli.execute(_final_arguments(tmp_path, marker))
    assert not cli._publication_commit_path(tmp_path / "receipt.json").exists()


def test_parent_junction_identity_replacement_is_rejected(monkeypatch, tmp_path: Path) -> None:
    cli = _load_cli()
    target = tmp_path / "canonical.sqlite"
    target.write_bytes(b"sqlite")
    anchor = cli._capture_path_anchor(target, "SQLite database")
    monkeypatch.setattr(cli, "_path_identity", lambda _path: (anchor.parent_identity[0], anchor.parent_identity[1] + 1))

    with pytest.raises(cli.CLIError, match="parent path was replaced"):
        cli._revalidate_path_anchor(anchor, "SQLite database")


def test_parent_symlink_or_reparse_replacement_is_rejected(monkeypatch, tmp_path: Path) -> None:
    cli = _load_cli()
    target = tmp_path / "writer-freeze.json"
    target.write_bytes(b"{}")
    anchor = cli._capture_path_anchor(target, "writer-freeze marker")
    monkeypatch.setattr(
        cli,
        "reject_symlink_components",
        lambda _path: (_ for _ in ()).throw(RuntimeError("reparse replacement")),
    )

    with pytest.raises(cli.CLIError, match="reparse point"):
        cli._revalidate_path_anchor(anchor, "writer-freeze marker")


def test_marker_path_replacement_after_finalization_fails_closed(monkeypatch, tmp_path: Path) -> None:
    cli, _source, _snapshot, sealed, marker = _prepare(monkeypatch, tmp_path)
    parity = _parity()
    monkeypatch.setattr(
        cli,
        "import_sqlite_to_postgres",
        lambda *_args, **_kwargs: ImportSummary(72, sealed.sha256, {}, {}),
    )

    def finalize(*_args, **_kwargs):
        replacement = marker.with_suffix(".replacement")
        replacement.write_bytes(marker.read_bytes())
        os.replace(replacement, marker)
        return ImportSummary(72, sealed.sha256, {name: 0 for name in parity}, {}, True, parity, 172)

    monkeypatch.setattr(cli, "finalize_sqlite_to_postgres_import", finalize)

    with pytest.raises(cli.CLIError, match="replaced|changed"):
        cli.execute(_final_arguments(tmp_path, marker))


def test_receipt_authentication_detects_tampering(monkeypatch, tmp_path: Path) -> None:
    cli, _source, _snapshot, sealed, _marker = _prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli,
        "import_sqlite_to_postgres",
        lambda *_args, **_kwargs: ImportSummary(81, sealed.sha256, {"jobs": 2}, {}),
    )
    receipt = json.loads(cli.execute(_arguments(tmp_path)).read_text(encoding="utf-8"))
    receipt["result"]["importedCounts"]["jobs"] = 200

    with pytest.raises(RuntimeError, match="signature is invalid"):
        cli.verify_receipt(receipt, key=IMPORT_KEY, expected_key_id=IMPORT_KEY_ID, label="brain import receipt")


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    [
        ("receipt", lambda value: value.__setitem__("purpose", "wrong"), "purpose"),
        ("receipt", lambda value: value.__setitem__("receiptSchema", "wrong"), "schema"),
        ("receipt", lambda value: value["command"].__setitem__("mode", "bulk-import"), "mode"),
        ("receipt", lambda value: value["command"].__setitem__("dryRun", True), "dryRun"),
        ("receipt", lambda value: value["command"].__setitem__("runKey", "unrelated-nonempty-run-key"), "run key"),
        ("receipt", lambda value: value["result"].__setitem__("terminalEventId", None), "terminal"),
        ("receipt", lambda value: value["destination"].__setitem__("unexpected", "x"), "destination"),
        ("receipt", lambda value: value["result"]["parity"]["jobs"].__setitem__("passed", False), "parity"),
        ("receipt", lambda value: value["result"]["parity"]["jobs"].__setitem__("source_count", "0"), "parity"),
        ("receipt", lambda value: value["result"]["parity"]["jobs"].__setitem__("unexpected", 0), "parity"),
        ("receipt", lambda value: value.__setitem__("unexpected", "x"), "schema"),
        (
            "receipt",
            lambda value: value["writer_freeze"].__setitem__(
                "frozenAt", value["writer_freeze"]["expiresAt"]
            ),
            "freeze interval",
        ),
        (
            "receipt",
            lambda value: value["writer_freeze"].__setitem__(
                "expiresAt",
                (
                    datetime.fromisoformat(value["writer_freeze"]["frozenAt"].replace("Z", "+00:00"))
                    + timedelta(seconds=901)
                ).isoformat().replace("+00:00", "Z"),
            ),
            "validity duration",
        ),
        (
            "receipt",
            lambda value: value["timestamps"].update(
                {"startedAt": value["timestamps"]["completedAt"], "completedAt": "2000-01-01T00:00:00Z"}
            ),
            "timestamp order",
        ),
        ("commit", lambda value: value.__setitem__("purpose", "wrong"), "purpose"),
        ("commit", lambda value: value.__setitem__("unexpected", "x"), "schema"),
    ],
)
def test_consumable_verifier_rejects_signed_semantic_malformations(
    monkeypatch,
    tmp_path: Path,
    target: str,
    mutation,
    message: str,
) -> None:
    cli, _source, _snapshot, sealed, marker = _prepare(monkeypatch, tmp_path)
    parity = _parity()
    monkeypatch.setattr(cli, "import_sqlite_to_postgres", lambda *_a, **_k: ImportSummary(82, sealed.sha256, {}, {}))
    monkeypatch.setattr(
        cli,
        "finalize_sqlite_to_postgres_import",
        lambda *_a, **_k: ImportSummary(82, sealed.sha256, {name: 0 for name in parity}, {}, True, parity, 182),
    )
    receipt_path = cli.execute(_final_arguments(tmp_path, marker))
    commit_path = cli._publication_commit_path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if target == "receipt":
        receipt.pop("authentication")
        mutation(receipt)
        receipt = _sign(receipt, IMPORT_KEY, IMPORT_KEY_ID)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        commit["receiptSha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        commit.pop("authentication")
        commit = _sign(commit, IMPORT_KEY, IMPORT_KEY_ID)
        commit_path.write_text(json.dumps(commit), encoding="utf-8")
    else:
        commit.pop("authentication")
        mutation(commit)
        commit_path.write_text(json.dumps(_sign(commit, IMPORT_KEY, IMPORT_KEY_ID)), encoding="utf-8")

    with pytest.raises(cli.CLIError, match=message):
        cli.verify_consumable_import_receipt(receipt_path)


def test_dry_run_is_non_promotable_and_uses_canonical_nonmutating_path(monkeypatch, tmp_path: Path) -> None:
    connection = _Connection()
    cli, _source, _snapshot, sealed, _marker = _prepare(monkeypatch, tmp_path, connection=connection)
    observed = []

    def dry_run(*_args, **kwargs):
        observed.append(kwargs["dry_run"])
        return ImportSummary(0, sealed.sha256, {}, {})

    monkeypatch.setattr(cli, "import_sqlite_to_postgres", dry_run)

    receipt = json.loads(cli.execute([*_arguments(tmp_path), "--dry-run"]).read_text(encoding="utf-8"))

    assert observed == [True]
    assert receipt["promotable"] is False
    assert receipt["result"]["status"] == "dry-run-non-promotable"
    assert all(not statement.startswith(("INSERT", "UPDATE", "DELETE")) for statement in connection.statements)


def test_main_redacts_dsn_and_password_from_failures(monkeypatch, tmp_path: Path, capsys) -> None:
    cli, _source, _snapshot, _sealed, _marker = _prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli,
        "_connect_postgres",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(f"could not connect with {DSN}")),
    )

    assert cli.main(_arguments(tmp_path)) == 1
    combined = capsys.readouterr().out + capsys.readouterr().err
    assert DSN not in combined
    assert "top-secret-password" not in combined


class _OpenConnection:
    def __init__(self, connection) -> None:
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        return False


def _provision_v8_identity_singleton(connection) -> None:
    """Create the V8 durable identity singleton owned by ``brain_schema_migrator``.

    The singleton is owned by the migrator but, exactly as real V8 does (schema_v8.sql grants
    SELECT to ``brain_schema_verifier``), grants only SELECT to the read-only verifier role. The
    CLI's scoped ``SET ROLE brain_schema_verifier`` therefore reaches it through the least-
    privileged channel -- never the table owner -- while an ordinary login role still cannot.
    """
    for role in ("brain_schema_migrator", "brain_schema_verifier"):
        connection.execute(
            f"DO $$ BEGIN CREATE ROLE {role} NOLOGIN; "
            "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        )
    connection.execute("GRANT USAGE, CREATE ON SCHEMA public TO brain_schema_migrator")
    connection.execute("GRANT USAGE ON SCHEMA public TO brain_schema_verifier")
    connection.execute("SET ROLE brain_schema_migrator")
    connection.execute(
        "CREATE TABLE public.applypilot_database_identity ("
        "singleton_id smallint PRIMARY KEY DEFAULT 1,"
        "database_incarnation_id uuid NOT NULL DEFAULT gen_random_uuid(),"
        "initialized_at timestamptz NOT NULL DEFAULT now())"
    )
    connection.execute("REVOKE ALL ON public.applypilot_database_identity FROM PUBLIC")
    connection.execute("GRANT SELECT ON public.applypilot_database_identity TO brain_schema_verifier")
    connection.execute("INSERT INTO public.applypilot_database_identity (singleton_id) VALUES (1)")
    connection.execute("RESET ROLE")


def test_real_terminal_event_recovers_after_receipt_publication_failure(monkeypatch, tmp_path: Path, fleet_db) -> None:
    connection = psycopg.connect(fleet_db, row_factory=dict_row)
    try:
        connection.execute(
            "CREATE TEMP TABLE brain_migration_sources (migration_source_id bigint PRIMARY KEY, "
            "source_namespace text NOT NULL, source_fingerprint text NOT NULL)"
        )
        connection.execute(
            "CREATE TEMP TABLE brain_migration_runs (migration_run_id bigint PRIMARY KEY, migration_source_id bigint, "
            "source_namespace text NOT NULL, run_key text NOT NULL)"
        )
        connection.execute(
            "CREATE TEMP TABLE brain_migration_run_events (migration_run_event_id bigint PRIMARY KEY, "
            "migration_run_id bigint, source_namespace text NOT NULL, event_type text NOT NULL, metadata jsonb)"
        )
        _provision_v8_identity_singleton(connection)
        identity = connection.execute(
            "SELECT current_database() AS database,(pg_control_system()).system_identifier::text AS system_identifier,"
            "(SELECT oid::text FROM pg_database WHERE datname=current_database()) AS database_oid,"
            "(SELECT database_incarnation_id::text FROM public.applypilot_database_identity WHERE singleton_id=1)"
            " AS database_incarnation_id,current_setting('server_version_num') AS server_version_num"
        ).fetchone()
        connection.commit()
        cli, _source, _snapshot, sealed, marker = _prepare(monkeypatch, tmp_path)
        arguments = _final_arguments(tmp_path, marker)
        for option, value in (
            ("--expected-database", identity["database"]),
            ("--expected-system-identifier", identity["system_identifier"]),
            ("--expected-database-oid", identity["database_oid"]),
            ("--expected-database-incarnation-id", identity["database_incarnation_id"]),
        ):
            arguments[arguments.index(option) + 1] = value
        monkeypatch.setattr(cli, "_connect_postgres", lambda *_args, **_kwargs: _OpenConnection(connection))
        destination = {
            "database": identity["database"],
            "systemIdentifier": identity["system_identifier"],
            "databaseOid": identity["database_oid"],
            "databaseIncarnationId": identity["database_incarnation_id"],
            "serverVersionNum": identity["server_version_num"],
        }
        durable_destination = cli._durable_destination_binding(destination)
        run_key = cli._run_key("release-2026-07-18", NONCE, "final-delta-finalize", durable_destination)
        parity = _parity(4)

        monkeypatch.setattr(
            cli,
            "import_sqlite_to_postgres",
            lambda *_args, **_kwargs: ImportSummary(101, sealed.sha256, {"jobs": 4}, {}),
        )

        def commit_terminal(*_args, **_kwargs):
            connection.execute(
                "INSERT INTO brain_migration_sources VALUES (1,%s,%s)",
                ("applypilot-sqlite", sealed.sha256),
            )
            connection.execute(
                "INSERT INTO brain_migration_runs VALUES (101,1,%s,%s)",
                ("applypilot-sqlite", run_key),
            )
            connection.execute(
                "INSERT INTO brain_migration_run_events VALUES (201,101,%s,'completed',%s)",
                (
                    "applypilot-sqlite",
                    json.dumps(
                        {
                            "source_sha256": sealed.sha256,
                            "independent_parity": parity,
                            "destination_binding": durable_destination,
                        }
                    ),
                ),
            )
            connection.commit()
            return ImportSummary(
                101,
                sealed.sha256,
                {name: int(record["target_count"]) for name, record in parity.items()},
                {},
                True,
                parity,
                201,
            )

        monkeypatch.setattr(cli, "finalize_sqlite_to_postgres_import", commit_terminal)
        original_publish = cli.atomic_write_no_overwrite
        monkeypatch.setattr(
            cli,
            "atomic_write_no_overwrite",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("simulated publication failure")),
        )

        with pytest.raises(cli.CLIError, match="recover-terminal"):
            cli.execute(arguments)
        assert connection.execute(
            "SELECT event_type FROM brain_migration_run_events WHERE migration_run_event_id=201"
        ).fetchone()["event_type"] == "completed"

        monkeypatch.setattr(cli, "atomic_write_no_overwrite", original_publish)
        recovered_path = cli.execute([*arguments, "--recover-terminal"])
        recovered = json.loads(recovered_path.read_text(encoding="utf-8"))
        assert recovered["result"]["recovered"] is True
        assert recovered["result"]["terminalEventId"] == 201
        assert recovered["result"]["parity"]["jobs"]["target_count"] == 4
    finally:
        try:
            connection.rollback()
            connection.execute("DROP TABLE IF EXISTS public.applypilot_database_identity")
            connection.commit()
        except Exception:
            pass
        connection.close()


# --- Canonical non-superuser PostgreSQL 18 login proofs -----------------------------------------
#
# These exercise the real repair: the durable identity singleton is SELECT-visible only to the
# ``brain_schema_migrator`` owner, and the canonical import identity is a NOINHERIT LOGIN that is
# *not* postgres and holds the read-only verifier role only via SET. They prove the pre/post reads succeed
# through the scoped SET ROLE, that a wrong/absent/mismatched singleton still fails before any
# write, and that neither the role nor the hardened search_path leaks after success, after a
# failure, or across a reused (pooled) connection.

_CANONICAL_LOGIN = "canonical_import_login"
_CANONICAL_LOGIN_PASSWORD = "canonical-login-disposable-test-only"


def _provision_canonical_login(admin, admin_dsn: str) -> str:
    """Provision a NOINHERIT LOGIN that reaches the singleton only via ``SET ROLE``.

    The login is a member of the read-only ``brain_schema_verifier`` WITH INHERIT FALSE, SET TRUE:
    it may assume the least-privileged role that V8 grants SELECT on the singleton for the guarded
    read, but never passively holds its privileges, and it is deliberately neither ``postgres`` nor
    a superuser. The V8 singleton is owned by the migrator and grants only SELECT to the verifier,
    so the login cannot read it directly and never assumes the table owner.
    """
    _provision_v8_identity_singleton(admin)
    admin.execute(
        pg_sql.SQL(
            "DO $$ BEGIN CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
            "NOREPLICATION NOBYPASSRLS; EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        ).format(pg_sql.Identifier(_CANONICAL_LOGIN))
    )
    admin.execute(
        pg_sql.SQL("ALTER ROLE {} PASSWORD {}").format(
            pg_sql.Identifier(_CANONICAL_LOGIN), pg_sql.Literal(_CANONICAL_LOGIN_PASSWORD)
        )
    )
    admin.execute(
        pg_sql.SQL("GRANT brain_schema_verifier TO {} WITH INHERIT FALSE, SET TRUE").format(
            pg_sql.Identifier(_CANONICAL_LOGIN)
        )
    )
    admin.execute(pg_sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(pg_sql.Identifier(_CANONICAL_LOGIN)))
    admin.commit()
    parameters = conninfo_to_dict(admin_dsn)
    parameters.update(user=_CANONICAL_LOGIN, password=_CANONICAL_LOGIN_PASSWORD)
    return make_conninfo(**parameters)


def _drop_canonical_login(admin) -> None:
    admin.rollback()
    admin.execute("DROP TABLE IF EXISTS public.applypilot_database_identity")
    if admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (_CANONICAL_LOGIN,)).fetchone() is not None:
        admin.execute(pg_sql.SQL("DROP OWNED BY {} CASCADE").format(pg_sql.Identifier(_CANONICAL_LOGIN)))
        admin.execute(pg_sql.SQL("DROP ROLE {}").format(pg_sql.Identifier(_CANONICAL_LOGIN)))
    admin.commit()


def _read_identity(admin) -> dict:
    return admin.execute(
        "SELECT current_database() AS database,(pg_control_system()).system_identifier::text AS system_identifier,"
        "(SELECT oid::text FROM pg_database WHERE datname=current_database()) AS database_oid,"
        "(SELECT database_incarnation_id::text FROM public.applypilot_database_identity WHERE singleton_id=1)"
        " AS database_incarnation_id,current_setting('server_version_num') AS server_version_num"
    ).fetchone()


def _bind_identity_arguments(arguments: list[str], identity: dict, *, incarnation: str | None = None) -> list[str]:
    bound = list(arguments)
    for option, value in (
        ("--expected-database", identity["database"]),
        ("--expected-system-identifier", identity["system_identifier"]),
        ("--expected-database-oid", identity["database_oid"]),
        ("--expected-database-incarnation-id", incarnation or identity["database_incarnation_id"]),
    ):
        bound[bound.index(option) + 1] = value
    return bound


def _session_role_and_search_path(conn) -> tuple[str, str]:
    row = conn.execute(
        "SELECT current_user AS role, current_setting('search_path') AS search_path"
    ).fetchone()
    return row["role"], row["search_path"]


def _assert_pg18_login(login_conn) -> None:
    caps = login_conn.execute(
        "SELECT rolcanlogin,rolinherit,rolsuper FROM pg_roles WHERE rolname=current_user"
    ).fetchone()
    assert caps == {"rolcanlogin": True, "rolinherit": False, "rolsuper": False}
    assert int(login_conn.execute("SELECT current_setting('server_version_num') AS v").fetchone()["v"]) >= 180000


@pytest.mark.parametrize("mode", ["bulk-import", "final-delta-finalize"])
def test_canonical_noinherit_login_reads_identity_via_scoped_role_without_leak(
    monkeypatch, tmp_path: Path, fleet_db, mode: str
) -> None:
    admin = psycopg.connect(fleet_db, row_factory=dict_row)
    try:
        login_dsn = _provision_canonical_login(admin, fleet_db)
        identity = _read_identity(admin)
        login_conn = psycopg.connect(login_dsn, row_factory=dict_row)
        try:
            _assert_pg18_login(login_conn)
            # The canonical login genuinely cannot read the singleton directly: only the scoped
            # SET ROLE in the CLI can, which is exactly what makes the run below a real proof.
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                login_conn.execute("SELECT count(*) FROM public.applypilot_database_identity")
            login_conn.rollback()

            cli, _source, _snapshot, sealed, marker = _prepare(monkeypatch, tmp_path)
            monkeypatch.setattr(
                cli, "import_sqlite_to_postgres",
                lambda *_a, **_k: ImportSummary(7, sealed.sha256, {"jobs": 3}, {}),
            )
            parity = _parity(3)
            monkeypatch.setattr(
                cli, "finalize_sqlite_to_postgres_import",
                lambda *_a, **_k: ImportSummary(
                    7, sealed.sha256, {name: int(r["target_count"]) for name, r in parity.items()},
                    {}, finalized=True, parity=parity, terminal_event_id=207,
                ),
            )
            monkeypatch.setattr(cli, "_connect_postgres", lambda *_a, **_k: _OpenConnection(login_conn))
            if mode == "bulk-import":
                arguments = _bind_identity_arguments(_arguments(tmp_path), identity)
            else:
                arguments = _bind_identity_arguments(_final_arguments(tmp_path, marker), identity)

            receipt = json.loads(cli.execute(arguments).read_text(encoding="utf-8"))

            # The connection is left idle (committed) after the post-operation read.
            assert login_conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
            # The pre- and post-operation durable reads both succeeded through the scoped role, and
            # the exact incarnation flowed into the receipt and the incarnation-bound run key.
            assert receipt["destination"]["databaseIncarnationId"] == identity["database_incarnation_id"]
            destination = {
                "database": identity["database"],
                "systemIdentifier": identity["system_identifier"],
                "databaseOid": identity["database_oid"],
                "databaseIncarnationId": identity["database_incarnation_id"],
                "serverVersionNum": identity["server_version_num"],
            }
            expected_run_key = cli._run_key(
                "release-2026-07-18", NONCE, mode, cli._durable_destination_binding(destination)
            )
            assert receipt["command"]["runKey"] == expected_run_key
            cli.verify_receipt(receipt, key=IMPORT_KEY, expected_key_id=IMPORT_KEY_ID, label="brain import receipt")
            # No leak: the reused connection is back to the login role and default search_path, and
            # can still not read the singleton without re-electing the role.
            assert _session_role_and_search_path(login_conn) == (_CANONICAL_LOGIN, '"$user", public')
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                login_conn.execute("SELECT count(*) FROM public.applypilot_database_identity")
            login_conn.rollback()
        finally:
            login_conn.close()
    finally:
        _drop_canonical_login(admin)
        admin.close()


def test_canonical_noinherit_login_recover_reads_identity_via_scoped_role(
    monkeypatch, tmp_path: Path, fleet_db
) -> None:
    admin = psycopg.connect(fleet_db, row_factory=dict_row)
    try:
        login_dsn = _provision_canonical_login(admin, fleet_db)
        identity = _read_identity(admin)
        login_conn = psycopg.connect(login_dsn, row_factory=dict_row)
        try:
            _assert_pg18_login(login_conn)
            cli, _source, _snapshot, sealed, marker = _prepare(monkeypatch, tmp_path)
            parity = _parity(5)
            monkeypatch.setattr(
                cli, "import_sqlite_to_postgres",
                lambda *_a, **_k: ImportSummary(9, sealed.sha256, {"jobs": 5}, {}),
            )
            monkeypatch.setattr(
                cli, "recover_finalized_sqlite_to_postgres_import",
                lambda *_a, **_k: ImportSummary(
                    9, sealed.sha256, {name: int(r["target_count"]) for name, r in parity.items()},
                    {}, finalized=True, parity=parity, terminal_event_id=209, recovered=True,
                ),
            )
            monkeypatch.setattr(cli, "_connect_postgres", lambda *_a, **_k: _OpenConnection(login_conn))
            arguments = _bind_identity_arguments(_final_arguments(tmp_path, marker, "--recover-terminal"), identity)

            receipt = json.loads(cli.execute(arguments).read_text(encoding="utf-8"))

            assert receipt["result"]["recovered"] is True
            assert receipt["destination"]["databaseIncarnationId"] == identity["database_incarnation_id"]
            assert _session_role_and_search_path(login_conn) == (_CANONICAL_LOGIN, '"$user", public')
        finally:
            login_conn.close()
    finally:
        _drop_canonical_login(admin)
        admin.close()


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("absent", "missing or ambiguous"),
        ("mismatched", "incarnation does not match"),
    ],
)
def test_canonical_noinherit_login_rejects_bad_singleton_before_writes_without_leak(
    monkeypatch, tmp_path: Path, fleet_db, scenario: str, message: str
) -> None:
    admin = psycopg.connect(fleet_db, row_factory=dict_row)
    try:
        login_dsn = _provision_canonical_login(admin, fleet_db)
        identity = _read_identity(admin)
        if scenario == "absent":
            # Empty the singleton by re-creating an unseeded migrator-owned table.
            admin.execute("SET ROLE brain_schema_migrator")
            admin.execute("DELETE FROM public.applypilot_database_identity")
            admin.execute("RESET ROLE")
            admin.commit()
        login_conn = psycopg.connect(login_dsn, row_factory=dict_row)
        try:
            _assert_pg18_login(login_conn)
            cli, _source, _snapshot, _sealed, _marker = _prepare(monkeypatch, tmp_path)
            monkeypatch.setattr(
                cli, "import_sqlite_to_postgres",
                lambda *_a, **_k: pytest.fail("a bad singleton must fail before any import"),
            )
            monkeypatch.setattr(cli, "_connect_postgres", lambda *_a, **_k: _OpenConnection(login_conn))
            if scenario == "mismatched":
                incarnation = "00000000-0000-0000-0000-0000000000aa"
            else:
                incarnation = identity["database_incarnation_id"]
            arguments = _bind_identity_arguments(_arguments(tmp_path), identity, incarnation=incarnation)

            with pytest.raises(cli.CLIError, match=message):
                cli.execute(arguments)

            # After the failure the scoped role and hardened search_path are fully torn down: no
            # dangling transaction remains (checked before any probe query reopens one) and the
            # reused connection is back to the login role and default search_path, so a pooled
            # connection is safe to reuse.
            assert login_conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
            assert _session_role_and_search_path(login_conn) == (_CANONICAL_LOGIN, '"$user", public')
        finally:
            login_conn.close()
    finally:
        _drop_canonical_login(admin)
        admin.close()
