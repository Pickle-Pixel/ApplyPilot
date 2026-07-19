from __future__ import annotations

import hashlib
import inspect
import sqlite3
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

import applypilot.brain.sqlite_to_postgres as importer_module
from applypilot.brain.policy_artifacts import ARTIFACT_MEDIA_TYPE, ArtifactDescriptor
from applypilot.brain.sqlite_to_postgres import _acquire_import_lock
from applypilot.brain.sqlite_to_postgres import _activate_controller
from applypilot.brain.sqlite_to_postgres import _commit_phase
from applypilot.brain.sqlite_to_postgres import _commit_terminal_phase
from applypilot.brain.sqlite_to_postgres import _bind_policy_artifact
from applypilot.brain.sqlite_to_postgres import _insert_observations
from applypilot.brain.sqlite_to_postgres import _insert_label_confidence, _insert_labels
from applypilot.brain.sqlite_to_postgres import _mark_run_failed
from applypilot.brain.sqlite_to_postgres import _migration_run_resume_event
from applypilot.brain.sqlite_to_postgres import _parity_metadata
from applypilot.brain.sqlite_to_postgres import _record_completed_batches
from applypilot.brain.sqlite_to_postgres import _require_artifact
from applypilot.brain.sqlite_to_postgres import _source_bounds
from applypilot.brain.sqlite_to_postgres import _verify_kg_artifacts
from applypilot.brain.sqlite_to_postgres import _validated_destination_binding
from applypilot.brain.sqlite_to_postgres import finalize_sqlite_to_postgres_import
from applypilot.brain.sqlite_to_postgres import recover_finalized_sqlite_to_postgres_import
from applypilot.brain.parity import ParityResult
from applypilot.brain.importer import SOURCE_TABLES


DESTINATION_BINDING = {
    "database": "brain",
    "systemIdentifier": "7460123456789012345",
    "databaseOid": "16384",
    "databaseIncarnationId": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
}


class _RecordingCursor:
    def __init__(self, connection: "_RecordingPostgres") -> None:
        self.connection = connection
        self._row: dict[str, Any] | None = None

    def __enter__(self) -> "_RecordingCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> "_RecordingCursor":
        normalized = " ".join(query.split())
        self.connection.statements.append((normalized, params))
        if "SELECT count(*) AS n FROM brain_label_events" in normalized:
            self._row = {"n": 0}
        elif "SELECT count(*) AS n FROM brain_migration_quarantine" in normalized:
            self._row = {"n": 0}
        elif "SELECT label_event_id FROM brain_label_events" in normalized:
            self._row = {"label_event_id": 42}
        elif "SELECT migration_run_event_id,event_type FROM brain_migration_run_events" in normalized:
            self._row = self.connection.run_head
        elif "JOIN brain_migration_sources source" in normalized and "event.metadata" in normalized:
            self._row = self.connection.terminal_row
        elif "SELECT current_user AS current_user" in normalized:
            self._row = {"current_user": self.connection.current_user}
        elif "pg_try_advisory_lock" in normalized:
            self._row = {"acquired": self.connection.lock_acquired}
        elif "pg_advisory_unlock" in normalized:
            self._row = {"released": True}
        elif "SELECT artifact_hash,byte_length,media_type" in normalized and "FROM brain_artifacts" in normalized:
            artifact_hash = params[0]
            self._row = self.connection.artifacts.get(artifact_hash)
        elif "INSERT INTO brain_policy_artifacts" in normalized:
            key = (str(params[0]), str(params[1]))
            self.connection.policy_artifacts.setdefault(key, str(params[2]))
            self._row = None
        elif "SELECT artifact_hash FROM brain_policy_artifacts" in normalized:
            artifact_hash = self.connection.policy_artifacts.get((str(params[0]), str(params[1])))
            self._row = None if artifact_hash is None else {"artifact_hash": artifact_hash}
        else:
            self._row = None
        return self

    def executemany(self, query: str, params: list[tuple[Any, ...]]) -> None:
        normalized = " ".join(query.split())
        self.connection.statements.extend((normalized, row) for row in params)

    def fetchone(self) -> dict[str, Any] | None:
        return self._row


def test_finalizer_does_not_accept_caller_supplied_parity_results() -> None:
    parameters = inspect.signature(finalize_sqlite_to_postgres_import).parameters

    assert "parity_results" not in parameters


def test_policy_artifact_binding_rejects_an_existing_different_hash() -> None:
    pg = _RecordingPostgres()
    pg.policy_artifacts[("policy-v1", "config")] = "a" * 64

    with pg.cursor() as cur, pytest.raises(Exception, match="binding conflicts"):
        _bind_policy_artifact(cur, "policy-v1", "config", "b" * 64)


def test_source_bounds_are_ordered_for_multi_digit_integer_keys() -> None:
    source = sqlite3.connect(":memory:")
    source.row_factory = sqlite3.Row
    source.execute("CREATE TABLE source_rows (id INTEGER PRIMARY KEY)")
    source.executemany("INSERT INTO source_rows VALUES (?)", [(1,), (10,)])

    key_start, key_end = _source_bounds(source, "source_rows", ("id",))

    assert key_start == "0:[1]"
    assert key_end == "1:[10]"
    assert key_start <= key_end


class _RecordingPostgres:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self.commits = 0
        self.rollbacks = 0
        self.run_head: dict[str, Any] | None = None
        self.current_user = "brain_schema_migrator"
        self.lock_acquired = True
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.policy_artifacts: dict[tuple[str, str], str] = {}
        self.terminal_row: dict[str, Any] | None = None

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def test_commit_phase_rechecks_source_before_and_after_commit() -> None:
    pg = _RecordingPostgres()
    events: list[str] = []

    def recheck() -> None:
        events.append("recheck")

    original_commit = pg.commit

    def commit() -> None:
        events.append("commit")
        original_commit()

    pg.commit = commit  # type: ignore[method-assign]

    _commit_phase(pg, recheck)

    assert events == ["recheck", "commit", "recheck"]
    assert pg.commits == 1


def test_controller_activation_sets_and_verifies_exact_role() -> None:
    pg = _RecordingPostgres()

    _activate_controller(pg)

    sql = [statement[0] for statement in pg.statements]
    assert sql[:3] == [
        "SET ROLE brain_schema_migrator",
        "SET search_path TO public",
        "SELECT current_user AS current_user",
    ]
    assert pg.commits == 1


def test_import_lock_fails_closed_when_source_is_already_owned() -> None:
    pg = _RecordingPostgres()
    pg.lock_acquired = False

    with pytest.raises(Exception, match="already owns source"):
        _acquire_import_lock(pg, "a" * 64)


def test_kg_accounting_derives_and_verifies_source_content_hash() -> None:
    source = sqlite3.connect(":memory:")
    source.row_factory = sqlite3.Row
    source.execute(
        "CREATE TABLE research_kg_artifacts (kg_version TEXT PRIMARY KEY, compact_kg_json BLOB, "
        "built_at TEXT, input_label_count INTEGER, inputs_sha TEXT)"
    )
    content = b'{"schemaVersion":"test-graph"}'
    digest = hashlib.sha256(content).hexdigest()
    source.execute(
        "INSERT INTO research_kg_artifacts VALUES (?,?,?,?,?)",
        ("test-graph", content, "2026-07-16T00:00:00Z", 1, digest),
    )
    pg = _RecordingPostgres()
    pg.artifacts[digest] = {
        "artifact_hash": digest,
        "byte_length": len(content),
        "media_type": "application/json",
    }

    assert _verify_kg_artifacts(pg, source) == 1


def test_batch_finalization_requires_every_independent_clean_parity_result() -> None:
    receipt = SimpleNamespace(
        table_counts={table.name: 0 for table in SOURCE_TABLES},
        sha256="a" * 64,
    )
    source = sqlite3.connect(":memory:")
    source.row_factory = sqlite3.Row

    with pytest.raises(Exception, match="missing independent parity result for jobs"):
        _record_completed_batches(_RecordingPostgres(), source, receipt, 1, {})

    failed = ParityResult(
        source_count=0,
        target_count=0,
        source_hash="a" * 64,
        target_hash="b" * 64,
        mismatch_count=1,
        unresolved_count=0,
    )
    with pytest.raises(Exception, match="independent parity failed for jobs"):
        _record_completed_batches(
            _RecordingPostgres(),
            source,
            receipt,
            1,
            {"jobs": failed},
        )


def test_terminal_parity_metadata_preserves_independent_counts_and_hashes() -> None:
    parity = ParityResult(
        source_count=7,
        target_count=7,
        source_hash="a" * 64,
        target_hash="a" * 64,
        mismatch_count=0,
        unresolved_count=0,
    )

    assert _parity_metadata({"jobs": parity}) == {
        "jobs": {
            "source_count": 7,
            "target_count": 7,
            "source_hash": "a" * 64,
            "target_hash": "a" * 64,
            "mismatch_count": 0,
            "unresolved_count": 0,
            "passed": True,
        }
    }


def test_terminal_recovery_reconstructs_only_completed_durable_parity() -> None:
    pg = _RecordingPostgres()
    durable_parity = {
        table.name: {
            "source_count": 3 if table.name == "jobs" else 0,
            "target_count": 3 if table.name == "jobs" else 0,
            "source_hash": "c" * 64,
            "target_hash": "c" * 64,
            "mismatch_count": 0,
            "unresolved_count": 0,
            "passed": True,
        }
        for table in SOURCE_TABLES
    }
    pg.terminal_row = {
        "migration_run_id": 91,
        "migration_run_event_id": 191,
        "event_type": "completed",
        "source_fingerprint": "b" * 64,
        "metadata": {
            "source_sha256": "b" * 64,
            "independent_parity": durable_parity,
            "destination_binding": DESTINATION_BINDING,
        },
    }

    result = recover_finalized_sqlite_to_postgres_import(
        pg,
        expected_sha256="b" * 64,
        run_key="stable-run-key",
        expected_destination_binding=DESTINATION_BINDING,
    )

    assert result.migration_run_id == 91
    assert result.finalized is True
    assert result.recovered is True
    assert result.imported["jobs"] == 3
    assert result.parity["jobs"]["source_hash"] == "c" * 64


@pytest.mark.parametrize(
    ("event_type", "metadata"),
    [
        ("started", {"source_sha256": "b" * 64, "independent_parity": {}, "destination_binding": DESTINATION_BINDING}),
        ("completed", {"source_sha256": "b" * 64, "independent_parity": {}, "destination_binding": DESTINATION_BINDING}),
    ],
)
def test_terminal_recovery_rejects_nonterminal_or_incomplete_metadata(event_type: str, metadata: dict[str, Any]) -> None:
    pg = _RecordingPostgres()
    pg.terminal_row = {
        "migration_run_id": 92,
        "migration_run_event_id": 192,
        "event_type": event_type,
        "source_fingerprint": "b" * 64,
        "metadata": metadata,
    }

    with pytest.raises(Exception, match="terminal recovery"):
        recover_finalized_sqlite_to_postgres_import(
            pg,
            expected_sha256="b" * 64,
            run_key="stable-run-key",
            expected_destination_binding=DESTINATION_BINDING,
        )


def test_destination_binding_requires_a_canonical_incarnation() -> None:
    valid = _validated_destination_binding(DESTINATION_BINDING)
    assert valid["databaseIncarnationId"] == DESTINATION_BINDING["databaseIncarnationId"]
    for bad in (None, "", "not-a-uuid", "3f2504e0-4f89-41d3-9a0c-0305e82c33"):
        with pytest.raises(Exception, match="canonical UUID"):
            _validated_destination_binding({**DESTINATION_BINDING, "databaseIncarnationId": bad})


def test_terminal_recovery_rejects_different_destination_binding() -> None:
    pg = _RecordingPostgres()
    parity = {
        table.name: {
            "source_count": 0,
            "target_count": 0,
            "source_hash": "c" * 64,
            "target_hash": "c" * 64,
            "mismatch_count": 0,
            "unresolved_count": 0,
            "passed": True,
        }
        for table in SOURCE_TABLES
    }
    pg.terminal_row = {
        "migration_run_id": 93,
        "migration_run_event_id": 193,
        "event_type": "completed",
        "source_fingerprint": "b" * 64,
        "metadata": {
            "source_sha256": "b" * 64,
            "independent_parity": parity,
            "destination_binding": DESTINATION_BINDING,
        },
    }

    with pytest.raises(Exception, match="destination"):
        recover_finalized_sqlite_to_postgres_import(
            pg,
            expected_sha256="b" * 64,
            run_key="stable-run-key",
            expected_destination_binding={**DESTINATION_BINDING, "databaseOid": "999"},
        )


def test_terminal_commit_fully_verifies_before_commit_without_postcommit_failure_window() -> None:
    pg = _RecordingPostgres()
    events: list[str] = []
    original_commit = pg.commit

    def commit() -> None:
        events.append("commit")
        original_commit()

    pg.commit = commit  # type: ignore[method-assign]

    _commit_terminal_phase(
        pg,
        recheck_source=lambda: events.append("recheck"),
        verify_final_source=lambda: events.append("verify-final"),
    )

    assert events == ["verify-final", "recheck", "commit"]


class _FakeGuard:
    events: list[str] = []

    def __init__(self, _path: object, _expected_sha256: str) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.events.append("guard-open")

    def recheck(self) -> None:
        self.events.append("recheck")

    def verify_final(self) -> None:
        self.events.append("verify-final")

    def close(self) -> None:
        self.events.append("guard-close")
        self.connection.close()


def _fake_receipt() -> SimpleNamespace:
    return SimpleNamespace(sha256="a" * 64, table_counts={"jobs": 0})


def test_dry_run_performs_final_full_source_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeGuard.events = []
    pg = _RecordingPostgres()
    monkeypatch.setattr(importer_module, "capture_sqlite_source_receipt", lambda _path: _fake_receipt())
    monkeypatch.setattr(importer_module, "SQLiteSourceGuard", _FakeGuard)

    result = importer_module.import_sqlite_to_postgres(
        pg, "sealed.db", expected_sha256="a" * 64, run_key="dry", dry_run=True
    )

    assert result.migration_run_id == 0
    assert result.imported == {}
    assert not result.finalized
    assert pg.commits == 0
    assert _FakeGuard.events == ["guard-open", "verify-final", "guard-close"]


def test_new_started_run_is_committed_before_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeGuard.events = []
    pg = _RecordingPostgres()
    events: list[str] = []
    original_commit = pg.commit

    def commit() -> None:
        events.append("commit")
        original_commit()

    def ensure_run(_pg: object, _receipt: object, _run_key: str) -> int:
        events.append("ensure-run")
        pg.run_head = {"migration_run_event_id": 8, "event_type": "started"}
        return 12

    def fail_after_start(_pg: object) -> dict[str, str]:
        events.append("import-phase")
        raise RuntimeError("expected failure")

    pg.commit = commit  # type: ignore[method-assign]
    monkeypatch.setattr(importer_module, "capture_sqlite_source_receipt", lambda _path: _fake_receipt())
    monkeypatch.setattr(importer_module, "SQLiteSourceGuard", _FakeGuard)
    monkeypatch.setattr(importer_module, "_ensure_source_and_run", ensure_run)
    monkeypatch.setattr(importer_module, "_source_job_map", fail_after_start)

    with pytest.raises(RuntimeError, match="expected failure"):
        importer_module.import_sqlite_to_postgres(
            pg, "sealed.db", expected_sha256="a" * 64, run_key="run-1"
        )

    ensure_index = events.index("ensure-run")
    import_index = events.index("import-phase")
    assert "commit" in events[ensure_index + 1 : import_index]
    assert len([statement for statement in pg.statements if "'failed'" in statement[0]]) == 1


def _label_source(*, job_url: str | None, item_id: str | None) -> sqlite3.Connection:
    source = sqlite3.connect(":memory:")
    source.row_factory = sqlite3.Row
    source.execute(
        """CREATE TABLE research_labels (
               id TEXT PRIMARY KEY,
               job_url TEXT,
               item_id TEXT,
               source_project_id TEXT,
               decision TEXT,
               rating INTEGER,
               reason TEXT,
               tags_json TEXT,
               method TEXT,
               created_at TEXT,
               raw_event_json TEXT
           )"""
    )
    source.execute(
        "INSERT INTO research_labels VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "label-1",
            job_url,
            item_id,
            "calibration",
            "downvote",
            1,
            "Not qualified",
            "[]",
            "typed_reason",
            "2026-05-25T00:00:00+00:00",
            "{}",
        ),
    )
    return source


def _insert_statements(pg: _RecordingPostgres, relation: str) -> list[tuple[str, tuple[Any, ...]]]:
    return [statement for statement in pg.statements if f"INSERT INTO {relation}" in statement[0]]


def test_migration_run_resume_event_starts_new_and_retryable_runs() -> None:
    assert _migration_run_resume_event(None) == ("started", None)
    assert _migration_run_resume_event({"migration_run_event_id": 4, "event_type": "planned"}) == ("started", 4)
    assert _migration_run_resume_event({"migration_run_event_id": 5, "event_type": "failed"}) == ("started", 5)
    assert _migration_run_resume_event({"migration_run_event_id": 6, "event_type": "started"}) is None


@pytest.mark.parametrize("event_type", ["completed", "aborted"])
def test_migration_run_resume_event_rejects_terminal_runs(event_type: str) -> None:
    with pytest.raises(Exception, match="terminal.*new run_key"):
        _migration_run_resume_event({"migration_run_event_id": 7, "event_type": event_type})


def test_mark_run_failed_appends_terminal_event_without_error_text() -> None:
    pg = _RecordingPostgres()
    pg.run_head = {"migration_run_event_id": 8, "event_type": "started"}

    _mark_run_failed(pg, 12, RuntimeError("secret-bearing error text"))

    failed = [statement for statement in pg.statements if "'failed'" in statement[0]]
    assert len(failed) == 1
    assert failed[0][1][0] == 12
    assert failed[0][1][-1] == 8
    assert "secret-bearing" not in repr(failed[0][1])
    assert pg.commits == 1


def test_item_scoped_label_without_job_url_is_imported_not_quarantined() -> None:
    source = _label_source(job_url=None, item_id="applypilot:untargeted-negative:item-1")
    pg = _RecordingPostgres()

    imported, quarantined = _insert_labels(pg, source, {}, 7)

    assert (imported, quarantined) == (1, 0)
    label_inserts = _insert_statements(pg, "brain_label_events")
    assert len(label_inserts) == 1
    assert label_inserts[0][1][2:5] == (None, "applypilot:untargeted-negative:item-1", None)
    assert not _insert_statements(pg, "brain_migration_quarantine")


def test_label_without_any_supported_endpoint_is_quarantined() -> None:
    source = _label_source(job_url=None, item_id=None)
    pg = _RecordingPostgres()

    imported, quarantined = _insert_labels(pg, source, {}, 7)

    assert (imported, quarantined) == (0, 1)
    assert not _insert_statements(pg, "brain_label_events")
    quarantine = _insert_statements(pg, "brain_migration_quarantine")
    assert len(quarantine) == 1
    assert "missing_label_endpoint" in quarantine[0][0]


def test_item_scoped_confidence_supersedes_original_label_without_job_url() -> None:
    source = _label_source(job_url=None, item_id="applypilot:untargeted-negative:item-1")
    source.execute(
        """CREATE TABLE research_label_confidence (
               label_id TEXT PRIMARY KEY,
               item_id TEXT,
               weight REAL NOT NULL,
               item_flip_rate REAL NOT NULL,
               method TEXT NOT NULL,
               imported_at TEXT NOT NULL,
               raw_json TEXT NOT NULL
           )"""
    )
    source.execute(
        "INSERT INTO research_label_confidence VALUES (?,?,?,?,?,?,?)",
        (
            "label-1",
            "applypilot:untargeted-negative:item-1",
            2 / 3,
            1 / 3,
            "confidence",
            "2026-05-26T00:00:00+00:00",
            "{}",
        ),
    )
    pg = _RecordingPostgres()

    imported, quarantined = _insert_label_confidence(pg, source, {}, 7)

    assert (imported, quarantined) == (1, 0)
    label_inserts = _insert_statements(pg, "brain_label_events")
    assert len(label_inserts) == 1
    assert label_inserts[0][1][2:5] == (None, "applypilot:untargeted-negative:item-1", None)
    assert label_inserts[0][1][7] == Decimal("0.6666666666666667")
    assert label_inserts[0][1][8] == Decimal("0.6666666666666666")
    assert label_inserts[0][1][-1] == 42
    assert not _insert_statements(pg, "brain_migration_quarantine")


def test_observation_delta_uses_idempotent_source_key_inserts() -> None:
    source = sqlite3.connect(":memory:")
    source.row_factory = sqlite3.Row
    source.execute(
        "CREATE TABLE jobs (url TEXT PRIMARY KEY, discovered_at TEXT, detail_scraped_at TEXT, "
        "scored_at TEXT, applied_at TEXT, site TEXT, source_board TEXT)"
    )
    source.executemany(
        "INSERT INTO jobs VALUES (?,?,?,?,?,?,?)",
        [
            ("https://jobs/1", "2026-07-16T00:00:00Z", None, None, None, "one", "board"),
            ("https://jobs/2", "2026-07-17T00:00:00Z", None, None, None, "two", "board"),
        ],
    )
    pg = _RecordingPostgres()

    assert _insert_observations(
        pg,
        source,
        {"https://jobs/1": "job-1", "https://jobs/2": "job-2"},
    ) == 2

    inserts = [statement for statement in pg.statements if "INSERT INTO brain_job_observations" in statement[0]]
    assert len(inserts) == 2
    assert all(
        "ON CONFLICT (source_namespace,source_observation_id) DO NOTHING" in query
        for query, _ in inserts
    )



def test_snapshot_roles_are_bindable_compiled_policy_roles() -> None:
    assert {"label_snapshot", "pairwise_snapshot", "outcome_snapshot"} <= (
        importer_module._BINDABLE_COMPILED_POLICY_ROLES
    )


@pytest.mark.parametrize(
    "target",
    [
        None,
        {
            "artifact_hash": hashlib.sha256(b"snapshot-reference").hexdigest(),
            "byte_length": len(b"snapshot-reference") + 1,
            "media_type": ARTIFACT_MEDIA_TYPE,
            "provenance": {"policy_source_id": "snapshot-reference"},
        },
        {
            "artifact_hash": hashlib.sha256(b"snapshot-reference").hexdigest(),
            "byte_length": len(b"snapshot-reference"),
            "media_type": "application/octet-stream",
            "provenance": {"policy_source_id": "snapshot-reference"},
        },
        {
            "artifact_hash": hashlib.sha256(b"snapshot-reference").hexdigest(),
            "byte_length": len(b"snapshot-reference"),
            "media_type": ARTIFACT_MEDIA_TYPE,
            "provenance": {"policy_source_id": "wrong"},
        },
    ],
)
def test_snapshot_artifact_fails_closed_on_missing_or_conflicting_ledger(
    target: dict[str, Any] | None,
) -> None:
    content = b"snapshot-reference"
    digest = hashlib.sha256(content).hexdigest()
    artifact = ArtifactDescriptor(
        role="outcome_snapshot",
        sha256=digest,
        byte_length=len(content),
        media_type=ARTIFACT_MEDIA_TYPE,
        content=content,
        policy_source_id=content.decode("utf-8"),
    )
    pg = _RecordingPostgres()
    if target is not None:
        pg.artifacts[digest] = target

    with pytest.raises(Exception, match="missing or inconsistent"):
        _require_artifact(pg, artifact)


def test_snapshot_artifact_accepts_exact_registered_policy_source_id() -> None:
    content = b"snapshot-reference"
    digest = hashlib.sha256(content).hexdigest()
    artifact = ArtifactDescriptor(
        role="label_snapshot",
        sha256=digest,
        byte_length=len(content),
        media_type=ARTIFACT_MEDIA_TYPE,
        content=content,
        policy_source_id=content.decode("utf-8"),
    )
    pg = _RecordingPostgres()
    pg.artifacts[digest] = {
        "artifact_hash": digest,
        "byte_length": len(content),
        "media_type": ARTIFACT_MEDIA_TYPE,
        "provenance": {"policy_source_id": artifact.policy_source_id},
    }

    _require_artifact(pg, artifact)
