from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import importlib.util
import json

from applypilot.fleet import pg_roles


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upgrade-brain-authority-existing-topology.py"


def test_cli_pins_candidate_source_before_stale_editable(tmp_path: Path) -> None:
    stale = tmp_path / "stale"
    package = stale / "applypilot" / "fleet"
    package.mkdir(parents=True)
    (stale / "applypilot" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "pg_roles.py").write_text(
        "raise RuntimeError('stale editable import won')\n", encoding="utf-8"
    )
    environment = {**os.environ, "PYTHONPATH": str(stale)}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "stale editable import won" not in result.stderr
    assert "--expected-database-name" in result.stdout
    assert "--expected-system-identifier" in result.stdout


def test_committed_receipt_uses_forward_v7_containment(tmp_path: Path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("authority_upgrade_cli", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt_path = (tmp_path / "receipt.json").resolve()
    rollback_path = (tmp_path / "rollback.sql").resolve()
    rollback_sql = "SELECT 1;\n"

    class FakeConnection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False

    receipt = pg_roles.BootstrapReceipt(
        database_name="postgres", session_user="postgres", topology={}, inventory={},
        effective_connect_grantees=(), rollback_sql=rollback_sql,
        escalation_required=False, bootstrapped_at="2001-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(module.psycopg, "connect", lambda *_args, **_kwargs: FakeConnection())
    def fake_upgrade(*_args, evidence_paths, **_kwargs):
        evidence_paths.rollback_sql_path.write_text(rollback_sql, encoding="utf-8")
        return receipt

    monkeypatch.setattr(module, "upgrade_brain_authority_existing_topology", fake_upgrade)
    monkeypatch.setattr(module, "_replace_durable", lambda path, payload: path.write_bytes(payload))
    monkeypatch.setenv("APPLYPILOT_ADMIN_PG_DSN", "postgresql://unused")
    monkeypatch.setenv("APPLYPILOT_BACKUP_PG_PASSWORD", "disposable-test-only")
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_HEX", (b"x" * 32).hex())
    monkeypatch.setenv("APPLYPILOT_ROLLBACK_HMAC_KEY_ID", "test-key")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--database-owner-role", "owner",
        "--controller-role", "controller", "--verifier-role", "brain_schema_verifier",
        "--migrator-role", "brain_schema_migrator", "--retired-admin-role", "retired",
        "--infrastructure-superuser-role", "postgres", "--expected-database-name", "postgres",
        "--expected-system-identifier", "1", "--receipt-path", str(receipt_path),
        "--rollback-sql", str(rollback_path)])
    assert module.main() == 0
    assert json.loads(receipt_path.read_text())["rollback_mode"] == "forward_v7_containment"
