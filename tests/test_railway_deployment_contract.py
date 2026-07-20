from pathlib import Path
import os
import subprocess


REPO = Path(__file__).resolve().parents[1]


def test_railway_entrypoint_runs_canonical_v3_worker_fail_closed() -> None:
    script = (REPO / "deploy" / "entrypoint.sh").read_text(encoding="utf-8")

    for required in (
        "require_env FLEET_PG_DSN",
        "require_env DEEPSEEK_API_KEY",
        "require_env APPLYPILOT_WORKER_ID",
        "require_env APPLYPILOT_WORKER_CONTRACT",
        "require_env APPLYPILOT_RELEASE_VERSION",
            'wait_for_file "${APPLYPILOT_DIR:-/data/applypilot}/profile.json"',
            'wait_for_file "${APPLYPILOT_DIR:-/data/applypilot}/resume.pdf"',
        'exec applypilot-fleet-apply "${worker_args[@]}"',
        '--worker-id "$APPLYPILOT_WORKER_ID"',
        '--machine-owner "$FLEET_MACHINE_OWNER"',
        ):
            assert required in script
    assert "python - <<'PY'" not in script
    assert "python3 - <<'PY'" in script

    assert "applypilot.apply.container_worker" not in script
    assert "${APPLYPILOT_WORKER_ID:-0}" not in script
    assert "require_env DATABASE_URL" not in script
    assert "export FLEET_PG_DSN=" not in script
    assert "--dsn" not in script
    assert "validate_runtime_principal" in script
    assert "conninfo_to_dict" in script
    assert "postgres" in script
    assert "fleet_worker" in script
    assert "current_user" in script
    assert "session_user" in script
    for forbidden in (
        "APPLYPILOT_ADMIN_PG_DSN",
        "APPLYPILOT_CONTROLLER_PG_DSN",
        "APPLYPILOT_SUPER_DSN",
        "DATABASE_URL",
        "DATABASE_PUBLIC_URL",
        "POSTGRES_URL",
        "POSTGRES_PUBLIC_URL",
        "PGHOST",
        "PGUSER",
        "PGPASSWORD",
        "PGDATABASE",
        "PGSERVICE",
        "PGPASSFILE",
    ):
        assert forbidden in script
    assert 'unset "$name"' in script


def test_railway_entrypoint_rejects_ambient_admin_database_variables_first() -> None:
    forbidden = (
        "APPLYPILOT_ADMIN_PG_DSN",
        "APPLYPILOT_CONTROLLER_PG_DSN",
        "DATABASE_URL",
        "DATABASE_PUBLIC_URL",
        "PGPASSWORD",
    )
    for name in forbidden:
        env = os.environ.copy()
        for key in forbidden:
            env.pop(key, None)
        result = subprocess.run(
            ["bash", "-c", f"export {name}=must-not-propagate; ./deploy/entrypoint.sh"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO,
        )
        assert result.returncode == 64
        assert name in result.stderr
        assert "required environment variable is not set" not in result.stderr


def test_docker_contract_documents_external_release_identity_and_assets() -> None:
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (REPO / ".dockerignore").read_text(encoding="utf-8")

    assert ".git" in dockerignore.splitlines()
    assert "APPLYPILOT_RELEASE_VERSION" in dockerfile
    assert "APPLYPILOT_WORKER_ID" in dockerfile
    assert "FLEET_PG_DSN" in dockerfile
    assert "DATABASE_URL" not in dockerfile
    assert "/data/applypilot" in dockerfile


def test_railway_runbook_documents_fail_closed_release_contract() -> None:
    runbook = (REPO / "docs" / "railway-canonical-worker-deployment.md").read_text(encoding="utf-8")

    for required in (
        "applypilot-fleet-apply",
        "APPLYPILOT_WORKER_ID",
        "APPLYPILOT_RELEASE_VERSION",
        "fleet_config.pinned_worker_version",
        "/data/applypilot",
        "profile.json",
        "resume.pdf",
        "Keep global and ATS lane pauses enabled",
        "Apply schema migrations",
        "fresh heartbeat",
        "unique mapped login role",
        "migration/admin-only",
        "not a fleet release path",
        "Tailscale/private gateway",
        "full Tailscale MagicDNS hostname",
        "raw Tailscale IP",
        "explicit password",
        "*.railway.internal",
        "validates the public hostname",
        "brain_schema_migrator",
        "brain_schema_verifier",
        "unique mapped login role",
        "REVOKE CONNECT ON DATABASE",
        "rollback SQL",
        "regrant manifest",
        "credential_forward_reconcile_required=true",
        "forces the role to `NOLOGIN`",
        "forward-reconciles a newly generated password",
        "DROP OWNED BY",
        "break-glass authority",
        "rollback-fleet-pg-role.py",
        "verify the rollback file's SHA-256",
        "--single-transaction --set=ON_ERROR_STOP=on",
        "already-established session",
        "in_doubt=true",
        "PostgreSQL 18",
        "historical `created_at`",
        "database `CONNECT`, `CREATE`, and `TEMPORARY`",
    ):
        assert required in runbook

    assert runbook.count("${{Postgres.DATABASE_URL}}") == 1


def test_railway_runbook_keeps_hba_closed_until_database_containment_finishes() -> None:
    runbook = (REPO / "docs" / "railway-canonical-worker-deployment.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(runbook.split())

    ordered_steps = (
        "commits authenticated phase-one database containment",
        "terminates affected non-break-glass, non-controller sessions",
        "repeats the idempotent rollback SQL in phase two",
        "asserts the final database containment postconditions",
        "Only then may it restore the recorded HBA backup",
    )
    positions = [normalized.index(step) for step in ordered_steps]

    assert positions == sorted(positions)
    assert (
        "HBA admission remains closed until both containment phases and the "
        "postconditions are durably evidenced."
    ) in normalized
    assert "Each containment phase is a separate fail-stop transaction" in normalized
