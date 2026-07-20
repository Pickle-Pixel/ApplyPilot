"""Static regression guard for the non-publishing container build gate (task C2).

This test never runs `docker build` and never touches the network. It only
parses `.github/workflows/ci.yml`, the `Dockerfile`, and `.dockerignore` as
text/YAML and asserts the properties the CI job promises:

- the `container-build` job exists, builds on GitHub-hosted Linux, and never
  grows a registry login or an actual push, and never gains access to
  repository environments or secrets;
- the workflow's triggers stay bounded to pull requests / workflow_dispatch /
  push-with-path-filters (never `push: branches: [main]` unconditionally, and
  never every branch);
- the Dockerfile's entrypoint contract (CMD) and the build context's
  PII/test/secret exclusions (.dockerignore) haven't silently drifted.

These are cheap, deterministic, whole-file assertions -- if a future edit
reintroduces `docker/login-action`, flips `push: true`, adds an
`environment:` key, or removes `tests`/`.git`/`.env` from `.dockerignore`,
this test fails without needing a Docker daemon.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

CONTAINER_JOB_NAME = "container-build"
WINDOWS_NATIVE_CI_NODES = (
    "tests/test_load_canary_home_spend_cap.py::"
    "test_native_exit_guard_stops_later_steps_after_a_failed_command",
    "tests/test_mapped_role_deployment_contract.py::"
    "test_hba_post_write_failure_restores_hba_and_outputs_byte_exact",
)


def _load_workflow() -> dict:
    text = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    assert isinstance(doc, dict)
    return doc


def _triggers(doc: dict) -> dict:
    # PyYAML's YAML-1.1 safe_load parses the bare top-level `on:` key as the
    # boolean `True`, not the string "on". Handle both so this test doesn't
    # silently pass on a doc it never actually inspected.
    if "on" in doc:
        return doc["on"]
    return doc[True]


def _container_job(doc: dict) -> dict:
    jobs = doc.get("jobs", {})
    assert CONTAINER_JOB_NAME in jobs, (
        f"expected a '{CONTAINER_JOB_NAME}' job in {CI_WORKFLOW_PATH}"
    )
    job = jobs[CONTAINER_JOB_NAME]
    assert isinstance(job, dict)
    return job


def test_workflow_yaml_parses() -> None:
    doc = _load_workflow()
    assert "jobs" in doc


def test_container_job_runs_on_hosted_linux() -> None:
    job = _container_job(_load_workflow())
    assert job.get("runs-on") == "ubuntu-latest"


def test_container_job_never_touches_environments_or_secrets() -> None:
    job = _container_job(_load_workflow())
    assert "environment" not in job, (
        "container-build must not access repository environments"
    )
    job_text = yaml.dump(job)
    assert "secrets." not in job_text, (
        "container-build must not reference repository secrets"
    )
    assert "secrets:" not in job_text, (
        "container-build must not accept/forward job-level secrets"
    )


def test_container_job_never_logs_in_or_pushes() -> None:
    job = _container_job(_load_workflow())
    steps = job.get("steps", [])
    assert steps, "container-build job has no steps"

    for step in steps:
        uses = step.get("uses", "") or ""
        assert "login-action" not in uses, (
            f"container-build must not log in to a registry (found step using {uses!r})"
        )

        if "build-push-action" in uses:
            with_block = step.get("with", {}) or {}
            # Must be explicitly non-publishing.
            assert with_block.get("push") is False, (
                "build-push-action step must set push: false"
            )

        run = step.get("run", "") or ""
        assert "docker login" not in run, "container-build must not call `docker login`"
        assert "docker push" not in run, "container-build must not call `docker push`"


def test_triggers_are_bounded() -> None:
    doc = _load_workflow()
    triggers = _triggers(doc)
    assert isinstance(triggers, dict)

    # Bounded means: no `on: push` with unrestricted branches (no branch/path
    # filter at all), and PRs/dispatch are present so the gate can actually run.
    assert "workflow_dispatch" in triggers
    assert "pull_request" in triggers

    push = triggers.get("push")
    if push is not None:
        assert push.get("branches"), "push trigger must be branch-scoped"
        assert push.get("paths"), "push trigger must be path-scoped"

    pr = triggers.get("pull_request")
    assert pr.get("branches"), "pull_request trigger must be branch-scoped"


def test_python_ci_installs_jobspy_and_excludes_windows_only_suite() -> None:
    job = _load_workflow()["jobs"]["test"]
    steps = {step["name"]: step for step in job["steps"]}
    install = steps["Install dependencies"]["run"]
    assert "pip install --no-deps python-jobspy" in install
    test_command = steps["Test"]["run"]
    assert "--ignore=tests/test_fleet_machine_blackout_scripts.py" in test_command


def test_python_ci_marks_service_postgres_as_disposable_before_tests() -> None:
    job = _load_workflow()["jobs"]["test"]
    steps = {step["name"]: step for step in job["steps"]}
    marker = steps["Mark disposable PostgreSQL service"]
    marker_script = marker["run"]
    assert "applypilot:disposable-postgres-test-database:v1" in marker_script
    assert "CREATE ROLE applypilot_disposable_cluster_marker" in marker_script
    assert "applypilot:disposable-postgres-test-cluster:v1:" in marker_script
    test_environment = steps["Test"]["env"]
    assert test_environment["APPLYPILOT_PGTEST_DISPOSABLE"] == "1"
    assert test_environment["APPLYPILOT_PGTEST_CLUSTER_MARKER"].startswith("ci-")


def test_python_ci_enables_and_verifies_tls_on_service_postgres() -> None:
    job = _load_workflow()["jobs"]["test"]
    steps = {step["name"]: step for step in job["steps"]}
    tls_step = steps["Enable TLS on disposable PostgreSQL service"]
    assert tls_step["env"]["POSTGRES_SERVICE_ID"] == "${{ job.services.postgres.id }}"

    tls_script = tls_step["run"]
    for required in (
        "docker exec -u 0",
        "openssl req -x509",
        "-days 1",
        "-subj /CN=localhost",
        '"$PGDATA/server.crt"',
        '"$PGDATA/server.key"',
        "chown postgres:postgres",
        'chmod 600 "$PGDATA/server.key"',
        "ssl=on",
        "docker restart",
        "for attempt in $(seq 1 30)",
        "pg_isready",
    ):
        assert required in tls_script

    marker = steps["Mark disposable PostgreSQL service"]
    assert marker["env"]["APPLYPILOT_PGTEST_DSN"].endswith("?sslmode=require")
    marker_script = marker["run"]
    assert "SHOW ssl" in marker_script
    assert "PostgreSQL service did not enable TLS" in marker_script

    test_environment = steps["Test"]["env"]
    assert test_environment["APPLYPILOT_PGTEST_DSN"].endswith("?sslmode=require")


def test_pg18_repair_contract_nodes_are_routed_to_windows_only() -> None:
    doc = _load_workflow()
    windows_job = doc["jobs"]["phase-a-windows-file"]
    packets = {
        packet["name"]: packet["tests"]
        for packet in windows_job["strategy"]["matrix"]["include"]
    }
    packet_nodes = packets["pg18-ci-repair-contracts"].split()
    assert packet_nodes == list(WINDOWS_NATIVE_CI_NODES)

    linux_job = doc["jobs"]["test"]
    assert linux_job["strategy"]["fail-fast"] is False
    test_command = {
        step["name"]: step for step in linux_job["steps"]
    }["Test"]["run"]
    deselected = [
        token.removeprefix("--deselect=")
        for token in test_command.split()
        if token.startswith("--deselect=")
    ]
    assert deselected == list(WINDOWS_NATIVE_CI_NODES)
    for node in WINDOWS_NATIVE_CI_NODES:
        test_file = node.split("::", 1)[0]
        assert f"--ignore={test_file}" not in test_command


def test_python_ci_matrix_matches_declared_support() -> None:
    job = _load_workflow()["jobs"]["test"]
    assert job["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
    assert project["requires-python"] == ">=3.11,<3.13"
    assert "Programming Language :: Python :: 3.13" not in project["classifiers"]


def test_dockerfile_entrypoint_contract_unchanged() -> None:
    text = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert 'CMD ["/app/entrypoint.sh"]' in text
    assert "COPY deploy/entrypoint.sh /app/entrypoint.sh" in text


def test_dockerignore_excludes_pii_and_secrets_sources() -> None:
    text = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines()}
    required = {".git", ".env", ".env.*", "tests", ".venv"}
    missing = required - lines
    assert not missing, f".dockerignore must keep excluding {missing}"
