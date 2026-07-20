import os
from pathlib import Path
import runpy
import subprocess
import sys

from applypilot.apply import pgqueue
from applypilot.fleet.software_version import current_sw_version


REPO = Path(__file__).resolve().parents[1]
FORBIDDEN_DATABASE_ENV_VARS = runpy.run_path(str(REPO / "fleet_agent_env.py"))[
    "FORBIDDEN_DATABASE_ENV_VARS"
]


def _run_version_script(fleet_db: str) -> str:
    env = os.environ.copy()
    for name in FORBIDDEN_DATABASE_ENV_VARS:
        env.pop(name, None)
    env["FLEET_PG_DSN"] = fleet_db
    result = subprocess.run(
        [sys.executable, str(REPO / "fleet-agent-version.py")],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def test_fleet_agent_version_script_reports_matching_pin(fleet_db, fleet_runtime_db) -> None:
    current = current_sw_version(repo=REPO)
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE fleet_config SET pinned_worker_version=%s WHERE id=1", (current,))
        conn.commit()

    assert _run_version_script(fleet_runtime_db) == f"OK|{current}|{current}|match"


def test_fleet_agent_version_script_reports_drift(fleet_db, fleet_runtime_db) -> None:
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE fleet_config SET pinned_worker_version='0.3.0+git.tree.deadbee' WHERE id=1")
        conn.commit()

    line = _run_version_script(fleet_runtime_db)

    assert line.startswith("OK|")
    assert line.endswith("|drift")
    assert "|0.3.0+git.tree.deadbee|" in line
