"""Standalone Linux regression for the authenticated HBA exchange boundary.

Run with a Python 3.10+ interpreter; no project dependencies are required.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
import tempfile
import types


def _load_rollback_module():
    psycopg = types.ModuleType("psycopg")
    psycopg.connect = None
    psycopg.sql = types.SimpleNamespace()
    rows = types.ModuleType("psycopg.rows")
    rows.dict_row = object()
    pg_roles = types.ModuleType("applypilot.fleet.pg_roles")
    pg_roles.authenticate_evidence_receipt = lambda receipt, **_kwargs: receipt
    pg_roles.verify_evidence_receipt = lambda *_args, **_kwargs: None
    applypilot = types.ModuleType("applypilot")
    fleet = types.ModuleType("applypilot.fleet")
    sys.modules.update(
        {
            "psycopg": psycopg,
            "psycopg.rows": rows,
            "applypilot": applypilot,
            "applypilot.fleet": fleet,
            "applypilot.fleet.pg_roles": pg_roles,
        }
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "rollback-fleet-pg-role.py"
    spec = importlib.util.spec_from_file_location("rollback_hba_posix_probe", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_rollback_module()
    assert module._hba_restore_supported(), "Linux renameat2 exchange is unavailable"
    with tempfile.TemporaryDirectory(prefix="applypilot-hba-exchange-") as directory:
        live = Path(directory) / "pg_hba.conf"
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
                current = live.stat()
                assert (current.st_dev, current.st_ino) == (before.st_dev, before.st_ino)
            return real_exchange(parent_descriptor, first_name, second_name)

        module._rename_exchange = inject_then_exchange
        try:
            module._replace_atomic(
                live,
                replacement,
                expected_identity=(before.st_dev, before.st_ino),
                expected_content=authenticated,
                expected_sha256=hashlib.sha256(authenticated).hexdigest(),
                replacement_mode=stat.S_IMODE(before.st_mode),
                replacement_owner=(before.st_uid, before.st_gid),
            )
        except RuntimeError as error:
            assert "immediately before atomic exchange" in str(error)
        else:
            raise AssertionError("same-inode mutation crossed the authenticated exchange boundary")
        assert injected
        assert live.read_bytes() == edited
        assert not list(Path(directory).glob(".applypilot-hba-rollback-*"))

        module._rename_exchange = real_exchange
        current = live.stat()
        module._replace_atomic(
            live,
            replacement,
            expected_identity=(current.st_dev, current.st_ino),
            expected_content=edited,
            expected_sha256=hashlib.sha256(edited).hexdigest(),
            replacement_mode=stat.S_IMODE(current.st_mode),
            replacement_owner=(os.getuid(), os.getgid()),
        )
        assert live.read_bytes() == replacement
        assert not list(Path(directory).glob(".applypilot-hba-rollback-*"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
