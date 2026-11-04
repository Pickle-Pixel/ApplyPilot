"""Shared test setup.

- APPLYPILOT_DIR points at a throwaway directory before applypilot.config is imported, so no test
  touches ~/.applypilot.
- python-jobspy is installed separately (see README), so CI's `pip install -e ".[dev]"` lacks it.
  A stub module stands in when it is missing; tests that reach JobSpy patch `scrape_jobs` anyway.
"""

import os
import sys
import tempfile
import types

os.environ.setdefault("APPLYPILOT_DIR", tempfile.mkdtemp(prefix="applypilot-tests-"))

try:
    import jobspy  # noqa: F401
except ImportError:
    _stub = types.ModuleType("jobspy")

    def _scrape_jobs(**kwargs):
        raise RuntimeError("python-jobspy is not installed; tests must patch scrape_jobs")

    _stub.scrape_jobs = _scrape_jobs
    sys.modules["jobspy"] = _stub
