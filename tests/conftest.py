"""Shared fixtures for ApplyPilot test suite.

@file conftest.py
@description Provides common fixtures and monkeypatch helpers for offline,
             deterministic testing. No live network or API calls.
"""

from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Ensure every test gets a clean environment.

    - Removes APPLY_BACKEND so default-selection tests are deterministic.
    - Points APPLYPILOT_DIR to a temp directory to avoid touching real data.
    - Creates required sub-directories so config.ensure_dirs() is unnecessary.
    """
    monkeypatch.delenv("APPLY_BACKEND", raising=False)
    monkeypatch.setenv("APPLYPILOT_DIR", str(tmp_path / "applypilot"))
    (tmp_path / "applypilot" / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "applypilot" / "apply-workers").mkdir(parents=True, exist_ok=True)
