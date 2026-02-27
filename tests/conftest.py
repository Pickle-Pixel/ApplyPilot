"""Pytest bootstrap checks for required runtime dependencies."""

from importlib.util import find_spec

import pytest


if find_spec("litellm") is None:
    raise pytest.UsageError(
        "Missing required dependency 'litellm'. Install dependencies with "
        '`python -m pip install -e ".[dev]"` before running tests.'
    )
