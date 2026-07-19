"""Adversarial image-package regression for the private authority-migrator.

Unlike ``test_authority_migrator_deployment.py`` -- which parses the wrappers
statically -- this module *reconstructs the image filesystem from the actual
Dockerfile ``COPY`` contract* and drives the **real** default Railway/Docker
entrypoint against it, in a fresh interpreter, so it proves behaviour rather than
strings.

The bug this guards against: the manifest binds three wrappers
(``Dockerfile.authority-migrator``, ``railway.toml``,
``scripts/run-private-authority-migrator.py``) by hash *relative to the /app
working directory*, and ``verify_package_manifest`` runs before any migration.
If the Dockerfile fails to ``COPY`` any bound wrapper into ``/app``, every
deployed startup fails closed with a missing-bound-wrapper error -- the image is
dead on arrival. A green static parse would not catch that, because the file
exists in the repo; it just never reaches the image.

What the tests below assert, end to end:

* the reconstructed ``/app`` tree contains every wrapper the manifest binds, at
  its exact manifest-relative path (filesystem check, not a string match);
* running the *real* entrypoint script from the reconstructed tree, with the
  exact argument vector Railway supplies and with ``repository-root`` /
  ``package-manifest`` left to default to ``/app``, gets *past* package
  verification (it stops only at the deliberately-unset target DSN);
* deleting any one bound wrapper from the reconstructed image, or drifting a
  single byte of it, makes that same default entrypoint fail closed with the
  expected ``PackageError``.

The entrypoint is executed as a genuine subprocess with a clean module table so
that ``_REPOSITORY_ROOT`` resolves to the reconstructed ``/app`` exactly as it
would inside the container, and ``applypilot`` is imported from the staged
``src`` tree -- faithful to the running image, not to this repo checkout.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile.authority-migrator"
RAILWAY_TOML = REPO_ROOT / "railway.toml"
PACKAGE_MANIFEST = REPO_ROOT / "deploy" / "authority-migrator" / "package-manifest.json"

# The image WORKDIR / repository root the wrapper resolves to at runtime.
IMAGE_WORKDIR = "/app"
ENTRYPOINT_IN_IMAGE = "/app/scripts/run-private-authority-migrator.py"


def _copy_pairs() -> list[tuple[str, str]]:
    """Parse ``COPY <src> <dest>`` pairs from the migrator Dockerfile.

    Returns the tracked repo-relative source and the absolute in-image
    destination for every COPY instruction, in file order.
    """
    pairs: list[tuple[str, str]] = []
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY "):
            continue
        parts = shlex.split(stripped)[1:]  # drop the COPY keyword
        assert len(parts) == 2, f"unexpected COPY arity: {stripped!r}"
        src, dest = parts
        assert dest.startswith(IMAGE_WORKDIR + "/"), (
            f"COPY destination must live under {IMAGE_WORKDIR}: {dest!r}"
        )
        pairs.append((src, dest))
    assert pairs, "Dockerfile.authority-migrator must contain COPY instructions"
    return pairs


def _reconstruct_image(stage_root: Path) -> None:
    """Materialise the image ``/app`` tree solely from the Dockerfile COPY set.

    Nothing that the Dockerfile does not explicitly COPY is present -- this is the
    filesystem the running container actually sees, not a copy of the repo.
    """
    for src, dest in _copy_pairs():
        relative = dest[len(IMAGE_WORKDIR) + 1:]
        target = stage_root / relative
        source = REPO_ROOT / src
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def _railway_entrypoint_argv() -> list[str]:
    """The argument vector Railway hands the entrypoint, target values filled in.

    Substitutes the ``${...}`` service-variable placeholders with values that
    pass the wrapper's own identifier validation, so execution reaches package
    verification and then the (intentionally unset) target-DSN lookup -- never a
    parse error. ``repository-root`` and ``package-manifest`` are deliberately
    left absent so they default to ``/app`` exactly as in production.
    """
    start = tomllib.loads(RAILWAY_TOML.read_text(encoding="utf-8"))["deploy"]["startCommand"]
    tokens = shlex.split(start)
    assert tokens[:2] == ["python3", ENTRYPOINT_IN_IMAGE], (
        f"unexpected start command shape: {tokens!r}"
    )
    argv = tokens[2:]  # drop `python3 <entrypoint>`
    substitutions = {
        "${AUTHORITY_MIGRATOR_EXPECTED_DB}": "appdb",
        "${AUTHORITY_MIGRATOR_EXPECTED_SYSID}": "1234567890",
    }
    resolved = [substitutions.get(tok, tok) for tok in argv]
    # No unresolved service-variable placeholders may leak into the vector.
    assert not any(tok.startswith("${") for tok in resolved), resolved
    # repository-root / package-manifest must stay defaulted (=> /app in image).
    assert "--repository-root" not in resolved
    assert "--package-manifest" not in resolved
    return resolved


def _bound_wrappers() -> list[str]:
    payload = json.loads(PACKAGE_MANIFEST.read_bytes())
    return sorted(payload["wrappers"])


@pytest.fixture
def stage_root():
    """A *shallow* staging directory for the reconstructed image.

    The full ``src`` tree carries deeply-nested migration files; combined with
    pytest's per-test ``tmp_path`` nesting that overruns the Windows 260-char
    path limit. Staging directly under the (already scratch-scoped) temp base
    keeps every reconstructed path short enough while remaining outside the
    tracked worktree.
    """
    root = Path(tempfile.mkdtemp(prefix="ami"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _run_entrypoint(stage_root: Path) -> subprocess.CompletedProcess[str]:
    """Run the staged entrypoint script exactly as the container would.

    Fresh interpreter, clean env (no target DSN), defaulted repository-root and
    package-manifest -- so ``_REPOSITORY_ROOT`` resolves to ``stage_root`` and
    ``applypilot`` loads from the staged ``src`` tree.
    """
    script = stage_root / "scripts" / "run-private-authority-migrator.py"
    env = {
        # Minimal, target-free environment: the DSN var is intentionally unset.
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, str(script), *_railway_entrypoint_argv()],
        capture_output=True,
        text=True,
        env={k: v for k, v in env.items() if v},
        check=False,
    )


# --- The image actually carries every bound wrapper --------------------------

def test_reconstructed_image_contains_every_bound_wrapper(stage_root) -> None:
    _reconstruct_image(stage_root)
    for relative in _bound_wrappers():
        staged = stage_root / relative
        assert staged.is_file(), (
            f"the Dockerfile COPY contract never places bound wrapper {relative!r} "
            f"into {IMAGE_WORKDIR}; the image would fail package verification"
        )
    # The package manifest and fleet migration manifest must also be present.
    assert (stage_root / "deploy/authority-migrator/package-manifest.json").is_file()
    assert (stage_root / "src/applypilot/fleet/migrations/manifest-v1.json").is_file()


# --- Default Railway/Docker entrypoint passes package verification -----------

def test_default_entrypoint_package_verification_succeeds_on_shipped_files(stage_root) -> None:
    _reconstruct_image(stage_root)
    result = _run_entrypoint(stage_root)
    combined = result.stdout + result.stderr
    # Package verification runs *before* target resolution. If it had failed we
    # would see a PackageError; instead the run must advance to the unset DSN.
    assert "PackageError" not in combined, (
        f"package verification failed on the shipped image:\n{combined}"
    )
    assert "bound wrapper is missing" not in combined, combined
    assert "wrapper hash drift" not in combined, combined
    # It gets exactly as far as the deliberately-unset target DSN, then stops.
    assert result.returncode == 1, combined
    assert "TargetError" in combined and "is not set" in combined, combined


# --- Adversarial: deletion of any bound wrapper fails closed -----------------

ENTRYPOINT_RELATIVE = "scripts/run-private-authority-migrator.py"


@pytest.mark.parametrize("victim", _bound_wrappers())
def test_deleting_a_bound_wrapper_fails_closed(stage_root, victim) -> None:
    _reconstruct_image(stage_root)
    (stage_root / victim).unlink()
    result = _run_entrypoint(stage_root)
    combined = result.stdout + result.stderr
    # Whichever wrapper is removed, the container never reaches a migration.
    assert result.returncode != 0, combined
    assert "authority migration complete" not in combined, combined
    assert "TargetError" not in combined, (
        f"deletion of {victim!r} slipped past package verification:\n{combined}"
    )
    if victim == ENTRYPOINT_RELATIVE:
        # Deleting the entrypoint wrapper itself means the interpreter cannot even
        # load the script -- a distinct, still-fail-closed startup failure.
        assert "PackageError" not in combined  # never got as far as verification
        assert "run-private-authority-migrator.py" in combined, combined
    else:
        # A missing non-entrypoint wrapper is caught by package verification.
        assert "PackageError" in combined, combined
        assert "bound wrapper is missing" in combined, combined


# --- Adversarial: one-byte drift of any bound wrapper fails closed -----------

@pytest.mark.parametrize("victim", _bound_wrappers())
def test_byte_drift_of_a_bound_wrapper_fails_closed(stage_root, victim) -> None:
    _reconstruct_image(stage_root)
    staged = stage_root / victim
    staged.write_bytes(staged.read_bytes() + b"\n# adversarial drift\n")
    result = _run_entrypoint(stage_root)
    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined
    assert "PackageError" in combined, combined
    assert "wrapper hash drift" in combined, combined
    assert "TargetError" not in combined, (
        f"byte drift of {victim!r} slipped past package verification:\n{combined}"
    )
