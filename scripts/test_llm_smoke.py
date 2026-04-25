"""Smoke test for the Ollama Cloud routing.

Runs a single tailor + cover-letter call against ONE master-lane job from
the live applypilot.db and prints a short report. No DB writes, no PDF
generation, no batch processing. Exit code is non-zero on any failure so
this can be wired into CI / cron health checks.

Usage:
    python scripts/test_llm_smoke.py            # uses one random master-lane job
    python scripts/test_llm_smoke.py --url URL  # use a specific job
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Ensure src/ is on the path when invoked directly.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from applypilot.config import RESUME_PATH, load_profile  # noqa: E402
from applypilot.database import get_connection  # noqa: E402
from applypilot.llm import get_stage_client  # noqa: E402
from applypilot.scoring.cover_letter import generate_cover_letter  # noqa: E402
from applypilot.scoring.tailor import tailor_resume  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke")


def _pick_job(url: str | None = None) -> dict:
    conn = get_connection()
    if url:
        row = conn.execute("SELECT * FROM jobs WHERE url = ?", (url,)).fetchone()
    else:
        # Master lane = jobs scored highly enough for tailoring; pick any
        # with a real description so the LLM has something to chew on.
        row = conn.execute(
            "SELECT * FROM jobs "
            "WHERE full_description IS NOT NULL AND length(full_description) > 500 "
            "AND COALESCE(fit_score, 0) >= 6 "
            "ORDER BY fit_score DESC, RANDOM() LIMIT 1"
        ).fetchone()
    if not row:
        raise SystemExit("No suitable job found in applypilot.db for smoke test.")
    return dict(zip(row.keys(), row)) if not isinstance(row, dict) else row


def _ping_stage(stage: str) -> tuple[str, str]:
    """One-line round-trip to confirm the route + model id are alive."""
    client, model = get_stage_client(stage)
    base = client.base_url
    t0 = time.time()
    out = client.chat(
        [{"role": "user", "content": "Reply with exactly: PONG"}],
        max_tokens=8,
        temperature=0.0,
        model=model,
    )
    dt = time.time() - t0
    log.info("[%s] %s @ %s -> %r (%.2fs)", stage, model, base, out.strip()[:40], dt)
    return model, out.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Specific job URL to use; default = highest-scoring master-lane job.")
    parser.add_argument("--skip-cover", action="store_true", help="Skip cover-letter step.")
    args = parser.parse_args()

    if not os.environ.get("OLLAMA_API_KEY"):
        log.error("OLLAMA_API_KEY is not set in this process environment. "
                  "On Windows it is a user-scope env var; new shells inherit it. "
                  "If running from VS Code, restart the integrated terminal.")
        return 2

    failures: list[str] = []

    # 1. Per-stage round-trip pings -- cheap canary, fails fast on auth/model id errors.
    for stage in ("TAILOR", "COVER_LETTER", "ENRICH"):
        try:
            _ping_stage(stage)
        except Exception as exc:  # noqa: BLE001
            log.error("ping %s FAILED: %s", stage, exc)
            failures.append(f"{stage} ping: {exc}")

    if failures:
        log.error("Stage pings failed; aborting before full tailor pass.")
        for f in failures:
            log.error("  - %s", f)
        return 1

    # 2. Real tailor + cover letter on one job.
    profile = load_profile()
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    job = _pick_job(args.url)
    log.info("Smoke job: %s @ %s | score=%s",
             (job.get("title") or "")[:60], job.get("site"), job.get("fit_score"))

    t0 = time.time()
    tailored, report = tailor_resume(resume_text, job, profile, max_retries=1, validation_mode="lenient")
    log.info("Tailor: %d chars in %.1fs (attempts=%d, status=%s)",
             len(tailored), time.time() - t0, report.get("attempts"), report.get("status"))
    if not tailored or len(tailored) < 200:
        failures.append("Tailor returned empty or too-short output.")
    log.info("Tailor sample: %s", tailored[:240].replace("\n", " | "))

    if not args.skip_cover:
        t1 = time.time()
        letter = generate_cover_letter(resume_text, job, profile, max_retries=1, validation_mode="lenient")
        log.info("Cover letter: %d chars in %.1fs", len(letter), time.time() - t1)
        if not letter or len(letter) < 200:
            failures.append("Cover letter returned empty or too-short output.")
        log.info("CL sample: %s", letter[:240].replace("\n", " | "))

    if failures:
        for f in failures:
            log.error("FAIL: %s", f)
        return 1

    log.info("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
