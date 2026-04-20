"""Score unscored jobs one at a time, committing after each, with structured logging and retry-safe error handling."""
import sys, os, time, logging, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from applypilot.config import load_env, ensure_dirs, RESUME_PATH
from applypilot.database import init_db, get_connection
from applypilot.scoring.scorer import score_job
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("score_batch.log", mode="a"),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger(__name__)

def main():
    load_env()
    ensure_dirs()
    init_db()
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    jobs = conn.execute(
        "SELECT url, title, site, location, full_description FROM jobs "
        "WHERE fit_score IS NULL AND full_description IS NOT NULL "
        "ORDER BY discovered_at DESC"
    ).fetchall()

    total = len(jobs)
    log.info("Found %d unscored jobs to process", total)
    scored = 0
    errors = 0
    t0 = time.time()

    for i, row in enumerate(jobs, 1):
        url, title, site, location, desc = row
        job = {"title": title, "site": site, "location": location,
               "full_description": desc, "url": url}
        try:
            result = score_job(resume_text, job)
            scored += 1
            if result["score"] == 0:
                errors += 1
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE jobs SET fit_score=?, score_reasoning=?, scored_at=? WHERE url=?",
                (result["score"], f"{result['keywords']}\n{result['reasoning']}", now, url),
            )
            conn.commit()
            elapsed = time.time() - t0
            rate = scored / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate / 60 if rate > 0 else 0
            log.info("[%d/%d] score=%d  %s  (%.1f/min, ETA %.0fm)",
                     i, total, result["score"], title[:50], rate*60, eta)
        except Exception:
            log.error("Error scoring job %d '%s':\n%s", i, title[:40], traceback.format_exc())
            errors += 1
            continue

    elapsed = time.time() - t0
    log.info("Done: %d scored (%d errors) in %.1f min", scored, errors, elapsed/60)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("Fatal error:\n%s", traceback.format_exc())
        sys.exit(1)
