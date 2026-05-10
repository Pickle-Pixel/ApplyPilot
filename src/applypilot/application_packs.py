"""Build human-facing application packs from ready ApplyPilot jobs."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import sqlite3

from applypilot.config import APPLICATION_PACK_DIR, load_avoid_companies, is_avoid_company
from applypilot.database import get_connection

MIN_PDF_BYTES = 1024
EXCLUDED_APPLY_STATUSES = {
    "manual",
    "avoid",
    "manual_review",
    "in_progress",
    "stale_in_progress_orphaned",
}


def _slug(text: str, fallback: str = "job") -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug[:72] or fallback


def _pdf_for(text_path: str | None) -> Path | None:
    if not text_path:
        return None
    return Path(text_path).with_suffix(".pdf")


def _valid_pdf(path: Path | None) -> bool:
    return bool(path and path.is_file() and path.stat().st_size >= MIN_PDF_BYTES)


def _blocker_reason_for_status(job: dict) -> str | None:
    raw_status = (job.get("apply_status") or "").strip().lower()
    if not raw_status:
        return None
    if raw_status == "failed" or raw_status.startswith("failed:"):
        reason = (job.get("apply_error") or "").strip()
        if reason:
            return f"already failed: {reason}"
        return "already failed"
    if raw_status in EXCLUDED_APPLY_STATUSES:
        return f"blocked status: {raw_status}"
    return f"blocked status: {raw_status}"


def _is_ready_status(raw_status: str | None) -> bool:
    return not (raw_status or "").strip()


def _write_next_steps(pack_dir: Path, job: dict, apply_url: str) -> None:
    dry_run_command = (
        f'.\\.venv\\Scripts\\applypilot.exe apply --dry-run --headless '
        f'--url "{apply_url}" --limit 1'
    )
    manual_checklist = "\n".join(
        [
            "- Open the apply URL in a signed-in browser if the dry-run hits a login gate.",
            "- Use `01_resume.pdf` as the primary upload.",
            "- Use `02_cover_letter.pdf` only when the form asks for a cover letter.",
            "- Do not submit until the final review page shows the correct name, email, phone, resume, and work authorization.",
            "- If CAPTCHA, SSO, or account setup blocks automation, finish that step manually and rerun dry-run first.",
        ]
    )
    content = f"""# {job["title"]}

Score: {job.get("fit_score", "N/A")}
Lane: {job.get("job_lane") or "primary"}
Source: {job.get("site") or "Unknown"}

Apply URL:
{apply_url}

Original Job URL:
{job["url"]}

## Artifacts

- Resume PDF: `01_resume.pdf`
- Cover Letter PDF: `02_cover_letter.pdf`

## Dry Run

```powershell
{dry_run_command}
```

## Manual Review Checklist

{manual_checklist}
"""
    (pack_dir / "00_NEXT_STEPS.md").write_text(content, encoding="utf-8")


def export_ready_packs(
    *,
    conn: sqlite3.Connection | None = None,
    output_root: Path | None = None,
    batch_name: str | None = None,
    min_score: int = 7,
    limit: int = 10,
    lane: str = "primary",
) -> dict:
    """Export ready application materials into ApplyPilot-owned pack folders."""
    conn = conn or get_connection()
    output_root = output_root or APPLICATION_PACK_DIR
    batch_name = batch_name or f"ready_{lane}_{datetime.now().strftime('%Y-%m-%d')}"
    batch_dir = output_root / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    lane_clause = "" if lane == "all" else "AND COALESCE(job_lane, 'primary') = ?"
    params: list[object] = [min_score]
    if lane != "all":
        params.append(lane)

    rows = conn.execute(
        f"""
        SELECT title, site, company, fit_score, job_lane, url, application_url,
               tailored_resume_path, cover_letter_path, apply_status, apply_attempts,
               apply_error
        FROM jobs
        WHERE fit_score >= ?
          AND tailored_resume_path IS NOT NULL AND tailored_resume_path <> ''
          AND cover_letter_path IS NOT NULL AND cover_letter_path <> ''
          AND applied_at IS NULL
          {lane_clause}
        ORDER BY fit_score DESC, title
        """,
        params,
    ).fetchall()

    jobs = [dict(row) for row in rows]
    exported: list[dict] = []
    skips: list[dict] = []

    # Hard-exclude anything matching the AVOID list — belt-and-suspenders
    # on top of the ingest-time filter.
    avoid_entries = load_avoid_companies()
    if avoid_entries:
        filtered: list[dict] = []
        for job in jobs:
            if is_avoid_company(job.get("company") or job.get("site") or "", avoid_entries):
                skips.append({
                    "title": job.get("title", ""),
                    "url": job.get("url", ""),
                    "reason": "company on AVOID list",
                })
                continue
            filtered.append(job)
        jobs = filtered

    for job in jobs:
        if len(exported) >= limit:
            break
        blocker_reason = None
        if not _is_ready_status(job.get("apply_status")):
            blocker_reason = _blocker_reason_for_status(job)
        if blocker_reason:
            skips.append({"title": job["title"], "url": job["url"], "reason": blocker_reason})
            continue
        resume_pdf = _pdf_for(job.get("tailored_resume_path"))
        cover_pdf = _pdf_for(job.get("cover_letter_path"))
        missing = []
        if not _valid_pdf(resume_pdf):
            missing.append("missing resume PDF")
        if not _valid_pdf(cover_pdf):
            missing.append("missing cover letter PDF")
        if missing:
            skips.append({"title": job["title"], "url": job["url"], "reason": "; ".join(missing)})
            continue

        pack_slug = _slug(f"{job.get('site') or 'source'}_{job['title']}")
        pack_dir = batch_dir / pack_slug
        pack_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resume_pdf, pack_dir / "01_resume.pdf")
        shutil.copy2(cover_pdf, pack_dir / "02_cover_letter.pdf")
        apply_url = job.get("application_url") or job["url"]
        _write_next_steps(pack_dir, job, apply_url)
        exported.append({"title": job["title"], "url": apply_url, "path": str(pack_dir)})

    index_lines = [f"# Ready Application Packs - {batch_name}", ""]
    if exported:
        index_lines.append("## Exported")
        for item in exported:
            index_lines.append(f"- [{item['title']}]({Path(item['path']).name}/00_NEXT_STEPS.md) - {item['url']}")
        index_lines.append("")
    if skips:
        index_lines.append("## Skipped")
        for item in skips:
            index_lines.append(f"- {item['title']} - {item['reason']}")
        index_lines.append("")
    if not exported and not skips:
        index_lines.append("No ready jobs matched the filters.")
    (batch_dir / "00_INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

    return {
        "output_dir": str(batch_dir),
        "exported": len(exported),
        "skipped": len(skips),
        "packs": exported,
        "skips": skips,
    }
