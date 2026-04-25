"""Canonical xlsx ingestion for ApplyPilot curated job lists.

Reads Darrin's WFH job xlsx (three-tab format) and inserts rows into the
jobs table under appropriate lanes:

    - "NON-VOICE MASTER"     -> job_lane="master"
    - "MIXED + VOICE BACKUP" -> job_lane="fallback"
    - "AVOID"                -> no rows inserted; entries are reported and
                                used to cross-check the active AVOID config
                                (~/.applypilot/avoid_companies.yaml).

Column expectations (both job tabs; MIXED tab has an extra "Modality" col):
    #, [Modality,] Rapid Hire, Company, Role, Pay, Type, Industry,
    Start Date, Priority, Apply Link, Source, Status, Notes / Group Intel

NEVER submits. DB-layer only (plus optional tailor/cover/pack when --run-pipeline).

Usage:
    python scripts/ingest_from_xlsx.py <xlsx_path>
    python scripts/ingest_from_xlsx.py <xlsx_path> --skip-company "Humana" \\
        --skip-company "Velera Solutions LLC"
    python scripts/ingest_from_xlsx.py <xlsx_path> --run-pipeline master \\
        --batch ready_master_2026-04-24
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from applypilot.config import (  # noqa: E402
    load_env,
    ensure_dirs,
    RESUME_PATH,
    TAILORED_DIR,
    COVER_LETTER_DIR,
    load_profile,
    load_avoid_companies,
    is_avoid_company,
)
from applypilot.database import init_db, get_connection  # noqa: E402
from applypilot.scoring.tailor import tailor_resume  # noqa: E402
from applypilot.scoring.cover_letter import generate_cover_letter  # noqa: E402
from applypilot.scoring.validator import validate_cover_letter  # noqa: E402
from applypilot.scoring.pdf import (  # noqa: E402
    batch_convert,
    convert_cover_letter_to_pdf,
)
from applypilot.application_packs import export_ready_packs  # noqa: E402

TAB_LANES = {
    "NON-VOICE MASTER": "master",
    "MIXED + VOICE BACKUP": "fallback",
}
AVOID_TAB = "AVOID"

VOICE_KEYWORDS = (
    "call center", "dialer", "outbound call", "inbound call",
    "phone rep", "voice agent", "bdr ", "sdr ",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(text: str, cap: int = 50) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text or "")[:cap].strip().replace(" ", "_")
    return cleaned or "job"


def _slug(text: str, cap: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()[:cap] or "x"


def _synth_description(row: dict, lane: str) -> str:
    company = row.get("Company", "") or ""
    role = row.get("Role", "") or ""
    pay = row.get("Pay") or "Not disclosed"
    industry = row.get("Industry") or ""
    typ = row.get("Type") or ""
    modality = row.get("Modality") or ("NON-VOICE" if lane == "master" else "MIXED/VOICE")
    notes = row.get("Notes / Group Intel") or ""
    start = row.get("Start Date") or ""
    priority = row.get("Priority") or ""
    return (
        f"Company: {company}\n"
        f"Role: {role}\n"
        f"Modality: {modality}\n"
        f"Pay: {pay}\n"
        f"Employment type: {typ}\n"
        f"Industry: {industry}\n"
        f"Start date signal: {start}\n"
        f"Priority: {priority}\n\n"
        f"Curator intel (human-vetted from sourcing group):\n{notes}\n\n"
        f"Format: Remote, US. Lane={lane}."
    )


def _is_voice_heuristic(row: dict) -> bool:
    """Voice-modality heuristic for the master (non-voice) lane.

    Checks in order:
      1. Explicit `Modality` column wins. "non-voice" / "back-office" / "chat" /
         "email" / "messaging" all suppress the heuristic. "voice" / "phone" /
         "call center" / "outbound" / "inbound" trigger it.
      2. Otherwise scan the Role title only for VOICE_KEYWORDS. Free-text
         Notes / Group Intel is curator commentary that often references
         background context (e.g. a candidate's prior call-center history) —
         substring scanning it produced false positives that killed legitimate
         non-voice WFM/analytical roles.
    """
    modality = (row.get("Modality") or "").strip().lower()
    if modality:
        non_voice_signals = ("non-voice", "back-office", "back office",
                             "chat", "email", "messaging", "async")
        voice_signals = ("voice", "phone", "call center", "outbound",
                         "inbound", "dialer")
        if any(s in modality for s in non_voice_signals):
            return False
        if any(s in modality for s in voice_signals):
            return True
    title = (row.get("Role", "") or "").lower()
    return any(kw in title for kw in VOICE_KEYWORDS)


def _load_sheet_rows(xlsx_path: Path, tab: str) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    if tab not in wb.sheetnames:
        return []
    ws = wb[tab]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    header = [str(h) if h is not None else "" for h in rows[0]]
    out: list[dict] = []
    for raw in rows[1:]:
        row = {header[i]: (raw[i] if i < len(raw) else None) for i in range(len(header))}
        # Drop rows without a company - happens on blank trailing rows
        if not (row.get("Company") and row.get("Role")):
            continue
        out.append({k: ("" if v is None else str(v)) for k, v in row.items()})
    return out


def _upsert_job(conn, row: dict, lane: str, skip_pairs: set[tuple[str, str]],
                avoid_entries: list[dict]) -> tuple[str, str]:
    """Return (status, url_or_reason).

    status in {"inserted", "skipped_reserved", "skipped_avoid",
               "skipped_no_url", "skipped_voice"}
    """
    company = (row.get("Company") or "").strip()
    role = (row.get("Role") or "").strip()
    if (company, role) in skip_pairs:
        return "skipped_reserved", f"{company} | {role}"
    if is_avoid_company(company, avoid_entries) or is_avoid_company(row.get("Source", ""), avoid_entries):
        return "skipped_avoid", f"{company} (on AVOID list)"
    # Voice heuristic only hard-filters MASTER lane; fallback can have voice rows.
    if lane == "master" and _is_voice_heuristic(row):
        return "skipped_voice", f"{company} | {role} (voice heuristic)"
    url = (row.get("Apply Link") or "").strip()
    if not url:
        return "skipped_no_url", f"{company} | {role}"

    # Uniquify URLs that would otherwise collide within this batch
    # (e.g. same career-page URL for multiple roles at one company).
    existing = conn.execute(
        "SELECT url, title, job_lane FROM jobs WHERE url = ?", (url,),
    ).fetchone()
    if existing:
        ex_title = (existing[1] or "").strip()
        ex_lane = existing[2] or "primary"
        if ex_title != role or ex_lane != lane:
            # Keep the old one; make the new one distinct via fragment.
            url = f"{url}#role={_slug(role)}"

    now = _now()
    pay = (row.get("Pay") or "").strip()
    desc_short = (row.get("Notes / Group Intel") or "")[:300]
    full_desc = _synth_description(row, lane)
    source = (row.get("Source") or f"curated_{lane}_list").strip() or f"curated_{lane}_list"

    # Master lane: pre-seed fit_score so packs can be built from curator trust.
    # Fallback lane: leave fit_score NULL so the scorer stage does real work on them.
    if lane == "master":
        fit_score_val = 8
        score_reason_val = "Pre-curated human-vetted non-voice master list pick."
        scored_at_val = now
    else:
        fit_score_val = None
        score_reason_val = None
        scored_at_val = None

    conn.execute(
        """
        INSERT INTO jobs (
            url, title, company, salary, description, full_description,
            location, site, strategy,
            discovered_at, detail_scraped_at,
            fit_score, score_reasoning, scored_at,
            application_url, job_lane
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            title = excluded.title,
            company = excluded.company,
            salary = excluded.salary,
            description = excluded.description,
            full_description = excluded.full_description,
            site = excluded.site,
            strategy = excluded.strategy,
            detail_scraped_at = excluded.detail_scraped_at,
            application_url = excluded.application_url,
            job_lane = excluded.job_lane
        """,
        (
            url, role, company, pay,
            desc_short, full_desc,
            "Remote - US",
            source, f"curated_{lane}_list",
            now, now,
            fit_score_val, score_reason_val, scored_at_val,
            (row.get("Apply Link") or url),
            lane,
        ),
    )
    return "inserted", url


def ingest_xlsx(xlsx_path: Path, skip_pairs: set[tuple[str, str]]) -> dict:
    conn = get_connection()
    avoid_entries = load_avoid_companies()
    summary = {
        "master_urls": [],
        "fallback_urls": [],
        "by_status": {"inserted": 0, "skipped_reserved": 0, "skipped_avoid": 0,
                      "skipped_voice": 0, "skipped_no_url": 0},
        "avoid_tab_entries": [],
        "notes": [],
    }

    for tab, lane in TAB_LANES.items():
        rows = _load_sheet_rows(xlsx_path, tab)
        summary["notes"].append(f"{tab}: {len(rows)} rows read")
        for row in rows:
            status, info = _upsert_job(conn, row, lane, skip_pairs, avoid_entries)
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            if status == "inserted":
                if lane == "master":
                    summary["master_urls"].append(info)
                else:
                    summary["fallback_urls"].append(info)
        conn.commit()

    # AVOID tab cross-check: ensure each listed company is represented in the
    # active yaml config. Report any drift.
    avoid_rows = _load_sheet_rows(xlsx_path, AVOID_TAB)
    summary["avoid_tab_entries"] = [
        {"company": r.get("Company", ""), "reason": r.get("Why Avoid", "")}
        for r in avoid_rows
    ]
    yaml_names = {e["name"].lower() for e in avoid_entries}
    missing = [r["Company"] for r in avoid_rows
               if r.get("Company") and r["Company"].lower() not in yaml_names
               and not is_avoid_company(r["Company"], avoid_entries)]
    if missing:
        summary["notes"].append(
            "WARNING: AVOID tab entries not covered by avoid_companies.yaml: "
            + ", ".join(missing)
        )

    return summary


def tailor_and_cover(urls: list[str], validation_mode: str = "normal") -> dict:
    conn = get_connection()
    profile = load_profile()
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    TAILORED_DIR.mkdir(parents=True, exist_ok=True)
    COVER_LETTER_DIR.mkdir(parents=True, exist_ok=True)
    out = {"tailored": 0, "cover": 0, "skipped_existing": 0, "errors": []}

    def _row(url: str) -> dict | None:
        r = conn.execute("SELECT * FROM jobs WHERE url=?", (url,)).fetchone()
        return dict(zip(r.keys(), r)) if r else None

    for url in urls:
        job = _row(url)
        if job is None:
            out["errors"].append((url, "row not found"))
            continue
        if job.get("tailored_resume_path") and job.get("cover_letter_path"):
            out["skipped_existing"] += 1
            continue

        prefix = f"{_safe(job.get('site') or 'curated', 20)}_{_safe(job['title'], 50)}"

        if not job.get("tailored_resume_path"):
            try:
                tailored_text, report = tailor_resume(
                    resume_text, job, profile, validation_mode=validation_mode,
                )
                status = report.get("status", "error")
                ok = {"approved", "approved_with_judge_warning"}
                if not tailored_text.strip() or status not in ok:
                    out["errors"].append((url, f"tailor failed: {status}"))
                    conn.execute(
                        "UPDATE jobs SET tailor_attempts=COALESCE(tailor_attempts,0)+? "
                        "WHERE url=?",
                        (report.get("attempts", 1), url),
                    )
                    conn.commit()
                    continue
                txt_path = TAILORED_DIR / f"{prefix}.txt"
                txt_path.write_text(tailored_text, encoding="utf-8")
                (TAILORED_DIR / f"{prefix}_JOB.txt").write_text(
                    f"Title: {job['title']}\nCompany: {job.get('company','')}\n"
                    f"Site: {job.get('site','')}\nURL: {job['url']}\n"
                    f"Fit: {job.get('fit_score','?')}\nLane: {job.get('job_lane','')}\n\n"
                    f"{job.get('full_description','')}",
                    encoding="utf-8",
                )
                conn.execute(
                    "UPDATE jobs SET tailored_resume_path=?, tailored_at=?, "
                    "tailor_attempts=COALESCE(tailor_attempts,0)+? WHERE url=?",
                    (str(txt_path), _now(), report.get("attempts", 1), url),
                )
                conn.commit()
                out["tailored"] += 1
                job = _row(url)
            except Exception as e:
                out["errors"].append((url, f"tailor exception: {e}"))
                continue

        if job and not job.get("cover_letter_path"):
            try:
                letter = generate_cover_letter(
                    resume_text, job, profile, validation_mode=validation_mode,
                )
                v = validate_cover_letter(letter, mode=validation_mode)
                if not v.get("passed"):
                    out["errors"].append(
                        (url, "cover validation failed: " + "; ".join(v.get("errors", []))),
                    )
                    conn.execute(
                        "UPDATE jobs SET cover_attempts=COALESCE(cover_attempts,0)+1 "
                        "WHERE url=?",
                        (url,),
                    )
                    conn.commit()
                    continue
                cl_path = COVER_LETTER_DIR / f"{prefix}_CL.txt"
                cl_path.write_text(letter, encoding="utf-8")
                try:
                    convert_cover_letter_to_pdf(cl_path)
                except Exception as pe:
                    out["errors"].append((url, f"cover pdf best-effort failed: {pe}"))
                conn.execute(
                    "UPDATE jobs SET cover_letter_path=?, cover_letter_at=?, "
                    "cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
                    (str(cl_path), _now(), url),
                )
                conn.commit()
                out["cover"] += 1
            except Exception as e:
                out["errors"].append((url, f"cover exception: {e}"))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx_path")
    ap.add_argument("--skip-company", action="append", default=[],
                    help="Skip by company name (repeat for multiple).")
    ap.add_argument("--skip-pair", action="append", default=[],
                    help='Skip exact Company|Role pair, e.g. "Humana|UM Admin".')
    ap.add_argument("--run-pipeline", choices=["master", "fallback", "none"],
                    default="none",
                    help="After ingest, run tailor+cover+pdf+pack for this lane.")
    ap.add_argument("--batch", default=None,
                    help="Pack batch name (default: ready_<lane>_YYYY-MM-DD).")
    ap.add_argument("--min-score", type=int, default=7)
    ap.add_argument("--validation", default="normal",
                    choices=["strict", "normal", "lenient"])
    args = ap.parse_args()

    load_env()
    ensure_dirs()
    init_db()

    xlsx_path = Path(args.xlsx_path).expanduser().resolve()
    if not xlsx_path.exists():
        print(f"xlsx not found: {xlsx_path}", file=sys.stderr)
        return 2

    skip_pairs: set[tuple[str, str]] = set()
    for raw in args.skip_pair:
        if "|" in raw:
            c, r = raw.split("|", 1)
            skip_pairs.add((c.strip(), r.strip()))
    # --skip-company expands to all roles for that company
    # We can't enumerate roles yet; handle at _upsert_job by additional check:
    company_skips = {c.strip().lower() for c in args.skip_company}

    # Monkey-patch: wrap _upsert_job to also honor company_skips
    orig_upsert = _upsert_job

    def wrapped_upsert(conn, row, lane, skp, avoid_entries):
        if (row.get("Company") or "").strip().lower() in company_skips:
            return "skipped_reserved", f"{row.get('Company','')} | {row.get('Role','')}"
        return orig_upsert(conn, row, lane, skp, avoid_entries)

    globals()["_upsert_job"] = wrapped_upsert

    print(f"[ingest] xlsx={xlsx_path}")
    t0 = time.time()
    summary = ingest_xlsx(xlsx_path, skip_pairs)
    print(f"[ingest] master inserts: {len(summary['master_urls'])}")
    print(f"[ingest] fallback inserts: {len(summary['fallback_urls'])}")
    print(f"[ingest] by status: {summary['by_status']}")
    for n in summary["notes"]:
        print("  NOTE", n)
    print(f"[avoid] tab entries cross-checked: {len(summary['avoid_tab_entries'])}")

    if args.run_pipeline != "none":
        urls = (summary["master_urls"] if args.run_pipeline == "master"
                else summary["fallback_urls"])
        if not urls:
            print(f"[pipeline] no urls for lane={args.run_pipeline}; nothing to do")
        else:
            print(f"[tailor+cover] running on {len(urls)} {args.run_pipeline} url(s)")
            res = tailor_and_cover(urls, validation_mode=args.validation)
            print(f"[tailor+cover] tailored={res['tailored']} cover={res['cover']} "
                  f"existing={res['skipped_existing']} errors={len(res['errors'])}")
            for e in res["errors"]:
                print(f"  ERR {e}")
            print("[pdf] batch_convert...")
            n = batch_convert()
            print(f"[pdf] new PDFs: {n}")
            batch_name = args.batch or (
                f"ready_{args.run_pipeline}_{datetime.now().strftime('%Y-%m-%d')}"
            )
            print(f"[pack] exporting batch={batch_name} lane={args.run_pipeline}")
            export_res = export_ready_packs(
                batch_name=batch_name, lane=args.run_pipeline,
                min_score=args.min_score, limit=200,
            )
            print(f"[pack] exported={export_res['exported']} "
                  f"skipped={export_res['skipped']} -> {export_res['output_dir']}")

    print(f"[done] elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
