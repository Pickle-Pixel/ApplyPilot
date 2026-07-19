"""Cost-quality aggregation and reporting for fleet apply metrics.

The summarize_* helpers intentionally accept already-loaded row dictionaries so
their behavior stays independent from the live database fetch helpers below.
"""
from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping


TERMINAL_STATUSES = {"applied", "failed", "blocked", "crash_unconfirmed"}
ROUTE_COST_PRECISION = 4


@dataclass
class CountCost:
    count: int = 0
    applied: int = 0
    terminal: int = 0
    cost: float = 0.0

    @property
    def success_pct(self) -> float:
        if not self.count:
            return 0.0
        return (self.applied / self.count) * 100.0

    @property
    def cost_per_applied(self) -> float:
        if not self.applied:
            return 0.0
        return self.cost / self.applied


@dataclass
class FailureBucket:
    count: int = 0
    cost: float = 0.0


@dataclass
class FleetQueueSummary:
    applied: int = 0
    terminal_attempts: int = 0
    queued_or_leased: int = 0
    total_cost_usd: float = 0.0
    cost_per_applied_all_in: float = 0.0
    cost_per_terminal_attempt: float = 0.0
    by_ats: dict[str, CountCost] = field(default_factory=dict)
    by_failure_bucket: dict[str, FailureBucket] = field(default_factory=dict)


@dataclass
class LocalJobsSummary:
    touched: int = 0
    applied: int = 0
    by_ats: dict[str, CountCost] = field(default_factory=dict)
    by_failure_bucket: dict[str, FailureBucket] = field(default_factory=dict)


@dataclass
class RouteSummary:
    available: bool = True
    by_route: dict[str, CountCost] = field(default_factory=dict)


@dataclass
class CostQualityReport:
    fleet: FleetQueueSummary
    local: LocalJobsSummary
    routes: RouteSummary = field(default_factory=RouteSummary)


def _money(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def classify_ats(url: str | None) -> str:
    token = (url or "").lower()
    if "ashbyhq.com" in token:
        return "ashby"
    if "greenhouse" in token or "grnh.se" in token:
        return "greenhouse"
    if "lever.co" in token:
        return "lever"
    if "workdayjobs.com" in token or "myworkdayjobs.com" in token:
        return "workday"
    if "smartrecruiters" in token:
        return "smartrecruiters"
    if "workable.com" in token:
        return "workable"
    return "other"


def classify_failure_bucket(status: str | None, apply_error: str | None) -> str:
    status_token = (status or "").strip().lower()
    error_token = (apply_error or "").strip().lower()

    if _contains_any(error_token, ("email", "otp", "auth", "login", "verification")):
        return "email_auth_related"
    if _contains_any(error_token, ("captcha", "challenge")):
        return "challenge_related"
    if status_token == "crash_unconfirmed" or _contains_any(
        error_token, ("browser", "no_result", "timeout")
    ):
        return "agent_browser_runtime"
    if _contains_any(
        error_token,
        ("expired", "not_eligible", "location", "already_applied", "excluded"),
    ):
        return "preflight_or_policy"
    return "other"


def summarize_fleet_queue(rows: Iterable[dict]) -> FleetQueueSummary:
    by_ats: dict[str, CountCost] = {}
    by_failure_bucket: dict[str, FailureBucket] = {}
    applied = 0
    terminal_attempts = 0
    queued_or_leased = 0
    total_cost = 0.0

    for row in rows:
        status = _status(_get(row, "status", "apply_status"))
        cost = _money(_get(row, "cost_usd", "est_cost_usd", "apply_cost_usd"))
        ats = classify_ats(_get(row, "application_url", "url"))
        apply_error = _get(row, "apply_error", "error")

        is_applied = status == "applied"
        is_terminal = status in TERMINAL_STATUSES
        applied += int(is_applied)
        terminal_attempts += int(is_terminal)
        queued_or_leased += int(not is_terminal)
        total_cost += cost

        _add_count_cost(by_ats, ats, cost=cost, is_applied=is_applied, is_terminal=is_terminal)
        if is_terminal and not is_applied:
            bucket = classify_failure_bucket(status, apply_error)
            _add_failure_bucket(by_failure_bucket, bucket, cost=cost)

    return FleetQueueSummary(
        applied=applied,
        terminal_attempts=terminal_attempts,
        queued_or_leased=queued_or_leased,
        total_cost_usd=total_cost,
        cost_per_applied_all_in=_divide(total_cost, applied),
        cost_per_terminal_attempt=_divide(total_cost, terminal_attempts),
        by_ats=dict(sorted(by_ats.items())),
        by_failure_bucket=dict(sorted(by_failure_bucket.items())),
    )


def summarize_local_jobs(rows: Iterable[dict]) -> LocalJobsSummary:
    by_ats: dict[str, CountCost] = {}
    by_failure_bucket: dict[str, FailureBucket] = {}
    touched = 0
    applied = 0

    for row in rows:
        status = _status(_get(row, "apply_status", "status"))
        cost = _money(_get(row, "cost_usd", "est_cost_usd", "apply_cost_usd"))
        ats = classify_ats(_get(row, "application_url", "url"))
        apply_error = _get(row, "apply_error", "error")

        is_applied = status == "applied"
        is_terminal = status in TERMINAL_STATUSES
        touched += 1
        applied += int(is_applied)

        _add_count_cost(by_ats, ats, cost=cost, is_applied=is_applied, is_terminal=is_terminal)
        if is_terminal and not is_applied:
            bucket = classify_failure_bucket(status, apply_error)
            _add_failure_bucket(by_failure_bucket, bucket, cost=cost)

    return LocalJobsSummary(
        touched=touched,
        applied=applied,
        by_ats=dict(sorted(by_ats.items())),
        by_failure_bucket=dict(sorted(by_failure_bucket.items())),
    )


def summarize_result_routes(
    rows: Iterable[dict], *, available: bool = True
) -> RouteSummary:
    by_route: dict[str, CountCost] = {}
    for row in rows:
        status = _status(_get(row, "status", "apply_status"))
        route = str(_get(row, "route") or "").strip().lower() or "unknown"
        cost = _money(_get(row, "cost_usd", "est_cost_usd"))
        count_value = _get(row, "event_count")
        event_count = int(count_value) if count_value is not None else 1
        is_applied = status == "applied"
        is_terminal = status in TERMINAL_STATUSES
        _add_count_cost(
            by_route,
            route,
            cost=cost,
            is_applied=is_applied,
            is_terminal=is_terminal,
            count=event_count,
        )
    for item in by_route.values():
        item.cost = round(item.cost, ROUTE_COST_PRECISION)
    return RouteSummary(available=available, by_route=dict(sorted(by_route.items())))


def fetch_fleet_queue_rows(pg_dsn: str) -> list[dict]:
    from psycopg.rows import dict_row
    import psycopg

    with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
        rows = conn.execute(
            """
            SELECT
                status::text AS status,
                apply_error,
                application_url,
                COALESCE(cumulative_cost_usd, 0) AS est_cost_usd
            FROM apply_queue
            """
        ).fetchall()
    return list(rows)


def fetch_fleet_result_event_rows(pg_dsn: str) -> tuple[list[dict], bool]:
    from psycopg import errors
    from psycopg.rows import dict_row
    import psycopg

    try:
        with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                SELECT
                    LOWER(BTRIM(status::text)) AS status,
                    COALESCE(
                        NULLIF(LOWER(BTRIM(route)), ''),
                        'unknown'
                    ) AS route,
                    COUNT(*) AS event_count,
                    COALESCE(SUM(est_cost_usd), 0) AS est_cost_usd
                FROM apply_result_events
                GROUP BY 1, 2
                """
            ).fetchall()
    except (errors.UndefinedColumn, errors.UndefinedTable):
        return [], False
    return list(rows), True


def fetch_local_job_rows(sqlite_path: str | Path) -> list[dict]:
    uri = Path(sqlite_path).expanduser().resolve(strict=False).as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT apply_status, apply_error, application_url
            FROM jobs
            WHERE apply_status IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def default_sqlite_path() -> Path:
    if os.environ.get("APPLYPILOT_DB_PATH"):
        return Path(os.environ["APPLYPILOT_DB_PATH"])
    return Path(os.environ.get("LOCALAPPDATA", "")) / "ApplyPilot" / "applypilot.db"


def build_report(
    *,
    pg_dsn: str | None = None,
    sqlite_path: str | Path | None = None,
) -> CostQualityReport:
    pg_dsn = pg_dsn or os.environ.get("FLEET_PG_DSN")
    if not pg_dsn:
        raise RuntimeError(
            "FLEET_PG_DSN is not set: the cost/quality report requires an explicit mapped "
            "fleet DSN (never the postgres admin or the shared fleet_worker role). Refusing to connect."
        )
    local_path = sqlite_path if sqlite_path is not None else default_sqlite_path()
    event_rows, route_metrics_available = fetch_fleet_result_event_rows(pg_dsn)
    return CostQualityReport(
        fleet=summarize_fleet_queue(fetch_fleet_queue_rows(pg_dsn)),
        local=summarize_local_jobs(fetch_local_job_rows(local_path)),
        routes=summarize_result_routes(
            event_rows,
            available=route_metrics_available,
        ),
    )


def _fmt_money(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"${value:.4f}"


def _escape_markdown_table_cell(value: str) -> str:
    normalized = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return normalized.replace("\\", "\\\\").replace("|", "\\|")


def render_report_markdown(report: CostQualityReport) -> str:
    fleet = report.fleet
    local = report.local
    ats_keys = sorted(set(local.by_ats) | set(fleet.by_ats))
    failure_keys = sorted(
        set(fleet.by_failure_bucket) | set(local.by_failure_bucket),
        key=lambda key: (-(fleet.by_failure_bucket.get(key) or FailureBucket()).cost, key),
    )

    lines = [
        "# Apply Cost Quality Report",
        "",
        "## Fleet Cost",
        "",
        f"- Applied: {fleet.applied}",
        f"- Terminal attempts: {fleet.terminal_attempts}",
        f"- Queued or leased: {fleet.queued_or_leased}",
        f"- Total recorded cost: {_fmt_money(fleet.total_cost_usd)}",
        "- All-in cost per successful apply: "
        f"{_fmt_money(fleet.cost_per_applied_all_in if fleet.applied else None)}",
        "- Cost per terminal attempt: "
        f"{_fmt_money(fleet.cost_per_terminal_attempt if fleet.terminal_attempts else None)}",
        "",
        "## Local History",
        "",
        f"- Touched: {local.touched}",
        f"- Applied: {local.applied}",
        "",
        "## ATS History",
        "",
        "| ATS | Local touched | Local applied | Local success | Fleet attempts | Fleet applied | Fleet cost/apply |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    if ats_keys:
        for ats in ats_keys:
            local_item = local.by_ats.get(ats, CountCost())
            fleet_item = fleet.by_ats.get(ats, CountCost())
            fleet_cost_per_apply = (
                fleet_item.cost_per_applied if fleet_item.applied else None
            )
            lines.append(
                f"| {ats} | {local_item.count} | {local_item.applied} | "
                f"{local_item.success_pct:.1f}% | {fleet_item.terminal} | "
                f"{fleet_item.applied} | {_fmt_money(fleet_cost_per_apply)} |"
            )
    else:
        lines.append("| n/a | 0 | 0 | 0.0% | 0 | 0 | n/a |")

    lines.extend(
        [
            "",
            "## Failure Buckets",
            "",
            "| Bucket | Fleet failures | Fleet cost | Local failures |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    if failure_keys:
        for bucket in failure_keys:
            fleet_item = fleet.by_failure_bucket.get(bucket)
            local_item = local.by_failure_bucket.get(bucket, FailureBucket())
            fleet_count = fleet_item.count if fleet_item else 0
            fleet_cost = _fmt_money(fleet_item.cost) if fleet_item else "n/a"
            lines.append(
                f"| {bucket} | {fleet_count} | {fleet_cost} | {local_item.count} |"
            )
    else:
        lines.append("| n/a | 0 | n/a | 0 |")

    lines.extend(
        [
            "",
            "## Result Event Routes",
            "",
            "Append-only result events; this table is not the canonical queue all-in denominator.",
            "Applied and Cost/applied are applied event-row counts and cost per applied event, not distinct applications.",
            "",
        ]
    )
    if not report.routes.available:
        lines.append(
            "Route metrics unavailable until the home schema migration runs."
        )
    else:
        lines.extend(
            [
                "| Route | Events | Applied | Cost | Cost/applied |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for route, item in sorted(report.routes.by_route.items()):
            lines.append(
                f"| {_escape_markdown_table_cell(route)} | {item.count} | "
                f"{item.applied} | "
                f"{_fmt_money(item.cost)} | "
                f"{_fmt_money(item.cost_per_applied) if item.applied else 'n/a'} |"
            )
        if not report.routes.by_route:
            lines.append(
                "No result events recorded; the placeholder row below is not an observation."
            )
            lines.append("| n/a | 0 | 0 | $0.0000 | n/a |")

    return "\n".join(lines) + "\n"


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _status(value: Any) -> str:
    return str(value or "").strip().lower()


def _get(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if hasattr(row, "get"):
            value = row.get(name)  # type: ignore[attr-defined]
            if value is not None:
                return value
            continue
        try:
            value = row[name]
        except (KeyError, TypeError):
            continue
        if value is not None:
            return value
    return None


def _add_count_cost(
    summary: dict[str, CountCost],
    key: str,
    *,
    cost: float,
    is_applied: bool,
    is_terminal: bool,
    count: int = 1,
) -> None:
    item = summary.setdefault(key, CountCost())
    item.count += count
    item.applied += count * int(is_applied)
    item.terminal += count * int(is_terminal)
    item.cost += cost


def _add_failure_bucket(summary: dict[str, FailureBucket], key: str, *, cost: float) -> None:
    item = summary.setdefault(key, FailureBucket())
    item.count += 1
    item.cost += cost


def _divide(numerator: float, denominator: int) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator
