"""Role-specific prompt helpers for resume tailoring and cover letters."""

from __future__ import annotations

from collections.abc import Iterable

INSURANCE_ROLE_KEYWORDS = (
    "insurance",
    "medicare",
    "medicaid",
    "aca",
    "health plan",
    "payer",
    "claims",
    "underwriting",
    "underwriter",
    "broker",
    "licensed agent",
    "enrollment",
)


def job_targets_insurance_domain(job: dict | None) -> bool:
    """Return True when the target role is explicitly insurance-domain work."""
    if not job:
        return False

    text = "\n".join(
        str(part)
        for part in (
            job.get("title"),
            job.get("description"),
            job.get("full_description"),
        )
        if part
    ).lower()

    return any(keyword in text for keyword in INSURANCE_ROLE_KEYWORDS)


def filter_role_sensitive_metrics(real_metrics: Iterable[str], *, insurance_role: bool) -> list[str]:
    """Drop insurance-license metrics for non-insurance roles."""
    metrics = [metric for metric in real_metrics if metric]
    if insurance_role:
        return metrics

    return [
        metric
        for metric in metrics
        if "license" not in metric.lower() and "licensed" not in metric.lower()
    ]
