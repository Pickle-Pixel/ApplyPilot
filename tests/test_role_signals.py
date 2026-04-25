from applypilot.scoring.role_signals import (
    filter_role_sensitive_metrics,
    job_targets_insurance_domain,
)


def test_job_targets_insurance_domain_detects_explicit_insurance_roles():
    job = {
        "title": "Medicare Enrollment Specialist",
        "full_description": "Support ACA enrollment operations and insurance compliance.",
    }

    assert job_targets_insurance_domain(job) is True


def test_job_targets_insurance_domain_skips_general_operations_roles():
    job = {
        "title": "Business Analyst, Support Operations",
        "full_description": "Build dashboards, improve workflows, and analyze staffing metrics.",
    }

    assert job_targets_insurance_domain(job) is False


def test_filter_role_sensitive_metrics_removes_license_mentions_for_non_insurance_roles():
    metrics = [
        "Promoted from entry-level to Real-Time Analyst in 2 months",
        "Licensed in 30+ states",
        "98% first-contact resolution at Jack Henry",
    ]

    assert filter_role_sensitive_metrics(metrics, insurance_role=False) == [
        "Promoted from entry-level to Real-Time Analyst in 2 months",
        "98% first-contact resolution at Jack Henry",
    ]


def test_filter_role_sensitive_metrics_keeps_license_mentions_for_insurance_roles():
    metrics = [
        "Licensed in 30+ states",
        "Supported ACA marketplace operations",
    ]

    assert filter_role_sensitive_metrics(metrics, insurance_role=True) == metrics
