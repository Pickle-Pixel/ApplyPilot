"""The apply prompt follows the profile: contract/part-time availability, hourly rate, location."""

import copy
from unittest import mock

from applypilot import config
from applypilot.apply import prompt as pm

PROFILE = {
    "personal": {"full_name": "Alex Doe", "email": "alex@example.com", "phone": "555-0100",
                 "city": "Austin", "country": "USA"},
    "work_authorization": {"legally_authorized_to_work": "Yes", "require_sponsorship": "No"},
    "availability": {"earliest_start_date": "Immediately", "available_for_full_time": "Yes",
                     "available_for_contract": "No"},
    "compensation": {"salary_expectation": "120000", "salary_currency": "USD"},
    "experience": {"years_of_experience_total": "5", "target_role": "software engineer"},
}


def _prompt(profile):
    config.APP_DIR.mkdir(parents=True, exist_ok=True)
    resume = config.APP_DIR / "resume"
    resume.with_suffix(".pdf").write_bytes(b"%PDF-1.4\n")
    job = {"url": "https://example.com/job", "title": "Engineer", "site": "Acme",
           "application_url": "https://example.com/job/apply", "fit_score": 8,
           "tailored_resume_path": str(resume)}
    with mock.patch.object(pm.config, "load_profile", return_value=profile), \
            mock.patch.object(pm.config, "load_search_config", return_value={}):
        return pm.build_prompt(job, tailored_resume="resume text")


def _with(**changes):
    profile = copy.deepcopy(PROFILE)
    for path, value in changes.items():
        section, key = path.split("__")
        profile[section][key] = value
    return profile


def test_full_time_only_by_default():
    assert "FULL-TIME salaried positions only" in _prompt(PROFILE)


def test_contract_availability_allows_contract_and_part_time_roles():
    prompt = _prompt(_with(availability__available_for_contract="Yes"))
    assert "FULL-TIME salaried positions only" not in prompt
    assert "Contract and part-time roles posted by an employer ARE valid applications" in prompt


def test_hourly_rate_is_used_for_rate_questions():
    prompt = _prompt(_with(availability__available_for_contract="Yes", compensation__hourly_rate="95"))
    assert "Hourly rate? -> $95 USD/hr" in prompt
    assert "Divide your annual answer by 2080" not in prompt


def test_without_hourly_rate_the_annual_conversion_stays():
    assert "Divide your annual answer by 2080" in _prompt(PROFILE)


def test_empty_city_falls_back_to_country():
    prompt = _prompt(_with(personal__city=""))
    assert "lives in ," not in prompt
    assert "lives in USA" in prompt
