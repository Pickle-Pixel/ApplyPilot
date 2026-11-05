"""`applypilot init` writes the contract/part-time fields that the apply prompt reads."""

from unittest import mock

import yaml

from applypilot import config
from applypilot.apply import prompt as pm
from applypilot.wizard import init as wizard


def _asker(answers: dict):
    """Stand-in for Prompt.ask / Confirm.ask: match a phrase in the question, else the default."""
    def ask(question, *args, default=None, **kwargs):
        for phrase, answer in answers.items():
            if phrase.lower() in question.lower():
                return answer
        return default if default is not None else ""
    return ask


def _run(step, answers: dict):
    config.APP_DIR.mkdir(parents=True, exist_ok=True)
    ask = _asker(answers)
    with mock.patch.object(wizard.Prompt, "ask", side_effect=ask), \
            mock.patch.object(wizard.Confirm, "ask", side_effect=ask):
        return step()


def test_contract_answers_reach_the_profile_and_the_apply_prompt():
    profile = _run(wizard._setup_profile, {"contract and part-time": True, "hourly rate": "95"})
    assert profile["availability"]["available_for_contract"] == "Yes"
    assert profile["compensation"]["hourly_rate"] == "95"
    rule = pm._build_role_type_rule(profile)
    assert "Contract and part-time roles posted by an employer ARE valid" in rule


def test_full_time_only_unless_the_user_says_otherwise():
    profile = _run(wizard._setup_profile, {})
    assert profile["availability"]["available_for_contract"] == "No"
    assert "hourly_rate" not in profile["compensation"]
    assert "FULL-TIME salaried positions only" in pm._build_role_type_rule(profile)


def test_generated_searches_yaml_documents_job_type_and_still_parses():
    _run(wizard._setup_searches, {"job titles": "Backend Engineer"})
    text = config.SEARCH_CONFIG_PATH.read_text(encoding="utf-8")
    assert "job_type" in text
    parsed = yaml.safe_load(text)
    # Documented, not set: searching every type stays the default.
    assert "job_type" not in parsed["locations"][0]
