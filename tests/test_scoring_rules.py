"""The candidate's own criteria in ~/.applypilot/scoring_rules.md reach the scoring prompt."""

from unittest import mock

from applypilot.scoring import scorer

JOB = {"title": "Engineer", "site": "Acme", "location": "Remote", "full_description": "Build things."}


def _system_prompt_sent(rules):
    client = mock.Mock()
    client.chat.return_value = "SCORE: 7\nKEYWORDS: x\nREASONING: y"
    with mock.patch.object(scorer, "get_client", return_value=client):
        scorer.score_job("resume", JOB, rules)
    messages = client.chat.call_args[0][0]
    return messages[0]["content"]


def test_without_rules_the_prompt_is_unchanged():
    assert _system_prompt_sent(None) == scorer.SCORE_PROMPT
    assert _system_prompt_sent("   \n") == scorer.SCORE_PROMPT


def test_rules_are_appended_to_the_system_prompt():
    prompt = _system_prompt_sent("Dealbreaker: Go as the main language.")
    assert prompt.startswith(scorer.SCORE_PROMPT)
    assert "Dealbreaker: Go as the main language." in prompt


def test_rules_file_is_optional(tmp_path):
    with mock.patch.object(scorer, "SCORING_RULES_PATH", tmp_path / "scoring_rules.md"):
        assert scorer.load_scoring_rules() is None
        (tmp_path / "scoring_rules.md").write_text("Only remote roles.", encoding="utf-8")
        assert scorer.load_scoring_rules() == "Only remote roles."
