"""F10: fabrication watchlist must be word-boundary and profile-aware."""
from applypilot.scoring.validator import find_watchlist_hits, validate_json_fields


def test_scalable_does_not_trip_scala():
    assert "scala" not in find_watchlist_hits("highly scalable systems", set())


def test_guardrails_does_not_trip_rails():
    assert "rails" not in find_watchlist_hits("implemented guardrails everywhere", set())


def test_real_rails_is_flagged():
    assert "rails" in find_watchlist_hits("Ruby on Rails developer", set())


def test_certification_prefix_flagged():
    assert "certif" in find_watchlist_hits("AWS Certified Solutions Architect", set())


def test_cplusplus_flagged_and_whitelistable():
    assert "c++" in find_watchlist_hits("C++ and Python experience", set())
    assert "c++" not in find_watchlist_hits("C++ and Python experience", {"c++"})


def test_csharp_flagged_and_whitelistable():
    assert "c#" in find_watchlist_hits("C# backend work", set())
    assert "c#" not in find_watchlist_hits("C# backend work", {"c#"})


def test_end_to_end_validate_json_fields_respects_profile():
    data = {
        "title": "Engineer", "summary": "Engineer",
        "skills": {"languages": ["C++", "Python"]},
        "experience": [{"company": "Acme", "bullets": ["did things"]}],
        "education": [{"school": "State U"}], "projects": [{"name": "P"}],
    }
    # No skills_boundary -> C++ flagged as fabricated.
    res_none = validate_json_fields(data, {"resume_facts": {}}, mode="normal")
    assert any("c++" in e.lower() for e in res_none["errors"])
    # C++ in profile -> not flagged.
    res_ok = validate_json_fields(
        data, {"resume_facts": {}, "skills_boundary": {"languages": ["C++"]}}, mode="normal")
    assert not any("c++" in e.lower() for e in res_ok["errors"])
