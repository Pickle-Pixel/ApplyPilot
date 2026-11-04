"""`job_type` in searches.yaml reaches every JobSpy call."""

from unittest import mock

import pytest

from applypilot.discovery import jobspy as js


class _Stop(Exception):
    """Raised by the fake scraper after recording its arguments."""


def _record_calls(target: str):
    calls = []

    def fake(*args, **kwargs):
        calls.append(args[0] if args else kwargs)
        raise _Stop()

    return calls, mock.patch.object(js, target, side_effect=fake)


def test_location_job_type_wins_over_default():
    assert js._resolve_job_type({"job_type": "contract"}, {"job_type": "fulltime"}) == "contract"


def test_default_job_type_applies_when_location_has_none():
    assert js._resolve_job_type({}, {"job_type": "parttime"}) == "parttime"


def test_no_job_type_means_any():
    assert js._resolve_job_type({}, {}) is None


@pytest.mark.parametrize("raw", ["part-time", "Part Time", "part_time", "PARTTIME"])
def test_common_spellings_normalise(raw):
    assert js._resolve_job_type({"job_type": raw}, {}) == "parttime"


def test_unknown_job_type_fails_loudly():
    with pytest.raises(ValueError, match="freelance"):
        js._resolve_job_type({"job_type": "freelance"}, {})


def test_full_crawl_builds_one_search_per_location_job_type():
    seen = []
    with mock.patch.object(js, "_run_one_search", side_effect=lambda s, *a, **k: seen.append(s) or {
        "new": 0, "existing": 0, "errors": 0, "filtered": 0, "total": 0, "label": ""}):
        js._full_crawl({  # the database lands in conftest's throwaway APPLYPILOT_DIR
            "queries": [{"query": "engineer", "tier": 1}],
            "locations": [
                {"location": "Remote", "remote": True, "job_type": "contract"},
                {"location": "Remote", "remote": True, "job_type": "parttime"},
                {"location": "Austin, TX"},
            ],
        })
    assert [s["job_type"] for s in seen] == ["contract", "parttime", None]


def test_job_type_reaches_every_board_including_glassdoor():
    calls, patch = _record_calls("_scrape_with_retry")
    search = {"query": "engineer", "location": "Remote", "remote": True, "job_type": "contract"}
    with patch:
        js._run_one_search(search, ["indeed", "linkedin", "glassdoor"], 10, 72, None, {}, 0,
                           [], [], {})
    assert len(calls) == 2  # non-Glassdoor boards, then Glassdoor on its own
    assert all(c["job_type"] == "contract" and c["is_remote"] for c in calls)


def test_no_job_type_key_without_a_filter():
    calls, patch = _record_calls("_scrape_with_retry")
    with patch:
        js._run_one_search({"query": "engineer", "location": "Austin, TX"}, ["indeed"], 10, 72,
                           None, {}, 0, [], [], {})
    assert "job_type" not in calls[0] and "is_remote" not in calls[0]


def test_single_query_search_passes_job_type():
    calls, patch = _record_calls("scrape_jobs")
    with patch:
        js.search_jobs("engineer", "Remote", remote_only=True, job_type="part-time")
    assert calls[0]["job_type"] == "parttime"
