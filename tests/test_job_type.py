"""`job_type` in searches.yaml reaches every JobSpy call."""

import logging
from unittest import mock

import pandas as pd
import pytest

from applypilot.database import init_db
from applypilot.discovery import jobspy as js


def _record_calls(target: str):
    """Patch ``target`` with a fake that records its arguments and finds nothing."""
    calls = []

    def fake(*args, **kwargs):
        calls.append(args[0] if args else kwargs)
        return pd.DataFrame()

    return calls, mock.patch.object(js, target, side_effect=fake)


def _board_result(rows: list[dict], warnings: dict | None = None) -> pd.DataFrame:
    """A DataFrame shaped like scrape_jobs' result, with its per-board warnings in ``attrs``."""
    df = pd.DataFrame(rows)
    df.attrs = {"errors": {}, "warnings": warnings or {}}
    return df


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


@pytest.mark.parametrize("value, keep", [
    ("contract", True),
    ("contract, parttime", True),   # JobSpy joins several types with ", "
    ("fulltime", False),
    ("Full-Time", False),
    (None, True),                   # the board did not say: not a mismatch
    (float("nan"), True),
    ("", True),
])
def test_row_filter_drops_only_a_known_different_job_type(value, keep):
    assert js._job_type_ok(value, "contract") is keep


def test_row_filter_is_off_without_a_job_type():
    assert js._job_type_ok("fulltime", None) is True


def test_rows_a_board_returned_despite_the_filter_are_dropped(caplog):
    # LinkedIn ignores JobSpy's job_type filter: it returns every type and records a warning.
    init_db()
    result = _board_result(
        [
            {"job_url": "https://example.com/1", "title": "A", "job_type": "contract"},
            {"job_url": "https://example.com/2", "title": "B", "job_type": "fulltime"},
            {"job_url": "https://example.com/3", "title": "C", "job_type": None},
        ],
        warnings={"linkedin": ["job_type isn't supported, ignoring it"]},
    )
    search = {"query": "engineer", "location": "Remote", "remote": True, "job_type": "contract"}
    with mock.patch.object(js, "_scrape_with_retry", return_value=result), \
            caplog.at_level(logging.WARNING, logger=js.log.name):
        stats = js._run_one_search(search, ["linkedin"], 10, 72, None, {}, 0, [], [], {})
    assert (stats["total"], stats["new"], stats["filtered"]) == (3, 2, 1)
    assert "linkedin: job_type isn't supported, ignoring it" in caplog.text
