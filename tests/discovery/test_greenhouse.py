"""Unit tests for Greenhouse ATS discovery module."""

import os
import sys
import sqlite3
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

# Ensure src is on sys.path for tests when running from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from applypilot.discovery.greenhouse import (
    GREENHOUSE_API_BASE,
    _location_ok,
    _store_jobs,
    _title_matches_query,
    fetch_jobs_api,
    load_employers,
    parse_api_response,
    search_employer,
)


class TestLoadEmployers:
    """Tests for load_employers function."""

    def test_loads_employers_from_yaml(self):
        """Test that employers are loaded from greenhouse.yaml config."""
        employers = load_employers()
        assert isinstance(employers, dict)
        assert len(employers) > 0
        # Check for known employers
        assert "scaleai" in employers
        assert employers["scaleai"]["name"] == "Scale AI"

    def test_returns_empty_dict_if_file_missing(self, tmp_path):
        """Test graceful handling of missing config file."""
        with patch("applypilot.discovery.greenhouse.CONFIG_DIR", tmp_path):
            employers = load_employers()
            assert employers == {}


class TestLocationFiltering:
    """Tests for location filtering functions."""

    def test_remote_jobs_always_accepted(self):
        """Remote jobs should pass any filter."""
        accept = ["San Francisco"]
        reject = ["New York"]

        remote_locations = [
            "Remote",
            "Anywhere",
            "Work from home",
            "WFH",
            "Distributed",
            "Fully remote",
        ]

        for loc in remote_locations:
            assert _location_ok(loc, accept, reject) is True

    def test_reject_locations_blocked(self):
        """Jobs in reject list should be filtered out."""
        accept = ["CA", "California"]
        reject = ["New York", "NYC"]

        assert _location_ok("New York, NY", accept, reject) is False
        assert _location_ok("NYC Office", accept, reject) is False
        assert _location_ok("San Francisco, CA", accept, reject) is True

    def test_accept_locations_required(self):
        """Non-remote jobs must match accept list."""
        accept = ["San Francisco", "California"]
        reject = []

        assert _location_ok("San Francisco, CA", accept, reject) is True
        assert _location_ok("Los Angeles, CA", accept, reject) is False
        assert _location_ok("", accept, reject) is True  # Unknown location passes

    def test_case_insensitive_matching(self):
        """Location matching should be case-insensitive."""
        accept = ["san francisco"]
        reject = ["new york"]

        assert _location_ok("San Francisco, CA", accept, reject) is True
        assert _location_ok("NEW YORK", accept, reject) is False


class TestTitleMatching:
    """Tests for query title matching."""

    def test_empty_query_matches_all(self):
        """Empty query should match any title."""
        assert _title_matches_query("Software Engineer", "") is True
        assert _title_matches_query("", "") is True

    def test_single_keyword_match(self):
        """Single keyword should match if in title."""
        assert _title_matches_query("Machine Learning Engineer", "machine learning") is True
        assert _title_matches_query("Software Engineer", "machine learning") is False

    def test_multiple_keywords_any_match(self):
        """Multiple keywords should match if any present."""
        assert _title_matches_query("Machine Learning Engineer", "machine learning AI") is True
        assert _title_matches_query("AI Researcher", "machine learning AI") is True
        assert _title_matches_query("Data Scientist", "machine learning AI") is False

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        assert _title_matches_query("MACHINE LEARNING Engineer", "machine learning") is True
        assert _title_matches_query("software engineer", "SOFTWARE") is True


class TestParseGreenhouseJobs:
    """Tests for API parsing functions (legacy HTML tests remain where relevant)."""

    def test_parse_simple_api_job(self):
        """Test parsing a simple job posting from API JSON."""
        api_response = {
            "jobs": [
                {
                    "id": 12345,
                    "title": "Software Engineer",
                    "location": {"name": "San Francisco, CA"},
                    "absolute_url": "https://boards.greenhouse.io/test/jobs/12345",
                    "content": "<p>Great role</p>",
                    "departments": [{"name": "Engineering"}],
                    "updated_at": "2026-02-27T00:00:00Z",
                }
            ]
        }

        jobs = parse_api_response(api_response, "Test Company", "")

        assert len(jobs) == 1
        job = jobs[0]
        assert job["title"] == "Software Engineer"
        assert job["company"] == "Test Company"
        assert job["location"] == "San Francisco, CA"
        assert job["department"] == "Engineering"
        assert job["strategy"] == "greenhouse"
        assert job["url"] == "https://boards.greenhouse.io/test/jobs/12345"
        assert job["job_id"] == 12345
        assert job["description"] == "Great role"
        assert job["updated_at"] == "2026-02-27T00:00:00Z"

    def test_parse_multiple_api_jobs(self):
        """Test parsing multiple jobs from API JSON."""
        api_response = {
            "jobs": [
                {
                    "id": 1,
                    "title": "Frontend Engineer",
                    "location": {"name": "Remote"},
                    "absolute_url": "https://.../1",
                    "content": "<p>a</p>",
                    "departments": [{"name": "Eng"}],
                    "updated_at": "2026-02-27T00:00:00Z",
                },
                {
                    "id": 2,
                    "title": "Backend Engineer",
                    "location": {"name": "New York, NY"},
                    "absolute_url": "https://.../2",
                    "content": "<p>b</p>",
                    "departments": [{"name": "Eng"}],
                    "updated_at": "2026-02-27T00:00:00Z",
                },
            ]
        }

        jobs = parse_api_response(api_response, "Test Company", "")

        assert len(jobs) == 2
        assert jobs[0]["title"] == "Frontend Engineer"
        assert jobs[1]["title"] == "Backend Engineer"

    def test_filter_by_query_api(self):
        """Test filtering API jobs by query string."""
        api_response = {
            "jobs": [
                {
                    "id": 1,
                    "title": "Machine Learning Engineer",
                    "location": {"name": "Remote"},
                    "absolute_url": "https://.../1",
                    "content": "<p>a</p>",
                    "departments": [],
                    "updated_at": "2026-02-27T00:00:00Z",
                },
                {
                    "id": 2,
                    "title": "Sales Representative",
                    "location": {"name": "Remote"},
                    "absolute_url": "https://.../2",
                    "content": "<p>b</p>",
                    "departments": [],
                    "updated_at": "2026-02-27T00:00:00Z",
                },
            ]
        }

        jobs = parse_api_response(api_response, "Test Company", "machine learning")

        assert len(jobs) == 1
        assert jobs[0]["title"] == "Machine Learning Engineer"

    def test_offices_and_absolute_url_api(self):
        """Test that offices field is parsed and absolute URLs are preserved in API response."""
        api_response = {
            "jobs": [
                {
                    "id": 123,
                    "title": "Test Job",
                    "location": {"name": "Remote"},
                    "absolute_url": "https://boards.greenhouse.io/test/jobs/123",
                    "content": "<p>x</p>",
                    "departments": [],
                    "offices": [{"name": "SF Office"}],
                    "updated_at": "2026-02-27T00:00:00Z",
                }
            ]
        }

        jobs = parse_api_response(api_response, "Test Company", "")

        assert len(jobs) == 1
        assert jobs[0]["url"] == "https://boards.greenhouse.io/test/jobs/123"
        assert jobs[0]["offices"] == ["SF Office"]

    def test_handles_empty_html(self):
        """Legacy: test handling of empty input via API parser (empty dict)."""
        jobs = parse_api_response({}, "Test Company", "")
        assert jobs == []

    def test_handles_no_job_posts(self):
        """Test handling of API response without jobs key or empty list."""
        jobs = parse_api_response({"jobs": []}, "Test Company", "")
        assert jobs == []


class TestFetchJobsAPI:
    """Tests for HTTP fetching functions using the API client."""

    @patch("applypilot.discovery.greenhouse.httpx.Client")
    def test_successful_fetch(self, mock_client_class):
        """Test successful API fetch returning JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobs": []}
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)

        mock_client_class.return_value = mock_client

        result = fetch_jobs_api("testcompany")

        assert result == {"jobs": []}
        mock_client.get.assert_called_once()

    @patch("applypilot.discovery.greenhouse.httpx.Client")
    def test_failed_fetch_returns_none(self, mock_client_class):
        """Test that failed fetches return None gracefully."""
        mock_client = Mock()
        mock_client.get.side_effect = Exception("Connection error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)

        mock_client_class.return_value = mock_client

        result = fetch_jobs_api("testcompany")

        assert result is None

    def test_url_format(self):
        """Test that API base URL composes correctly."""
        assert f"{GREENHOUSE_API_BASE}/scaleai/jobs" == "https://boards-api.greenhouse.io/v1/boards/scaleai/jobs"
        assert f"{GREENHOUSE_API_BASE}/stripe/jobs" == "https://boards-api.greenhouse.io/v1/boards/stripe/jobs"


class TestSearchEmployer:
    """Tests for employer search function."""

    @patch("applypilot.discovery.greenhouse.fetch_jobs_api")
    @patch("applypilot.discovery.greenhouse.parse_api_response")
    def test_search_with_location_filter(self, mock_parse, mock_fetch):
        """Test searching with location filter enabled."""
        # fetch_jobs_api returns API dict, parse_api_response returns normalized job list
        mock_fetch.return_value = {"jobs": []}
        mock_parse.return_value = [
            {
                "title": "Engineer",
                "company": "Test",
                "location": "San Francisco, CA",
                "department": "Engineering",
                "url": "https://example.com/job",
                "strategy": "greenhouse",
            }
        ]

        employer = {"name": "Test Company"}
        jobs = search_employer(
            "test",
            employer,
            "engineer",
            location_filter=True,
            accept_locs=["San Francisco"],
            reject_locs=["New York"],
        )

        assert len(jobs) == 1
        mock_fetch.assert_called_once_with("test", content=True)
        mock_parse.assert_called_once_with({"jobs": []}, "Test Company", "engineer")

    @patch("applypilot.discovery.greenhouse.fetch_jobs_api")
    def test_search_no_api_returns_empty(self, mock_fetch):
        """Test that empty result is returned if API fetch fails."""
        mock_fetch.return_value = None

        employer = {"name": "Test Company"}
        jobs = search_employer("test", employer, "engineer")

        assert jobs == []


class TestStoreJobs:
    """Tests for database storage functions."""

    def test_store_new_jobs(self, tmp_path):
        """Test storing new jobs in database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Create jobs table
        conn.execute("""
            CREATE TABLE jobs (
                url TEXT PRIMARY KEY,
                title TEXT,
                salary TEXT,
                description TEXT,
                location TEXT,
                site TEXT,
                strategy TEXT,
                discovered_at TEXT,
                full_description TEXT,
                application_url TEXT,
                detail_scraped_at TEXT,
                detail_error TEXT
            )
        """)

        jobs = [
            {
                "title": "Test Job",
                "company": "Test Company",
                "location": "Remote",
                "department": "Engineering",
                "url": "https://example.com/job1",
                "strategy": "greenhouse",
            }
        ]

        with patch("applypilot.discovery.greenhouse.get_connection", return_value=conn):
            new, existing = _store_jobs(jobs)

        assert new == 1
        assert existing == 0

        # Verify job was stored
        cursor = conn.execute("SELECT title, site FROM jobs WHERE url = ?", ("https://example.com/job1",))
        row = cursor.fetchone()
        assert row[0] == "Test Job"
        assert row[1] == "Test Company"

    def test_store_duplicate_jobs(self, tmp_path):
        """Test that duplicate jobs are counted as existing."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Create jobs table
        conn.execute("""
            CREATE TABLE jobs (
                url TEXT PRIMARY KEY,
                title TEXT,
                salary TEXT,
                description TEXT,
                location TEXT,
                site TEXT,
                strategy TEXT,
                discovered_at TEXT,
                full_description TEXT,
                application_url TEXT,
                detail_scraped_at TEXT,
                detail_error TEXT
            )
        """)

        jobs = [
            {
                "title": "Test Job",
                "company": "Test Company",
                "location": "Remote",
                "department": "Engineering",
                "url": "https://example.com/job1",
                "strategy": "greenhouse",
            }
        ]

        with patch("applypilot.discovery.greenhouse.get_connection", return_value=conn):
            # Store first time
            new, existing = _store_jobs(jobs)
            assert new == 1

            # Store again - should be duplicate
            new, existing = _store_jobs(jobs)
            assert new == 0
            assert existing == 1

    def test_store_multiple_jobs(self, tmp_path):
        """Test storing multiple jobs at once."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE jobs (
                url TEXT PRIMARY KEY,
                title TEXT,
                salary TEXT,
                description TEXT,
                location TEXT,
                site TEXT,
                strategy TEXT,
                discovered_at TEXT,
                full_description TEXT,
                application_url TEXT,
                detail_scraped_at TEXT,
                detail_error TEXT
            )
        """)

        jobs = [
            {
                "title": "Job 1",
                "company": "Company A",
                "location": "Remote",
                "department": "Engineering",
                "url": "https://example.com/job1",
                "strategy": "greenhouse",
            },
            {
                "title": "Job 2",
                "company": "Company B",
                "location": "NYC",
                "department": "Sales",
                "url": "https://example.com/job2",
                "strategy": "greenhouse",
            },
        ]

        with patch("applypilot.discovery.greenhouse.get_connection", return_value=conn):
            new, existing = _store_jobs(jobs)

        assert new == 2
        assert existing == 0

        count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        assert count == 2


class TestIntegration:
    """Integration-style tests."""

    @patch("applypilot.discovery.greenhouse.fetch_jobs_api")
    @patch("applypilot.discovery.greenhouse.get_connection")
    def test_end_to_end_search_and_store(self, mock_get_conn, mock_fetch, tmp_path):
        """Test full flow from fetch to parse to store using the API client."""
        # Setup mock API response
        api_response = {
            "jobs": [
                {
                    "id": 123,
                    "title": "ML Engineer",
                    "location": {"name": "Remote"},
                    "absolute_url": "https://boards.greenhouse.io/test/jobs/123",
                    "content": "<p>ML role</p>",
                    "departments": [{"name": "Engineering"}],
                    "updated_at": "2026-02-27T00:00:00Z",
                }
            ]
        }
        mock_fetch.return_value = api_response

        # Setup mock DB
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE jobs (
                url TEXT PRIMARY KEY,
                title TEXT,
                salary TEXT,
                description TEXT,
                location TEXT,
                site TEXT,
                strategy TEXT,
                discovered_at TEXT,
                full_description TEXT,
                application_url TEXT,
                detail_scraped_at TEXT,
                detail_error TEXT
            )
        """)
        mock_get_conn.return_value = conn

        # Run search (returns jobs but doesn't store them - search_all does that)
        employer = {"name": "TestCorp"}
        jobs = search_employer("testcorp", employer, "ML")

        # Verify jobs returned
        assert len(jobs) == 1
        assert jobs[0]["title"] == "ML Engineer"

        # Manually store jobs to test _store_jobs integration
        from applypilot.discovery.greenhouse import _store_jobs

        new, existing = _store_jobs(jobs)

        # Verify stored in DB
        assert new == 1
        assert existing == 0
        count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        assert count == 1
