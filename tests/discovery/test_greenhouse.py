"""Unit tests for Greenhouse ATS discovery module."""

import sqlite3
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from applypilot.discovery.greenhouse import (
    GREENHOUSE_BASE_URL,
    _location_ok,
    _store_jobs,
    _title_matches_query,
    fetch_greenhouse_board,
    load_employers,
    parse_greenhouse_jobs,
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
    """Tests for HTML parsing functions."""

    def test_parse_simple_job_posting(self):
        """Test parsing a simple job posting HTML."""
        html = """
        <html>
        <body>
            <div class="job-posts--table--department">
                <h3 class="section-header">Engineering</h3>
                <table>
                    <tr class="job-post">
                        <td>
                            <a href="/jobs/12345">
                                <p class="body--medium">Software Engineer</p>
                                <p class="body--metadata">San Francisco, CA</p>
                            </a>
                        </td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """

        jobs = parse_greenhouse_jobs(html, "Test Company", "")

        assert len(jobs) == 1
        assert jobs[0]["title"] == "Software Engineer"
        assert jobs[0]["company"] == "Test Company"
        assert jobs[0]["location"] == "San Francisco, CA"
        assert jobs[0]["department"] == "Engineering"
        assert jobs[0]["strategy"] == "greenhouse"
        assert jobs[0]["url"].endswith("/jobs/12345")

    def test_parse_multiple_jobs(self):
        """Test parsing multiple job postings."""
        html = """
        <html>
        <body>
            <div class="job-posts--table--department">
                <h3 class="section-header">Engineering</h3>
                <table>
                    <tr class="job-post">
                        <td>
                            <a href="/jobs/1">
                                <p class="body--medium">Frontend Engineer</p>
                                <p class="body--metadata">Remote</p>
                            </a>
                        </td>
                    </tr>
                    <tr class="job-post">
                        <td>
                            <a href="/jobs/2">
                                <p class="body--medium">Backend Engineer</p>
                                <p class="body--metadata">New York, NY</p>
                            </a>
                        </td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """

        jobs = parse_greenhouse_jobs(html, "Test Company", "")

        assert len(jobs) == 2
        assert jobs[0]["title"] == "Frontend Engineer"
        assert jobs[1]["title"] == "Backend Engineer"

    def test_filter_by_query(self):
        """Test filtering jobs by query string."""
        html = """
        <html>
        <body>
            <div class="job-posts--table--department">
                <table>
                    <tr class="job-post">
                        <td>
                            <a href="/jobs/1">
                                <p class="body--medium">Machine Learning Engineer</p>
                                <p class="body--metadata">Remote</p>
                            </a>
                        </td>
                    </tr>
                    <tr class="job-post">
                        <td>
                            <a href="/jobs/2">
                                <p class="body--medium">Sales Representative</p>
                                <p class="body--metadata">Remote</p>
                            </a>
                        </td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """

        jobs = parse_greenhouse_jobs(html, "Test Company", "machine learning")

        assert len(jobs) == 1
        assert jobs[0]["title"] == "Machine Learning Engineer"

    def test_handles_absolute_urls(self):
        """Test that absolute URLs are preserved."""
        html = """
        <html>
        <body>
            <div class="job-posts--table--department">
                <table>
                    <tr class="job-post">
                        <td>
                            <a href="https://boards.greenhouse.io/test/jobs/123">
                                <p class="body--medium">Test Job</p>
                                <p class="body--metadata">Remote</p>
                            </a>
                        </td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """

        jobs = parse_greenhouse_jobs(html, "Test Company", "")

        assert len(jobs) == 1
        assert jobs[0]["url"] == "https://boards.greenhouse.io/test/jobs/123"

    def test_handles_relative_urls(self):
        """Test that relative URLs are converted to absolute."""
        html = """
        <html>
        <body>
            <div class="job-posts--table--department">
                <table>
                    <tr class="job-post">
                        <td>
                            <a href="/jobs/123">
                                <p class="body--medium">Test Job</p>
                                <p class="body--metadata">Remote</p>
                            </a>
                        </td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """

        jobs = parse_greenhouse_jobs(html, "Test Company", "")

        assert len(jobs) == 1
        assert jobs[0]["url"].startswith("https://")
        assert "/jobs/123" in jobs[0]["url"]

    def test_handles_empty_html(self):
        """Test handling of empty HTML."""
        jobs = parse_greenhouse_jobs("", "Test Company", "")
        assert jobs == []

    def test_handles_no_job_posts(self):
        """Test handling of HTML without job posts."""
        html = "<html><body><h1>No jobs here</h1></body></html>"
        jobs = parse_greenhouse_jobs(html, "Test Company", "")
        assert jobs == []


class TestFetchGreenhouseBoard:
    """Tests for HTTP fetching functions."""

    @patch("applypilot.discovery.greenhouse.httpx.Client")
    def test_successful_fetch(self, mock_client_class):
        """Test successful HTML fetch."""
        mock_response = Mock()
        mock_response.text = "<html>Test</html>"
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)

        mock_client_class.return_value = mock_client

        result = fetch_greenhouse_board("testcompany")

        assert result == "<html>Test</html>"
        mock_client.get.assert_called_once()

    @patch("applypilot.discovery.greenhouse.httpx.Client")
    def test_failed_fetch_returns_none(self, mock_client_class):
        """Test that failed fetches return None gracefully."""
        mock_client = Mock()
        mock_client.get.side_effect = Exception("Connection error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)

        mock_client_class.return_value = mock_client

        result = fetch_greenhouse_board("testcompany")

        assert result is None

    def test_url_format(self):
        """Test that URL is formatted correctly."""
        assert GREENHOUSE_BASE_URL.format(company="scaleai") == "https://job-boards.greenhouse.io/scaleai"
        assert GREENHOUSE_BASE_URL.format(company="stripe") == "https://job-boards.greenhouse.io/stripe"


class TestSearchEmployer:
    """Tests for employer search function."""

    @patch("applypilot.discovery.greenhouse.fetch_greenhouse_board")
    @patch("applypilot.discovery.greenhouse.parse_greenhouse_jobs")
    def test_search_with_location_filter(self, mock_parse, mock_fetch):
        """Test searching with location filter enabled."""
        mock_fetch.return_value = "<html>Test</html>"
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
        mock_fetch.assert_called_once_with("test")
        mock_parse.assert_called_once_with("<html>Test</html>", "Test Company", "engineer")

    @patch("applypilot.discovery.greenhouse.fetch_greenhouse_board")
    def test_search_no_html_returns_empty(self, mock_fetch):
        """Test that empty result is returned if fetch fails."""
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

    @patch("applypilot.discovery.greenhouse.fetch_greenhouse_board")
    @patch("applypilot.discovery.greenhouse.get_connection")
    def test_end_to_end_search_and_store(self, mock_get_conn, mock_fetch, tmp_path):
        """Test full flow from fetch to parse to store."""
        # Setup mock HTML
        html = """
        <html>
        <body>
            <div class="job-posts--table--department">
                <h3 class="section-header">Engineering</h3>
                <table>
                    <tr class="job-post">
                        <td>
                            <a href="/jobs/123">
                                <p class="body--medium">ML Engineer</p>
                                <p class="body--metadata">Remote</p>
                            </a>
                        </td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """
        mock_fetch.return_value = html

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
