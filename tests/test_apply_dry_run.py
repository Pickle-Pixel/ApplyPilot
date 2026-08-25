"""Safety regressions for ApplyPilot dry-run mode."""

import unittest
from unittest.mock import patch

from applypilot.apply import launcher


class TestApplyDryRun(unittest.TestCase):
    @patch.object(launcher, "cleanup_worker")
    @patch.object(launcher, "launch_chrome", return_value=object())
    @patch.object(launcher, "update_state")
    @patch.object(launcher, "add_event")
    @patch.object(launcher, "mark_result")
    @patch.object(launcher, "release_lock")
    @patch.object(launcher, "run_job", return_value=("applied", 100))
    @patch.object(launcher, "acquire_job")
    def test_dry_run_never_marks_applied(
        self, acquire_job, run_job, release_lock, mark_result,
        add_event, update_state, launch_chrome, cleanup_worker,
    ):
        acquire_job.return_value = {
            "url": "https://example.test/job/1",
            "title": "Engineer",
            "site": "Example",
        }

        applied, failed = launcher.worker_loop(limit=1, dry_run=True)

        self.assertEqual((applied, failed), (0, 0))
        release_lock.assert_called_once_with("https://example.test/job/1")
        mark_result.assert_not_called()


if __name__ == "__main__":
    unittest.main()
