import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from applypilot.apply import chrome


class ChromeProfileSetupTests(unittest.TestCase):
    def test_missing_source_profile_creates_clean_worker_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker_root = root / "chrome-workers"
            missing_source = root / "missing-google-profile"

            with patch.object(chrome.config, "CHROME_WORKER_DIR", worker_root), patch.object(
                chrome.config, "get_chrome_user_data", return_value=missing_source
            ):
                profile_dir = chrome.setup_worker_profile(0)

            self.assertEqual(profile_dir, worker_root / "worker-0")
            self.assertTrue((profile_dir / "Default").is_dir())


if __name__ == "__main__":
    unittest.main()
