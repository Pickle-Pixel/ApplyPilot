import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_config_module(*, home: Path, cwd: Path, env: dict[str, str]):
    original_env = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        sys.modules.pop("applypilot.config", None)
        with patch("pathlib.Path.home", return_value=home), patch("pathlib.Path.cwd", return_value=cwd):
            return importlib.import_module("applypilot.config")
    finally:
        os.environ.clear()
        os.environ.update(original_env)


class ConfigSetupTests(unittest.TestCase):
    def test_wsl_prefers_windows_userprofile_applypilot_directory_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            linux_home = root / "linux-home"
            windows_home = root / "windows-home"
            project_dir = windows_home / "Projects" / "dev" / "ApplyPilot"
            windows_applypilot = windows_home / ".applypilot"

            linux_home.mkdir(parents=True)
            project_dir.mkdir(parents=True)
            windows_applypilot.mkdir(parents=True)

            config = load_config_module(
                home=linux_home,
                cwd=project_dir,
                env={
                    "USERPROFILE": str(windows_home),
                    "WSL_DISTRO_NAME": "Ubuntu",
                },
            )

            self.assertEqual(config.APP_DIR, windows_applypilot)

    def test_get_chrome_path_rejects_directory_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_dir = root / "chrome-dir"
            bad_dir.mkdir()

            config = load_config_module(
                home=root / "linux-home",
                cwd=root,
                env={"CHROME_PATH": str(bad_dir)},
            )

            with patch("applypilot.config.platform.system", return_value="Windows"), patch(
                "applypilot.config.shutil.which", return_value=None
            ), patch.dict(
                os.environ,
                {
                    "CHROME_PATH": str(bad_dir),
                    "PROGRAMFILES": str(root / "Program Files"),
                    "PROGRAMFILES(X86)": str(root / "Program Files (x86)"),
                    "LOCALAPPDATA": str(root / "LocalAppData"),
                },
                clear=True,
            ):
                with self.assertRaises(FileNotFoundError):
                    config.get_chrome_path()

    def test_load_env_clears_removed_provider_keys_from_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            app_dir = home / ".applypilot"
            home.mkdir(parents=True)
            app_dir.mkdir(parents=True)
            (app_dir / ".env").write_text(
                "LLM_URL=http://localhost:11434/v1\n"
                "LLM_MODEL=glm-5.1:cloud\n",
                encoding="utf-8",
            )

            config = load_config_module(home=home, cwd=root, env={})

            with patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "stale-gemini",
                    "OPENAI_API_KEY": "stale-openai",
                },
                clear=True,
            ):
                config.load_env()

                self.assertNotIn("GEMINI_API_KEY", os.environ)
                self.assertNotIn("OPENAI_API_KEY", os.environ)
                self.assertEqual(os.environ["LLM_URL"], "http://localhost:11434/v1")


if __name__ == "__main__":
    unittest.main()
