"""F16: re-running init must not destroy existing .env keys."""
from applypilot.wizard.init import _merge_env


def test_merge_preserves_unknown_keys():
    out = _merge_env("CAPSOLVER_API_KEY=abc\nGEMINI_API_KEY=old", {"GEMINI_API_KEY": "new"})
    assert "CAPSOLVER_API_KEY=abc" in out
    assert "GEMINI_API_KEY=new" in out
    assert "GEMINI_API_KEY=old" not in out
    # Each key appears exactly once.
    assert out.count("CAPSOLVER_API_KEY=") == 1
    assert out.count("GEMINI_API_KEY=") == 1


def test_merge_appends_new_keys():
    out = _merge_env("CHROME_PATH=/usr/bin/chrome", {"GEMINI_API_KEY": "k"})
    assert "CHROME_PATH=/usr/bin/chrome" in out
    assert "GEMINI_API_KEY=k" in out


def test_merge_from_empty():
    out = _merge_env("", {"GEMINI_API_KEY": "k", "LLM_MODEL": "m"})
    assert "GEMINI_API_KEY=k" in out
    assert "LLM_MODEL=m" in out


def test_merge_keeps_comments():
    out = _merge_env("# my notes\nCAPSOLVER_API_KEY=x", {"LLM_MODEL": "m"})
    assert "# my notes" in out
    assert "CAPSOLVER_API_KEY=x" in out
