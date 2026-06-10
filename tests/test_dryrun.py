"""F1: apply --dry-run must be side-effect-free."""
import applypilot.config as config
import applypilot.apply.prompt as prompt_mod
from applypilot.apply.launcher import _build_claude_cmd


def _fixture_profile():
    return {
        "personal": {
            "full_name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "5551234567",
            "city": "Remoteville",
        },
        "work_authorization": {},
        "compensation": {"salary_expectation": "100000"},
        "experience": {},
        "availability": {},
        "eeo": {},
        "skills_boundary": {},
    }


def _make_job(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    return {
        "url": "https://example.com/job1",
        "title": "Software Engineer",
        "site": "linkedin",
        "application_url": None,
        "fit_score": 8,
        "tailored_resume_path": str(tmp_path / "x.txt"),
    }


def _build(tmp_path, monkeypatch, dry_run):
    monkeypatch.setattr(config, "load_profile", _fixture_profile)
    monkeypatch.setattr(config, "load_search_config", lambda: {})
    monkeypatch.setattr(config, "APPLY_WORKER_DIR", tmp_path)
    job = _make_job(tmp_path)
    return prompt_mod.build_prompt(job=job, tailored_resume="résumé text", dry_run=dry_run)


def test_dryrun_prompt_uses_dryrun_result_code(tmp_path, monkeypatch):
    out = _build(tmp_path, monkeypatch, dry_run=True)
    assert "RESULT:DRYRUN" in out
    # The email-only step must not instruct a real send.
    assert "Output RESULT:APPLIED. Done." not in out
    assert "do NOT send any email" in out


def test_non_dryrun_prompt_unchanged(tmp_path, monkeypatch):
    out = _build(tmp_path, monkeypatch, dry_run=False)
    assert "RESULT:DRYRUN" not in out
    # Real email-only path retains the send_email instruction.
    assert "send_email with subject" in out
    assert "Output RESULT:APPLIED. Done." in out


def test_dryrun_disallows_send_email_tool():
    real = _build_claude_cmd("claude-x", "/tmp/mcp.json", dry_run=False)
    dry = _build_claude_cmd("claude-x", "/tmp/mcp.json", dry_run=True)
    real_disallowed = real[real.index("--disallowedTools") + 1]
    dry_disallowed = dry[dry.index("--disallowedTools") + 1]
    assert "mcp__gmail__send_email" not in real_disallowed
    assert "mcp__gmail__send_email" in dry_disallowed


def test_dangerous_builtin_tools_disallowed():
    """F2: host/network built-ins are denied even outside dry-run."""
    cmd = _build_claude_cmd("claude-x", "/tmp/mcp.json", dry_run=False)
    disallowed = cmd[cmd.index("--disallowedTools") + 1]
    for tool in ("Bash", "Edit", "Write", "WebFetch", "WebSearch"):
        assert tool in disallowed
    # Browser + read must remain available (not in the deny list as standalone tokens).
    assert "mcp__playwright__" not in disallowed


def test_prompt_injection_guard_present(tmp_path, monkeypatch):
    out = _build(tmp_path, monkeypatch, dry_run=False)
    assert "suspected_prompt_injection" in out
