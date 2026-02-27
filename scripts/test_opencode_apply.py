#!/usr/bin/env python3
"""Test OpenCode backend with a real job application (dry-run).

This script tests the end-to-end application flow using OpenCode backend
without actually submitting the application (--dry-run mode).

Prerequisites:
1. Resume and profile configured:
   - Run: applypilot init
   - Or manually create ~/.applypilot/profile.json and ~/.applypilot/resume.txt
2. OpenCode installed: curl -fsSL https://opencode.ai/install | bash
3. MCP servers registered:
   - opencode mcp add playwright --provider=openai --url="$LLM_URL" --api-key="$LLM_API_KEY"
   - opencode mcp add gmail --provider=openai --url="$LLM_URL" --api-key="$LLM_API_KEY"
4. Environment configured in ~/.applypilot/.env:
   - APPLY_BACKEND=opencode
   - GEMINI_API_KEY (for scoring/tailoring)
   - LLM_URL, LLM_API_KEY, LLM_MODEL (for MCP servers)

Usage:
    # Test with a specific job URL (dry-run, no actual submission)
    python3 scripts/test_opencode_apply.py --url "https://example.com/jobs/123"

    # Test with OpenCode backend explicitly
    APPLY_BACKEND=opencode python3 scripts/test_opencode_apply.py --url "https://example.com/jobs/123"

    # Test with verbose output
    python3 scripts/test_opencode_apply.py --url "https://example.com/jobs/123" --verbose
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_prerequisites() -> list[str]:
    """Check that all prerequisites are met. Returns list of issues."""
    issues = []

    # Check OpenCode is installed
    if not subprocess.run(["which", "opencode"], capture_output=True).returncode == 0:
        issues.append("❌ OpenCode not found. Install: curl -fsSL https://opencode.ai/install | bash")
    else:
        result = subprocess.run(["opencode", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ OpenCode version: {result.stdout.strip()}")
        else:
            issues.append("❌ OpenCode found but --version failed")

    # Check MCP servers
    result = subprocess.run(["opencode", "mcp", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        issues.append("❌ Failed to list MCP servers. Run: opencode mcp list")
    else:
        servers = result.stdout
        required = ["playwright", "gmail"]
        for server in required:
            if server in servers:
                print(f"✓ MCP server '{server}' registered")
            else:
                issues.append(f"❌ MCP server '{server}' not found. Register: opencode mcp add {server} --provider=...")

    # Check profile and resume exist
    app_dir = Path.home() / ".applypilot"
    profile_path = app_dir / "profile.json"
    resume_txt = app_dir / "resume.txt"
    resume_pdf = app_dir / "resume.pdf"
    
    if not profile_path.exists():
        issues.append(f"❌ Profile not found at {profile_path}. Run: applypilot init")
    else:
        print(f"✓ Profile found: {profile_path}")
    
    if not (resume_txt.exists() or resume_pdf.exists()):
        issues.append(f"❌ Resume not found at {app_dir}/resume.txt or resume.pdf. Run: applypilot init")
    else:
        resume = resume_txt if resume_txt.exists() else resume_pdf
        print(f"✓ Resume found: {resume}")
    # Check environment
    backend = os.environ.get("APPLY_BACKEND", "claude")
    if backend != "opencode":
        issues.append(f"⚠️ APPLY_BACKEND={backend} (not opencode). Set: export APPLY_BACKEND=opencode")
    else:
        print("✓ APPLY_BACKEND=opencode")
    if not os.environ.get("GEMINI_API_KEY"):
        issues.append("⚠️ GEMINI_API_KEY not set (needed for scoring/tailoring)")
    else:
        print("✓ GEMINI_API_KEY set")

    return issues

def test_apply(url: str, verbose: bool = False) -> int:
    """Test applying to a job with dry-run mode."""
    print(f"\n{'=' * 60}")
    print(f"Testing OpenCode Backend Apply")
    print(f"{'=' * 60}")
    print(f"Job URL: {url}")
    print(f"Mode: DRY-RUN (no actual submission)")
    print()

    # Check prerequisites
    print("Checking prerequisites...")
    issues = check_prerequisites()

    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
        print("\nPlease fix these issues before testing.")
        return 1

    print("\n✓ All prerequisites met!")
    print()

    # Run applypilot apply --dry-run
    cmd = [
        sys.executable,
        "-m",
        "applypilot",
        "apply",
        "--url",
        url,
        "--dry-run",
    ]

    if verbose:
        cmd.append("--verbose")

    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    # Set up environment
    env = os.environ.copy()
    env["APPLY_BACKEND"] = "opencode"
    env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        print(f"\n{'=' * 60}")
        if result.returncode == 0:
            print("✓ Test completed successfully!")
            print("\nThe OpenCode backend worked through the full application flow.")
            print("Check the output above to verify it navigated the form correctly.")
        else:
            print(f"❌ Test failed with exit code: {result.returncode}")
            print("\nPossible issues:")
            print("- Job URL is no longer valid")
            print("- Form structure changed")
            print("- CAPTCHA or bot detection")
            print("- OpenCode backend error")

        print(f"{'=' * 60}")
        return result.returncode

    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return 1
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Test OpenCode backend with real job application (dry-run)")
    parser.add_argument("--url", required=True, help="Job URL to test (e.g., https://example.com/jobs/123)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    return test_apply(args.url, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
