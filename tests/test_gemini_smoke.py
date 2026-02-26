import os

import pytest

litellm = pytest.importorskip("litellm")


def _gemini_smoke_model() -> str:
    raw = os.getenv("GEMINI_SMOKE_MODEL", "gemini-2.0-flash").strip()
    if raw.startswith("gemini/"):
        return raw
    if raw.startswith("models/"):
        raw = raw.split("/", 1)[1]
    return f"gemini/{raw}"


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
            if isinstance(part, (str, dict))
        ).strip()
    return ""


@pytest.mark.smoke
def test_gemini_smoke_completion_returns_non_empty_content() -> None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("Set GEMINI_API_KEY to run Gemini smoke tests.")

    prompt = os.getenv("GEMINI_SMOKE_PROMPT", "Reply with a single word: ready.")
    response = litellm.completion(
        model=_gemini_smoke_model(),
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        timeout=60,
        num_retries=1,
    )

    choices = getattr(response, "choices", None)
    assert choices, "Gemini smoke call returned no choices."

    content = choices[0].message.content
    assert _content_text(content), "Gemini smoke call returned empty choices[0].message.content."
