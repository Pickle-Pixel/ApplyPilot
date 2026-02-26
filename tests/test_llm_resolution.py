import logging

import pytest

from applypilot.llm import resolve_llm_config


def test_only_gemini_api_key_selects_gemini() -> None:
    cfg = resolve_llm_config({"GEMINI_API_KEY": "g-key"})
    assert cfg.provider == "gemini"
    assert cfg.base_url == ""
    assert cfg.model == "gemini-2.0-flash"


def test_only_openai_api_key_selects_openai() -> None:
    cfg = resolve_llm_config({"OPENAI_API_KEY": "o-key"})
    assert cfg.provider == "openai"


def test_gemini_model_override_without_prefix_is_normalized() -> None:
    cfg = resolve_llm_config({"GEMINI_API_KEY": "g-key", "LLM_MODEL": "gemini-2.5-flash"})
    assert cfg.model == "gemini-2.5-flash"


def test_gemini_model_override_google_models_prefix_is_normalized() -> None:
    cfg = resolve_llm_config({"GEMINI_API_KEY": "g-key", "LLM_MODEL": "models/gemini-2.5-flash"})
    assert cfg.model == "gemini-2.5-flash"


def test_gemini_model_override_gemini_prefix_is_stripped() -> None:
    cfg = resolve_llm_config({"GEMINI_API_KEY": "g-key", "LLM_MODEL": "gemini/gemini-2.5-flash"})
    assert cfg.model == "gemini-2.5-flash"


def test_llm_url_with_keys_selects_local() -> None:
    cfg = resolve_llm_config(
        {
            "LLM_URL": "http://127.0.0.1:8080/v1",
            "GEMINI_API_KEY": "g-key",
            "OPENAI_API_KEY": "o-key",
            "ANTHROPIC_API_KEY": "a-key",
        }
    )
    assert cfg.provider == "local"


def test_multiple_keys_selects_deterministically_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        cfg = resolve_llm_config(
            {
                "GEMINI_API_KEY": "g-key",
                "OPENAI_API_KEY": "o-key",
                "ANTHROPIC_API_KEY": "a-key",
            }
        )
    assert cfg.provider == "gemini"
    assert any(
        "Multiple LLM providers configured" in rec.message and "Using 'gemini' based on precedence" in rec.message
        for rec in caplog.records
    )


def test_missing_everything_raises_clear_error() -> None:
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        resolve_llm_config({})
