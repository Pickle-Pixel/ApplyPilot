import pytest

from applypilot.llm import resolve_llm_config


def test_infers_provider_from_first_configured_source() -> None:
    cfg = resolve_llm_config(
        {
            "GEMINI_API_KEY": "g-key",
            "OPENAI_API_KEY": "o-key",
            "ANTHROPIC_API_KEY": "a-key",
            "LLM_URL": "http://127.0.0.1:8080/v1",
        }
    )
    assert cfg.provider == "gemini"
    assert cfg.model == "gemini/gemini-3.0-flash"
    assert cfg.api_key == "g-key"


def test_unprefixed_model_uses_inferred_provider() -> None:
    cfg = resolve_llm_config({"LLM_MODEL": "gpt-4o-mini", "OPENAI_API_KEY": "o-key"})
    assert cfg.provider == "openai"
    assert cfg.model == "openai/gpt-4o-mini"


def test_requires_model_provider_prefix_without_inferable_provider() -> None:
    with pytest.raises(RuntimeError, match="must include a provider prefix"):
        resolve_llm_config({"LLM_MODEL": "gpt-4o-mini", "LLM_API_KEY": "generic"})


def test_provider_and_api_key_come_from_model_contract() -> None:
    cfg = resolve_llm_config({"LLM_MODEL": "gemini/gemini-3.0-flash", "GEMINI_API_KEY": "g-key"})
    assert cfg.provider == "gemini"
    assert cfg.api_base is None
    assert cfg.model == "gemini/gemini-3.0-flash"
    assert cfg.api_key == "g-key"


def test_uses_generic_api_key_for_unmapped_provider() -> None:
    cfg = resolve_llm_config({"LLM_MODEL": "vertex_ai/gemini-3.0-flash", "LLM_API_KEY": "v-key"})
    assert cfg.provider == "vertex_ai"
    assert cfg.api_key == "v-key"


def test_llm_url_infers_local_default_model_and_allows_missing_api_key() -> None:
    cfg = resolve_llm_config(
        {
            "LLM_URL": "http://127.0.0.1:8080/v1/",
        }
    )
    assert cfg.provider == "openai"
    assert cfg.model == "openai/local-model"
    assert cfg.api_base == "http://127.0.0.1:8080/v1"
    assert cfg.api_key == ""


def test_missing_everything_raises_clear_error() -> None:
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        resolve_llm_config({})
