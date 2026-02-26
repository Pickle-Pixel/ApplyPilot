import pytest

from applypilot.llm import resolve_llm_config


def test_requires_llm_model() -> None:
    with pytest.raises(RuntimeError, match="LLM_MODEL is required"):
        resolve_llm_config({"GEMINI_API_KEY": "g-key"})


def test_requires_model_provider_prefix() -> None:
    with pytest.raises(RuntimeError, match="must include a provider prefix"):
        resolve_llm_config({"LLM_MODEL": "gpt-4o-mini", "OPENAI_API_KEY": "o-key"})


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


def test_llm_url_allows_missing_api_key() -> None:
    cfg = resolve_llm_config(
        {
            "LLM_MODEL": "openai/local-model",
            "LLM_URL": "http://127.0.0.1:8080/v1/",
        }
    )
    assert cfg.provider == "openai"
    assert cfg.api_base == "http://127.0.0.1:8080/v1"
    assert cfg.api_key == ""
