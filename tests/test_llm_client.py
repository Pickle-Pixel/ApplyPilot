import os

from applypilot.llm import LLMClient, LLMConfig


def test_client_init_does_not_mutate_provider_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    LLMClient(
        LLMConfig(
            provider="openai",
            api_base=None,
            model="openai/gpt-4o-mini",
            api_key="test-key",
        )
    )
    assert "OPENAI_API_KEY" not in os.environ


def test_build_completion_args_does_not_include_reasoning_effort_by_default() -> None:
    client = LLMClient(
        LLMConfig(
            provider="openai",
            api_base=None,
            model="openai/gpt-4o-mini",
            api_key="test-key",
        )
    )
    args = client._build_completion_args(
        messages=[{"role": "user", "content": "hello"}],
        temperature=None,
        max_output_tokens=128,
        response_kwargs=None,
    )
    assert "reasoning_effort" not in args
    assert args["max_tokens"] == 128


def test_build_completion_args_uses_litellm_native_gemini_model_prefix() -> None:
    client = LLMClient(
        LLMConfig(
            provider="gemini",
            api_base=None,
            model="gemini/gemini-3.0-flash",
            api_key="g-key",
        )
    )
    args = client._build_completion_args(
        messages=[{"role": "user", "content": "hello"}],
        temperature=None,
        max_output_tokens=64,
        response_kwargs=None,
    )
    assert args["model"] == "gemini/gemini-3.0-flash"


def test_build_completion_args_includes_api_key_for_remote_provider() -> None:
    client = LLMClient(
        LLMConfig(
            provider="gemini",
            api_base=None,
            model="gemini/gemini-3.0-flash",
            api_key="g-key",
        )
    )
    args = client._build_completion_args(
        messages=[{"role": "user", "content": "hello"}],
        temperature=None,
        max_output_tokens=64,
        response_kwargs=None,
    )
    assert args["api_key"] == "g-key"


def test_build_completion_args_sets_local_api_base_and_api_key() -> None:
    client = LLMClient(
        LLMConfig(
            provider="openai",
            api_base="http://127.0.0.1:8080/v1",
            model="openai/local-model",
            api_key="local-key",
        )
    )
    args = client._build_completion_args(
        messages=[{"role": "user", "content": "hello"}],
        temperature=None,
        max_output_tokens=64,
        response_kwargs=None,
    )
    assert args["api_base"] == "http://127.0.0.1:8080/v1"
    assert args["api_key"] == "local-key"
