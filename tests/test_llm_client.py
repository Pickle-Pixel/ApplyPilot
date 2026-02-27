import os
from types import SimpleNamespace

import applypilot.llm as llm_module
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
    assert llm_module.litellm.suppress_debug_info is True


def _mock_response(content: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_chat_passes_defaults_without_temperature(monkeypatch) -> None:
    client = LLMClient(
        LLMConfig(
            provider="openai",
            api_base=None,
            model="openai/gpt-4o-mini",
            api_key="test-key",
        )
    )
    captured: dict[str, object] = {}

    def _fake_completion(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _mock_response()

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)
    response = client.chat([{"role": "user", "content": "hello"}], max_output_tokens=128)

    assert response == "hello"
    assert captured["model"] == "openai/gpt-4o-mini"
    assert captured["max_tokens"] == 128
    assert captured["timeout"] == 120
    assert captured["num_retries"] == 5
    assert captured["drop_params"] is True
    assert captured["api_key"] == "test-key"
    assert captured["api_base"] is None
    assert "temperature" not in captured
    assert "reasoning_effort" not in captured


def test_chat_supports_temperature_and_typed_extra(monkeypatch) -> None:
    client = LLMClient(
        LLMConfig(
            provider="gemini",
            api_base=None,
            model="gemini/gemini-3.0-flash",
            api_key="g-key",
        )
    )
    captured: dict[str, object] = {}

    def _fake_completion(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _mock_response("ok")

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)
    response = client.chat(
        [{"role": "user", "content": "hello"}],
        max_output_tokens=64,
        temperature=0.2,
        top_p=0.9,
        stop=["\n\n"],
        response_format={"type": "json_object"},
    )

    assert response == "ok"
    assert captured["model"] == "gemini/gemini-3.0-flash"
    assert captured["api_key"] == "g-key"
    assert captured["temperature"] == 0.2
    assert captured["top_p"] == 0.9
    assert captured["stop"] == ["\n\n"]
    assert captured["response_format"] == {"type": "json_object"}


def test_chat_sets_local_api_base_and_api_key(monkeypatch) -> None:
    client = LLMClient(
        LLMConfig(
            provider="openai",
            api_base="http://127.0.0.1:8080/v1",
            model="openai/local-model",
            api_key="local-key",
        )
    )
    captured: dict[str, object] = {}

    def _fake_completion(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _mock_response()

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)
    _ = client.chat([{"role": "user", "content": "hello"}], max_output_tokens=64)

    assert captured["api_base"] == "http://127.0.0.1:8080/v1"
    assert captured["api_key"] == "local-key"
