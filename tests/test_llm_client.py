from applypilot.llm import LLMClient, LLMConfig, _normalize_thinking_level


def test_normalize_thinking_level_accepts_supported_levels() -> None:
    assert _normalize_thinking_level("none") == "none"
    assert _normalize_thinking_level("low") == "low"
    assert _normalize_thinking_level("medium") == "medium"
    assert _normalize_thinking_level("high") == "high"


def test_normalize_thinking_level_defaults_minimal_to_low() -> None:
    assert _normalize_thinking_level("minimal") == "low"


def test_normalize_thinking_level_defaults_invalid_value_to_low() -> None:
    assert _normalize_thinking_level("max") == "low"


def test_build_response_args_applies_reasoning_effort_for_openai() -> None:
    client = LLMClient(
        LLMConfig(
            provider="openai",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
    )
    args = client._build_response_args(
        messages=[{"role": "user", "content": "hello"}],
        temperature=None,
        max_output_tokens=128,
        thinking_level="medium",
        response_kwargs=None,
    )
    assert args["reasoning_effort"] == "medium"
