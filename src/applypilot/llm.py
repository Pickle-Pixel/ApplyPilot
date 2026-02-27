"""Unified LLM client for ApplyPilot using LiteLLM.

Runtime contract:
  - If set, LLM_MODEL must be a fully-qualified LiteLLM model string
    (for example: openai/gpt-4o-mini, anthropic/claude-3-5-haiku-latest,
    gemini/gemini-3.0-flash).
  - If LLM_MODEL is unset, provider is inferred by first configured source:
    GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, then LLM_URL.
  - Credentials come from provider env vars or generic LLM_API_KEY.
  - LLM_URL is optional for custom OpenAI-compatible endpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os
from typing import Any, Literal, TypedDict, Unpack
import warnings

import litellm

log = logging.getLogger(__name__)

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds
_INFERRED_SOURCE_ORDER: tuple[tuple[str, str], ...] = (
    ("gemini", "GEMINI_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "LLM_URL"),
)
_DEFAULT_MODEL_BY_PROVIDER = {
    "gemini": "gemini/gemini-3.0-flash",
    "openai": "openai/gpt-5-mini",
    "anthropic": "anthropic/claude-haiku-4-5",
}
_DEFAULT_LOCAL_MODEL = "openai/local-model"


@dataclass(frozen=True)
class LLMConfig:
    """LLM configuration consumed by LLMClient."""

    provider: str
    api_base: str | None
    model: str
    api_key: str


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LiteLLMExtra(TypedDict, total=False):
    stop: str | list[str]
    top_p: float
    seed: int
    stream: bool
    response_format: dict[str, Any]
    tools: list[dict[str, Any]]
    tool_choice: str | dict[str, Any]
    fallbacks: list[str]


def _env_get(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _provider_from_model(model: str) -> str:
    provider, _, model_name = model.partition("/")
    if not provider or not model_name:
        raise RuntimeError("LLM_MODEL must include a provider prefix (for example 'openai/gpt-4o-mini').")
    return provider


def _infer_provider_and_source(env: Mapping[str, str]) -> tuple[str, str] | None:
    for provider, env_key in _INFERRED_SOURCE_ORDER:
        if _env_get(env, env_key):
            return provider, env_key
    return None


def resolve_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    """Resolve LLM configuration from environment."""
    env_map = env if env is not None else os.environ

    model = _env_get(env_map, "LLM_MODEL")
    local_url = _env_get(env_map, "LLM_URL")
    inferred = _infer_provider_and_source(env_map)
    if model:
        if "/" in model:
            provider = _provider_from_model(model)
        elif inferred:
            provider, _ = inferred
            model = f"{provider}/{model}"
        else:
            raise RuntimeError("LLM_MODEL must include a provider prefix (for example 'openai/gpt-4o-mini').")
    else:
        if not inferred:
            raise RuntimeError(
                "No LLM provider configured. Set one of GEMINI_API_KEY, OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, LLM_URL, or LLM_MODEL."
            )
        provider, source = inferred
        if source == "LLM_URL":
            model = _DEFAULT_LOCAL_MODEL
        else:
            model = _DEFAULT_MODEL_BY_PROVIDER[provider]

    provider_api_key_env = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    api_key_env = provider_api_key_env.get(provider, "LLM_API_KEY")
    api_key = _env_get(env_map, api_key_env) or _env_get(env_map, "LLM_API_KEY")

    if not api_key and not local_url:
        key_help = f"{api_key_env} or LLM_API_KEY" if provider in provider_api_key_env else "LLM_API_KEY"
        raise RuntimeError(
            f"Missing credentials for LLM_MODEL '{model}'. Set {key_help}, or set LLM_URL for "
            "a local OpenAI-compatible endpoint."
        )

    return LLMConfig(
        provider=provider,
        api_base=local_url.rstrip("/") if local_url else None,
        model=model,
        api_key=api_key,
    )


class LLMClient:
    """Thin wrapper around LiteLLM completion()."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.provider = config.provider
        self.model = config.model
        litellm.suppress_debug_info = True

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int = 10000,
        temperature: float | None = None,
        timeout: int = _TIMEOUT,
        num_retries: int = _MAX_RETRIES,
        drop_params: bool = True,
        **extra: Unpack[LiteLLMExtra],
    ) -> str:
        """Send a completion request and return plain text content."""
        try:
            if temperature is None:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    timeout=timeout,
                    num_retries=num_retries,
                    drop_params=drop_params,
                    api_key=self.config.api_key or None,
                    api_base=self.config.api_base or None,
                    **extra,
                )
            else:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    num_retries=num_retries,
                    drop_params=drop_params,
                    api_key=self.config.api_key or None,
                    api_base=self.config.api_base or None,
                    **extra,
                )

            choices = getattr(response, "choices", None)
            if not choices:
                raise RuntimeError("LLM response contained no choices.")
            content = response.choices[0].message.content
            text = content.strip() if isinstance(content, str) else str(content).strip()

            if not text:
                raise RuntimeError("LLM response contained no text content.")
            return text
        except Exception as exc:  # pragma: no cover - provider SDK exception types vary by backend/version.
            raise RuntimeError(f"LLM request failed ({self.provider}/{self.model}): {exc}") from exc

    def close(self) -> None:
        """No-op. LiteLLM completion() is stateless per call."""
        return None


_instance: LLMClient | None = None


def get_client() -> LLMClient:
    """Return (or create) the module-level LLMClient singleton."""
    global _instance
    if _instance is None:
        try:
            from applypilot.config import load_env

            load_env()
        except ModuleNotFoundError:
            log.debug("python-dotenv not installed; skipping .env auto-load in llm.get_client().")
        config = resolve_llm_config()
        log.info("LLM provider: %s  model: %s", config.provider, config.model)
        _instance = LLMClient(config)
    return _instance


def validate_api_key(provider: str, api_key: str, model: str = "", endpoint: str = "") -> tuple[bool, str]:
    """Validate an API key by making a minimal test request.

    Args:
        provider: "gemini", "openai", or "local"
        api_key: The API key to validate
        model: Optional model name override
        endpoint: Required for "local" provider

    Returns:
        (is_valid, error_message) - error_message is empty if valid
    """
    try:
        if provider == "gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
            test_model = model or "gemini-2.0-flash"
            client = LLMClient(base_url, test_model, api_key)
        elif provider == "openai":
            base_url = "https://api.openai.com/v1"
            test_model = model or "gpt-4o-mini"
            client = LLMClient(base_url, test_model, api_key)
        elif provider == "local":
            if not endpoint:
                return False, "Local endpoint URL is required"
            base_url = endpoint.rstrip("/")
            test_model = model or "local-model"
            client = LLMClient(base_url, test_model, api_key)
        else:
            return False, f"Unknown provider: {provider}"

        # Simple test request
        response = client.ask("Reply with only the word 'ok'.", temperature=0.0, max_tokens=10)
        client.close()

        if response and len(response.strip()) > 0:
            return True, ""
        return False, "Empty response from API"

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return False, "Invalid API key"
        elif e.response.status_code == 403:
            return False, "API key lacks required permissions"
        elif e.response.status_code == 429:
            # Rate limited but key is valid
            return True, ""
        return False, f"API error: {e.response.status_code}"
    except httpx.ConnectError:
        return False, "Could not connect to API endpoint"
    except httpx.TimeoutException:
        return False, "API request timed out"
    except Exception as e:
        return False, f"Validation failed: {str(e)}"
