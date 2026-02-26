"""Unified LLM client for ApplyPilot using LiteLLM.

Runtime contract:
  - LLM_MODEL must be a fully-qualified LiteLLM model string
    (for example: openai/gpt-4o-mini, anthropic/claude-3-5-haiku-latest,
    gemini/gemini-3.0-flash).
  - Credentials come from provider env vars (GEMINI_API_KEY, OPENAI_API_KEY,
    ANTHROPIC_API_KEY) or generic LLM_API_KEY.
  - LLM_URL is optional for custom OpenAI-compatible endpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os

import litellm

log = logging.getLogger(__name__)

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds


@dataclass(frozen=True)
class LLMConfig:
    """LLM configuration consumed by LLMClient."""

    provider: str
    api_base: str | None
    model: str
    api_key: str


def _env_get(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _provider_from_model(model: str) -> str:
    provider, _, model_name = model.partition("/")
    if not provider or not model_name:
        raise RuntimeError(
            "LLM_MODEL must include a provider prefix (for example 'openai/gpt-4o-mini')."
        )
    return provider


def resolve_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    """Resolve LLM configuration from environment."""
    env_map = env if env is not None else os.environ

    model = _env_get(env_map, "LLM_MODEL")
    if not model:
        raise RuntimeError(
            "LLM_MODEL is required. Set it to a LiteLLM model string like "
            "'openai/gpt-4o-mini'."
        )
    provider = _provider_from_model(model)
    local_url = _env_get(env_map, "LLM_URL")
    provider_api_key_env = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    api_key_env = provider_api_key_env.get(provider, "LLM_API_KEY")
    api_key = _env_get(env_map, api_key_env) or _env_get(env_map, "LLM_API_KEY")

    if not api_key and not local_url:
        key_help = (
            f"{api_key_env} or LLM_API_KEY"
            if provider in provider_api_key_env
            else "LLM_API_KEY"
        )
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

    def _build_completion_args(
        self,
        messages: list[dict],
        temperature: float | None,
        max_output_tokens: int,
        response_kwargs: Mapping[str, object] | None,
    ) -> dict:
        args: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "timeout": _TIMEOUT,
            "num_retries": _MAX_RETRIES,  # Delegate retry handling to LiteLLM.
        }
        if temperature is not None:
            args["temperature"] = temperature

        if self.config.api_key:
            args["api_key"] = self.config.api_key

        if self.config.api_base:
            args["api_base"] = self.config.api_base

        if response_kwargs:
            args.update(response_kwargs)
        return args

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_output_tokens: int = 10000,
        response_kwargs: Mapping[str, object] | None = None,
    ) -> str:
        """Send a completion request and return plain text content."""
        litellm.suppress_debug_info = True

        try:
            response = litellm.completion(
                **self._build_completion_args(
                    messages=messages,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    response_kwargs=response_kwargs,
                )
            )

            choices = getattr(response, "choices", None)
            if not choices:
                raise RuntimeError("LLM response contained no choices.")
            text = response.choices[0].message.content.strip()

            if not text:
                raise RuntimeError("LLM response contained no text content.")
            return text
        except Exception as exc:  # pragma: no cover - provider SDK exception types vary by backend/version.
            raise RuntimeError(f"LLM request failed ({self.provider}/{self.model}): {exc}") from exc

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

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
