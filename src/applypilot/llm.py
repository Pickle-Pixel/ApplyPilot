"""Unified LLM client for ApplyPilot using LiteLLM.

Auto-detects provider from environment:
  GEMINI_API_KEY      -> Google Gemini (default: gemini-2.0-flash)
  OPENAI_API_KEY      -> OpenAI (default: gpt-4o-mini)
  ANTHROPIC_API_KEY   -> Anthropic Claude (default: claude-3-5-haiku-latest)
  LLM_URL             -> Local OpenAI-compatible endpoint

LLM_MODEL env var overrides the model name for any provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os

import litellm

log = logging.getLogger(__name__)

_OPENAI_BASE = "https://api.openai.com/v1"
_ANTHROPIC_BASE = "https://api.anthropic.com/v1"
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_PROVIDER_API_ENV_KEY = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_DEFAULT_MODEL_BY_PROVIDER = {
    "local": "local-model",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds

_THINKING_LEVELS = {"none", "low", "medium", "high"}


@dataclass(frozen=True)
class LLMConfig:
    """Normalized LLM configuration consumed by LLMClient."""

    provider: str
    base_url: str
    model: str
    api_key: str


def _env_get(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _normalize_thinking_level(thinking_level: str) -> str:
    level = (thinking_level or "low").strip().lower()
    if level not in _THINKING_LEVELS:
        log.warning("Invalid thinking_level '%s', defaulting to 'low'.", thinking_level)
        return "low"
    return level


def _provider_model(provider: str, model: str) -> str:
    if provider == "local":
        return model
    if model.startswith(f"{provider}/"):
        return model
    return f"{provider}/{model}"


def _default_model(provider: str) -> str:
    return _DEFAULT_MODEL_BY_PROVIDER[provider]


def resolve_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    """Resolve provider configuration from environment with deterministic precedence."""
    env_map = env if env is not None else os.environ

    model_override = _env_get(env_map, "LLM_MODEL")
    local_url = _env_get(env_map, "LLM_URL")
    gemini_key = _env_get(env_map, "GEMINI_API_KEY")
    openai_key = _env_get(env_map, "OPENAI_API_KEY")
    anthropic_key = _env_get(env_map, "ANTHROPIC_API_KEY")
    llm_provider = _env_get(env_map, "LLM_PROVIDER").lower()

    providers_present = {
        "local": bool(local_url),
        "gemini": bool(gemini_key),
        "openai": bool(openai_key),
        "anthropic": bool(anthropic_key),
    }
    precedence = ["local", "gemini", "openai", "anthropic"]
    configured = [provider for provider in precedence if providers_present[provider]]

    if not configured:
        raise RuntimeError(
            "No LLM provider configured. "
            "Set one of LLM_URL, GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
        )

    chosen = ""
    override_aliases = {
        "local": "local",
        "gemini": "gemini",
        "openai": "openai",
        "anthropic": "anthropic",
    }

    # Optional override only when multiple providers are configured.
    if len(configured) > 1 and llm_provider:
        overridden = override_aliases.get(llm_provider)
        if overridden and overridden in configured:
            chosen = overridden
            log.warning(
                "Multiple LLM providers configured (%s). Using '%s' via LLM_PROVIDER override.",
                ", ".join(configured),
                chosen,
            )
        else:
            log.warning(
                "Ignoring LLM_PROVIDER='%s' because it is not configured. "
                "Using precedence: LLM_URL > GEMINI_API_KEY > OPENAI_API_KEY > ANTHROPIC_API_KEY.",
                llm_provider,
            )

    if not chosen:
        chosen = configured[0]
        if len(configured) > 1:
            log.warning(
                "Multiple LLM providers configured (%s). Using '%s' based on precedence: "
                "LLM_URL > GEMINI_API_KEY > OPENAI_API_KEY > ANTHROPIC_API_KEY.",
                ", ".join(configured),
                chosen,
            )
    model = model_override or _default_model(chosen)

    if chosen == "local":
        return LLMConfig(
            provider="local",
            base_url=local_url.rstrip("/"),
            model=model,
            api_key=_env_get(env_map, "LLM_API_KEY"),
        )
    if chosen == "gemini":
        return LLMConfig(
            provider="gemini",
            base_url=_GEMINI_BASE,
            model=model,
            api_key=gemini_key,
        )
    if chosen == "openai":
        return LLMConfig(
            provider="openai",
            base_url=_OPENAI_BASE,
            model=model,
            api_key=openai_key,
        )
    return LLMConfig(
        provider="anthropic",
        base_url=_ANTHROPIC_BASE,
        model=model,
        api_key=anthropic_key,
    )


def _extract_text_content(resp: object) -> str:
    output_text = getattr(resp, "output_text", None)
    if output_text is None and isinstance(resp, dict):
        output_text = resp.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(resp, "output", None)
    if output is None and isinstance(resp, dict):
        output = resp.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if isinstance(item, dict):
                content = item.get("content")
            else:
                content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                else:
                    text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        text = "".join(chunks).strip()
        if text:
            return text

    choices = getattr(resp, "choices", None)
    if choices is None and isinstance(resp, dict):
        choices = resp.get("choices", [])
    if not choices:
        raise RuntimeError("LLM response contained no choices.")

    first = choices[0]
    if isinstance(first, dict):
        message = first.get("message", {})
    else:
        message = getattr(first, "message", {})

    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        text = "".join(chunks).strip()
        if text:
            return text
    raise RuntimeError("LLM response contained no text content.")


class LLMClient:
    """Thin wrapper around LiteLLM responses()."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.provider = config.provider
        self.model = config.model
        self._apply_provider_env()

    def _apply_provider_env(self) -> None:
        env_key = _PROVIDER_API_ENV_KEY.get(self.provider)
        if env_key and self.config.api_key:
            os.environ[env_key] = self.config.api_key

    def _build_response_args(
        self,
        messages: list[dict],
        temperature: float | None,
        max_output_tokens: int,
        thinking_level: str | None,
        response_kwargs: Mapping[str, object] | None,
    ) -> dict:
        args: dict = {
            "model": _provider_model(self.provider, self.model),
            "messages": messages,
            "max_output_tokens": max_output_tokens,
            "timeout": _TIMEOUT,
            "num_retries": _MAX_RETRIES,  # Delegate retry handling to LiteLLM.
        }
        if temperature is not None:
            args["temperature"] = temperature

        if self.provider == "local":
            args["model"] = self.model
            args["api_base"] = self.config.base_url
            if self.config.api_key:
                args["api_key"] = self.config.api_key
        if thinking_level is not None:
            level = _normalize_thinking_level(thinking_level)
            args["reasoning_effort"] = level

        if response_kwargs:
            args.update(response_kwargs)
        return args

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_output_tokens: int = 10000,
        thinking_level: str | None = None,
        response_kwargs: Mapping[str, object] | None = None,
    ) -> str:
        """Send a responses request and return plain text content."""
        # Suppress LiteLLM's verbose multiline info logs (e.g. request traces).
        if hasattr(litellm, 'set_verbose'):
            litellm.set_verbose(False)
        if hasattr(litellm, 'suppress_debug_info'):
            litellm.suppress_debug_info = True

        try:
            response = litellm.responses(
                **self._build_response_args(
                    messages=messages,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    thinking_level=thinking_level,
                    response_kwargs=response_kwargs,
                )
            )
            return _extract_text_content(response)
        except Exception as exc:  # pragma: no cover - provider SDK exception types vary by backend/version.
            raise RuntimeError(f"LLM request failed ({self.provider}/{self.model}): {exc}") from exc

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        """No-op. LiteLLM responses() is stateless per call."""
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
