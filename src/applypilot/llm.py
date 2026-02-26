"""
Unified LLM client for ApplyPilot.

Auto-detects provider from environment:
  GEMINI_API_KEY  -> Google Gemini (default: gemini-2.0-flash)
  OPENAI_API_KEY  -> OpenAI (default: gpt-4o-mini)
  ANTHROPIC_API_KEY / CLAUDE_API_KEY -> Anthropic Claude (default: claude-3-5-haiku-latest)
  LLM_URL         -> Local llama.cpp / Ollama compatible endpoint

LLM_MODEL env var overrides the model name for any provider.
"""

from dataclasses import dataclass
import json
import logging
import os
import time
from collections.abc import Mapping

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

_OPENAI_BASE = "https://api.openai.com/v1"
_ANTHROPIC_BASE = "https://api.anthropic.com/v1"


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


def resolve_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    """Resolve provider configuration from environment with deterministic precedence.

    Reads env at call time (not module import time) so that load_env() called
    in _bootstrap() is always visible here.
    """
    env_map = env if env is not None else os.environ

    model_override = _env_get(env_map, "LLM_MODEL")
    local_url = _env_get(env_map, "LLM_URL")
    gemini_key = _env_get(env_map, "GEMINI_API_KEY")
    openai_key = _env_get(env_map, "OPENAI_API_KEY")
    anthropic_key = _env_get(env_map, "ANTHROPIC_API_KEY") or _env_get(env_map, "CLAUDE_API_KEY")
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
        "claude": "anthropic",
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

    if chosen == "local":
        return LLMConfig(
            provider="local",
            base_url=local_url.rstrip("/"),
            model=model_override or "local-model",
            api_key=_env_get(env_map, "LLM_API_KEY"),
        )
    if chosen == "gemini":
        return LLMConfig(
            provider="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model=model_override or "gemini-2.0-flash",
            api_key=gemini_key,
        )
    if chosen == "openai":
        return LLMConfig(
            provider="openai",
            base_url=_OPENAI_BASE,
            model=model_override or "gpt-4o-mini",
            api_key=openai_key,
        )
    return LLMConfig(
        provider="anthropic",
        base_url=_ANTHROPIC_BASE,
        model=model_override or "claude-3-5-haiku-latest",
        api_key=anthropic_key,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds

# Base wait on first 429/503 (doubles each retry, caps at 60s).
# Gemini free tier is 15 RPM = 4s minimum between requests; 10s gives headroom.
_RATE_LIMIT_BASE_WAIT = 10


_GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"
_GEMINI_THINKING_LEVELS = {"none", "minimal", "low", "medium", "high"}
_GEMINI_COMPAT_REASONING_EFFORT = {
    "none": "none",
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
}
_GEMINI_25_THINKING_BUDGET = {
    "none": 0,
    "minimal": 1024,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}
_GEMINI_NATIVE_THINKING_LEVEL = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
}


class LLMClient:
    """Thin LLM client supporting OpenAI-compatible and native Gemini endpoints.

    For Gemini keys, starts on the OpenAI-compat layer. On a 403 (which
    happens with preview/experimental models not exposed via compat), it
    automatically switches to the native generateContent API and stays there
    for the lifetime of the process.
    """

    def __init__(self, provider: str, base_url: str, model: str, api_key: str) -> None:
        self.provider = provider
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=_TIMEOUT)
        # True once we've confirmed the native Gemini API works for this model
        self._use_native_gemini: bool = False
        self._is_gemini: bool = provider == "gemini"
        self._is_anthropic: bool = provider == "anthropic"

    @staticmethod
    def _normalize_thinking_level(thinking_level: str) -> str:
        level = (thinking_level or "low").strip().lower()
        if level not in _GEMINI_THINKING_LEVELS:
            log.warning("Invalid thinking_level '%s', defaulting to 'low'.", thinking_level)
            return "low"
        return level

    def _gemini_native_thinking_config(self, thinking_level: str) -> dict:
        level = self._normalize_thinking_level(thinking_level)
        if "2.5" in self.model:
            return {"thinkingBudget": _GEMINI_25_THINKING_BUDGET[level]}
        return {"thinkingLevel": _GEMINI_NATIVE_THINKING_LEVEL[level]}

    # -- Native Gemini API --------------------------------------------------

    def _chat_native_gemini(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        thinking_level: str,
    ) -> str:
        """Call the native Gemini generateContent API.

        Used automatically when the OpenAI-compat endpoint returns 403,
        which happens for preview/experimental models not exposed via compat.

        Converts OpenAI-style messages to Gemini's contents/systemInstruction
        format transparently.
        """
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                contents.append({"role": "model", "parts": [{"text": text}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "thinkingConfig": self._gemini_native_thinking_config(thinking_level),
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"{_GEMINI_NATIVE_BASE}/models/{self.model}:generateContent"
        resp = self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # -- OpenAI-compat API --------------------------------------------------

    def _chat_compat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        thinking_level: str,
    ) -> str:
        """Call the OpenAI-compatible endpoint."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._is_gemini:
            level = self._normalize_thinking_level(thinking_level)
            payload["reasoning_effort"] = _GEMINI_COMPAT_REASONING_EFFORT[level]

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

        # 403 on Gemini compat = model not available on compat layer.
        # Raise a specific sentinel so chat() can switch to native API.
        if resp.status_code == 403 and self._is_gemini:
            raise _GeminiCompatForbidden(resp)

        return self._handle_compat_response(resp)

    def _chat_anthropic(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Anthropic Messages API."""
        system_chunks: list[str] = []
        anth_messages: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            if role == "system":
                system_chunks.append(content)
                continue
            if role not in ("user", "assistant"):
                role = "user"
            anth_messages.append({"role": role, "content": content})

        payload: dict = {
            "model": self.model,
            "messages": anth_messages or [{"role": "user", "content": ""}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_chunks:
            payload["system"] = "\n\n".join(system_chunks)

        resp = self._client.post(
            f"{self.base_url}/messages",
            json=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        text_blocks = [
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(part for part in text_blocks if part).strip()
        if text:
            return text

        raise RuntimeError("Anthropic response did not include text content.")

    @staticmethod
    def _handle_compat_response(resp: httpx.Response) -> str:
        resp.raise_for_status()
        data = resp.json()
        if resp.status_code == 200:
            # Intentionally log the full JSON payload for every successful
            # chat/completions response to aid truncation/debug analysis.
            log.info("LLM compat full response JSON:\n%s", json.dumps(data, indent=2, ensure_ascii=False))
        return data["choices"][0]["message"]["content"]

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 10000,
        thinking_level: str = "low",
    ) -> str:
        """Send a chat completion request and return the assistant message text.

        thinking_level applies to Gemini requests and defaults to "low".
        """
        # Qwen3 optimization: prepend /no_think to skip chain-of-thought
        # reasoning, saving tokens on structured extraction tasks.
        if "qwen" in self.model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        for attempt in range(_MAX_RETRIES):
            try:
                if self._is_anthropic:
                    return self._chat_anthropic(messages, temperature, max_tokens)

                # Route to native Gemini if we've already confirmed it's needed
                if self._use_native_gemini:
                    return self._chat_native_gemini(messages, temperature, max_tokens, thinking_level)

                return self._chat_compat(messages, temperature, max_tokens, thinking_level)

            except _GeminiCompatForbidden as exc:
                # Model not available on OpenAI-compat layer — switch to native.
                log.warning(
                    "Gemini compat endpoint returned 403 for model '%s'. "
                    "Switching to native generateContent API. "
                    "(Preview/experimental models are often compat-only on native.)",
                    self.model,
                )
                self._use_native_gemini = True
                # Retry immediately with native — don't count as a rate-limit wait
                try:
                    return self._chat_native_gemini(messages, temperature, max_tokens, thinking_level)
                except httpx.HTTPStatusError as native_exc:
                    raise RuntimeError(
                        f"Both Gemini endpoints failed. Compat: 403 Forbidden. "
                        f"Native: {native_exc.response.status_code} — "
                        f"{native_exc.response.text[:200]}"
                    ) from native_exc

            except httpx.HTTPStatusError as exc:
                resp = exc.response
                if resp.status_code in (429, 503, 529) and attempt < _MAX_RETRIES - 1:
                    # Respect Retry-After header if provided (Gemini sends this).
                    retry_after = (
                        resp.headers.get("Retry-After")
                        or resp.headers.get("X-RateLimit-Reset-Requests")
                    )
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except (ValueError, TypeError):
                            wait = _RATE_LIMIT_BASE_WAIT * (2 ** attempt)
                    else:
                        wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)

                    log.warning(
                        "LLM rate limited (HTTP %s). Waiting %ds before retry %d/%d. "
                        "Tip: Gemini free tier = 15 RPM. Consider a paid account "
                        "or switching to a local model.",
                        resp.status_code, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning(
                        "LLM request timed out, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError("LLM request failed after all retries")

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


class _GeminiCompatForbidden(Exception):
    """Sentinel: Gemini OpenAI-compat returned 403. Switch to native API."""
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Gemini compat 403: {response.text[:200]}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: LLMClient | None = None


def get_client() -> LLMClient:
    """Return (or create) the module-level LLMClient singleton."""
    global _instance
    if _instance is None:
        config = resolve_llm_config()
        log.info("LLM provider: %s  model: %s", config.provider, config.model)
        _instance = LLMClient(
            provider=config.provider,
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
        )
    return _instance
