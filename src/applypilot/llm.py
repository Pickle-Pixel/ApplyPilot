"""
Unified LLM client for ApplyPilot.

Auto-detects provider from environment:
  GEMINI_API_KEY  -> Google Gemini (default: gemini-2.0-flash)
  OPENAI_API_KEY  -> OpenAI (default: gpt-4o-mini)
  OLLAMA_API_KEY  -> Ollama Cloud (OpenAI-compatible, https://ollama.com/v1)
  LLM_URL         -> Cerebras / local llama.cpp / self-hosted Ollama

LLM_MODEL env var overrides the model name for the default provider.

Per-stage model routing (preferred for the pipeline):
  APPLYPILOT_MODEL_TAILOR
  APPLYPILOT_MODEL_COVER_LETTER
  APPLYPILOT_MODEL_ENRICH
  APPLYPILOT_MODEL_SCORE_FALLBACK
Resolved by get_stage_client(stage). Models containing ":cloud" auto-route
to Ollama Cloud when OLLAMA_API_KEY is set; otherwise the default client
is used. Use client.chat(messages, model=<id>) to override per call.
"""

import logging
import os
import time

import httpx

log = logging.getLogger(__name__)

_GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"
OLLAMA_CLOUD_BASE = "https://ollama.com/v1"


def _detect_provider() -> tuple[str, str, str]:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    ollama_cloud_key = os.environ.get("OLLAMA_API_KEY", "")
    local_url = os.environ.get("LLM_URL", "")
    model_override = os.environ.get("LLM_MODEL", "")

    if gemini_key and not local_url:
        model = model_override or "gemini-2.0-flash"
        if "/" in model:
            model = model.split("/", 1)[1]
        return (_GEMINI_COMPAT_BASE, model, gemini_key)

    if openai_key and not local_url:
        return ("https://api.openai.com/v1", model_override or "gpt-4o-mini", openai_key)

    if local_url:
        return (local_url.rstrip("/"), model_override or "local-model", os.environ.get("LLM_API_KEY", ""))

    if ollama_cloud_key:
        return (OLLAMA_CLOUD_BASE, model_override or "kimi-k2.6:cloud", ollama_cloud_key)

    raise RuntimeError(
        "No LLM provider configured. "
        "Set GEMINI_API_KEY, OPENAI_API_KEY, OLLAMA_API_KEY, or LLM_URL in your environment."
    )


_MAX_RETRIES = 5
_TIMEOUT = 120
_RATE_LIMIT_BASE_WAIT = 10


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=_TIMEOUT)
        self._use_native_gemini: bool = False
        self._is_gemini: bool = base_url.startswith(_GEMINI_COMPAT_BASE)

    def _chat_native_gemini(self, messages, temperature, max_tokens, model=None) -> str:
        contents = []
        system_parts = []
        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        url = f"{_GEMINI_NATIVE_BASE}/models/{model or self.model}:generateContent"
        resp = self._client.post(url, json=payload, headers={"Content-Type": "application/json"}, params={"key": self.api_key})
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _chat_compat(self, messages, temperature, max_tokens, model=None) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        active_model = model or self.model
        payload = {"model": active_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        _is_local = any(h in self.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        if _is_local and any(tag in active_model.lower() for tag in ("qwen", "glm")):
            payload["max_tokens"] = max(payload["max_tokens"], 2048)
        resp = self._client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        if resp.status_code == 403 and self._is_gemini:
            raise _GeminiCompatForbidden(resp)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def chat(self, messages, temperature=0.0, max_tokens=4096, model=None) -> str:
        active_model = model or self.model
        if "qwen" in active_model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        for attempt in range(_MAX_RETRIES):
            try:
                if self._use_native_gemini:
                    return self._chat_native_gemini(messages, temperature, max_tokens, model=active_model)
                return self._chat_compat(messages, temperature, max_tokens, model=active_model)

            except _GeminiCompatForbidden:
                log.warning("Gemini compat 403 for %s -- switching to native.", active_model)
                self._use_native_gemini = True
                try:
                    return self._chat_native_gemini(messages, temperature, max_tokens, model=active_model)
                except httpx.HTTPStatusError as native_exc:
                    raise RuntimeError(
                        f"Both Gemini endpoints failed. Native: {native_exc.response.status_code} "
                        f"{native_exc.response.text[:200]}"
                    ) from native_exc

            except httpx.HTTPStatusError as exc:
                resp = exc.response
                if resp.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                    retry_after = resp.headers.get("Retry-After") or resp.headers.get("X-RateLimit-Reset-Requests")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except (ValueError, TypeError):
                            wait = _RATE_LIMIT_BASE_WAIT * (2 ** attempt)
                    else:
                        wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning("LLM rate limited (HTTP %s). Wait %ds (retry %d/%d).", resp.status_code, wait, attempt + 1, _MAX_RETRIES)
                    time.sleep(wait)
                    continue
                raise

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning("LLM timeout, retry in %ds (attempt %d/%d)", wait, attempt + 1, _MAX_RETRIES)
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError("LLM request failed after all retries")

    def ask(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


class _GeminiCompatForbidden(Exception):
    def __init__(self, response):
        self.response = response
        super().__init__(f"Gemini compat 403: {response.text[:200]}")


_instance: "LLMClient | None" = None


def get_client() -> LLMClient:
    global _instance
    if _instance is None:
        base_url, model, api_key = _detect_provider()
        log.info("LLM provider: %s  model: %s", base_url, model)
        _instance = LLMClient(base_url, model, api_key)
    return _instance


_ollama_cloud_client: "LLMClient | None" = None


def _get_ollama_cloud_client(model: str) -> LLMClient:
    global _ollama_cloud_client
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. It is persisted as a Windows user-scope "
            "env var on this machine; new shells inherit it. If running from an "
            "IDE, restart the integrated terminal."
        )
    if _ollama_cloud_client is None:
        log.info("LLM provider: %s (Ollama Cloud) initial model: %s", OLLAMA_CLOUD_BASE, model)
        _ollama_cloud_client = LLMClient(OLLAMA_CLOUD_BASE, model, api_key)
    return _ollama_cloud_client


def get_stage_client(stage: str):
    """Return (client, model) for a pipeline stage.

    Resolution order for the model id:
      1. APPLYPILOT_MODEL_<STAGE> (e.g. APPLYPILOT_MODEL_TAILOR)
      2. APPLYPILOT_MODEL (global override)
      3. LLM_MODEL (legacy global)
      4. The default client's bound model

    If the resolved model contains ':cloud' and OLLAMA_API_KEY is set,
    routes to Ollama Cloud. Otherwise routes to the default client.
    """
    stage_env = f"APPLYPILOT_MODEL_{stage.upper()}"
    model = (
        os.environ.get(stage_env)
        or os.environ.get("APPLYPILOT_MODEL")
        or os.environ.get("LLM_MODEL")
        or ""
    )
    client = get_client()
    if model and ":cloud" in model:
        if os.environ.get("OLLAMA_API_KEY"):
            return _get_ollama_cloud_client(model), model
        return client, client.model
    return client, (model or client.model)
