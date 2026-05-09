"""
Unified LLM client for ApplyPilot.

Auto-detects provider from environment:
  GEMINI_API_KEY  -> Google Gemini (default: gemini-2.0-flash)
  OPENAI_API_KEY  -> OpenAI (default: gpt-4o-mini)
  OLLAMA_API_KEY  -> Ollama Cloud (OpenAI-compatible, https://ollama.com/v1)
  LLM_URL         -> Cerebras / local llama.cpp / self-hosted Ollama

LLM_MODEL env var overrides the model name for the default provider.

Per-stage model routing with fallback chains (preferred for the pipeline):
  APPLYPILOT_MODEL_TAILOR
  APPLYPILOT_MODEL_COVER_LETTER
  APPLYPILOT_MODEL_ENRICH
  APPLYPILOT_MODEL_SCORE
  APPLYPILOT_MODEL_SCORE_FALLBACK  (legacy alias, still honored)

Each value may be a single model id OR a comma-separated chain of fallback
tiers. Chain entries use a provider prefix to pick the client:
  - bare id or '*:cloud'           -> Ollama Cloud         (OLLAMA_API_KEY)
  - 'nvidia/<model>'               -> NVIDIA cloud         (NVIDIA_API_KEY)
  - 'cerebras/<model>'             -> Cerebras             (CEREBRAS_API_KEY
                                                            or LLM_API_KEY)
  - 'ollama:<model>'               -> local Ollama         (OLLAMA_HOST or
                                                            localhost:11434)
  - 'groq/<model>'                 -> Groq cloud (reserved, GROQ_API_KEY)

Use get_stage_client(stage, skip=N) to walk the chain (skip past first N
tiers), or chat_with_fallback(stage, messages, ...) for stage-aware retry
that auto-advances on 401/429/5xx/timeout. Both are backwards compatible
with single-model env vars (no commas).
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
# 2026-04-28: bumped from 120 -> 300 because Kimi K2.6 with max_tokens=6144
# can take 90-180s on a long cover-letter prompt (reasoning model).
_TIMEOUT = 300
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


# ── Provider abstraction (multi-tier fallback) ────────────────────────────
#
# Each chain entry like 'nvidia/llama-3.1-405b-instruct' or 'ollama:llama3.1:8b'
# is parsed into (provider, model_name). The provider determines which OpenAI-
# compatible base URL and API key env var to use. Auth is checked at resolve
# time so unconfigured tiers are skipped silently — the actual API call only
# happens when a caller invokes .chat() on the returned client.

NVIDIA_CLOUD_BASE = "https://integrate.api.nvidia.com/v1"
CEREBRAS_CLOUD_BASE = "https://api.cerebras.ai/v1"
GROQ_CLOUD_BASE = "https://api.groq.com/openai/v1"
OLLAMA_LOCAL_DEFAULT = "http://localhost:11434/v1"


def _parse_model_id(model_id: str) -> tuple[str, str]:
    """Return (provider, model_name) for a chain entry.

    Provider is one of: 'ollama_cloud', 'nvidia', 'cerebras', 'ollama_local',
    'groq', or 'default' (route through the auto-detected get_client()).
    """
    mid = model_id.strip()
    if "/" in mid:
        prefix, rest = mid.split("/", 1)
        if prefix == "nvidia":
            return ("nvidia", rest)
        if prefix == "cerebras":
            return ("cerebras", rest)
        if prefix == "groq":
            return ("groq", rest)
    if mid.startswith("ollama:"):
        return ("ollama_local", mid[len("ollama:"):])
    # 2026-04-28: explicit prefix for Ollama Cloud catalog slugs that don't
    # carry a ':cloud' suffix (e.g. 'deepseek-v4-pro', 'kimi-k2:1t').
    if mid.startswith("ocloud:"):
        return ("ollama_cloud", mid[len("ocloud:"):])
    if ":cloud" in mid:
        return ("ollama_cloud", mid)
    return ("default", mid)


# Per-(base_url, api_key) client cache — reused across stages so we don't
# rebuild httpx connection pools on every resolve.
_client_cache: "dict[tuple[str, str], LLMClient]" = {}


def _get_or_make_client(base_url: str, api_key: str, placeholder_model: str) -> LLMClient:
    """Fetch (or build) the LLMClient bound to (base_url, api_key)."""
    key = (base_url, api_key)
    client = _client_cache.get(key)
    if client is None:
        log.info("LLM client init: %s (model placeholder: %s)", base_url, placeholder_model)
        client = LLMClient(base_url, placeholder_model, api_key)
        _client_cache[key] = client
    return client


def _resolve_provider_client(provider: str, model: str) -> "LLMClient | None":
    """Build the LLMClient for a tier, or return None if its auth is missing.

    Never makes an outbound API call — this is a config check only. The actual
    API call is the caller's responsibility (which is when failures surface).
    """
    if provider == "ollama_cloud":
        api_key = os.environ.get("OLLAMA_API_KEY", "")
        if not api_key:
            return None
        return _get_or_make_client(OLLAMA_CLOUD_BASE, api_key, model)

    if provider == "nvidia":
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            return None
        return _get_or_make_client(NVIDIA_CLOUD_BASE, api_key, model)

    if provider == "cerebras":
        # CEREBRAS_API_KEY is preferred; fall back to legacy LLM_API_KEY
        # since Darrin's existing .env binds Cerebras under that name.
        api_key = os.environ.get("CEREBRAS_API_KEY") or os.environ.get("LLM_API_KEY", "")
        if not api_key:
            return None
        return _get_or_make_client(CEREBRAS_CLOUD_BASE, api_key, model)

    if provider == "groq":
        # Reserved tier — parser is wired but no live integration yet.
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return None
        return _get_or_make_client(GROQ_CLOUD_BASE, api_key, model)

    if provider == "ollama_local":
        host = os.environ.get("OLLAMA_HOST", OLLAMA_LOCAL_DEFAULT).rstrip("/")
        if not host.endswith("/v1"):
            host = host + "/v1"
        # Local Ollama needs no auth; key in cache is "" so all local-Ollama
        # tiers across stages share the same client.
        return _get_or_make_client(host, "", model)

    if provider == "default":
        # Auto-detected provider (Cerebras / Gemini / OpenAI / Ollama Cloud
        # depending on env). Raises if nothing is configured.
        try:
            return get_client()
        except RuntimeError:
            return None

    return None


def get_stage_client(stage: str, skip: int = 0) -> tuple["LLMClient", str]:
    """Return (client, model_id) for a pipeline stage, walking the chain.

    APPLYPILOT_MODEL_<STAGE> is the primary source. It may be a single
    model id or a comma-separated fallback chain. `skip=N` advances past
    the first N tiers (use after a tier fails at call time).

    Resolution order for the env value:
      1. APPLYPILOT_MODEL_<STAGE>
      2. APPLYPILOT_MODEL_<STAGE>_FALLBACK (legacy alias)
      3. APPLYPILOT_MODEL (global override)
      4. LLM_MODEL (legacy global)
      5. Auto-detected default client + its bound model

    Tiers whose auth is missing are skipped silently (logged at DEBUG).
    Raises RuntimeError if every reachable tier is unconfigured AND no
    default client is available.
    """
    stage_env = f"APPLYPILOT_MODEL_{stage.upper()}"
    legacy_env = f"APPLYPILOT_MODEL_{stage.upper()}_FALLBACK"
    raw = (
        os.environ.get(stage_env)
        or os.environ.get(legacy_env)
        or os.environ.get("APPLYPILOT_MODEL")
        or os.environ.get("LLM_MODEL")
        or ""
    ).strip()

    if not raw:
        # No stage env at all — fall straight back to the auto-detected client.
        if skip > 0:
            raise RuntimeError(
                f"Stage {stage}: no fallback tiers configured (skip={skip})."
            )
        client = get_client()
        return client, client.model

    tiers = [m.strip() for m in raw.split(",") if m.strip()]

    for idx in range(skip, len(tiers)):
        model_id = tiers[idx]
        provider, model_name = _parse_model_id(model_id)
        client = _resolve_provider_client(provider, model_name)
        if client is not None:
            return client, model_name
        log.debug(
            "Stage %s: tier %d (%s) skipped — provider %s unconfigured.",
            stage, idx, model_id, provider,
        )

    # All explicit tiers were unresolvable. For backwards-compat with the
    # prior single-':cloud'-model-without-OLLAMA_API_KEY behavior, fall back
    # to the auto-detected default client on first attempt only. A skip > 0
    # caller is in cross-tier retry mode and must not silently re-use a
    # provider that already failed.
    if skip == 0:
        try:
            client = get_client()
            log.warning(
                "Stage %s: all explicit tiers unconfigured; using default client (%s @ %s).",
                stage, client.model, client.base_url,
            )
            return client, client.model
        except RuntimeError:
            pass

    raise RuntimeError(
        f"Stage {stage}: all fallback tiers exhausted "
        f"(skip={skip}, total={len(tiers)})."
    )


# ── Stage-aware fallback chat helper ──────────────────────────────────────
#
# Wraps client.chat() with cross-tier retry. On any HTTPStatusError or
# TimeoutException, advances `skip` and re-resolves to the next tier.
# Total attempts capped at _FALLBACK_MAX_ATTEMPTS to bound the blast radius
# when every tier is dark.

_FALLBACK_MAX_ATTEMPTS = 3


def chat_with_fallback(
    stage: str,
    messages: list,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_attempts: int = _FALLBACK_MAX_ATTEMPTS,
) -> str:
    """Stage-aware chat with cross-tier fallback on transient/auth failures.

    The underlying LLMClient.chat() already retries 429/503 internally with
    backoff; this wrapper handles the next layer up — when chat() exhausts
    its retries (or hits 401/4xx/5xx that don't retry), we advance to the
    next tier in APPLYPILOT_MODEL_<STAGE> and try again.

    Each attempt is logged on a stage-specific logger
    (`applypilot.llm.stage.<stage>`) with which tier was used and elapsed
    time, so it is easy to tell which provider actually served a given run.

    Args:
        stage:        Stage name (e.g. 'TAILOR', 'COVER_LETTER', 'ENRICH', 'SCORE').
        messages:     Chat-format message list passed straight to .chat().
        temperature:  Sampling temperature (passed through).
        max_tokens:   Token budget (passed through).
        max_attempts: Hard cap on total tries across all tiers (default 3).

    Returns:
        The string response from the first tier that succeeds.

    Raises:
        RuntimeError: If every tier within max_attempts failed.
    """
    stage_log = logging.getLogger(f"applypilot.llm.stage.{stage.lower()}")
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        try:
            client, model = get_stage_client(stage, skip=attempt)
        except RuntimeError as exc:
            last_exc = exc
            stage_log.warning(
                "Stage %s: no more tiers at skip=%d — %s", stage, attempt, exc,
            )
            break

        t0 = time.time()
        try:
            resp = client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
            )
            elapsed = time.time() - t0
            stage_log.info(
                "Stage %s tier=%d OK model=%s base=%s elapsed=%.1fs",
                stage, attempt, model, client.base_url, elapsed,
            )
            return resp
        except httpx.HTTPStatusError as exc:
            elapsed = time.time() - t0
            last_exc = exc
            status = exc.response.status_code
            stage_log.warning(
                "Stage %s tier=%d FAIL model=%s base=%s status=%s elapsed=%.1fs",
                stage, attempt, model, client.base_url, status, elapsed,
            )
            continue
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            elapsed = time.time() - t0
            last_exc = exc
            stage_log.warning(
                "Stage %s tier=%d FAIL model=%s base=%s err=%s elapsed=%.1fs",
                stage, attempt, model, client.base_url,
                type(exc).__name__, elapsed,
            )
            continue
        except RuntimeError as exc:
            # LLMClient.chat raises RuntimeError once internal retries are
            # exhausted; treat that as a tier failure and advance.
            elapsed = time.time() - t0
            last_exc = exc
            stage_log.warning(
                "Stage %s tier=%d FAIL model=%s base=%s err=RuntimeError elapsed=%.1fs",
                stage, attempt, model, client.base_url, elapsed,
            )
            continue

    raise RuntimeError(
        f"Stage {stage}: all fallback tiers failed within "
        f"{max_attempts} attempts."
    ) from last_exc
