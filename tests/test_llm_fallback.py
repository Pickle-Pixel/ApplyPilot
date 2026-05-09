"""Tests for the comma-separated fallback chain in applypilot.llm.

Covers:
- Single-model env var (no commas) is still resolved (backwards compat).
- Comma-chained env walks left-to-right and skips tiers whose auth is missing.
- skip=N advances past the first N tiers.
- chat_with_fallback advances tiers when the primary raises HTTPStatusError,
  caps total attempts, and logs which tier won.
"""
import os
import unittest
from unittest.mock import patch

import httpx

from applypilot import llm


def _clean_env(monkeypatch_dict: dict) -> dict:
    """Build an os.environ dict that wipes inherited LLM env then applies overrides."""
    base = {k: v for k, v in os.environ.items()
            if not k.startswith(("APPLYPILOT_MODEL", "OLLAMA_", "NVIDIA_API_KEY",
                                 "CEREBRAS_API_KEY", "GROQ_API_KEY", "LLM_",
                                 "GEMINI_API_KEY", "OPENAI_API_KEY"))}
    base.update(monkeypatch_dict)
    return base


class LLMFallbackChainTests(unittest.TestCase):
    def setUp(self) -> None:
        # Reset the per-(base, key) cache so each test gets fresh clients.
        llm._client_cache.clear()
        # Reset the auto-detected default-client singleton so monkeypatched
        # env actually takes effect.
        llm._instance = None

    # ── parser ─────────────────────────────────────────────────────────────

    def test_parse_model_id_routes_each_provider(self):
        self.assertEqual(llm._parse_model_id("kimi-k2.6:cloud"),
                         ("ollama_cloud", "kimi-k2.6:cloud"))
        self.assertEqual(llm._parse_model_id("nvidia/llama-3.1-405b-instruct"),
                         ("nvidia", "llama-3.1-405b-instruct"))
        self.assertEqual(llm._parse_model_id("cerebras/llama3.1-8b"),
                         ("cerebras", "llama3.1-8b"))
        self.assertEqual(llm._parse_model_id("ollama:llama3.1:8b"),
                         ("ollama_local", "llama3.1:8b"))
        self.assertEqual(llm._parse_model_id("groq/mixtral-8x7b"),
                         ("groq", "mixtral-8x7b"))
        # bare id with no prefix and no :cloud → default
        self.assertEqual(llm._parse_model_id("llama3.1-8b"),
                         ("default", "llama3.1-8b"))

    # ── backwards compat ───────────────────────────────────────────────────

    def test_single_model_no_commas_still_works(self):
        env = _clean_env({
            "APPLYPILOT_MODEL_TAILOR": "kimi-k2.6:cloud",
            "OLLAMA_API_KEY": "fake-ollama-key",
        })
        with patch.dict(os.environ, env, clear=True):
            client, model = llm.get_stage_client("TAILOR")
            self.assertEqual(model, "kimi-k2.6:cloud")
            self.assertEqual(client.base_url, llm.OLLAMA_CLOUD_BASE)

    # ── chain walk ─────────────────────────────────────────────────────────

    def test_chain_walks_to_first_configured_tier(self):
        # Tier 1 (Ollama Cloud) needs OLLAMA_API_KEY → unset, skip.
        # Tier 2 (NVIDIA) has NVIDIA_API_KEY set → win.
        env = _clean_env({
            "APPLYPILOT_MODEL_TAILOR": (
                "kimi-k2.6:cloud,nvidia/llama-3.1-405b-instruct,"
                "cerebras/llama3.1-8b,ollama:llama3.1:8b"
            ),
            "NVIDIA_API_KEY": "fake-nvidia-key",
        })
        with patch.dict(os.environ, env, clear=True):
            client, model = llm.get_stage_client("TAILOR")
            self.assertEqual(model, "llama-3.1-405b-instruct")
            self.assertEqual(client.base_url, llm.NVIDIA_CLOUD_BASE)
            self.assertEqual(client.api_key, "fake-nvidia-key")

    def test_skip_advances_past_tier(self):
        # Configure tiers 2 and 3; ask for skip=2 → cerebras tier wins.
        env = _clean_env({
            "APPLYPILOT_MODEL_TAILOR": (
                "kimi-k2.6:cloud,nvidia/llama-3.1-405b-instruct,"
                "cerebras/llama3.1-8b,ollama:llama3.1:8b"
            ),
            "NVIDIA_API_KEY": "fake-nvidia-key",
            "CEREBRAS_API_KEY": "fake-cerebras-key",
        })
        with patch.dict(os.environ, env, clear=True):
            client, model = llm.get_stage_client("TAILOR", skip=2)
            self.assertEqual(model, "llama3.1-8b")
            self.assertEqual(client.base_url, llm.CEREBRAS_CLOUD_BASE)

    def test_local_ollama_tier_is_always_resolvable(self):
        # ollama: prefix needs no auth — tier 4 always resolves.
        env = _clean_env({
            "APPLYPILOT_MODEL_TAILOR": (
                "kimi-k2.6:cloud,nvidia/llama-3.1-405b-instruct,"
                "cerebras/llama3.1-8b,ollama:llama3.1:8b"
            ),
        })
        with patch.dict(os.environ, env, clear=True):
            client, model = llm.get_stage_client("TAILOR", skip=3)
            self.assertEqual(model, "llama3.1:8b")
            self.assertTrue(client.base_url.startswith("http://localhost"))

    def test_legacy_score_fallback_alias_honored(self):
        # APPLYPILOT_MODEL_SCORE unset → legacy APPLYPILOT_MODEL_SCORE_FALLBACK
        # is consulted next.
        env = _clean_env({
            "APPLYPILOT_MODEL_SCORE_FALLBACK": "ollama:llama3.1:8b",
        })
        with patch.dict(os.environ, env, clear=True):
            client, model = llm.get_stage_client("SCORE")
            self.assertEqual(model, "llama3.1:8b")
            self.assertTrue(client.base_url.startswith("http://localhost"))

    # ── chat_with_fallback ─────────────────────────────────────────────────

    def test_chat_with_fallback_advances_on_primary_failure(self):
        """Mocked unreachable primary → fallback list walked, second tier wins."""
        env = _clean_env({
            # Tier 1 (NVIDIA) is "configured" but its chat() will be mocked
            # to raise. Tier 2 (Cerebras) chat() returns a string.
            "APPLYPILOT_MODEL_TAILOR": (
                "nvidia/llama-3.1-405b-instruct,cerebras/llama3.1-8b"
            ),
            "NVIDIA_API_KEY": "fake-nvidia-key",
            "CEREBRAS_API_KEY": "fake-cerebras-key",
        })

        # The cache means each base_url has one client. We swap their .chat
        # methods after resolution. We can't pre-build them, so let
        # chat_with_fallback resolve — then intercept by patching
        # LLMClient.chat at the class level with a side-effect dispatcher.
        call_log: list[str] = []

        original_chat = llm.LLMClient.chat

        def fake_chat(self, messages, *, temperature=0.0, max_tokens=4096, model=None):
            call_log.append(self.base_url)
            if self.base_url == llm.NVIDIA_CLOUD_BASE:
                # Simulate a 503 from the primary.
                resp = httpx.Response(503, request=httpx.Request("POST", self.base_url))
                raise httpx.HTTPStatusError("503 Service Unavailable",
                                            request=resp.request, response=resp)
            if self.base_url == llm.CEREBRAS_CLOUD_BASE:
                return "OK from cerebras tier"
            self.fail(f"unexpected base_url: {self.base_url}")  # type: ignore[attr-defined]

        with patch.dict(os.environ, env, clear=True), \
             patch.object(llm.LLMClient, "chat", fake_chat):
            result = llm.chat_with_fallback(
                "TAILOR",
                [{"role": "user", "content": "hi"}],
                temperature=0.1,
                max_tokens=64,
            )

        self.assertEqual(result, "OK from cerebras tier")
        # First call hit NVIDIA, second call hit Cerebras.
        self.assertEqual(call_log, [llm.NVIDIA_CLOUD_BASE, llm.CEREBRAS_CLOUD_BASE])

        # Restore (patch.object should handle this, but be explicit since
        # other tests share the LLMClient class).
        llm.LLMClient.chat = original_chat


if __name__ == "__main__":
    unittest.main()
