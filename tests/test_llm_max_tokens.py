import unittest
from unittest.mock import Mock

from applypilot.llm import LLMClient


class LLMMaxTokensTests(unittest.TestCase):
    def test_glm_cloud_on_local_ollama_gets_higher_token_budget(self):
        client = LLMClient("http://localhost:11434/v1", "glm-5.1:cloud", "")
        client._client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        client._client.post.return_value = response

        client._chat_compat(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.1,
            max_tokens=512,
        )

        payload = client._client.post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()
