"""Tests for llm.py against mocked HTTP responses — no live Anthropic API
key needed. Verifies request construction and key resolution; does not
verify Anthropic's actual server responds as documented.
"""

import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import llm  # noqa: E402


def _json_response(payload: dict):
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    return cm


class ApiKeyResolutionTests(unittest.TestCase):
    def setUp(self):
        self._env = patch.dict("os.environ", {}, clear=False)
        self._env.start()
        import os

        os.environ.pop("ANTHROPIC_API_KEY", None)
        self._tmp = tempfile.TemporaryDirectory()
        self.key_path = Path(self._tmp.name) / "anthropic_api_key.txt"

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_prefers_environment_variable(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            self.assertEqual(llm.resolve_api_key(self.key_path), "env-key")

    def test_falls_back_to_secrets_file(self):
        self.key_path.write_text("file-key\n")
        self.assertEqual(llm.resolve_api_key(self.key_path), "file-key")

    def test_environment_variable_wins_over_file(self):
        self.key_path.write_text("file-key")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            self.assertEqual(llm.resolve_api_key(self.key_path), "env-key")

    def test_strips_whitespace_and_takes_first_line(self):
        self.key_path.write_text("  file-key  \nsome other note\n")
        self.assertEqual(llm.resolve_api_key(self.key_path), "file-key")

    def test_raises_a_clear_error_when_nothing_is_configured(self):
        with self.assertRaises(llm.LLMNotConfigured):
            llm.resolve_api_key(self.key_path)

    def test_blank_file_is_treated_as_not_configured(self):
        self.key_path.write_text("   \n")
        with self.assertRaises(llm.LLMNotConfigured):
            llm.resolve_api_key(self.key_path)


class CompleteTests(unittest.TestCase):
    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_sends_expected_request_shape(self, mock_urlopen):
        mock_urlopen.return_value = _json_response(
            {"content": [{"type": "text", "text": "hello"}]}
        )

        result = llm.complete("Say hi", system="Be brief.", api_key="test-key")

        self.assertEqual(result, "hello")
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, llm.ANTHROPIC_ENDPOINT)
        self.assertEqual(request.get_header("X-api-key"), "test-key")
        self.assertEqual(request.get_header("Anthropic-version"), llm.ANTHROPIC_VERSION)
        body = json.loads(request.data)
        self.assertEqual(body["messages"], [{"role": "user", "content": "Say hi"}])
        self.assertEqual(body["system"], "Be brief.")
        self.assertEqual(body["model"], llm.DEFAULT_MODEL)

    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_omits_system_when_not_given(self, mock_urlopen):
        mock_urlopen.return_value = _json_response({"content": [{"type": "text", "text": "x"}]})

        llm.complete("hi", api_key="test-key")

        body = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertNotIn("system", body)

    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_concatenates_multiple_text_blocks(self, mock_urlopen):
        mock_urlopen.return_value = _json_response(
            {"content": [{"type": "text", "text": "one "}, {"type": "text", "text": "two"}]}
        )

        self.assertEqual(llm.complete("hi", api_key="test-key"), "one two")

    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_ignores_non_text_content_blocks(self, mock_urlopen):
        mock_urlopen.return_value = _json_response(
            {"content": [{"type": "tool_use", "id": "x"}, {"type": "text", "text": "actual text"}]}
        )

        self.assertEqual(llm.complete("hi", api_key="test-key"), "actual text")

    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_http_error_is_reported_with_the_response_body(self, mock_urlopen):
        error_body = json.dumps({"error": {"message": "invalid x-api-key"}}).encode()
        http_error = urllib.error.HTTPError(
            url=llm.ANTHROPIC_ENDPOINT, code=401, msg="Unauthorized", hdrs=MagicMock(), fp=None
        )
        http_error.read = lambda: error_body
        mock_urlopen.side_effect = http_error

        with self.assertRaises(RuntimeError) as ctx:
            llm.complete("hi", api_key="test-key")
        self.assertIn("401", str(ctx.exception))
        self.assertIn("invalid x-api-key", str(ctx.exception))

    @patch("photo_grouping.llm.resolve_api_key", return_value="resolved-key")
    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_resolves_key_when_not_passed_explicitly(self, mock_urlopen, mock_resolve):
        mock_urlopen.return_value = _json_response({"content": [{"type": "text", "text": "x"}]})

        llm.complete("hi")

        mock_resolve.assert_called_once()
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.get_header("X-api-key"), "resolved-key")


class OpenAiKeyResolutionTests(unittest.TestCase):
    """Mirrors ApiKeyResolutionTests, for the OpenAI counterpart."""

    def setUp(self):
        self._env = patch.dict("os.environ", {}, clear=False)
        self._env.start()
        import os

        os.environ.pop("OPENAI_API_KEY", None)
        self._tmp = tempfile.TemporaryDirectory()
        self.key_path = Path(self._tmp.name) / "openai_api_key.txt"

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_prefers_environment_variable(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            self.assertEqual(llm.resolve_openai_key(self.key_path), "env-key")

    def test_falls_back_to_secrets_file(self):
        self.key_path.write_text("file-key\n")
        self.assertEqual(llm.resolve_openai_key(self.key_path), "file-key")

    def test_returns_none_not_a_raise_when_unconfigured(self):
        # Unlike resolve_api_key(), this is only ever one candidate among
        # several in resolve_provider_and_key() — "not this one" isn't an
        # error on its own.
        self.assertIsNone(llm.resolve_openai_key(self.key_path))


class ResolveProviderAndKeyTests(unittest.TestCase):
    def setUp(self):
        self._env = patch.dict("os.environ", {}, clear=False)
        self._env.start()
        import os

        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AUTOBIO_LLM_PROVIDER"):
            os.environ.pop(var, None)
        self._tmp = tempfile.TemporaryDirectory()
        self.anthropic_path = Path(self._tmp.name) / "anthropic_api_key.txt"
        self.openai_path = Path(self._tmp.name) / "openai_api_key.txt"

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _resolve(self):
        return llm.resolve_provider_and_key(
            anthropic_key_path=self.anthropic_path, openai_key_path=self.openai_path
        )

    def test_anthropic_wins_when_both_configured(self):
        self.anthropic_path.write_text("ant-key")
        self.openai_path.write_text("oai-key")

        self.assertEqual(self._resolve(), ("anthropic", "ant-key"))

    def test_falls_back_to_openai_when_only_that_is_configured(self):
        self.openai_path.write_text("oai-key")

        self.assertEqual(self._resolve(), ("openai", "oai-key"))

    def test_raises_a_clear_error_naming_both_options_when_neither_is_configured(self):
        with self.assertRaises(llm.LLMNotConfigured) as ctx:
            self._resolve()
        message = str(ctx.exception)
        self.assertIn("ANTHROPIC_API_KEY", message)
        self.assertIn("OPENAI_API_KEY", message)

    def test_forced_provider_env_var_picks_openai_even_if_anthropic_is_also_configured(self):
        self.anthropic_path.write_text("ant-key")
        self.openai_path.write_text("oai-key")

        with patch.dict("os.environ", {"AUTOBIO_LLM_PROVIDER": "openai"}):
            self.assertEqual(self._resolve(), ("openai", "oai-key"))

    def test_forced_provider_env_var_picks_anthropic(self):
        self.anthropic_path.write_text("ant-key")

        with patch.dict("os.environ", {"AUTOBIO_LLM_PROVIDER": "anthropic"}):
            self.assertEqual(self._resolve(), ("anthropic", "ant-key"))

    def test_forced_provider_raises_clearly_if_its_key_is_missing(self):
        with patch.dict("os.environ", {"AUTOBIO_LLM_PROVIDER": "openai"}):
            with self.assertRaises(llm.LLMNotConfigured) as ctx:
                self._resolve()
        self.assertIn("openai", str(ctx.exception).lower())

    def test_unknown_forced_provider_raises_clearly(self):
        with patch.dict("os.environ", {"AUTOBIO_LLM_PROVIDER": "gemini"}):
            with self.assertRaises(llm.LLMNotConfigured) as ctx:
                self._resolve()
        self.assertIn("gemini", str(ctx.exception))


class OpenAiCompleteTests(unittest.TestCase):
    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_sends_expected_request_shape(self, mock_urlopen):
        mock_urlopen.return_value = _json_response(
            {"choices": [{"message": {"content": "hello from gpt"}}]}
        )

        result = llm.complete("Say hi", system="Be brief.", provider="openai", api_key="test-key")

        self.assertEqual(result, "hello from gpt")
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, llm.OPENAI_ENDPOINT)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        body = json.loads(request.data)
        self.assertEqual(
            body["messages"], [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say hi"}]
        )
        self.assertEqual(body["model"], llm.DEFAULT_OPENAI_MODEL)

    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_omits_system_message_when_not_given(self, mock_urlopen):
        mock_urlopen.return_value = _json_response({"choices": [{"message": {"content": "x"}}]})

        llm.complete("hi", provider="openai", api_key="test-key")

        body = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])

    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_empty_choices_returns_empty_string_not_a_crash(self, mock_urlopen):
        mock_urlopen.return_value = _json_response({"choices": []})

        self.assertEqual(llm.complete("hi", provider="openai", api_key="test-key"), "")

    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_http_error_is_reported_with_the_response_body(self, mock_urlopen):
        error_body = json.dumps({"error": {"message": "invalid api key"}}).encode()
        http_error = urllib.error.HTTPError(
            url=llm.OPENAI_ENDPOINT, code=401, msg="Unauthorized", hdrs=MagicMock(), fp=None
        )
        http_error.read = lambda: error_body
        mock_urlopen.side_effect = http_error

        with self.assertRaises(RuntimeError) as ctx:
            llm.complete("hi", provider="openai", api_key="test-key")
        self.assertIn("401", str(ctx.exception))
        self.assertIn("invalid api key", str(ctx.exception))

    @patch("photo_grouping.llm.resolve_openai_key", return_value="resolved-oai-key")
    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_resolves_key_from_provider_when_api_key_not_given(self, mock_urlopen, mock_resolve):
        mock_urlopen.return_value = _json_response({"choices": [{"message": {"content": "x"}}]})

        llm.complete("hi", provider="openai")

        mock_resolve.assert_called_once()
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.get_header("Authorization"), "Bearer resolved-oai-key")


class ProviderDispatchTests(unittest.TestCase):
    """complete()'s own provider-selection logic, independent of either
    provider's request-building details (covered above)."""

    @patch("photo_grouping.llm.resolve_provider_and_key", return_value=("openai", "auto-key"))
    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_auto_detects_provider_when_neither_api_key_nor_provider_given(self, mock_urlopen, mock_resolve):
        mock_urlopen.return_value = _json_response({"choices": [{"message": {"content": "x"}}]})

        llm.complete("hi")

        mock_resolve.assert_called_once()
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, llm.OPENAI_ENDPOINT)
        self.assertEqual(request.get_header("Authorization"), "Bearer auto-key")

    @patch("photo_grouping.llm.urllib.request.urlopen")
    def test_explicit_api_key_without_provider_defaults_to_anthropic(self, mock_urlopen):
        # The original, pre-multi-provider behavior — every existing
        # caller that passes api_key=... without naming a provider must
        # keep hitting Anthropic exactly as before.
        mock_urlopen.return_value = _json_response({"content": [{"type": "text", "text": "x"}]})

        llm.complete("hi", api_key="test-key")

        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, llm.ANTHROPIC_ENDPOINT)

    def test_unknown_explicit_provider_raises(self):
        with self.assertRaises(llm.LLMNotConfigured):
            llm.complete("hi", provider="gemini", api_key="test-key")


if __name__ == "__main__":
    unittest.main()
