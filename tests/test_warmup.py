"""Tests for warm_up() across every provider.

Offline tests intercept the transport (``urlopen``, ``_post``, ``_get_client``)
so no network access is needed. Live tests call ``self.skipTest()`` when the
provider they need is not available.

Usage:
    python -m unittest discover -s tests
    python -m unittest tests.test_warmup.TestWarmUpOpenAiCompat
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _path in (str(_PROJECT_ROOT), str(_TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Imported as a top-level module: unittest discovery puts tests/ on sys.path,
# and so does running this file directly.
from test_providers import (  # noqa: E402
    ProviderRegistryIsolation,
    _make_script,
    _make_text_file,
    _first_ollama_model,
    _ollama_available,
)

from unified_ai_client.models import ProviderConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_http_response(payload: dict) -> MagicMock:
    """Build a urlopen() return value usable as a context manager."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


_ALL_PROVIDER_CLASSES = (
    ("ollama", "unified_ai_client.providers.ollama", "OllamaProvider"),
    ("google", "unified_ai_client.providers.google", "GoogleProvider"),
    ("anthropic", "unified_ai_client.providers.anthropic", "AnthropicProvider"),
    ("openai", "unified_ai_client.providers.openai", "OpenAiProvider"),
    ("mistral", "unified_ai_client.providers.mistral", "MistralProvider"),
    ("cohere", "unified_ai_client.providers.cohere", "CohereProvider"),
    ("meta", "unified_ai_client.providers.meta", "MetaProvider"),
    ("groq", "unified_ai_client.providers.groq", "GroqProvider"),
    ("xai", "unified_ai_client.providers.xai", "XAiProvider"),
    ("lmstudio", "unified_ai_client.providers.lmstudio", "LmStudioProvider"),
    ("llamacpp", "unified_ai_client.providers.llamacpp", "LlamaCppProvider"),
    ("script", "unified_ai_client.providers.script", "ScriptProvider"),
)


# ---------------------------------------------------------------------------
# 1. The contract itself
# ---------------------------------------------------------------------------

class TestWarmUpContract(unittest.TestCase):
    """warm_up() must exist everywhere and default to a harmless no-op."""

    def test_warm_up_present_on_every_provider(self) -> None:
        import importlib
        for name, module_path, class_name in _ALL_PROVIDER_CLASSES:
            with self.subTest(provider=name):
                cls = getattr(importlib.import_module(module_path), class_name)
                self.assertTrue(
                    callable(getattr(cls, "warm_up", None)),
                    f"{class_name} must expose a callable warm_up()",
                )

    def test_base_provider_default_is_a_false_no_op(self) -> None:
        """A provider that does not override warm_up must return False quietly.

        "Nothing to warm up" is a legitimate answer, so the default must not be
        abstract and must not raise: a third-party subclass written before
        warm_up existed keeps working untouched.
        """
        from unified_ai_client.providers.base import BaseProvider

        class MinimalProvider(BaseProvider):
            def call(self, request):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def preload_model(self, model, keep_alive="15m", context_size=None,
                              extra_options=None):  # type: ignore[no-untyped-def]
                pass

            def get_embedding(self, model, text):  # type: ignore[no-untyped-def]
                raise NotImplementedError

        provider = MinimalProvider()
        self.assertIs(provider.warm_up("any-model"), False)
        self.assertIs(provider.warm_up("any-model", ["/tmp/a.pdf"]), False)


# ---------------------------------------------------------------------------
# 2. Google — the only provider that does substantial work
# ---------------------------------------------------------------------------

class TestWarmUpGoogle(unittest.TestCase):
    """The client is built, the model is validated, and files are pre-uploaded."""

    def _provider(self, **config_kwargs):
        from unified_ai_client.providers.google import GoogleProvider
        return GoogleProvider(
            config=ProviderConfig(**config_kwargs), api_key="dummy-key"
        )

    def test_warm_up_validates_model_and_uploads_files(self) -> None:
        provider = self._provider()
        fake_client = MagicMock()

        with patch.object(provider, "_get_client", return_value=fake_client), \
             patch.object(provider, "_upload_file") as upload:
            result = provider.warm_up(
                "gemini-2.5-flash", ["/tmp/a.pdf", "/tmp/b.png"]
            )

        self.assertIs(result, True)
        fake_client.models.get.assert_called_once_with(model="gemini-2.5-flash")
        self.assertEqual(upload.call_count, 2)
        self.assertEqual(
            [c.args[0] for c in upload.call_args_list],
            ["/tmp/a.pdf", "/tmp/b.png"],
        )

    def test_warm_up_accepts_a_single_path_string(self) -> None:
        provider = self._provider()
        with patch.object(provider, "_get_client", return_value=MagicMock()), \
             patch.object(provider, "_upload_file") as upload:
            provider.warm_up("gemini-2.5-flash", "/tmp/only.pdf")
        upload.assert_called_once()
        self.assertEqual(upload.call_args.args[0], "/tmp/only.pdf")

    def test_warm_up_without_files_uploads_nothing(self) -> None:
        provider = self._provider()
        with patch.object(provider, "_get_client", return_value=MagicMock()), \
             patch.object(provider, "_upload_file") as upload:
            result = provider.warm_up("gemini-2.5-flash")
        self.assertIs(result, True)
        upload.assert_not_called()

    def test_warm_up_honours_upload_poll_timeout_from_config(self) -> None:
        """The polling budget must come from config, as it does in call()."""
        provider = self._provider(extra_options={"upload_poll_timeout": 42})
        with patch.object(provider, "_get_client", return_value=MagicMock()), \
             patch.object(provider, "_upload_file") as upload:
            provider.warm_up("gemini-2.5-flash", "/tmp/a.pdf")
        self.assertEqual(upload.call_args.args[1], 42)

    def test_preload_model_no_longer_raises(self) -> None:
        """Regression: Google used to raise NotImplementedError here.

        The README documents preloading as a no-op on providers that do not
        support it, which is what Anthropic and the OpenAI-compatible providers
        already did. Consumers should not need a provider check around it.
        """
        provider = self._provider()
        self.assertIsNone(provider.preload_model("gemini-2.5-flash"))


# ---------------------------------------------------------------------------
# 3. Ollama
# ---------------------------------------------------------------------------

class TestWarmUpOllama(unittest.TestCase):
    """Ollama delegates to its own official warm-up path."""

    def test_warm_up_delegates_to_preload_model(self) -> None:
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(ProviderConfig(url="http://localhost:11434"))
        with patch.object(provider, "preload_model") as preload:
            result = provider.warm_up("gemma4:12b")

        self.assertIs(result, True)
        preload.assert_called_once_with("gemma4:12b", "15m")

    def test_warm_up_uses_keep_alive_from_config(self) -> None:
        """keep_alive must come from config, not from the signature default.

        call() already reads it from extra_options; a warm-up that ignored it
        would load the model with a different residency than every subsequent
        call, defeating the point.
        """
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            ProviderConfig(
                url="http://localhost:11434",
                extra_options={"keep_alive": "45m"},
            )
        )
        with patch.object(provider, "preload_model") as preload:
            provider.warm_up("gemma4:12b")

        preload.assert_called_once_with("gemma4:12b", "45m")


# ---------------------------------------------------------------------------
# 4. OpenAI-compatible cloud providers and Anthropic
# ---------------------------------------------------------------------------

class TestWarmUpOpenAiCompat(unittest.TestCase):
    """Cloud providers must warm up with a free metadata GET."""

    def _warm_up_and_capture_request(self, provider, model: str = "some-model"):
        """Run warm_up with urlopen intercepted, returning the Request sent."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["request"] = req
            return _fake_http_response({"data": []})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = provider.warm_up(model)
        return result, captured["request"]

    def test_openai_warm_up_hits_models_endpoint(self) -> None:
        from unified_ai_client.providers.openai import OpenAiProvider

        provider = OpenAiProvider(ProviderConfig(), api_key="sk-fake")
        result, req = self._warm_up_and_capture_request(provider)

        self.assertIs(result, True)
        self.assertEqual(req.get_method(), "GET")
        self.assertEqual(req.full_url, "https://api.openai.com/v1/models")
        self.assertEqual(req.get_header("Authorization"), "Bearer sk-fake")
        self.assertIsNone(req.data, "A metadata GET must not carry a body")

    def test_every_cloud_compat_provider_composes_a_clean_models_url(self) -> None:
        """No provider may emit a doubled or missing /v1 segment.

        Cohere is the reason this test exists: its DEFAULT_URL used to end in
        /v1 while the endpoint paths add their own, so every request went to
        /compatibility/v1/v1/... The base URL must stop before the version.
        """
        import importlib

        expected = {
            "openai": "https://api.openai.com/v1/models",
            "mistral": "https://api.mistral.ai/v1/models",
            "cohere": "https://api.cohere.ai/compatibility/v1/models",
            "meta": "https://api.llama-api.com/v1/models",
            "groq": "https://api.groq.com/openai/v1/models",
            "xai": "https://api.x.ai/v1/models",
        }

        for name, module_path, class_name in _ALL_PROVIDER_CLASSES:
            if name not in expected:
                continue
            with self.subTest(provider=name):
                cls = getattr(importlib.import_module(module_path), class_name)
                provider = cls(ProviderConfig(), api_key="fake")
                _, req = self._warm_up_and_capture_request(provider)
                self.assertEqual(req.full_url, expected[name])
                self.assertNotIn("/v1/v1/", req.full_url)

    def test_anthropic_warm_up_hits_models_endpoint(self) -> None:
        from unified_ai_client.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(ProviderConfig(), api_key="sk-ant-fake")
        result, req = self._warm_up_and_capture_request(provider)

        self.assertIs(result, True)
        self.assertEqual(req.get_method(), "GET")
        self.assertEqual(req.full_url, "https://api.anthropic.com/v1/models")
        # Anthropic authenticates with its own headers, not a Bearer token.
        self.assertEqual(req.get_header("X-api-key"), "sk-ant-fake")
        self.assertIsNotNone(req.get_header("Anthropic-version"))


# ---------------------------------------------------------------------------
# 5. Local OpenAI-compatible servers
# ---------------------------------------------------------------------------

class TestWarmUpLocalServers(unittest.TestCase):
    """LM Studio and llama.cpp need a real completion to force the model load."""

    def _capture_post(self, provider, model: str = "local-model"):
        captured = {}

        def fake_post(endpoint: str, payload: dict, timeout: int) -> dict:
            captured["endpoint"] = endpoint
            captured["payload"] = payload
            return {"choices": [{"message": {"content": ""}}], "usage": {}}

        with patch.object(provider, "_post", side_effect=fake_post):
            result = provider.warm_up(model)
        return result, captured

    def test_lmstudio_warm_up_sends_minimal_completion(self) -> None:
        from unified_ai_client.providers.lmstudio import LmStudioProvider

        provider = LmStudioProvider(ProviderConfig())
        result, captured = self._capture_post(provider)

        self.assertIs(result, True)
        self.assertEqual(captured["endpoint"], "/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "local-model")
        self.assertEqual(captured["payload"]["max_tokens"], 1)
        self.assertEqual(captured["payload"]["temperature"], 0.0)
        self.assertIs(captured["payload"]["stream"], False)

    def test_llamacpp_warm_up_sends_minimal_completion(self) -> None:
        from unified_ai_client.providers.llamacpp import LlamaCppProvider

        provider = LlamaCppProvider(ProviderConfig())
        result, captured = self._capture_post(provider)

        self.assertIs(result, True)
        self.assertEqual(captured["endpoint"], "/v1/chat/completions")
        self.assertEqual(captured["payload"]["max_tokens"], 1)

    def test_local_override_wins_over_inherited_metadata_get(self) -> None:
        """The local servers must not fall back to GET /v1/models.

        A metadata GET returns instantly without loading anything, which would
        leave the load cost exactly where warm_up is supposed to remove it.
        """
        from unified_ai_client.providers.lmstudio import LmStudioProvider
        from unified_ai_client.providers.llamacpp import LlamaCppProvider

        for cls in (LmStudioProvider, LlamaCppProvider):
            with self.subTest(provider=cls.__name__):
                provider = cls(ProviderConfig())
                with patch.object(provider, "_get") as get, \
                     patch.object(provider, "_post") as post:
                    provider.warm_up("local-model")
                get.assert_not_called()
                post.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Script provider — the extended stdin/stdout protocol
# ---------------------------------------------------------------------------

_WARMING_SCRIPT = '''\
from __future__ import annotations
import json, sys

def main() -> None:
    req = json.loads(sys.stdin.read())
    mode = req.get("mode")
    if mode == "warm_up":
        # Echo back what we received so the test can inspect the payload.
        print(json.dumps({
            "warmed_up": True,
            "seen_files": req.get("file_path"),
        }))
    elif mode == "preload":
        with open(SIDECAR, "w", encoding="utf-8") as fh:
            json.dump(req, fh)
        print(json.dumps({"preloaded": True}))
    else:
        print(f"Unsupported mode: {mode}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

_LEGACY_SCRIPT = '''\
from __future__ import annotations
import json, sys

def main() -> None:
    req = json.loads(sys.stdin.read())
    if req.get("mode") != "generate":
        print("This script only implements generate", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({"text": "ok", "input_tokens": 1, "output_tokens": 1}))

if __name__ == "__main__":
    main()
'''

_DECLINING_SCRIPT = '''\
from __future__ import annotations
import json, sys

def main() -> None:
    json.loads(sys.stdin.read())
    print(json.dumps({"warmed_up": False}))

if __name__ == "__main__":
    main()
'''


class TestWarmUpScript(unittest.TestCase):
    """Scripts can now react to warm-up and preload instead of being assumed idle."""

    def setUp(self) -> None:
        super().setUp()
        from unified_ai_client.providers.script import ScriptProvider
        self.provider = ScriptProvider(ProviderConfig(timeout=30))
        self._temp_paths: list[str] = []

    def tearDown(self) -> None:
        for path in self._temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        super().tearDown()

    def _script(self, source: str, sidecar: str = "") -> str:
        path = _make_script(source.replace("SIDECAR", repr(sidecar)))
        self._temp_paths.append(path)
        return path

    def test_script_implementing_warm_up_reports_true(self) -> None:
        script = self._script(_WARMING_SCRIPT)
        self.assertIs(self.provider.warm_up(script), True)

    def test_script_receives_the_file_paths(self) -> None:
        """file_paths must reach the script, normalised to a list."""
        script = self._script(_WARMING_SCRIPT)
        attachment = _make_text_file("content")
        self._temp_paths.append(attachment)

        captured = {}

        from unified_ai_client.providers import script as script_module
        real_run = script_module._run_script

        def spy(cmd, payload, timeout):
            captured["payload"] = payload
            return real_run(cmd, payload, timeout)

        with patch.object(script_module, "_run_script", side_effect=spy):
            self.provider.warm_up(script, attachment)

        self.assertEqual(captured["payload"]["mode"], "warm_up")
        self.assertEqual(captured["payload"]["file_path"], [attachment])

    def test_script_may_decline_by_reporting_false(self) -> None:
        """Implementing the mode is not the same as having work to do."""
        script = self._script(_DECLINING_SCRIPT)
        self.assertIs(self.provider.warm_up(script), False)

    def test_legacy_script_warm_up_returns_false_without_raising(self) -> None:
        """A script that only implements 'generate' must not break warm_up.

        It exits non-zero on the unknown mode, which the provider reads as
        "nothing to warm up" rather than as an error.
        """
        script = self._script(_LEGACY_SCRIPT)
        self.assertIs(self.provider.warm_up(script), False)

    def test_script_implementing_preload_receives_the_settings(self) -> None:
        sidecar = os.path.join(tempfile.mkdtemp(), "preload.json")
        self._temp_paths.append(sidecar)
        script = self._script(_WARMING_SCRIPT, sidecar=sidecar)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.provider.preload_model(script, keep_alive="30m", context_size=8000)

        self.assertEqual(
            [w for w in caught if issubclass(w.category, UserWarning)],
            [],
            "A script that implements 'preload' must not trigger the warning",
        )
        with open(sidecar, encoding="utf-8") as fh:
            received = json.load(fh)
        self.assertEqual(received["mode"], "preload")
        self.assertEqual(received["keep_alive"], "30m")
        self.assertEqual(received["context_size"], 8000)

    def test_legacy_script_preload_still_warns(self) -> None:
        """The signal that a preload call did nothing must survive."""
        script = self._script(_LEGACY_SCRIPT)
        with self.assertWarns(UserWarning):
            self.provider.preload_model(script)


# ---------------------------------------------------------------------------
# 7. client.warm_up() — the public entry point
# ---------------------------------------------------------------------------

class TestClientWarmUp(ProviderRegistryIsolation):
    """The router must never let a warm-up failure reach the caller."""

    def test_warm_up_delegates_to_the_provider(self) -> None:
        from unified_ai_client import warm_up
        from unified_ai_client.client import get_provider

        provider = get_provider("google")
        with patch.object(provider, "warm_up", return_value=True) as pw:
            result = warm_up("google", "gemini-2.5-flash", "/tmp/a.pdf")

        self.assertIs(result, True)
        pw.assert_called_once_with("gemini-2.5-flash", "/tmp/a.pdf")

    def test_warm_up_swallows_provider_errors(self) -> None:
        """A failed warm-up is a missed optimisation, not an error.

        The call_ai() that follows has its own retries and will surface the
        real failure with a useful message.
        """
        from unified_ai_client import warm_up
        from unified_ai_client.client import get_provider

        provider = get_provider("google")
        with patch.object(
            provider, "warm_up", side_effect=RuntimeError("no network")
        ):
            with self.assertLogs("unified_ai_client.client", level="WARNING"):
                result = warm_up("google", "gemini-2.5-flash")

        self.assertIs(result, False)

    def test_warm_up_rejects_an_unknown_provider_quietly(self) -> None:
        from unified_ai_client import warm_up
        with self.assertLogs("unified_ai_client.client", level="WARNING"):
            self.assertIs(warm_up("nonexistent_provider_xyz", "m"), False)

    def test_warm_up_registers_the_cleanup_handler(self) -> None:
        """A process that only warms up must still clean up its uploads.

        Google's warm-up can upload files; without the atexit registration they
        would be left behind on Google's servers.
        """
        from unified_ai_client import client as client_module

        with patch.object(client_module, "_register_cleanup") as register:
            with self.assertLogs("unified_ai_client.client", level="WARNING"):
                client_module.warm_up("nonexistent_provider_xyz", "m")
        register.assert_called_once()


# ---------------------------------------------------------------------------
# 8. Live warm-up tests
# ---------------------------------------------------------------------------

class TestWarmUpLive(unittest.TestCase):
    """Real warm-up calls, skipped when the provider is not available."""

    def test_ollama_live_warm_up(self) -> None:
        if not _ollama_available():
            self.skipTest("Ollama not reachable at localhost:11434")
        model = _first_ollama_model()
        if not model:
            self.skipTest("No chat-capable Ollama models installed")
        from unified_ai_client import warm_up
        self.assertIs(warm_up("ollama", model), True)

    def test_google_live_warm_up_populates_the_upload_cache(self) -> None:
        from unified_ai_client.config import load_secrets
        if not load_secrets(os.getcwd()).get("google_api_key"):
            self.skipTest(
                "google_api_key not found in secrets.json or environment variables"
            )
        from unified_ai_client import warm_up
        from unified_ai_client.client import get_provider

        tmp = _make_text_file("Warm-up upload test.")
        try:
            result = warm_up("google", "gemini-2.5-flash", tmp)
            self.assertIs(result, True)
            provider = get_provider("google")
            self.assertIn(
                os.path.abspath(tmp),
                provider._uploaded_files,
                "warm_up must leave the file in the same cache call_ai() reads",
            )
        finally:
            os.unlink(tmp)

    def test_lmstudio_live_warm_up(self) -> None:
        import urllib.request
        try:
            with urllib.request.urlopen(
                "http://localhost:1234/v1/models", timeout=2
            ) as resp:
                models = json.loads(resp.read()).get("data", [])
        except Exception:
            self.skipTest("LM Studio not reachable at localhost:1234")
        if not models:
            self.skipTest("No models available in LM Studio")
        from unified_ai_client import warm_up
        self.assertIs(warm_up("lmstudio", models[0]["id"]), True)


if __name__ == "__main__":
    unittest.main()
