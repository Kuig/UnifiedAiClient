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
    RequiresCredential,
    _make_script,
    _make_text_file,
    _first_ollama_model,
    _ollama_available,
    _PROVIDER_CLASSES,
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


# Same 12 providers as _PROVIDER_CLASSES, reshaped to the (name, dotted
# module path, class name) triple this file's own tests expect. Derived
# rather than hand-typed a second time: until 0.5.8 this was an independent
# copy, unsynchronized with test_provider_contracts.py's table.
_ALL_PROVIDER_CLASSES = tuple(
    (name, f"unified_ai_client.providers.{module}", class_name)
    for name, (module, class_name) in _PROVIDER_CLASSES.items()
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

    def _attachment(self, suffix: str) -> str:
        """A real file on disk, since warm_up() validates before uploading.

        The upload itself is patched out, but the path has to exist: an
        attachment warm_up() cannot use must not reach Google's quota.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(b"x")
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_warm_up_validates_model_and_uploads_files(self) -> None:
        provider = self._provider()
        fake_client = MagicMock()
        paths = [self._attachment(".pdf"), self._attachment(".png")]

        with patch.object(provider, "_get_client", return_value=fake_client), \
             patch.object(provider, "_upload_file") as upload:
            result = provider.warm_up("gemini-2.5-flash", paths)

        self.assertIs(result, True)
        fake_client.models.get.assert_called_once_with(model="gemini-2.5-flash")
        self.assertEqual(upload.call_count, 2)
        self.assertEqual([c.args[0] for c in upload.call_args_list], paths)

    def test_warm_up_accepts_a_single_path_string(self) -> None:
        provider = self._provider()
        path = self._attachment(".pdf")
        with patch.object(provider, "_get_client", return_value=MagicMock()), \
             patch.object(provider, "_upload_file") as upload:
            provider.warm_up("gemini-2.5-flash", path)
        upload.assert_called_once()
        self.assertEqual(upload.call_args.args[0], path)

    def test_warm_up_refuses_a_file_it_could_not_use(self) -> None:
        """A bad attachment must not reach the upload, and so not the quota.

        warm_up() uploads into the same cache call() reads, so a file call()
        would refuse would otherwise sit on Google's quota until cleanup(),
        with client.warm_up() reporting only False.
        """
        provider = self._provider()
        with patch.object(provider, "_get_client", return_value=MagicMock()), \
             patch.object(provider, "_upload_file") as upload:
            with self.assertRaises(FileNotFoundError):
                provider.warm_up("gemini-2.5-flash", "no-such-file.pdf")
        upload.assert_not_called()

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
            provider.warm_up("gemini-2.5-flash", self._attachment(".pdf"))
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
        preload.assert_called_once_with("gemma4:12b", "15m", timeout=None)

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

        preload.assert_called_once_with("gemma4:12b", "45m", timeout=None)

    def test_explicit_keep_alive_beats_config(self) -> None:
        """The argument is the top of the precedence chain.

        Config still wins over the default, but an explicit value has to win
        over config, or a caller cannot pin one model without reconfiguring the
        provider for every other call in the process.
        """
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            ProviderConfig(
                url="http://localhost:11434",
                extra_options={"keep_alive": "45m"},
            )
        )
        with patch.object(provider, "preload_model") as preload:
            provider.warm_up("gemma4:12b", keep_alive=-1)

        preload.assert_called_once_with("gemma4:12b", -1, timeout=None)

    def test_keep_alive_and_timeout_reach_the_wire(self) -> None:
        """The whole point of A2: no configure_provider() in sight."""
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(ProviderConfig(url="http://localhost:11434"))
        captured: dict = {}

        def fake_post(endpoint: str, payload: dict, timeout: int) -> dict:
            captured["endpoint"] = endpoint
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {}

        with patch.object(provider, "_post", side_effect=fake_post):
            provider.warm_up("gemma4:12b", keep_alive=-1, timeout=600)

        self.assertEqual(captured["endpoint"], "/api/chat")
        self.assertEqual(captured["payload"]["keep_alive"], -1)
        self.assertEqual(captured["payload"]["messages"], [])
        self.assertEqual(captured["timeout"], 600)

    def test_warm_up_falls_back_to_configured_timeout(self) -> None:
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            ProviderConfig(url="http://localhost:11434", timeout=123)
        )
        with patch.object(provider, "_post", return_value={}) as post:
            provider.warm_up("gemma4:12b")

        self.assertEqual(post.call_args.args[2], 123)

    def test_warm_up_logs_the_effective_values(self) -> None:
        """A1: the numbers that went on the wire must be readable from a log."""
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(ProviderConfig(url="http://localhost:11434"))
        with patch.object(provider, "_post", return_value={}):
            with self.assertLogs(
                "unified_ai_client.providers.ollama", level="DEBUG"
            ) as cm:
                provider.warm_up("gemma4:12b", keep_alive=-1, timeout=600)

        self.assertTrue(
            any("keep_alive=-1" in line and "600" in line for line in cm.output),
            f"no log line named both the keep_alive and the timeout: {cm.output}",
        )


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
    elif mode == "unload":
        with open(SIDECAR, "w", encoding="utf-8") as fh:
            json.dump(req, fh)
        print(json.dumps({}))
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

        with self.assertNoLogs(
            "unified_ai_client.providers.script", level="WARNING"
        ):
            self.provider.preload_model(script, keep_alive="30m", context_size=8000)

        with open(sidecar, encoding="utf-8") as fh:
            received = json.load(fh)
        self.assertEqual(received["mode"], "preload")
        self.assertEqual(received["keep_alive"], "30m")
        self.assertEqual(received["context_size"], 8000)

    def test_legacy_script_preload_still_warns(self) -> None:
        """The signal that a preload call did nothing must survive.

        Moved off the stdlib `warnings` module onto `logging`, per
        CLAUDE.md's "logs through stdlib logging only" rule: `warnings.warn`
        was the one place in the library that did not, and it bypassed
        set_verbosity("silent") entirely since it never touched the
        unified_ai_client logger tree.
        """
        script = self._script(_LEGACY_SCRIPT)
        with self.assertLogs(
            "unified_ai_client.providers.script", level="WARNING"
        ) as cm:
            self.provider.preload_model(script)
        self.assertTrue(any("preload" in line for line in cm.output))


# ---------------------------------------------------------------------------
# 7. client.warm_up() — the public entry point
# ---------------------------------------------------------------------------

class TestClientWarmUp(ProviderRegistryIsolation):
    """The router must never let a warm-up failure reach the caller."""

    def test_warm_up_delegates_to_the_provider(self) -> None:
        from unified_ai_client import warm_up
        from unified_ai_client.registry import get_provider

        provider = get_provider("google")
        with patch.object(provider, "warm_up", return_value=True) as pw:
            result = warm_up("google", "gemini-2.5-flash", "/tmp/a.pdf")

        self.assertIs(result, True)
        pw.assert_called_once_with(
            "gemini-2.5-flash", "/tmp/a.pdf", keep_alive=None, timeout=None
        )

    def test_warm_up_swallows_provider_errors(self) -> None:
        """A failed warm-up is a missed optimisation, not an error.

        The call_ai() that follows has its own retries and will surface the
        real failure with a useful message.
        """
        from unified_ai_client import warm_up
        from unified_ai_client.registry import get_provider

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

        # warm_up() imports _register_cleanup by name from registry.py, so the
        # binding client.py actually calls lives on the client module, not on
        # registry: patching registry._register_cleanup here would leave
        # client.py's already-bound reference untouched.
        with patch.object(client_module, "_register_cleanup") as register:
            with self.assertLogs("unified_ai_client.client", level="WARNING"):
                client_module.warm_up("nonexistent_provider_xyz", "m")
        register.assert_called_once()


# ---------------------------------------------------------------------------
# 8. Live warm-up tests
# ---------------------------------------------------------------------------

class TestWarmUpLive(RequiresCredential, unittest.TestCase):
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
        self._require_credential("google_api_key")
        from unified_ai_client import warm_up

        tmp = _make_text_file("Warm-up upload test.")
        try:
            result = warm_up("google", "gemini-2.5-flash", tmp)
            self.assertIs(result, True)
            from unified_ai_client.providers import google as google_mod
            self.assertIn(
                os.path.abspath(tmp),
                google_mod._UPLOADED_FILES,
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


# ---------------------------------------------------------------------------
# 9. Unloading
# ---------------------------------------------------------------------------


class TestUnloadModel(ProviderRegistryIsolation):
    """unload_model() is the named counterpart to preload_model()."""

    def test_base_provider_unload_is_a_concrete_no_op(self) -> None:
        """Like cleanup(), and unlike preload_model(), it must not be abstract.

        A third-party provider written before this hook existed has to keep
        instantiating and answering "nothing to release".
        """
        from unified_ai_client.providers.base import BaseProvider

        class MinimalProvider(BaseProvider):
            def call(self, request):  # noqa: ANN001, ANN201
                raise NotImplementedError

            def preload_model(self, model, keep_alive="15m", context_size=None,
                              extra_options=None):  # noqa: ANN001, ANN201
                raise NotImplementedError

            def get_embedding(self, model, text):  # noqa: ANN001, ANN201
                raise NotImplementedError

        provider = MinimalProvider()
        self.assertIs(provider.SUPPORTS_UNLOAD, False)
        self.assertIsNone(provider.unload_model("anything"))

    def test_ollama_sends_the_documented_unload_request(self) -> None:
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(ProviderConfig(url="http://localhost:11434"))
        captured: dict = {}

        def fake_post(endpoint: str, payload: dict, timeout: int) -> dict:
            captured["endpoint"] = endpoint
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {}

        with patch.object(provider, "_post", side_effect=fake_post):
            provider.unload_model("gemma4:12b")

        self.assertEqual(captured["endpoint"], "/api/chat")
        self.assertEqual(
            captured["payload"],
            {"model": "gemma4:12b", "messages": [], "keep_alive": 0},
        )

    def test_ollama_unload_does_not_wait_the_full_provider_timeout(self) -> None:
        """An unload queues behind a running generation; it must not stall exit."""
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            ProviderConfig(url="http://localhost:11434", timeout=600)
        )
        with patch.object(provider, "_post", return_value={}) as post:
            provider.unload_model("gemma4:12b")

        self.assertLess(post.call_args.args[2], 600)

        with patch.object(provider, "_post", return_value={}) as post:
            provider.unload_model("gemma4:12b", timeout=5)
        self.assertEqual(post.call_args.args[2], 5)

    def test_script_forwards_the_unload_mode(self) -> None:
        from unified_ai_client.providers.script import ScriptProvider

        sidecar = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        sidecar.close()
        provider = ScriptProvider(ProviderConfig(timeout=30))
        script = _make_script(
            _WARMING_SCRIPT.replace("SIDECAR", repr(sidecar.name))
        )
        try:
            provider.unload_model(script)
            with open(sidecar.name, encoding="utf-8") as fh:
                payload = json.load(fh)
        finally:
            os.unlink(sidecar.name)

        self.assertEqual(payload["mode"], "unload")
        self.assertEqual(payload["timeout"], 30)

    def test_script_declining_the_mode_is_not_an_error(self) -> None:
        """Most scripts hold nothing between calls; that is the normal case."""
        from unified_ai_client.providers.script import ScriptProvider

        provider = ScriptProvider(ProviderConfig(timeout=30))
        script = _make_script(_LEGACY_SCRIPT.replace("SIDECAR", repr("")))

        self.assertIsNone(provider.unload_model(script))

    def test_client_unload_swallows_provider_errors(self) -> None:
        """Failing to free VRAM must not bring down the caller."""
        from unified_ai_client import unload_model
        from unified_ai_client.registry import get_provider

        provider = get_provider("ollama")
        with patch.object(
            provider, "unload_model", side_effect=RuntimeError("server gone")
        ):
            with self.assertLogs("unified_ai_client.registry", level="WARNING") as cm:
                self.assertIsNone(unload_model("ollama", "gemma4:12b"))

        self.assertTrue(any("server gone" in line for line in cm.output))

    def test_client_unload_on_a_provider_without_residency(self) -> None:
        """Every cloud provider inherits the no-op and must not raise."""
        from unified_ai_client import unload_model

        self.assertIsNone(unload_model("google", "gemini-2.5-flash"))


# ---------------------------------------------------------------------------
# Cleanup arming
# ---------------------------------------------------------------------------

class TestCleanupIsArmedWhenAModelIsTracked(ProviderRegistryIsolation):
    """Tracking a resident model and arming its release are the same event.

    Regression: preload_model() recorded the model in _LOADED_MODELS but never
    called _register_cleanup(), so a process that only preloaded — the very
    thing keep_alive=-1 is for — exited with the model still in VRAM and no
    atexit hook to free it.
    """

    def setUp(self) -> None:
        super().setUp()
        from unified_ai_client import registry as client_module

        self._client = client_module
        self._was_registered = client_module._CLEANUP_REGISTERED
        client_module._CLEANUP_REGISTERED = False

    def tearDown(self) -> None:
        self._client._CLEANUP_REGISTERED = self._was_registered
        super().tearDown()

    def test_preload_model_arms_atexit(self) -> None:
        from unified_ai_client import preload_model
        from unified_ai_client.providers.ollama import OllamaProvider

        with patch.object(OllamaProvider, "preload_model", return_value=None):
            with patch.object(self._client, "atexit") as fake_atexit:
                preload_model("ollama", "gemma4:12b", keep_alive=-1)

        self.assertTrue(self._client._CLEANUP_REGISTERED)
        fake_atexit.register.assert_called_once_with(self._client.cleanup)

    def test_the_model_is_tracked_for_release(self) -> None:
        """Arming is only useful if cleanup() has something to drain."""
        from unified_ai_client import preload_model
        from unified_ai_client.providers.ollama import OllamaProvider

        with patch.object(OllamaProvider, "preload_model", return_value=None):
            preload_model("ollama", "gemma4:12b")

        self.assertIn(("ollama", "gemma4:12b"), self._client._LOADED_MODELS)

    def test_a_provider_without_residency_arms_nothing_here(self) -> None:
        """Cloud providers hold no model, so this path must stay a no-op.

        They arm cleanup through call_ai/warm_up/get_embedding instead, for the
        remote files they do hold.
        """
        from unified_ai_client.registry import _record_loaded, get_provider

        provider = get_provider("google")
        _record_loaded("google", provider, "gemini-2.5-flash")

        self.assertFalse(self._client._CLEANUP_REGISTERED)


class TestCleanupKeepsTrackingAFailedUnload(ProviderRegistryIsolation):
    """A model whose unload fails must stay tracked, not vanish with the rest.

    Regression: _LOADED_MODELS.clear() used to run before the unload attempts,
    so a single provider raising lost the tracking for every model in the
    same cleanup() call, itself included. The next cleanup() then had nothing
    left to retry, and the model stayed resident for good.
    """

    def test_the_failed_model_survives_the_successful_ones(self) -> None:
        from unified_ai_client import registry as client_module
        from unified_ai_client.providers.ollama import OllamaProvider

        client_module._LOADED_MODELS.clear()
        client_module._LOADED_MODELS.update(
            {("ollama", "stuck-model"), ("ollama", "released-model")}
        )

        def fake_unload(self, model, *, timeout=None):
            if model == "stuck-model":
                raise RuntimeError("server unreachable")

        with patch.object(OllamaProvider, "unload_model", fake_unload):
            client_module.cleanup()

        self.assertIn(("ollama", "stuck-model"), client_module._LOADED_MODELS)
        self.assertNotIn(("ollama", "released-model"), client_module._LOADED_MODELS)


if __name__ == "__main__":
    unittest.main()
