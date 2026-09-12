"""Smoke tests for all UnifiedAiClient providers.

Tests that need a provider which is not available (missing API key, offline
server, model that does not support the feature under test) call
``self.skipTest()`` so the suite reports SKIP instead of failing.

Usage:
    python -m unittest discover -s tests     # from project root
    python tests/test_providers.py           # direct execution
    python -m unittest tests.test_providers.TestDispatch.test_dispatch
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure the project root is in sys.path so unified_ai_client is importable
# both when running this file directly from tests/ and from the project root.
# This is needed when the package is not pip-installed in the active Python.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers — disposable temp files and Ollama availability probes
# ---------------------------------------------------------------------------

def _make_text_file(content: str = "Hello from UnifiedAiClient test.") -> str:
    """Write content to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def _make_script(code: str) -> str:
    """Write a Python script to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    tmp.write(code)
    tmp.close()
    return tmp.name


def _ollama_available() -> bool:
    """Quick check whether Ollama is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _first_ollama_model() -> str | None:
    """Return the smallest Ollama chat model available.

    Embedding models (e.g. nomic-embed-text, mxbai-embed-*) return HTTP 400 on
    /api/chat, so they are excluded by name pattern.

    The smallest is chosen rather than the first because every live test in this
    suite loads it: a smaller model means a shorter load, less VRAM pressure and
    less chance of the server evicting something else to make room. All of these
    tests assert on the shape of the response, never on its content, so the
    choice of model does not affect what they verify.
    """
    import urllib.request, json as _json
    _EMBED_PATTERNS = ("embed", "embedding", "bge-", "e5-")
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            data = _json.loads(r.read())
        chat = [
            m for m in data.get("models", [])
            if not any(p in m["name"].lower() for p in _EMBED_PATTERNS)
        ]
        if not chat:
            return None
        return min(chat, key=lambda m: m.get("size", 0))["name"]
    except Exception:
        return None


def _first_ollama_embed_model() -> str | None:
    """Return the first Ollama embedding-capable model available.

    Looks for models whose name matches known embedding model patterns
    (bge-, e5-, embed, embedding, nomic-embed, mxbai-embed, etc.).
    Returns None if no embedding model is found.
    """
    import urllib.request, json as _json
    _EMBED_PATTERNS = ("embed", "embedding", "bge-", "e5-", "nomic-", "mxbai-")
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            data = _json.loads(r.read())
        for m in data.get("models", []):
            name: str = m["name"]
            if any(p in name.lower() for p in _EMBED_PATTERNS):
                return name
        return None
    except Exception:
        return None


class ProviderRegistryIsolation(unittest.TestCase):
    """Base class that restores the global provider registries after each test.

    ``configure_provider()`` writes into ``registry._PROVIDER_CONFIGS`` and
    invalidates ``registry._PROVIDERS``. Tests run in an arbitrary order, so any
    test that touches those registries must leave them exactly as it found
    them or it silently changes the outcome of the next one.

    Two more registries have the same problem and are restored here as well:
    ``registry._LOADED_MODELS``, which any ``call_ai()`` against a provider with
    a residency concept writes to, and ``google._UPLOADED_FILES``. A leaked
    entry in the first makes the atexit ``cleanup()`` chase a model that was
    never really loaded; one in the second makes a later test count an upload
    it did not make. ``registry._BUILT_PROVIDERS`` is restored for the same
    reason: ``cleanup()`` walks it by name and would otherwise reach providers
    a later test never asked for.
    """

    def setUp(self) -> None:
        super().setUp()
        from unified_ai_client import registry as _client
        from unified_ai_client.providers import google as _google
        self._saved_configs = dict(_client._PROVIDER_CONFIGS)
        self._saved_providers = dict(_client._PROVIDERS)
        self._saved_loaded = set(_client._LOADED_MODELS)
        self._saved_built = set(_client._BUILT_PROVIDERS)
        self._saved_uploads = dict(_google._UPLOADED_FILES)

    def tearDown(self) -> None:
        from unified_ai_client import registry as _client
        from unified_ai_client.providers import google as _google
        _client._PROVIDER_CONFIGS.clear()
        _client._PROVIDER_CONFIGS.update(self._saved_configs)
        _client._PROVIDERS.clear()
        _client._PROVIDERS.update(self._saved_providers)
        _client._LOADED_MODELS.clear()
        _client._LOADED_MODELS.update(self._saved_loaded)
        _client._BUILT_PROVIDERS.clear()
        _client._BUILT_PROVIDERS.update(self._saved_built)
        _google._UPLOADED_FILES.clear()
        _google._UPLOADED_FILES.update(self._saved_uploads)
        super().tearDown()


# Every provider, keyed by the name call_ai() accepts, derived from the
# library's own registry rather than hand-typed a second time. Until 0.5.8
# this was an independent copy in test_provider_contracts.py, and test_warmup.py
# kept a third, unsynchronized one of its own: a 13th provider added only to
# unified_ai_client.registry would not have failed either. Both now derive
# from this one.
from unified_ai_client.registry import _PROVIDER_SPECS  # noqa: E402

_PROVIDER_CLASSES: dict[str, tuple[str, str]] = {
    name: (spec.module.rsplit(".", 1)[-1], spec.class_name)
    for name, spec in _PROVIDER_SPECS.items()
}


class RequiresCredential:
    """Skip the test when the named secret is not configured.

    Replaces the ``load_secrets(...).get(key) or self.skipTest(...)`` guard
    that used to be hand-copied verbatim in every live test needing one
    specific provider's key.
    """

    def _require_credential(self, secrets_key: str) -> None:
        from unified_ai_client.config import load_secrets
        if not load_secrets(os.getcwd()).get(secrets_key):
            self.skipTest(
                f"{secrets_key} not found in secrets.json or environment variables"
            )


class TestRequiresCredential(RequiresCredential, unittest.TestCase):
    """The mixin itself must actually skip, not silently pass through."""

    def test_a_missing_key_raises_skip_test(self) -> None:
        with patch("unified_ai_client.config.load_secrets", return_value={}):
            with self.assertRaises(unittest.SkipTest):
                self._require_credential("does_not_exist_api_key")


_ECHO_SCRIPT = '''\
from __future__ import annotations
import json, sys

def main() -> None:
    req = json.loads(sys.stdin.read())
    mode = req.get("mode", "generate")
    if mode == "generate":
        files = req.get("file_path") or []
        thinking = req.get("thinking", False)
        result = {
            "text": f"Echo: {req['prompt']} (files={len(files)})",
            "input_tokens": 1,
            "output_tokens": 2,
            "reasoning_tokens": 10 if thinking else 0,
            "reasoning_text": "I thought about it." if thinking else "",
        }
        print(json.dumps(result))
    elif mode == "embed":
        print(json.dumps({"embedding": [0.1, 0.2, 0.3]}))
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------------------

class TestImports(unittest.TestCase):
    """The public modules and every provider class must be importable."""

    def test_import_models(self) -> None:
        from unified_ai_client.models import AiRequest, AiResponse, ProviderConfig
        self.assertTrue(AiRequest and AiResponse and ProviderConfig)

    def test_import_file_utils(self) -> None:
        from unified_ai_client.file_utils import (
            classify_file,
            normalize_file_paths,
            encode_file_base64,
            get_mime_type,
            audio_format_name,
            format_text_attachment,
            inline_text_attachments,
        )
        self.assertTrue(classify_file)

    def test_import_client(self) -> None:
        from unified_ai_client import (
            call_ai, cleanup, preload_model, unload_model, get_embedding,
        )
        self.assertTrue(call_ai)

    def test_import_providers(self) -> None:
        """Every provider named in the registry must actually import.

        Looped over unified_ai_client.registry._PROVIDER_SPECS instead of
        12 hand-typed import lines, so a provider added there is covered
        here for free.
        """
        from unified_ai_client.registry import _PROVIDER_SPECS
        for name, spec in _PROVIDER_SPECS.items():
            with self.subTest(provider=name):
                module = importlib.import_module(spec.module)
                self.assertTrue(getattr(module, spec.class_name))


# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

class TestModels(unittest.TestCase):
    """Dataclass construction and default values."""

    def test_airequest_construction(self) -> None:
        from unified_ai_client.models import AiRequest
        r = AiRequest(
            provider="ollama",
            model="llava",
            prompt="test",
            file_path=["a.jpg", "b.txt"],
        )
        self.assertEqual(r.file_path, ["a.jpg", "b.txt"])
        self.assertFalse(
            hasattr(r, "image_path"),
            "Old 'image_path' field must not exist on AiRequest",
        )

    def test_airesponse_defaults(self) -> None:
        from unified_ai_client.models import AiResponse
        r = AiResponse(text="hello")
        self.assertEqual(r.reasoning_text, "")
        self.assertEqual(r.reasoning_tokens, 0)


# ---------------------------------------------------------------------------
# 3. File utils unit tests
# ---------------------------------------------------------------------------

class TestFileUtils(unittest.TestCase):
    """Classification, path normalisation and text inlining."""

    def test_classify_file(self) -> None:
        from unified_ai_client.file_utils import classify_file
        self.assertEqual(classify_file("photo.jpg"), "image")
        self.assertEqual(classify_file("PHOTO.PNG"), "image")
        self.assertEqual(classify_file("audio.mp3"), "audio")
        self.assertEqual(classify_file("audio.wav"), "audio")
        self.assertEqual(classify_file("doc.pdf"), "document")
        self.assertEqual(classify_file("notes.md"), "text")
        self.assertEqual(classify_file("data.csv"), "text")
        self.assertEqual(classify_file("binary.bin"), "unknown")

    def test_classify_file_covers_common_source_and_config_types(self) -> None:
        """Anything landing in 'unknown' is now refused, so coverage matters."""
        from unified_ai_client.file_utils import classify_file
        for name in (
            "Service.cs", "widget.tsx", "app.jsx", "theme.scss", "Card.vue",
            "records.jsonl", "diagram.svg", "fix.patch", "api.proto",
            "schema.graphql", "main.tf", "build.gradle", "util.hpp",
            "worker.dart", "analysis.jl", "settings.conf", "local.env",
        ):
            with self.subTest(name=name):
                self.assertEqual(classify_file(name), "text")

    def test_classify_file_recognises_extensionless_names(self) -> None:
        """A Dockerfile has no extension but is plainly text."""
        from unified_ai_client.file_utils import classify_file
        for name in (
            "Dockerfile", "Makefile", "LICENSE", "README", "CHANGELOG",
            ".gitignore", ".dockerignore", ".editorconfig", ".env",
        ):
            with self.subTest(name=name):
                self.assertEqual(classify_file(name), "text")

    def test_extensionless_names_match_case_insensitively(self) -> None:
        from unified_ai_client.file_utils import classify_file
        for name in ("dockerfile", "DOCKERFILE", "makefile", "license"):
            with self.subTest(name=name):
                self.assertEqual(classify_file(name), "text")

    def test_classify_file_still_refuses_genuine_unknowns(self) -> None:
        """The wider list must not quietly turn into 'everything is text'."""
        from unified_ai_client.file_utils import classify_file
        for name in (
            "archive.zip", "sheet.xlsx", "report.docx", "lib.so", "app.exe",
            "Dockerfile.dev", "notes",
        ):
            with self.subTest(name=name):
                self.assertEqual(classify_file(name), "unknown")

    def test_full_paths_classify_by_basename(self) -> None:
        """The name lookup must not be confused by parent directories."""
        from unified_ai_client.file_utils import classify_file
        self.assertEqual(classify_file("/srv/app/Dockerfile"), "text")
        self.assertEqual(classify_file(os.path.join("repo", ".gitignore")), "text")
        self.assertEqual(classify_file(os.path.join("src", "Service.cs")), "text")

    def test_normalize_file_paths(self) -> None:
        from unified_ai_client.file_utils import normalize_file_paths
        self.assertEqual(normalize_file_paths(None), [])
        self.assertEqual(normalize_file_paths("a.jpg"), ["a.jpg"])
        self.assertEqual(
            normalize_file_paths(["a.jpg", "b.pdf"]), ["a.jpg", "b.pdf"]
        )

    def test_inline_text_attachments(self) -> None:
        from unified_ai_client.file_utils import inline_text_attachments
        tmp = _make_text_file("line one\nline two")
        try:
            result = inline_text_attachments("My prompt", [tmp])
            self.assertIn("My prompt", result)
            self.assertIn(os.path.basename(tmp), result)
            self.assertIn("line one", result)
            # Prompt must appear only ONCE
            self.assertEqual(result.count("My prompt"), 1)
        finally:
            os.unlink(tmp)

    def test_inline_text_attachments_multiple(self) -> None:
        from unified_ai_client.file_utils import inline_text_attachments
        t1 = _make_text_file("file one content")
        t2 = _make_text_file("file two content")
        try:
            result = inline_text_attachments("My prompt", [t1, t2])
            self.assertIn("file one content", result)
            self.assertIn("file two content", result)
            # Prompt must appear exactly once regardless of how many files
            self.assertEqual(result.count("My prompt"), 1)
        finally:
            os.unlink(t1)
            os.unlink(t2)

    def test_audio_format_name(self) -> None:
        from unified_ai_client.file_utils import audio_format_name
        self.assertEqual(audio_format_name("track.mp3"), "mp3")
        self.assertEqual(audio_format_name("clip.wav"), "wav")
        self.assertEqual(audio_format_name("sound.flac"), "flac")
        self.assertEqual(audio_format_name("file.m4a"), "mp4")


# ---------------------------------------------------------------------------
# 4. Config loading — load_secrets (os.environ + secrets.json) and load_config
# ---------------------------------------------------------------------------

class TestSecrets(unittest.TestCase):
    """load_secrets() merges os.environ over secrets.json."""

    def test_load_secrets_from_env_var(self) -> None:
        """Environment variables are read and returned as snake_case keys."""
        from unified_ai_client.config import load_secrets
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-test-key-xyz"}):
            result = load_secrets("/nonexistent/path/no_secrets_here")
        self.assertEqual(result.get("google_api_key"), "env-test-key-xyz")

    def test_load_secrets_env_wins_over_json(self) -> None:
        """os.environ takes priority over secrets.json when both define a key."""
        import shutil
        from unified_ai_client.config import load_secrets
        tmp_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmp_dir, "secrets.json"), "w", encoding="utf-8") as f:
                json.dump({"google_api_key": "from-secrets-json"}, f)
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "from-env-var"}):
                result = load_secrets(tmp_dir)
            self.assertEqual(
                result["google_api_key"],
                "from-env-var",
                "os.environ must win over secrets.json",
            )
        finally:
            shutil.rmtree(tmp_dir)

    def test_load_secrets_no_sources(self) -> None:
        """load_secrets returns an empty dict with no file and no env vars."""
        from unified_ai_client.config import load_secrets
        with patch.dict(os.environ, {}, clear=True):
            result = load_secrets("/nonexistent/path/xyz")
        self.assertEqual(result, {})


class TestConfigLoading(unittest.TestCase):
    """load_config() maps known fields and sweeps the rest into extra_options."""

    def test_config_dynamic_extra_options(self) -> None:
        """Unrecognized configuration keys must land in extra_options."""
        from unified_ai_client.config import load_config
        from unified_ai_client.models import ProviderConfig

        tmp_config_fd, tmp_config_path = tempfile.mkstemp(suffix=".json")
        try:
            config_data = {
                "google": {
                    "url": "http://google-api-mock",
                    "timeout": 45,
                    "sleep_time": 5,
                    "disable_safety": True,
                    "upload_poll_timeout": 20,
                    "custom_app_setting": "hello_world"
                }
            }
            with os.fdopen(tmp_config_fd, "w", encoding="utf-8") as f:
                json.dump(config_data, f)

            cfg = load_config(tmp_config_path, ProviderConfig, section="google")
            self.assertEqual(cfg.url, "http://google-api-mock")
            self.assertEqual(cfg.timeout, 45)
            self.assertEqual(cfg.sleep_time, 5)
            self.assertIsInstance(cfg.extra_options, dict)
            self.assertIs(cfg.extra_options.get("disable_safety"), True)
            self.assertEqual(cfg.extra_options.get("upload_poll_timeout"), 20)
            self.assertEqual(cfg.extra_options.get("custom_app_setting"), "hello_world")
        finally:
            os.unlink(tmp_config_path)


# ---------------------------------------------------------------------------
# 5. Dispatch — get_provider() resolves the correct class
# ---------------------------------------------------------------------------

class TestDispatch(ProviderRegistryIsolation):
    """get_provider() must return the right adapter for every provider name.

    One subTest per name in _PROVIDER_CLASSES instead of a hand-typed
    test_dispatch_<name> per provider: adding a provider to the registry
    covers it here automatically, with nothing to remember to add by hand.
    """

    def test_dispatch(self) -> None:
        from unified_ai_client.registry import get_provider
        for name, (module, class_name) in _PROVIDER_CLASSES.items():
            with self.subTest(provider=name):
                mod = importlib.import_module(f"unified_ai_client.providers.{module}")
                self.assertIsInstance(get_provider(name), getattr(mod, class_name))

    def test_dispatch_invalid(self) -> None:
        from unified_ai_client.registry import get_provider
        with self.assertRaises(ValueError):
            get_provider("nonexistent_provider_xyz")


# ---------------------------------------------------------------------------
# 6. Live tests — Ollama (most likely available locally)
# ---------------------------------------------------------------------------

class TestOllamaLive(ProviderRegistryIsolation):
    """End-to-end calls against a local Ollama server.

    These are deliberately gentle. Ollama serves one model at a time on the
    developer machine that also runs the suite, so a test class that generates
    freely competes with whatever else is loaded. Three rules keep it cheap:

    - every generation is capped with ``max_tokens``, since all of these tests
      assert on the *shape* of the response, never on its content;
    - ``max_retries=0``, so a failing call fails once instead of being sent four
      times over 35 seconds of backoff onto a server that is already struggling;
    - the first transport failure fails its own test and makes the rest of the
      class skip, so one sick server produces one error rather than a cascade.
    """

    # Enough for "PONG" or a one-word answer, which is all these tests read.
    _MAX_TOKENS = 32
    # Thinking needs room for a trace, or reasoning_tokens is always 0.
    _MAX_TOKENS_THINKING = 160

    # Set by _live_call on the first transport failure. Class-level on purpose:
    # it has to outlive the test that discovered the server was unhealthy.
    _server_unhealthy: str | None = None

    # Model actually exercised, so tearDownClass can hand back its VRAM.
    _model_used: str | None = None

    @classmethod
    def tearDownClass(cls) -> None:
        """Release the model these tests loaded.

        ProviderRegistryIsolation restores client._LOADED_MODELS after every
        test, which is right for isolation but means the atexit cleanup no
        longer knows about the model this class loaded. Without this it would
        sit in VRAM until Ollama's own idle timer expires, long after the suite
        has finished.
        """
        if cls._model_used:
            from unified_ai_client import unload_model
            unload_model("ollama", cls._model_used)   # never raises
            cls._model_used = None
        super().tearDownClass()

    def _require_model(self) -> str:
        """Skip unless Ollama is reachable and has a chat-capable model."""
        if not _ollama_available():
            self.skipTest("Ollama not reachable at localhost:11434")
        if TestOllamaLive._server_unhealthy:
            self.skipTest(
                f"an earlier live call failed ({TestOllamaLive._server_unhealthy}); "
                f"not sending more work to this server"
            )
        model = _first_ollama_model()
        if not model:
            self.skipTest("No chat-capable Ollama models installed")
        TestOllamaLive._model_used = model
        return model

    def _live_call(self, model: str, **kwargs):
        """Run one capped, un-retried live generation.

        Args:
            model: Model identifier to call.
            **kwargs: Passed to ``call_ai``. ``temperature``, ``max_tokens``,
                ``timeout`` and ``max_retries`` get gentle defaults.

        Returns:
            The ``AiResponse``.
        """
        from unified_ai_client import call_ai

        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("max_tokens", self._MAX_TOKENS)
        kwargs.setdefault("timeout", 60)
        try:
            return call_ai(provider="ollama", model=model, max_retries=0, **kwargs)
        except Exception as exc:
            if "400" in str(exc):
                self.skipTest(
                    f"Model '{model}' rejected the request (400), likely embed-only"
                )
            # Report this one, and spare the server the remaining tests.
            TestOllamaLive._server_unhealthy = f"{type(exc).__name__}: {exc}"
            raise

    def test_ollama_live_generate(self) -> None:
        model = self._require_model()
        response = self._live_call(
            model, prompt="Reply with exactly the word PONG and nothing else."
        )
        self.assertIsInstance(response.text, str)
        self.assertGreater(len(response.text), 0)

    def test_ollama_live_with_text_file(self) -> None:
        model = self._require_model()
        tmp = _make_text_file("The sky is blue.")
        try:
            response = self._live_call(
                model,
                prompt="What colour is mentioned in the attached file? Reply in one word.",
                file_path=tmp,
            )
            self.assertIsInstance(response.text, str)
            self.assertGreater(len(response.text), 0)
        finally:
            os.unlink(tmp)

    def test_ollama_live_thinking(self) -> None:
        model = self._require_model()
        # thinking=True may silently fall back on models that don't support it
        response = self._live_call(
            model,
            prompt="What is 2+2?",
            thinking=True,
            max_tokens=self._MAX_TOKENS_THINKING,
        )
        self.assertIsInstance(response.text, str)
        # reasoning_text may be empty if the model doesn't support thinking
        self.assertIsInstance(response.reasoning_text, str)

    def test_ollama_live_embedding(self) -> None:
        if not _ollama_available():
            self.skipTest("Ollama not reachable at localhost:11434")
        model = _first_ollama_embed_model()
        if not model:
            self.skipTest(
                "No embedding-capable Ollama models installed "
                "(need bge-m3, embeddinggemma, nomic-embed-text, or similar)"
            )
        import urllib.error
        from unified_ai_client import get_embedding
        try:
            vec = get_embedding(provider="ollama", model=model, text="hello world")
        except (RuntimeError, urllib.error.HTTPError) as exc:
            self.skipTest(f"Model '{model}' does not support embeddings ({exc})")
        self.assertIsInstance(vec, list)
        self.assertGreater(len(vec), 0)
        self.assertTrue(all(isinstance(x, float) for x in vec))

    def test_ollama_live_reasoning_tokens_thinking_true(self) -> None:
        """With thinking=True, reasoning_tokens must be > 0 when a trace exists.

        If the model responds but produces no thinking output (empty
        reasoning_text and zero reasoning_tokens) the test is skipped: the model
        does not support the 'think' parameter.
        """
        model = self._require_model()
        response = self._live_call(
            model,
            prompt="What is 2+2? Think step by step.",
            thinking=True,
            max_tokens=self._MAX_TOKENS_THINKING,
        )
        if not response.reasoning_text and response.reasoning_tokens == 0:
            self.skipTest(
                f"Model '{model}' produced no thinking output, 'think' not supported"
            )
        self.assertIsInstance(response.text, str)
        self.assertGreater(len(response.text), 0)
        self.assertIs(
            response.reasoning_is_summary,
            False,
            "Ollama returns the raw thinking transcript, not a provider-written summary",
        )
        self.assertGreater(
            response.reasoning_tokens,
            0,
            f"reasoning_tokens must be > 0 when thinking text is present, "
            f"got {response.reasoning_tokens}",
        )

    def test_ollama_live_reasoning_tokens_thinking_false(self) -> None:
        """With thinking=False, reasoning_tokens must be a non-negative integer.

        Some models produce a small thinking trace even when not asked to. The
        result must never be negative and must never raise.
        """
        model = self._require_model()
        response = self._live_call(model, prompt="What is 2+2?", thinking=False)
        self.assertIsInstance(response.text, str)
        self.assertGreaterEqual(
            response.reasoning_tokens,
            0,
            f"reasoning_tokens must be >= 0, got {response.reasoning_tokens}",
        )

    def test_ollama_live_tool_calling(self) -> None:
        """Live Ollama tool calling with get_weather, over two turns.

        The second call must pass the full conversation history including the
        assistant's intermediate tool_calls turn so the model can link the tool
        result back to its own request.

        Uses whichever model ``_require_model`` picks rather than naming one:
        pinning a second, larger model here would load it alongside the one the
        rest of the class already has resident, for no extra coverage. A model
        that does not do tool calling simply skips.
        """
        from unified_ai_client import ToolDefinition, ToolResult

        model = self._require_model()

        tools = [
            ToolDefinition(
                name="get_weather",
                description="Returns the current weather for a given city.",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. Rome",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ]

        prompt = "What is the weather in Rome right now? Use the get_weather tool."

        response = self._live_call(model, prompt=prompt, tools=tools)

        if not response.tool_calls:
            self.skipTest(
                f"'{model}' did not produce a tool call, model may not support it"
            )

        tc = response.tool_calls[0]
        self.assertEqual(tc.name, "get_weather")
        self.assertIn("location", tc.arguments)

        weather_result = (
            f"The weather in {tc.arguments['location']} is 22 degrees Celsius and sunny."
        )

        # Build the conversation history for the second turn:
        # [user turn 1] -> [assistant turn with tool_calls] -> [tool result]
        # The assistant message must be in Ollama's format so the model knows
        # which tool result corresponds to which call.
        assistant_tool_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                },
            ],
        }

        final = self._live_call(
            model,
            prompt=prompt,  # stored in history but not re-appended (tool_results present)
            messages=[
                {"role": "user", "content": prompt},
                assistant_tool_message,
            ],
            tools=tools,
            tool_results=[
                ToolResult(call_id=tc.id, name=tc.name, content=weather_result),
            ],
        )

        self.assertIsInstance(final.text, str)
        self.assertGreater(len(final.text), 0, "Final response must contain text")


# ---------------------------------------------------------------------------
# 7. Script provider
# ---------------------------------------------------------------------------

class TestScriptProvider(ProviderRegistryIsolation):
    """The subprocess provider and its stdin/stdout JSON protocol."""

    def test_script_generate(self) -> None:
        script = _make_script(_ECHO_SCRIPT)
        try:
            from unified_ai_client import call_ai
            response = call_ai(
                provider="script",
                model=script,
                prompt="Hello",
                timeout=15,
            )
            self.assertIn("Echo: Hello", response.text)
            self.assertIn("(files=0)", response.text)
            self.assertEqual(response.input_tokens, 1)
            self.assertEqual(response.output_tokens, 2)
        finally:
            os.unlink(script)

    def test_script_generate_with_file(self) -> None:
        script = _make_script(_ECHO_SCRIPT)
        tmp = _make_text_file("attached content")
        try:
            from unified_ai_client import call_ai
            response = call_ai(
                provider="script",
                model=script,
                prompt="Summarize",
                file_path=tmp,
                timeout=15,
            )
            self.assertIn("(files=1)", response.text)
        finally:
            os.unlink(script)
            os.unlink(tmp)

    def test_script_generate_with_reasoning(self) -> None:
        script = _make_script(_ECHO_SCRIPT)
        try:
            from unified_ai_client import call_ai
            response = call_ai(
                provider="script",
                model=script,
                prompt="Think",
                thinking=True,
                timeout=15,
            )
            self.assertEqual(response.reasoning_text, "I thought about it.")
            self.assertEqual(response.reasoning_tokens, 10)
            self.assertIs(response.reasoning_is_summary, False)
        finally:
            os.unlink(script)

    def test_script_subprocess_is_run_as_utf8(self) -> None:
        """Regression: stdin/stdout used to be decoded with the locale encoding.

        `subprocess.run(text=True)` without an explicit `encoding` falls back
        to `locale.getpreferredencoding(False)`, which is cp1252 on Windows,
        the platform this project develops on. The documented JSON payload
        happens to survive that anyway, because `json.dumps()` defaults to
        `ensure_ascii=True` and escapes every non-ASCII character before it
        ever reaches the pipe — but a script's stderr on a crash is raw text,
        not JSON, and that path is where the mojibake or UnicodeDecodeError
        actually surfaces. Asserted directly on the call rather than through
        an end-to-end pipe, since the failure mode depends on the parent's
        locale, the child's own stdio encoding, and this shell's console
        encoding all at once, none of which a portable test controls.
        """
        from unified_ai_client.providers.script import _run_script

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout='{"text": "ok"}', stderr="",
            )
            _run_script(["python", "fake.py"], {"mode": "generate"}, 30)

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs.get("encoding"), "utf-8")
        self.assertEqual(kwargs.get("errors"), "replace")

    def test_script_embed(self) -> None:
        script = _make_script(_ECHO_SCRIPT)
        try:
            from unified_ai_client import get_embedding
            vec = get_embedding(provider="script", model=script, text="hello")
            self.assertEqual(vec, [0.1, 0.2, 0.3])
        finally:
            os.unlink(script)

    def test_script_nonzero_exit(self) -> None:
        """A script that exits non-zero must raise RuntimeError, not crash silently."""
        failing_script = _make_script(
            "import sys; print('error details', file=sys.stderr); sys.exit(1)\n"
        )
        try:
            from unified_ai_client import call_ai
            with self.assertRaises(RuntimeError) as ctx:
                call_ai(
                    provider="script",
                    model=failing_script,
                    prompt="test",
                    max_retries=1,
                    timeout=10,
                )
            self.assertIn("error details", str(ctx.exception))
        finally:
            os.unlink(failing_script)

    def test_script_provider_forwards_new_params(self) -> None:
        """ScriptProvider must forward top_k, top_p, max_tokens and extra_options."""
        script_content = """\
from __future__ import annotations
import json, sys

def main() -> None:
    req = json.loads(sys.stdin.read())
    result = {
        "text": f"top_k={req.get('top_k')} top_p={req.get('top_p')} max_tokens={req.get('max_tokens')} extra={req.get('extra_options')}",
        "input_tokens": 1,
        "output_tokens": 2
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()
"""
        script_path = _make_script(script_content)
        try:
            from unified_ai_client import call_ai
            response = call_ai(
                provider="script",
                model=script_path,
                prompt="test",
                top_k=42,
                top_p=0.8,
                max_tokens=150,
                extra_options={"some_custom_option": "xyz"},
                timeout=10,
            )
            self.assertIn("top_k=42", response.text)
            self.assertIn("top_p=0.8", response.text)
            self.assertIn("max_tokens=150", response.text)
            self.assertIn("some_custom_option", response.text)
        finally:
            os.unlink(script_path)


# ---------------------------------------------------------------------------
# 8. Ollama provider — offline unit tests
# ---------------------------------------------------------------------------

class TestModelResidency(ProviderRegistryIsolation):
    """cleanup() releases the VRAM that keep_alive=-1 would otherwise hold."""

    def _tracked(self) -> set:
        from unified_ai_client import registry as _client
        return set(_client._LOADED_MODELS)

    def test_call_ai_tracks_a_local_model(self) -> None:
        from unified_ai_client import call_ai
        from unified_ai_client.registry import get_provider
        from unified_ai_client.models import AiResponse

        provider = get_provider("ollama")
        with patch.object(provider, "call", return_value=AiResponse(text="hi")):
            call_ai("ollama", "gemma4:12b", "hello")

        self.assertIn(("ollama", "gemma4:12b"), self._tracked())

    def test_a_cloud_model_is_not_tracked(self) -> None:
        """A cloud endpoint holds nothing, so there is nothing to release."""
        from unified_ai_client import call_ai
        from unified_ai_client.registry import get_provider
        from unified_ai_client.models import AiResponse

        provider = get_provider("google")
        with patch.object(provider, "call", return_value=AiResponse(text="hi")):
            call_ai("google", "gemini-2.5-flash", "hello")

        self.assertNotIn(("google", "gemini-2.5-flash"), self._tracked())

    def test_cleanup_unloads_tracked_models(self) -> None:
        from unified_ai_client import cleanup
        from unified_ai_client.registry import get_provider, _LOADED_MODELS

        provider = get_provider("ollama")
        _LOADED_MODELS.clear()
        _LOADED_MODELS.add(("ollama", "gemma4:12b"))
        with patch.object(provider, "unload_model") as unload:
            cleanup()

        unload.assert_called_once_with("gemma4:12b")
        self.assertEqual(self._tracked(), set())

    def test_cleanup_can_leave_models_resident(self) -> None:
        from unified_ai_client import cleanup
        from unified_ai_client.registry import get_provider, _LOADED_MODELS

        provider = get_provider("ollama")
        _LOADED_MODELS.add(("ollama", "gemma4:12b"))
        with patch.object(provider, "unload_model") as unload:
            cleanup(unload_models=False)

        unload.assert_not_called()
        self.assertIn(("ollama", "gemma4:12b"), self._tracked())

    def test_cleanup_survives_a_failing_unload(self) -> None:
        """One provider that cannot release must not block the others,
        and the one that failed must stay tracked for the next cleanup()
        to retry instead of being cleared along with the one that succeeded.
        """
        from unified_ai_client import cleanup
        from unified_ai_client.registry import get_provider, _LOADED_MODELS

        ollama = get_provider("ollama")
        _LOADED_MODELS.clear()
        _LOADED_MODELS.update({("ollama", "a"), ("ollama", "b")})
        with patch.object(
            ollama, "unload_model", side_effect=[RuntimeError("boom"), None]
        ) as unload:
            cleanup()

        self.assertEqual(unload.call_count, 2)
        # sorted(_LOADED_MODELS) processes "a" before "b", so the
        # side_effect list makes "a" the one that raised.
        self.assertEqual(self._tracked(), {("ollama", "a")})

    def test_explicit_unload_is_not_repeated_at_exit(self) -> None:
        from unified_ai_client import cleanup, unload_model
        from unified_ai_client.registry import get_provider, _LOADED_MODELS

        provider = get_provider("ollama")
        _LOADED_MODELS.clear()
        _LOADED_MODELS.add(("ollama", "gemma4:12b"))
        with patch.object(provider, "unload_model") as unload:
            unload_model("ollama", "gemma4:12b")
            self.assertEqual(self._tracked(), set())
            cleanup()

        self.assertEqual(unload.call_count, 1)

    def test_cleanup_unloads_after_the_instance_was_evicted(self) -> None:
        """The registry outlives the instance, and the unload has to follow it.

        configure_provider() drops the cached instance. The model it loaded is
        still resident, so cleanup() must resolve the provider by name rather
        than reuse whatever instance happened to be cached when it started.
        """
        from unified_ai_client import cleanup
        from unified_ai_client.registry import (
            configure_provider, get_provider, _LOADED_MODELS,
        )

        from unified_ai_client.providers.ollama import OllamaProvider

        get_provider("ollama")
        _LOADED_MODELS.clear()
        _LOADED_MODELS.add(("ollama", "gemma4:12b"))

        # Evict, and deliberately do not resolve the provider again: cleanup()
        # has to rebuild it. Patching the class rather than an instance is what
        # makes this test fail if cleanup() reuses its own cache snapshot.
        configure_provider("ollama", timeout=42)

        with patch.object(OllamaProvider, "unload_model") as unload:
            cleanup()

        unload.assert_called_once_with("gemma4:12b")

    def test_cleanup_still_purges_remote_resources(self) -> None:
        """The pre-existing half of cleanup() must survive the restructure."""
        from unified_ai_client import cleanup
        from unified_ai_client.registry import get_provider

        provider = get_provider("google")
        with patch.object(provider, "cleanup") as purge:
            cleanup()

        purge.assert_called_once_with()


class TestGoogleUploadCache(ProviderRegistryIsolation):
    """The upload cache must outlive the provider instance that filled it."""

    def test_cache_survives_reconfiguration(self) -> None:
        """configure_provider() evicts the instance; the references must stay.

        Held on the instance, they went with it: the files stayed on the remote
        quota with nothing left to delete them, and the next call re-uploaded
        every one of them.
        """
        from unified_ai_client.registry import configure_provider, get_provider
        from unified_ai_client.providers import google as google_mod

        first = get_provider("google")
        google_mod._UPLOADED_FILES["/tmp/whatever.pdf"] = object()

        configure_provider("google", timeout=120)
        second = get_provider("google")

        self.assertIsNot(first, second, "configure_provider must evict the instance")
        self.assertIn("/tmp/whatever.pdf", google_mod._UPLOADED_FILES)

    def test_cleanup_deletes_what_an_evicted_instance_uploaded(self) -> None:
        from unified_ai_client.registry import cleanup, configure_provider, get_provider
        from unified_ai_client.providers import google as google_mod
        from unified_ai_client.providers.google import GoogleProvider

        get_provider("google")
        # A live test earlier in the run may have left real uploads here, and
        # this test counts delete calls. Isolation restores them in tearDown.
        google_mod._UPLOADED_FILES.clear()
        ref = MagicMock()
        ref.name = "files/abc123"
        google_mod._UPLOADED_FILES["/tmp/whatever.pdf"] = ref

        # Evict, and deliberately do not resolve the provider again. Patching
        # the class rather than an instance is what makes this test fail if
        # cleanup() only visits the instances still in the cache: the upload
        # would then sit on the remote quota with nothing able to delete it.
        configure_provider("google", timeout=120)

        client = MagicMock()
        with patch.object(GoogleProvider, "_get_client", return_value=client):
            cleanup(unload_models=False)

        client.files.delete.assert_called_once_with(name="files/abc123")
        self.assertEqual(google_mod._UPLOADED_FILES, {})

    def test_a_failed_delete_is_logged_and_does_not_stop_the_others(self) -> None:
        """Regression: a delete failure was swallowed with no trace at all,
        for the one outcome cleanup() exists to prevent: the file stays on
        Google's quota. Must still not raise, and must not stop the delete
        of the other uploads in the same cleanup() call.
        """
        from unified_ai_client.registry import cleanup, configure_provider, get_provider
        from unified_ai_client.providers import google as google_mod
        from unified_ai_client.providers.google import GoogleProvider

        get_provider("google")
        google_mod._UPLOADED_FILES.clear()
        bad_ref = MagicMock()
        bad_ref.name = "files/bad"
        good_ref = MagicMock()
        good_ref.name = "files/good"
        google_mod._UPLOADED_FILES["/tmp/bad.pdf"] = bad_ref
        google_mod._UPLOADED_FILES["/tmp/good.pdf"] = good_ref

        configure_provider("google", timeout=120)

        client = MagicMock()
        client.files.delete.side_effect = [RuntimeError("quota service down"), None]
        with patch.object(GoogleProvider, "_get_client", return_value=client):
            with self.assertLogs(
                "unified_ai_client.providers.google", level="DEBUG"
            ) as cm:
                cleanup(unload_models=False)

        self.assertEqual(client.files.delete.call_count, 2)
        self.assertTrue(any("bad" in line for line in cm.output))
        self.assertEqual(google_mod._UPLOADED_FILES, {})


class TestOllamaOffline(unittest.TestCase):
    """Payload construction and response parsing, with _post intercepted."""

    def test_ollama_provider_options_mapping(self) -> None:
        """Config and call parameters must map onto Ollama's options dict."""
        from unified_ai_client.providers.ollama import OllamaProvider
        from unified_ai_client.models import ProviderConfig, AiRequest

        config = ProviderConfig(
            url="http://mock-ollama:11434",
            extra_options={
                "context_size": 2048,
                "keep_alive": "30m",
                "my_custom_option": 123,
            },
        )
        provider = OllamaProvider(config=config)

        captured_payload: dict = {}

        def fake_post(endpoint: str, payload: dict, timeout: int) -> dict:
            captured_payload.update(payload)
            return {
                "message": {"content": "response", "thinking": ""},
                "eval_count": 10,
                "prompt_eval_count": 5,
            }

        request = AiRequest(
            provider="ollama",
            model="mock-model",
            prompt="hello",
            top_k=50,
            top_p=0.9,
            max_tokens=100,
            extra_options={"my_custom_option": 456, "visual_token_budget": 70},
        )

        with patch.object(provider, "_post", side_effect=fake_post):
            provider.call(request)

        self.assertEqual(captured_payload["keep_alive"], "30m")
        options = captured_payload["options"]
        self.assertEqual(options["top_k"], 50)
        self.assertEqual(options["top_p"], 0.9)
        self.assertEqual(options["num_ctx"], 2048)
        self.assertEqual(options["num_predict"], 100)
        self.assertEqual(options["my_custom_option"], 456)
        self.assertEqual(options["visual_token_budget"], 70)

    def test_ollama_skips_file_processing_on_a_tool_result_continuation(self) -> None:
        """Regression: an attachment was read and base64-encoded even when
        tool_results meant the result could never be attached to this turn.

        _process_files_for_message() used to run unconditionally, before the
        `if not request.tool_results` guard around using its result, so the
        encoding cost was paid and then thrown away.
        """
        from unified_ai_client.providers.ollama import OllamaProvider
        from unified_ai_client.models import AiRequest, ProviderConfig, ToolResult

        provider = OllamaProvider(config=ProviderConfig())

        def fake_post(endpoint: str, payload: dict, timeout: int) -> dict:
            return {
                "message": {"content": "response", "thinking": ""},
                "eval_count": 10,
                "prompt_eval_count": 5,
            }

        request = AiRequest(
            provider="ollama",
            model="mock-model",
            prompt="hello",
            file_path=__file__,  # any real path; must not be read
            tool_results=[ToolResult(call_id="1", name="f", content="done")],
        )

        with patch.object(provider, "_post", side_effect=fake_post):
            with patch.object(
                provider, "_process_files_for_message"
            ) as mock_process:
                provider.call(request)

        mock_process.assert_not_called()

    def test_ollama_tool_payload(self) -> None:
        """Ollama provider must build tools in OpenAI-compatible format."""
        from unified_ai_client.providers.ollama import OllamaProvider
        from unified_ai_client.models import AiRequest, ProviderConfig, ToolDefinition

        provider = OllamaProvider(ProviderConfig(url="http://localhost:11434"))

        tool = ToolDefinition(
            name="get_weather",
            description="Returns weather.",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        request = AiRequest(
            provider="ollama", model="gemma4:12b", prompt="Weather in Rome?",
            tools=[tool],
        )

        captured_payload: dict = {}

        def fake_post(endpoint: str, payload: dict, timeout: int) -> dict:
            captured_payload.update(payload)
            return {
                "message": {"content": "", "tool_calls": []},
                "eval_count": 0,
                "prompt_eval_count": 0,
            }

        with patch.object(provider, "_post", side_effect=fake_post):
            provider.call(request)

        self.assertIn("tools", captured_payload)
        self.assertEqual(captured_payload["tools"][0]["type"], "function")
        self.assertEqual(
            captured_payload["tools"][0]["function"]["name"], "get_weather"
        )

    def test_ollama_parse_tool_calls(self) -> None:
        """Ollama provider must parse message.tool_calls."""
        from unified_ai_client.providers.ollama import OllamaProvider
        from unified_ai_client.models import AiRequest, ProviderConfig

        provider = OllamaProvider(ProviderConfig(url="http://localhost:11434"))
        request = AiRequest(provider="ollama", model="gemma4:12b", prompt="Weather?")

        def fake_post(endpoint: str, payload: dict, timeout: int) -> dict:
            return {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"location": "Rome"},
                            },
                        },
                    ],
                },
                "eval_count": 0,
                "prompt_eval_count": 0,
            }

        with patch.object(provider, "_post", side_effect=fake_post):
            resp = provider.call(request)

        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].name, "get_weather")
        self.assertEqual(resp.tool_calls[0].arguments, {"location": "Rome"})


# ---------------------------------------------------------------------------
# 9. Google provider — offline unit tests
# ---------------------------------------------------------------------------

class TestGoogleOffline(unittest.TestCase):
    """Thinking configuration built per model family, without network access."""

    def test_google_thinking_config_offline(self) -> None:
        """Thinking must be minimised when False and raised when True.

        Gemini 3.x uses thinking_level, Gemini 2.5 uses thinking_budget, and
        every other model falls back to a small budget.
        """
        from unified_ai_client.providers.google import GoogleProvider
        from unified_ai_client.models import ProviderConfig

        provider = GoogleProvider(
            config=ProviderConfig(sleep_time=0), api_key="dummy_key"
        )

        # 1. thinking=False (explicitly disabled/minimized)
        cfg_g3_off = provider._build_thinking_config(
            thinking=False, model_name="gemini-3.5-flash"
        )
        self.assertIsNotNone(cfg_g3_off)
        self.assertEqual(cfg_g3_off.thinking_level, "MINIMAL")
        self.assertIs(cfg_g3_off.include_thoughts, True)

        cfg_g25_off = provider._build_thinking_config(
            thinking=False, model_name="gemini-2.5-pro"
        )
        self.assertIsNotNone(cfg_g25_off)
        self.assertEqual(cfg_g25_off.thinking_budget, 0)
        self.assertIs(cfg_g25_off.include_thoughts, True)

        cfg_other_off = provider._build_thinking_config(
            thinking=False, model_name="gemini-1.5-pro"
        )
        self.assertIsNotNone(cfg_other_off)
        self.assertEqual(cfg_other_off.thinking_budget, 0)
        self.assertIs(cfg_other_off.include_thoughts, True)

        # 2. thinking=True (enabled)
        cfg_g3_on = provider._build_thinking_config(
            thinking=True, model_name="gemini-3.5-flash"
        )
        self.assertIsNotNone(cfg_g3_on)
        self.assertEqual(cfg_g3_on.thinking_level, "HIGH")
        self.assertIs(cfg_g3_on.include_thoughts, True)

        cfg_g25_on = provider._build_thinking_config(
            thinking=True, model_name="gemini-2.5-pro"
        )
        self.assertIsNotNone(cfg_g25_on)
        self.assertEqual(cfg_g25_on.thinking_budget, 24576)
        self.assertIs(cfg_g25_on.include_thoughts, True)

        cfg_other_on = provider._build_thinking_config(
            thinking=True, model_name="gemini-1.5-pro"
        )
        self.assertIsNotNone(cfg_other_on)
        self.assertEqual(cfg_other_on.thinking_budget, 1024)
        self.assertIs(cfg_other_on.include_thoughts, True)

        # 3. thinking="default" (provider defaults, but capture thoughts)
        cfg_default = provider._build_thinking_config(
            thinking="default", model_name="gemini-2.5-pro"
        )
        self.assertIsNotNone(cfg_default)
        self.assertIs(cfg_default.include_thoughts, True)
        self.assertIsNone(getattr(cfg_default, "thinking_budget", None))
        self.assertIsNone(getattr(cfg_default, "thinking_level", None))


class TestReasoningContract(unittest.TestCase):
    """The reasoning_is_summary flag must stay conservative across providers."""

    def test_reasoning_is_summary_offline(self) -> None:
        """Only Google may report a summarised trace.

        The flag tells consumers whether reasoning_text is the model's raw chain
        of thought or a summary the provider wrote about it. Consumers that
        measure trace length or composition cannot compare the two, so the
        default must be conservative (False = raw).
        """
        from unified_ai_client.models import AiResponse

        # Default is raw: a provider that never sets the flag reports a raw trace.
        self.assertIs(AiResponse(text="x").reasoning_is_summary, False)
        self.assertIs(
            AiResponse(text="x", reasoning_text="raw trace").reasoning_is_summary,
            False,
        )

        # Google sets it from the presence of thought parts, not from
        # request.thinking: a model may emit thoughts even when thinking was
        # not explicitly requested.
        providers_dir = _PROJECT_ROOT / "unified_ai_client" / "providers"
        google_src = (providers_dir / "google.py").read_text(encoding="utf-8")
        self.assertIn("reasoning_is_summary=bool(reasoning_text)", google_src)

        # Raw-trace providers must not set the flag at all.
        for name in ("ollama", "anthropic", "openai_compat", "script"):
            src = (providers_dir / f"{name}.py").read_text(encoding="utf-8")
            self.assertNotIn(
                "reasoning_is_summary",
                src,
                f"{name} returns a raw trace and must leave "
                f"reasoning_is_summary at its default",
            )


# ---------------------------------------------------------------------------
# 9b. Live smoke test shared by the three cloud chat providers
# ---------------------------------------------------------------------------

class TestLivePongSmoke(RequiresCredential, unittest.TestCase):
    """The same trivial live round-trip, once, for every cloud chat provider.

    Until 0.5.8 this was the same test body copied three times
    (TestGoogleLive.test_google_live_generate, TestAnthropicLive,
    TestOpenAiLive), differing only in provider/model/secrets key. Ollama
    does not participate: its live tests go through _require_model() and
    _live_call(), a different mechanism not worth forcing into this shape.
    """

    _CASES: dict[str, tuple[str, str]] = {
        "google": ("gemini-2.5-flash", "google_api_key"),
        "anthropic": ("claude-3-5-haiku-latest", "anthropic_api_key"),
        "openai": ("gpt-4o-mini", "openai_api_key"),
    }

    def test_replies_to_a_trivial_prompt(self) -> None:
        from unified_ai_client import call_ai
        for provider, (model, secrets_key) in self._CASES.items():
            with self.subTest(provider=provider):
                self._require_credential(secrets_key)
                response = call_ai(
                    provider=provider,
                    model=model,
                    prompt="Reply with exactly the word PONG and nothing else.",
                    temperature=0.0,
                    timeout=30,
                )
                self.assertIsInstance(response.text, str)
                self.assertGreater(len(response.text), 0)


# ---------------------------------------------------------------------------
# 10. Google (skip if no API key)
# ---------------------------------------------------------------------------

class TestGoogleLive(RequiresCredential, unittest.TestCase):
    """End-to-end calls against the Google AI API."""

    def setUp(self) -> None:
        super().setUp()
        self._require_credential("google_api_key")

    def test_google_live_with_text_file(self) -> None:
        tmp = _make_text_file("The sky is blue.")
        try:
            from unified_ai_client import call_ai
            response = call_ai(
                provider="google",
                model="gemini-2.5-flash",
                prompt="What colour is mentioned in the attached file? Reply in one word.",
                file_path=tmp,
                temperature=0.0,
                timeout=30,
            )
            self.assertIsInstance(response.text, str)
            self.assertGreater(len(response.text), 0)
        finally:
            os.unlink(tmp)

    def test_google_live_thinking(self) -> None:
        from unified_ai_client import call_ai
        response = call_ai(
            provider="google",
            model="gemini-2.5-flash",
            prompt="What is 2+2? Think step by step.",
            thinking=True,
            temperature=0.0,
            timeout=60,
        )
        self.assertIsInstance(response.text, str)
        self.assertIsInstance(response.reasoning_text, str)

    def test_google_live_thinking_default(self) -> None:
        from unified_ai_client import call_ai
        response = call_ai(
            provider="google",
            model="gemini-2.5-flash",
            prompt="What is 2+2? Think step by step.",
            thinking="default",
            temperature=0.0,
            timeout=60,
        )
        self.assertIsInstance(response.text, str)
        self.assertIsInstance(response.reasoning_text, str)

    def test_google_live_tool_calling(self) -> None:
        """Live Google test: gemini-2.5-flash tool calling (two-turn).

        The second call passes the full conversation history including the
        assistant's intermediate tool_calls turn so the provider converts it to
        function_call Parts and Google can link the result back.
        """
        from unified_ai_client import call_ai, ToolDefinition, ToolResult

        tools = [
            ToolDefinition(
                name="get_weather",
                description="Returns the current weather for a given city.",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. Rome",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ]

        prompt = "What is the weather in Rome right now? Use the get_weather tool."

        response = call_ai(
            provider="google",
            model="gemini-2.5-flash",
            prompt=prompt,
            tools=tools,
            temperature=0.0,
            timeout=60,
        )

        if not response.tool_calls:
            self.skipTest("gemini-2.5-flash did not produce a tool call")

        tc = response.tool_calls[0]
        self.assertEqual(tc.name, "get_weather")
        self.assertIn("location", tc.arguments)

        weather_result = (
            f"The weather in {tc.arguments['location']} is 22 degrees Celsius and sunny."
        )

        assistant_tool_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                },
            ],
        }

        final = call_ai(
            provider="google",
            model="gemini-2.5-flash",
            prompt=prompt,
            messages=[
                {"role": "user", "content": prompt},
                assistant_tool_message,
            ],
            tools=tools,
            tool_results=[
                ToolResult(call_id=tc.id, name=tc.name, content=weather_result),
            ],
            temperature=0.0,
            timeout=60,
        )

        self.assertIsInstance(final.text, str)
        self.assertGreater(len(final.text), 0, "Final response must contain text")


# ---------------------------------------------------------------------------
# 11. Anthropic
# ---------------------------------------------------------------------------

class TestAnthropicOffline(unittest.TestCase):
    """Anthropic payload construction and response parsing."""

    def _provider(self):
        from unified_ai_client.providers.anthropic import AnthropicProvider
        from unified_ai_client.models import ProviderConfig
        return AnthropicProvider(
            ProviderConfig(url="https://api.anthropic.com"), api_key="fake-key"
        )

    def test_anthropic_tool_payload(self) -> None:
        """Anthropic provider must build tools in input_schema format."""
        from unified_ai_client.models import AiRequest, ToolDefinition

        provider = self._provider()
        tool = ToolDefinition(
            name="get_weather",
            description="Returns weather.",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        request = AiRequest(
            provider="anthropic", model="claude-opus-4-5", prompt="Weather in Rome?",
            tools=[tool],
        )

        captured_payload: dict = {}

        def fake_post(payload: dict, timeout: int) -> dict:
            captured_payload.update(payload)
            return {
                "content": [{"type": "text", "text": "Sunny."}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

        with patch.object(provider, "_post", side_effect=fake_post):
            provider.call(request)

        self.assertIn("tools", captured_payload)
        self.assertEqual(captured_payload["tools"][0]["name"], "get_weather")
        self.assertIn("input_schema", captured_payload["tools"][0])
        self.assertNotIn("parameters", captured_payload["tools"][0])

    def test_anthropic_parse_tool_calls(self) -> None:
        """Anthropic provider must parse tool_use content blocks."""
        from unified_ai_client.models import AiRequest

        provider = self._provider()
        request = AiRequest(
            provider="anthropic", model="claude-opus-4-5", prompt="Weather?"
        )

        def fake_post(payload: dict, timeout: int) -> dict:
            return {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "get_weather",
                        "input": {"location": "Rome"},
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

        with patch.object(provider, "_post", side_effect=fake_post):
            resp = provider.call(request)

        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].id, "toolu_01")
        self.assertEqual(resp.tool_calls[0].name, "get_weather")
        self.assertEqual(resp.tool_calls[0].arguments, {"location": "Rome"})


# ---------------------------------------------------------------------------
# 12. OpenAI-compatible providers
# ---------------------------------------------------------------------------

class TestOpenAiCompatOffline(unittest.TestCase):
    """Payload construction and response parsing for the shared base class."""

    def _provider(self):
        from unified_ai_client.providers.openai_compat import OpenAiCompatProvider
        from unified_ai_client.models import ProviderConfig
        return OpenAiCompatProvider(ProviderConfig(url="http://localhost:8080"))

    def test_openai_compat_tool_payload(self) -> None:
        """OpenAI-compat provider must build the correct tools payload."""
        from unified_ai_client.models import AiRequest, ToolDefinition

        provider = self._provider()
        tool = ToolDefinition(
            name="get_weather",
            description="Returns current weather.",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        request = AiRequest(
            provider="openai", model="gpt-4o-mini", prompt="Weather in Rome?",
            tools=[tool],
        )

        captured_payload: dict = {}

        def fake_post(endpoint: str, payload: dict, timeout: int) -> dict:
            captured_payload.update(payload)
            return {
                "choices": [{"message": {"content": "Sunny.", "tool_calls": None}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        with patch.object(provider, "_post", side_effect=fake_post):
            resp = provider.call(request)

        self.assertIn("tools", captured_payload)
        self.assertEqual(captured_payload["tools"][0]["type"], "function")
        self.assertEqual(
            captured_payload["tools"][0]["function"]["name"], "get_weather"
        )
        self.assertEqual(resp.tool_calls, [])
        self.assertEqual(resp.text, "Sunny.")

    def test_openai_compat_parse_tool_calls(self) -> None:
        """OpenAI-compat provider must parse tool calls from the response."""
        from unified_ai_client.models import AiRequest, ToolDefinition

        provider = self._provider()
        tool = ToolDefinition(
            name="get_weather",
            description="Returns current weather.",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        request = AiRequest(
            provider="openai", model="gpt-4o-mini", prompt="Weather in Rome?",
            tools=[tool],
        )

        def fake_post(endpoint: str, payload: dict, timeout: int) -> dict:
            return {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": json.dumps({"location": "Rome"}),
                                },
                            },
                        ],
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        with patch.object(provider, "_post", side_effect=fake_post):
            resp = provider.call(request)

        self.assertEqual(len(resp.tool_calls), 1)
        tc = resp.tool_calls[0]
        self.assertEqual(tc.id, "call_xyz")
        self.assertEqual(tc.name, "get_weather")
        self.assertEqual(tc.arguments, {"location": "Rome"})
        self.assertEqual(resp.text, "")

    def test_openai_compat_tool_results_in_messages(self) -> None:
        """Tool results must be serialized as role:tool messages."""
        from unified_ai_client.models import AiRequest, ToolResult

        provider = self._provider()
        request = AiRequest(
            provider="openai", model="gpt-4o-mini", prompt="What now?",
            tool_results=[
                ToolResult(call_id="call_xyz", name="get_weather", content="22C, sunny"),
            ],
        )

        captured_messages: list = []

        def fake_post(endpoint: str, payload: dict, timeout: int) -> dict:
            captured_messages.extend(payload["messages"])
            return {
                "choices": [{"message": {"content": "Great weather!", "tool_calls": None}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            }

        with patch.object(provider, "_post", side_effect=fake_post):
            provider.call(request)

        tool_msgs = [m for m in captured_messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "call_xyz")
        self.assertEqual(tool_msgs[0]["content"], "22C, sunny")


# ---------------------------------------------------------------------------
# 13. Tool calling data types
# ---------------------------------------------------------------------------

class TestToolTypes(unittest.TestCase):
    """The public tool-calling dataclasses and their defaults."""

    def test_import_tool_types(self) -> None:
        from unified_ai_client import ToolDefinition, ToolCall, ToolResult
        self.assertTrue(ToolDefinition and ToolCall and ToolResult)

    def test_tool_definition_construction(self) -> None:
        from unified_ai_client import ToolDefinition
        td = ToolDefinition(
            name="get_weather",
            description="Get current weather.",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        self.assertEqual(td.name, "get_weather")
        self.assertEqual(td.parameters["type"], "object")

    def test_tool_call_construction(self) -> None:
        from unified_ai_client import ToolCall
        tc = ToolCall(id="call_abc", name="get_weather", arguments={"location": "Rome"})
        self.assertEqual(tc.id, "call_abc")
        self.assertEqual(tc.arguments["location"], "Rome")

    def test_tool_result_construction(self) -> None:
        from unified_ai_client import ToolResult
        tr = ToolResult(call_id="call_abc", name="get_weather", content="22C, sunny")
        self.assertEqual(tr.call_id, "call_abc")
        self.assertEqual(tr.name, "get_weather")
        self.assertEqual(tr.content, "22C, sunny")

    def test_airesponse_tool_calls_default(self) -> None:
        from unified_ai_client import AiResponse
        self.assertEqual(AiResponse(text="hello").tool_calls, [])

    def test_airequest_tools_fields(self) -> None:
        from unified_ai_client.models import AiRequest, ToolDefinition, ToolResult
        td = ToolDefinition(
            name="f", description="d",
            parameters={"type": "object", "properties": {}},
        )
        tr = ToolResult(call_id="c1", name="f", content="result")
        r = AiRequest(
            provider="ollama", model="m", prompt="p",
            tools=[td], tool_results=[tr],
        )
        self.assertIsNotNone(r.tools)
        self.assertEqual(len(r.tools), 1)
        self.assertIsNotNone(r.tool_results)
        self.assertEqual(len(r.tool_results), 1)


class TestConfigurationReachesTheCall(ProviderRegistryIsolation):
    """A registered timeout has to apply to generation, not just to warm-up."""

    def _capture_request(self, **call_kwargs):
        """Run call_ai against a patched transport and return the AiRequest."""
        from unified_ai_client import call_ai
        from unified_ai_client.models import AiResponse
        from unified_ai_client.providers.ollama import OllamaProvider

        seen = {}

        def fake_call(self, request):
            seen["request"] = request
            return AiResponse(text="ok")

        with patch.object(OllamaProvider, "call", fake_call):
            call_ai(provider="ollama", model="m", prompt="p", **call_kwargs)
        return seen["request"]

    def test_configured_timeout_reaches_generation(self) -> None:
        """Regression: configure_provider(timeout=...) was ignored by call_ai.

        call_ai defaulted timeout to a hard 300 that went straight into the
        AiRequest, and every adapter reads request.timeout. config.timeout was
        consulted only by warm-up, preload and embeddings, so a caller who
        registered a longer deadline got 300 seconds with nothing to say so.
        """
        from unified_ai_client import configure_provider

        configure_provider("ollama", timeout=600)
        self.assertEqual(self._capture_request().timeout, 600)

    def test_an_explicit_timeout_still_wins(self) -> None:
        from unified_ai_client import configure_provider

        configure_provider("ollama", timeout=600)
        self.assertEqual(self._capture_request(timeout=30).timeout, 30)

    def test_the_default_is_still_300_when_nothing_is_registered(self) -> None:
        self.assertEqual(self._capture_request().timeout, 300)


class TestFilePathWithToolResultsIsFlagged(ProviderRegistryIsolation):
    """A file that will not be attached must not vanish without a trace.

    Every adapter treats tool_results as "the user turn is already in
    messages", so file_path is never attached alongside it. The warning is
    the caller's only signal that this happened, since dropping the file
    itself is unchanged and stays out of scope for this fix.
    """

    def _call(self, **call_kwargs):
        from unified_ai_client import call_ai
        from unified_ai_client.models import AiResponse, ToolResult
        from unified_ai_client.providers.ollama import OllamaProvider

        with patch.object(OllamaProvider, "call", lambda self, request: AiResponse(text="ok")):
            call_ai(
                provider="ollama", model="m", prompt="p",
                tool_results=[ToolResult(call_id="1", name="f", content="done")],
                **call_kwargs,
            )

    def test_file_path_with_tool_results_logs_a_warning(self) -> None:
        with self.assertLogs("unified_ai_client.client", level="WARNING") as cm:
            self._call(file_path=__file__)
        self.assertTrue(any("file_path" in line for line in cm.output))

    def test_no_warning_without_a_file_path(self) -> None:
        with self.assertNoLogs("unified_ai_client.client", level="WARNING"):
            self._call()


class TestConfigureProviderMergesTheFile(ProviderRegistryIsolation):
    """configure_provider() must add to config.json, not replace it."""

    def setUp(self) -> None:
        super().setUp()
        from unified_ai_client import registry as _client

        self._client = _client
        self._saved_path = _client._effective_config_path
        self._tmp = tempfile.mkdtemp()
        path = os.path.join(self._tmp, "config.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {"ollama": {"url": "http://box:11434", "timeout": 600,
                            "keep_alive": "30m"}},
                fh,
            )
        _client._effective_config_path = path

    def tearDown(self) -> None:
        import shutil

        self._client._effective_config_path = self._saved_path
        shutil.rmtree(self._tmp, ignore_errors=True)
        super().tearDown()

    def test_registering_one_extra_keeps_the_rest_of_the_section(self) -> None:
        """Regression: the whole config.json section was silently discarded.

        get_provider() uses the programmatic config *instead of* the file, so
        one configure_provider() call for a single extra dropped the url and
        the timeout the file had set, sending requests to localhost with no log
        to explain it.
        """
        from unified_ai_client import configure_provider, get_provider

        configure_provider("ollama", context_size=8000)
        provider = get_provider("ollama")

        self.assertEqual(provider.base_url, "http://box:11434")
        self.assertEqual(provider.config.timeout, 600)
        self.assertEqual(provider.config.extra_options["context_size"], 8000)
        self.assertEqual(provider.config.extra_options["keep_alive"], "30m")

    def test_an_explicit_value_still_overrides_the_file(self) -> None:
        from unified_ai_client import configure_provider, get_provider

        configure_provider("ollama", url="http://other:11434")
        self.assertEqual(get_provider("ollama").base_url, "http://other:11434")

    def test_a_provider_absent_from_the_file_gets_the_defaults(self) -> None:
        from unified_ai_client import configure_provider, get_provider

        configure_provider("lmstudio", context_size=4096)
        provider = get_provider("lmstudio")

        self.assertEqual(provider.base_url, "http://localhost:1234")
        self.assertEqual(provider.config.timeout, 300)


if __name__ == "__main__":
    unittest.main()
