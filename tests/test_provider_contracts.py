"""Provider-level contracts: endpoint resolution, credentials, reasoning, embeddings.

These cover the parts of a provider that are decided before any content is
generated: which URL it talks to, whether it has the credentials to talk at
all, how the unified `thinking` flag reaches each API, and which providers
serve embeddings. Everything here runs offline, with the transport intercepted.

Usage:
    python -m unittest discover -s tests
    python -m unittest tests.test_provider_contracts.TestProviderUrlResolution
"""
from __future__ import annotations

import base64
import importlib
import io
import os
import shutil
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _path in (str(_PROJECT_ROOT), str(_TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Imported as a top-level module: unittest discovery puts tests/ on sys.path.
from test_providers import ProviderRegistryIsolation  # noqa: E402

from unified_ai_client.models import AiRequest, ProviderConfig  # noqa: E402


# Every provider, keyed by the name call_ai() accepts. The single source the
# credential, capability and file-support tables below are all driven from, so a
# renamed class is one edit rather than three.
_PROVIDER_CLASSES: dict[str, tuple[str, str]] = {
    "ollama": ("ollama", "OllamaProvider"),
    "google": ("google", "GoogleProvider"),
    "anthropic": ("anthropic", "AnthropicProvider"),
    "openai": ("openai", "OpenAiProvider"),
    "mistral": ("mistral", "MistralProvider"),
    "cohere": ("cohere", "CohereProvider"),
    "meta": ("meta", "MetaProvider"),
    "groq": ("groq", "GroqProvider"),
    "xai": ("xai", "XAiProvider"),
    "lmstudio": ("lmstudio", "LmStudioProvider"),
    "llamacpp": ("llamacpp", "LlamaCppProvider"),
    "script": ("script", "ScriptProvider"),
}


def _provider_class(module: str, class_name: str):
    """Import a provider class by module and class name."""
    return getattr(
        importlib.import_module(f"unified_ai_client.providers.{module}"), class_name
    )


def _provider_class_by_name(name: str):
    """Import a provider class by the registry name call_ai() accepts."""
    return _provider_class(*_PROVIDER_CLASSES[name])


# ---------------------------------------------------------------------------
# 1. Endpoint resolution
# ---------------------------------------------------------------------------

class TestProviderUrlResolution(ProviderRegistryIsolation):
    """An unset url falls back to the provider's own default, nothing else."""

    _DEFAULTS = {
        "ollama": "http://localhost:11434",
        "anthropic": "https://api.anthropic.com",
        "openai": "https://api.openai.com",
        "mistral": "https://api.mistral.ai",
        "cohere": "https://api.cohere.ai/compatibility",
        "meta": "https://api.llama-api.com",
        "groq": "https://api.groq.com/openai",
        "xai": "https://api.x.ai",
        "lmstudio": "http://localhost:1234",
        "llamacpp": "http://localhost:8080",
    }

    # google reaches its endpoint through the genai SDK and script spawns a
    # subprocess, so neither declares a DEFAULT_URL to fall back to. Naming them
    # here keeps the table below an assertion about every HTTP provider rather
    # than a list that quietly stops covering new ones.
    _NO_DEFAULT_URL = {"google", "script"}

    def test_table_covers_every_http_provider(self) -> None:
        self.assertEqual(
            set(self._DEFAULTS), set(_PROVIDER_CLASSES) - self._NO_DEFAULT_URL,
            "a provider was added or removed without updating this table",
        )

    def test_providers_without_a_default_url_declare_none(self) -> None:
        """The exclusion above must stay a fact, not an oversight."""
        for name in self._NO_DEFAULT_URL:
            with self.subTest(provider=name):
                cls = _provider_class_by_name(name)
                self.assertFalse(hasattr(cls, "DEFAULT_URL"))

    def test_provider_config_url_defaults_to_none(self) -> None:
        """None is the 'unset' marker, so no real URL doubles as a sentinel."""
        self.assertIsNone(ProviderConfig().url)

    def test_unset_url_resolves_to_each_provider_default(self) -> None:
        from unified_ai_client.client import get_provider
        for name, expected in self._DEFAULTS.items():
            with self.subTest(provider=name):
                self.assertEqual(get_provider(name).base_url, expected)

    def test_explicit_url_is_never_rewritten(self) -> None:
        """Regression: an explicit Ollama URL used to be silently redirected.

        `http://localhost:11434` was the ProviderConfig default and doubled as
        a "not set" sentinel, so configuring the openai adapter to talk to a
        local Ollama server (which does serve /v1/chat/completions) sent the
        request, and the OpenAI key with it, to api.openai.com instead.
        """
        from unified_ai_client import configure_provider
        from unified_ai_client.client import get_provider

        configure_provider("openai", url="http://localhost:11434")
        self.assertEqual(get_provider("openai").base_url, "http://localhost:11434")

    def test_explicit_url_does_not_mutate_the_registered_config(self) -> None:
        """Regression: the provider used to rewrite the registry's own object.

        `_PROVIDER_CONFIGS` holds the very ProviderConfig the provider is built
        from, so rewriting `self.config.url` corrupted the user's registered
        configuration beyond recovery.
        """
        from unified_ai_client import configure_provider
        from unified_ai_client.client import get_provider, _PROVIDER_CONFIGS

        configure_provider("openai", url="http://localhost:11434")
        get_provider("openai")
        self.assertEqual(_PROVIDER_CONFIGS["openai"].url, "http://localhost:11434")

    def test_trailing_slash_is_stripped(self) -> None:
        cls = _provider_class("openai", "OpenAiProvider")
        provider = cls(ProviderConfig(url="https://example.test/"), api_key="k")
        self.assertEqual(provider.base_url, "https://example.test")


class TestProviderNameRoundTrip(ProviderRegistryIsolation):
    """The name a provider reports must be the one call_ai() accepts.

    ``BaseProvider.provider_name`` derives the name from the class name, while
    ``get_provider()`` holds the authoritative mapping. Nothing else ties the
    two together, so a class renamed for style would quietly start naming a
    provider the caller cannot pass — inside the very refusal messages the
    attachment policy asks them to act on.
    """

    def test_every_registered_provider_reports_its_registry_name(self) -> None:
        from unified_ai_client.client import get_provider

        for name in _PROVIDER_CLASSES:
            if name == "script":
                continue  # needs a script_path in config to instantiate
            with self.subTest(provider=name):
                self.assertEqual(get_provider(name).provider_name, name)


# ---------------------------------------------------------------------------
# 2. API key handling
# ---------------------------------------------------------------------------

class TestApiKeyHandling(ProviderRegistryIsolation):
    """A missing key must fail with a message, not a type error."""

    _CLOUD = {
        "anthropic": "anthropic_api_key",
        "openai": "openai_api_key",
        "mistral": "mistral_api_key",
        "cohere": "cohere_api_key",
        "meta": "meta_api_key",
        "groq": "groq_api_key",
        "xai": "xai_api_key",
    }

    _LOCAL = ("lmstudio", "llamacpp")

    # The three that take neither path: google checks its credential in the lazy
    # client getter rather than in _require_api_key, ollama talks to a local
    # server, and script spawns a subprocess. Listing them makes the coverage
    # assertion below exhaustive instead of merely long.
    _NO_KEY_CHECK = {"google", "ollama", "script"}

    def test_tables_cover_every_registered_provider(self) -> None:
        self.assertEqual(
            set(self._CLOUD) | set(self._LOCAL) | self._NO_KEY_CHECK,
            set(_PROVIDER_CLASSES),
            "a provider was added or removed without updating these tables",
        )

    def test_cloud_providers_reject_a_missing_key(self) -> None:
        """The error must name the secrets key, so the fix is obvious."""
        for name, secrets_key in self._CLOUD.items():
            with self.subTest(provider=name):
                provider = _provider_class_by_name(name)(ProviderConfig(), api_key="")
                with self.assertRaises(ValueError) as ctx:
                    provider._require_api_key()
                message = str(ctx.exception)
                self.assertIn(secrets_key, message)
                # The name the caller would pass to call_ai(), not one derived
                # from the secrets key, which need not match it.
                self.assertIn(f"'{name}'", message)

    def test_local_providers_accept_a_missing_key(self) -> None:
        """A local server without credentials is normal, not an error."""
        for name in self._LOCAL:
            with self.subTest(provider=name):
                provider = _provider_class_by_name(name)(ProviderConfig())
                self.assertIsNone(provider._require_api_key())

    def test_an_explicit_url_lifts_the_key_requirement(self) -> None:
        """Pointing a cloud adapter elsewhere is supported and needs no key.

        Aiming the OpenAI adapter at a local Ollama or LM Studio server that
        serves /v1/chat/completions is exactly what the url override is for;
        demanding a cloud credential for it would make that impossible.
        """
        for name in self._CLOUD:
            with self.subTest(provider=name):
                provider = _provider_class_by_name(name)(
                    ProviderConfig(url="http://localhost:11434"), api_key=""
                )
                self.assertIsNone(provider._require_api_key())

    def test_none_is_normalised_to_empty_string(self) -> None:
        """`get_provider` reads absent keys as None; the signatures say str.

        An unset `x-api-key` header made urllib fail with "expected string or
        bytes-like object, got 'NoneType'", which tells someone who merely
        forgot the key nothing at all.
        """
        for module, class_name in (
            ("anthropic", "AnthropicProvider"),
            ("openai", "OpenAiProvider"),
        ):
            with self.subTest(provider=module):
                provider = _provider_class(module, class_name)(
                    ProviderConfig(), api_key=None
                )
                self.assertEqual(provider.api_key, "")

    def test_providers_without_a_key_inherit_a_harmless_check(self) -> None:
        """The three no-key providers must survive the inherited method.

        `_require_api_key()` moved onto BaseProvider, so ollama, script and
        google now inherit it where before they simply had no such method. It
        returns before touching `api_key`, which none of them defines: without
        that early exit this raises AttributeError on every request they make.
        """
        for name in sorted(self._NO_KEY_CHECK):
            with self.subTest(provider=name):
                cls = _provider_class_by_name(name)
                provider = cls(ProviderConfig())
                self.assertFalse(cls.REQUIRES_API_KEY)
                self.assertIsNone(provider._require_api_key())

    def test_get_provider_never_passes_none(self) -> None:
        from unified_ai_client.client import get_provider
        with patch.dict(os.environ, {}, clear=True):
            for name in self._CLOUD:
                with self.subTest(provider=name):
                    self.assertIsInstance(get_provider(name).api_key, str)


# ---------------------------------------------------------------------------
# 3. Anthropic thinking
# ---------------------------------------------------------------------------

class TestAnthropicThinking(unittest.TestCase):
    """The thinking payload shape depends on the model generation."""

    def _provider(self):
        return _provider_class("anthropic", "AnthropicProvider")(
            ProviderConfig(), api_key="fake-key"
        )

    def test_model_version_parsing(self) -> None:
        """The version sits before or after the role depending on the era."""
        provider = self._provider()
        cases = {
            "claude-3-5-haiku-latest": (3, 5),
            "claude-sonnet-4-5": (4, 5),
            "claude-opus-4-6": (4, 6),
            "claude-opus-5": (5, 0),
            "claude-fable-5": (5, 0),
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(provider._model_version(model), expected)

    def test_dated_ids_do_not_read_the_date_as_a_minor(self) -> None:
        """Regression: the release date was parsed as the minor version.

        `claude-opus-4-20250514` carries no minor, but the pattern used to take
        the date as one and return (4, 20250514). That clears every threshold,
        so the whole Claude 4.0 line was sent the adaptive form that
        `_ADAPTIVE_SINCE` exists to withhold, and thinking=True was a hard 400
        on exactly the models people are calling today.
        """
        provider = self._provider()
        cases = {
            "claude-opus-4-20250514": (4, 0),
            "claude-sonnet-4-20250514": (4, 0),
            "claude-opus-4-1-20250805": (4, 1),
            "claude-3-5-haiku-20241022": (3, 5),
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(provider._model_version(model), expected)
                self.assertFalse(provider._uses_adaptive_thinking(model))

    def test_older_models_get_the_budget_form(self) -> None:
        """Regression: 'adaptive' is a 400 on everything below Claude 4.6.

        The provider used to send it unconditionally, so thinking=True was
        broken on every model up to and including 4.5.
        """
        provider = self._provider()
        for model in ("claude-3-5-haiku-latest", "claude-sonnet-4-5"):
            with self.subTest(model=model):
                payload = provider._build_thinking_payload(True, model, 8192)
                self.assertEqual(payload["type"], "enabled")
                self.assertIn("budget_tokens", payload)
                self.assertLess(payload["budget_tokens"], 8192)
                self.assertGreaterEqual(payload["budget_tokens"], 1024)

    def test_newer_models_get_the_adaptive_form(self) -> None:
        provider = self._provider()
        for model in ("claude-sonnet-4-6", "claude-opus-4-7", "claude-opus-5"):
            with self.subTest(model=model):
                self.assertEqual(
                    provider._build_thinking_payload(True, model, 8192),
                    {"type": "adaptive"},
                )

    def test_budget_respects_the_documented_floor(self) -> None:
        """budget_tokens has a documented minimum of 1024."""
        provider = self._provider()
        payload = provider._build_thinking_payload(True, "claude-sonnet-4-5", 512)
        self.assertEqual(payload["budget_tokens"], 1024)

    def test_thinking_false(self) -> None:
        """Only the newer form can be explicitly disabled.

        On older models thinking is off unless enabled, so omitting the field
        is what "off" means there.
        """
        provider = self._provider()
        self.assertEqual(
            provider._build_thinking_payload(False, "claude-opus-5", 8192),
            {"type": "disabled"},
        )
        self.assertIsNone(
            provider._build_thinking_payload(False, "claude-sonnet-4-5", 8192)
        )

    def test_thinking_default_sends_nothing(self) -> None:
        provider = self._provider()
        for model in ("claude-sonnet-4-5", "claude-opus-5"):
            with self.subTest(model=model):
                self.assertIsNone(
                    provider._build_thinking_payload("default", model, 8192)
                )

    def test_unrecognisable_model_assumes_the_newer_form(self) -> None:
        """Guessing new breaks only models on their way out."""
        provider = self._provider()
        self.assertEqual(
            provider._build_thinking_payload(True, "some-unknown-model", 8192),
            {"type": "adaptive"},
        )

    def test_payload_reaches_the_request(self) -> None:
        provider = self._provider()
        captured: dict = {}

        def fake_post(payload: dict, timeout: int) -> dict:
            captured.update(payload)
            return {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        request = AiRequest(
            provider="anthropic", model="claude-opus-5", prompt="p", thinking=True
        )
        with patch.object(provider, "_post", side_effect=fake_post):
            provider.call(request)

        self.assertEqual(captured["thinking"], {"type": "adaptive"})


# ---------------------------------------------------------------------------
# 4. OpenAI-compatible reasoning control
# ---------------------------------------------------------------------------

class TestOpenAiCompatReasoning(unittest.TestCase):
    """thinking maps to reasoning_effort, but only when asked explicitly."""

    def _capture(self, module, class_name, thinking, extra_options=None,
                 api_key="fake") -> dict:
        """Run call() with _post intercepted and return the payload sent."""
        cls = _provider_class(module, class_name)
        provider = (
            cls(ProviderConfig(), api_key=api_key)
            if api_key
            else cls(ProviderConfig())
        )
        captured: dict = {}

        def fake_post(endpoint: str, payload: dict, timeout: int) -> dict:
            captured.update(payload)
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

        request = AiRequest(
            provider="x", model="m", prompt="p",
            thinking=thinking, extra_options=extra_options,
        )
        with patch.object(provider, "_post", side_effect=fake_post):
            provider.call(request)
        return captured

    def test_supported_providers_map_thinking(self) -> None:
        for module, class_name in (
            ("openai", "OpenAiProvider"),
            ("mistral", "MistralProvider"),
            ("groq", "GroqProvider"),
            ("xai", "XAiProvider"),
        ):
            with self.subTest(provider=module):
                self.assertEqual(
                    self._capture(module, class_name, True)["reasoning_effort"],
                    "high",
                )
                self.assertEqual(
                    self._capture(module, class_name, False)["reasoning_effort"],
                    "none",
                )

    def test_default_sends_nothing(self) -> None:
        """Critical: a non-reasoning model rejects the parameter outright.

        "default" is what call_ai() sends unless the caller says otherwise, so
        leaving it out is what keeps plain models such as gpt-4o working.
        """
        captured = self._capture("openai", "OpenAiProvider", "default")
        self.assertNotIn("reasoning_effort", captured)

    def test_providers_without_the_control_send_nothing(self) -> None:
        for module, class_name, api_key in (
            ("cohere", "CohereProvider", "fake"),
            ("meta", "MetaProvider", "fake"),
            ("lmstudio", "LmStudioProvider", ""),
            ("llamacpp", "LlamaCppProvider", ""),
        ):
            with self.subTest(provider=module):
                captured = self._capture(module, class_name, True, api_key=api_key)
                self.assertNotIn("reasoning_effort", captured)

    def test_extra_options_overrides_the_mapping(self) -> None:
        """The unified lever is coarse on purpose; extra_options is the escape."""
        captured = self._capture(
            "openai", "OpenAiProvider", True,
            extra_options={"reasoning_effort": "medium"},
        )
        self.assertEqual(captured["reasoning_effort"], "medium")


# ---------------------------------------------------------------------------
# 5. Embeddings
# ---------------------------------------------------------------------------

class TestGoogleEmbeddings(unittest.TestCase):
    """Google does serve embeddings, so the provider implements them."""

    def _provider(self, **config_kwargs):
        return _provider_class("google", "GoogleProvider")(
            config=ProviderConfig(**config_kwargs), api_key="dummy-key"
        )

    def test_embedding_returns_floats(self) -> None:
        provider = self._provider()
        fake_client = MagicMock()
        fake_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.1, 0.2, 0.3])]
        )

        with patch.object(provider, "_get_client", return_value=fake_client):
            vector = provider.get_embedding("gemini-embedding-001", "hello")

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        self.assertTrue(all(isinstance(x, float) for x in vector))
        fake_client.models.embed_content.assert_called_once()

    def test_embedding_raises_when_the_api_returns_nothing(self) -> None:
        provider = self._provider()
        fake_client = MagicMock()
        fake_client.models.embed_content.return_value = MagicMock(embeddings=[])

        with patch.object(provider, "_get_client", return_value=fake_client):
            with self.assertRaises(RuntimeError):
                provider.get_embedding("gemini-embedding-001", "hello")

    def test_embedding_config_comes_from_extra_options(self) -> None:
        provider = self._provider(
            extra_options={
                "task_type": "RETRIEVAL_QUERY",
                "output_dimensionality": 256,
            }
        )
        fake_client = MagicMock()
        fake_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.1])]
        )

        with patch.object(provider, "_get_client", return_value=fake_client):
            provider.get_embedding("gemini-embedding-001", "hello")

        config = fake_client.models.embed_content.call_args.kwargs["config"]
        self.assertEqual(config.task_type, "RETRIEVAL_QUERY")
        self.assertEqual(config.output_dimensionality, 256)

    def test_anthropic_still_declines_embeddings(self) -> None:
        """Anthropic offers no embeddings API of its own."""
        provider = _provider_class("anthropic", "AnthropicProvider")(
            ProviderConfig(), api_key="fake"
        )
        with self.assertRaises(NotImplementedError):
            provider.get_embedding("any", "text")


class TestGoogleEmbeddingsLive(unittest.TestCase):
    """Real embedding call, skipped without a key."""

    def test_google_live_embedding(self) -> None:
        from unified_ai_client.config import load_secrets
        if not load_secrets(os.getcwd()).get("google_api_key"):
            self.skipTest(
                "google_api_key not found in secrets.json or environment variables"
            )
        from unified_ai_client import get_embedding

        vector = get_embedding(
            provider="google", model="gemini-embedding-001", text="hello world"
        )
        self.assertIsInstance(vector, list)
        self.assertGreater(len(vector), 0)
        self.assertTrue(all(isinstance(x, float) for x in vector))


# ---------------------------------------------------------------------------
# 6. File handling: what each provider accepts, and how it refuses the rest
# ---------------------------------------------------------------------------

class FileFixtureCase(ProviderRegistryIsolation):
    """Cleans up the temp files a test creates, and the registry it touches.

    Registry isolation matters here because the end-to-end tests call
    ``call_ai()``, which caches provider instances in ``client._PROVIDERS``.
    These classes sort before ``TestProviderUrlResolution``, which reads
    ``get_provider(name).base_url``, so a leaked instance would decide a later
    test's result.
    """

    def setUp(self) -> None:
        super().setUp()
        self._paths: list[str] = []

    def tearDown(self) -> None:
        for path in self._paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        super().tearDown()

    def make(self, suffix: str, data: bytes = b"x") -> str:
        """Write a temp file with the given extension and return its path."""
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(data)
        tmp.close()
        self._paths.append(tmp.name)
        return tmp.name

    def sample(self, suffix: str) -> str:
        """A shared file for the given extension, created at most once.

        ``validate_files()`` reads only the path's existence and extension, so
        every check for a given class wants the same file rather than its own.
        """
        cache = self.__dict__.setdefault("_samples", {})
        if suffix not in cache:
            cache[suffix] = self.make(suffix)
        return cache[suffix]


class TestUnloadSupportMatrix(unittest.TestCase):
    """Which providers hold a model resident, and therefore can release one.

    The table is the specification, and it covers every registered provider so
    that a provider added without a row here fails the suite rather than
    silently defaulting to "holds nothing".
    """

    _SUPPORTS_UNLOAD = {
        "ollama": True,
        "script": True,
        "google": False,
        "anthropic": False,
        "openai": False,
        "mistral": False,
        "cohere": False,
        "meta": False,
        "groq": False,
        "xai": False,
        "lmstudio": False,
        "llamacpp": False,
    }

    def test_table_covers_every_registered_provider(self) -> None:
        self.assertEqual(
            set(self._SUPPORTS_UNLOAD), set(_PROVIDER_CLASSES),
            "a provider was added or removed without updating this table",
        )

    def test_declared_support_matches_the_table(self) -> None:
        for name, expected in self._SUPPORTS_UNLOAD.items():
            with self.subTest(provider=name):
                cls = _provider_class_by_name(name)
                self.assertIs(cls.SUPPORTS_UNLOAD, expected)

    def test_every_provider_exposes_the_hook(self) -> None:
        """Concrete on the base class, so no provider can fail to have it."""
        for name in self._SUPPORTS_UNLOAD:
            with self.subTest(provider=name):
                cls = _provider_class_by_name(name)
                self.assertTrue(callable(cls.unload_model))

    def test_only_declaring_providers_override_it(self) -> None:
        """The flag and the implementation must not drift apart.

        A provider that overrides unload_model without declaring the flag would
        never be asked to unload; one that declares it without overriding would
        silently do nothing at cleanup.
        """
        from unified_ai_client.providers.base import BaseProvider

        for name, declared in self._SUPPORTS_UNLOAD.items():
            with self.subTest(provider=name):
                cls = _provider_class_by_name(name)
                overrides = cls.unload_model is not BaseProvider.unload_model
                self.assertIs(overrides, declared)


class TestFileSupportMatrix(FileFixtureCase):
    """Every provider declares what it can carry, and refuses the rest.

    The table is the specification: each provider maps to the file classes it
    can transmit natively. 'text' is deliberately absent everywhere, being
    inlined into the prompt rather than carried in a native block.
    """

    _SUPPORT = {
        "google": {"image", "audio", "document"},
        "openai": {"image", "audio", "document"},
        "anthropic": {"image", "document"},
        "llamacpp": {"image", "audio"},
        "ollama": {"image"},
        "mistral": {"image"},
        "cohere": {"image"},
        "meta": {"image"},
        "groq": {"image"},
        "xai": {"image"},
        "lmstudio": {"image"},
        # Empty by contract: only the script knows what it can open, so it
        # declares nothing and never calls validate_files(). See
        # docs/script-protocol.md.
        "script": frozenset(),
    }

    _SAMPLE = {"image": ".png", "audio": ".mp3", "document": ".pdf"}

    def test_table_covers_every_registered_provider(self) -> None:
        self.assertEqual(
            set(self._SUPPORT), set(_PROVIDER_CLASSES),
            "a provider was added or removed without updating this table",
        )

    def test_declared_support_matches_the_table(self) -> None:
        for name, expected in self._SUPPORT.items():
            with self.subTest(provider=name):
                cls = _provider_class_by_name(name)
                self.assertEqual(set(cls.SUPPORTED_FILE_TYPES), expected)

    def test_unsupported_classes_raise(self) -> None:
        """The refusal must name the file and the provider, not just fail."""
        from unified_ai_client.exceptions import UnsupportedFileError
        from unified_ai_client.file_utils import validate_files

        for name, supported in self._SUPPORT.items():
            for file_type, suffix in self._SAMPLE.items():
                if file_type in supported:
                    continue
                with self.subTest(provider=name, file_type=file_type):
                    path = self.sample(suffix)
                    with self.assertRaises(UnsupportedFileError) as ctx:
                        validate_files([path], name, frozenset(supported))
                    message = str(ctx.exception)
                    self.assertIn(path, message)
                    self.assertIn(name, message)

    def test_supported_classes_pass(self) -> None:
        from unified_ai_client.file_utils import validate_files

        for name, supported in self._SUPPORT.items():
            for file_type in supported:
                with self.subTest(provider=name, file_type=file_type):
                    path = self.sample(self._SAMPLE[file_type])
                    self.assertEqual(
                        validate_files([path], name, frozenset(supported)),
                        [(path, file_type)],
                    )

    def test_script_passes_unsupported_files_straight_through(self) -> None:
        """The one provider that must NOT refuse: only the script knows.

        Its empty SUPPORTED_FILE_TYPES row above says what it declares; this
        says what it does. The passthrough is a public contract
        (docs/script-protocol.md), and without this the row would read as
        "refuses everything", which is the opposite of the truth.
        """
        from unified_ai_client.models import AiRequest
        from unified_ai_client.providers import script as script_module
        from unified_ai_client.providers.script import ScriptProvider

        audio = self.sample(".mp3")
        captured: dict = {}

        def fake_run(cmd, payload, timeout):
            captured.update(payload)
            return {"text": "ok"}

        provider = ScriptProvider(ProviderConfig())
        request = AiRequest(provider="script", model="fake.py", prompt="hi",
                            file_path=audio)
        with patch.object(script_module, "_run_script", side_effect=fake_run):
            provider.call(request)

        self.assertEqual(captured["file_path"], [audio])

    def test_text_is_accepted_everywhere(self) -> None:
        """No provider has a native text block, so inlining must stay open."""
        from unified_ai_client.file_utils import validate_files

        for name, supported in self._SUPPORT.items():
            with self.subTest(provider=name):
                path = self.sample(".md")
                self.assertEqual(
                    validate_files([path], name, frozenset(supported)),
                    [(path, "text")],
                )

    def test_unknown_extensions_raise(self) -> None:
        """An unrecognised file is refused rather than guessed at as text."""
        from unified_ai_client.exceptions import UnsupportedFileError
        from unified_ai_client.file_utils import validate_files

        path = self.make(".bin", bytes([0, 1, 2]))
        with self.assertRaises(UnsupportedFileError):
            validate_files([path], "groq", frozenset({"image"}))

    def test_a_missing_file_is_reported_before_anything_is_read(self) -> None:
        from unified_ai_client.file_utils import validate_files

        good = self.make(".png")
        with self.assertRaises(FileNotFoundError):
            validate_files([good, "no-such-file.png"], "groq", frozenset({"image"}))

    def test_script_validates_nothing(self) -> None:
        """The script owns its own type policy; the library must not guess."""
        cls = _provider_class_by_name("script")
        self.assertEqual(set(cls.SUPPORTED_FILE_TYPES), set())


class TestUnsupportedFilesAreNotRetried(FileFixtureCase):
    """A rejected attachment must fail at once, not after the backoff budget."""

    def test_with_retry_reraises_immediately(self) -> None:
        from unified_ai_client.exceptions import UnsupportedFileError
        from unified_ai_client.retry import with_retry

        calls = {"n": 0}

        def always_unsupported() -> None:
            calls["n"] += 1
            raise UnsupportedFileError("nope")

        with self.assertRaises(UnsupportedFileError):
            with_retry(always_unsupported, max_retries=3, base_delay=0.01)
        self.assertEqual(calls["n"], 1)

    def test_a_bad_attachment_fails_call_ai_without_backoff(self) -> None:
        """The end-to-end path, not just the validator.

        Validating in the right place is not enough: the error also has to be
        one with_retry() declines to retry. Raising a plain FileNotFoundError
        here cost three backoff rounds before the caller heard about a path that
        was never going to appear.
        """
        import time
        from unified_ai_client import call_ai
        from unified_ai_client.exceptions import UnsupportedFileError

        cases = (
            ("missing path", "no-such-file.png", FileNotFoundError),
            ("unsupported type", self.make(".mp3"), UnsupportedFileError),
        )
        for label, path, expected in cases:
            with self.subTest(case=label):
                started = time.monotonic()
                with self.assertRaises(expected):
                    call_ai(
                        provider="groq",
                        model="irrelevant",
                        prompt="hi",
                        file_path=path,
                        max_retries=3,
                        retry_base_delay=5.0,
                    )
                self.assertLess(
                    time.monotonic() - started,
                    2.0,
                    "attachment errors must surface immediately, not after backoff",
                )

    def test_ordinary_failures_are_still_retried(self) -> None:
        """The non-retry path must not disable retrying in general."""
        from unified_ai_client.retry import with_retry

        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return "ok"

        self.assertEqual(with_retry(flaky, max_retries=3, base_delay=0.01), "ok")
        self.assertEqual(calls["n"], 3)


class TestTextAttachmentDecoding(FileFixtureCase):
    """A text attachment that is not text must not become placeholder prose."""

    def test_undecodable_text_file_raises(self) -> None:
        from unified_ai_client.exceptions import FileDecodeError
        from unified_ai_client.file_utils import format_text_attachment

        path = self.make(".txt", bytes([0xFF, 0xFE, 0x00, 0x80]) + b"binary")
        with self.assertRaises(FileDecodeError):
            format_text_attachment(path)

    def test_no_placeholder_text_survives_in_the_library(self) -> None:
        """The old fallback reached the model as if it were file content."""
        import unified_ai_client.file_utils as file_utils

        source = Path(file_utils.__file__).read_text(encoding="utf-8")
        self.assertNotIn("could not be read as text", source)


class TestNativeBlockShapes(FileFixtureCase):
    """The block a provider emits must match what its API actually accepts."""

    def test_openai_pdf_uses_the_chat_completions_file_block(self) -> None:
        """'input_file' is the Responses API name and is rejected here."""
        provider = _provider_class("openai", "OpenAiProvider")(
            ProviderConfig(), api_key="fake"
        )
        path = self.make(".pdf", b"%PDF-1.4 fake")
        block = provider._build_native_block(path, "document")

        self.assertEqual(block["type"], "file")
        self.assertIn("file_data", block["file"])
        self.assertTrue(
            block["file"]["file_data"].startswith("data:application/pdf;base64,")
        )

    def test_llamacpp_audio_uses_an_input_audio_block(self) -> None:
        provider = _provider_class("llamacpp", "LlamaCppProvider")(ProviderConfig())
        path = self.make(".wav", b"RIFFfake")
        block = provider._build_native_block(path, "audio")

        self.assertEqual(block["type"], "input_audio")
        self.assertEqual(block["input_audio"]["format"], "wav")
        self.assertTrue(block["input_audio"]["data"])

    def test_image_blocks_carry_a_data_url(self) -> None:
        provider = _provider_class("groq", "GroqProvider")(
            ProviderConfig(), api_key="fake"
        )
        path = self.make(".png", b"\x89PNG")
        block = provider._build_native_block(path, "image")

        self.assertEqual(block["type"], "image_url")
        self.assertTrue(
            block["image_url"]["url"].startswith("data:image/png;base64,")
        )


class TestEmptyPromptWithAnAttachment(FileFixtureCase):
    """prompt='' plus a file must not send an empty text block or Part.

    Regression: anthropic and google both appended a text block unconditionally,
    so `call_ai(m, prompt="", file_path="photo.png")` -- a legitimate "describe
    this" call with the instruction in the system prompt -- produced an empty
    text block or Part that the API rejects.
    """

    def test_anthropic_omits_the_empty_text_block_with_an_image(self) -> None:
        provider = _provider_class("anthropic", "AnthropicProvider")(
            ProviderConfig(), api_key="k"
        )
        path = self.make(".png", b"\x89PNG")
        content = provider._build_user_content("", [path])
        self.assertTrue(all(block["type"] != "text" for block in content))

    def test_anthropic_still_inlines_a_text_attachment(self) -> None:
        """The guard is "nothing to say", not "the prompt string was empty"."""
        provider = _provider_class("anthropic", "AnthropicProvider")(
            ProviderConfig(), api_key="k"
        )
        path = self.make(".md", b"important context")
        content = provider._build_user_content("", [path])
        text_blocks = [b for b in content if b["type"] == "text"]
        self.assertEqual(len(text_blocks), 1)
        self.assertIn("important context", text_blocks[0]["text"])

    def test_anthropic_never_sends_an_empty_content_array(self) -> None:
        """The safety valve: genuinely nothing to say still needs one block."""
        provider = _provider_class("anthropic", "AnthropicProvider")(
            ProviderConfig(), api_key="k"
        )
        content = provider._build_user_content("", [])
        self.assertEqual(len(content), 1)

    def _google_call(self, prompt: str, file_path: str | None):
        from types import SimpleNamespace
        from unified_ai_client.providers.google import GoogleProvider

        provider = GoogleProvider(ProviderConfig(), api_key="fake-key")
        fake_response = MagicMock()
        fake_response.candidates = []
        fake_response.prompt_feedback = None
        fake_response.text = "ok"
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        request = AiRequest(
            provider="google", model="gemini-2.5-flash", prompt=prompt,
            file_path=file_path, timeout=30,
        )
        # _upload_file is patched with a real .uri/.mime_type, which
        # _build_parts_for_files needs to build a Part: a bare MagicMock
        # fails pydantic validation, and the real upload path would poll for
        # ACTIVE state for real seconds against a mock that never satisfies it.
        fake_ref = SimpleNamespace(uri="gs://bucket/f", mime_type="image/png")
        with patch.object(GoogleProvider, "_get_client", return_value=fake_client), \
                patch.object(provider, "_upload_file", return_value=fake_ref):
            provider.call(request)

        contents = fake_client.models.generate_content.call_args.kwargs["contents"]
        return contents[-1].parts

    def test_google_omits_the_empty_text_part_with_an_image(self) -> None:
        path = self.make(".png", b"\x89PNG")
        parts = self._google_call("", path)
        self.assertFalse(any(p.text == "" for p in parts))

    def test_google_never_sends_an_empty_parts_list(self) -> None:
        parts = self._google_call("", None)
        self.assertEqual(len(parts), 1)


class TestToolMessageKeepsItsCallId(ProviderRegistryIsolation):
    """A history 'tool' message without tool_call_id is a 400 waiting to happen.

    docs/tool-calling.md only documents the two-turn exchange, where
    tool_results carries the id for the library. A consumer building a third
    turn has to reconstruct the 'tool' message itself, and until now that id
    was read from the history dict and then silently dropped.
    """

    def test_the_id_reaches_the_payload(self) -> None:
        history = [{
            "role": "tool",
            "content": "22°C and sunny",
            "tool_call_id": "call_abc123",
        }]
        payload = _capture_http_payload(
            "openai", ProviderConfig(), messages=history
        )
        self.assertEqual(payload["messages"][0]["tool_call_id"], "call_abc123")

    def test_a_missing_id_is_not_invented(self) -> None:
        """No fallback: a wrong link is worse than a visible KeyError upstream."""
        history = [{"role": "tool", "content": "22°C and sunny"}]
        payload = _capture_http_payload(
            "openai", ProviderConfig(), messages=history
        )
        self.assertNotIn("tool_call_id", payload["messages"][0])


class TestHistoryMessageKeepsEverything(FileFixtureCase):
    """text + files + tool_calls on one history message must all survive.

    Regression, the same shape in four places: anthropic returned early inside
    its tool_calls branch without ever looking at file_paths; google's tool_calls
    branch was the 'if' half of an if/else whose 'else' appended the text Part;
    ollama's file branch built its own entry dict and never checked tool_calls.
    Three different bugs from the same cause -- an exclusive branch where the
    fields should have been independent -- which is exactly the shape that
    produced the .mp3 bug CLAUDE.md documents.
    """

    def _history(self, path: str) -> list[dict]:
        return [{
            "role": "assistant",
            "content": "let me check the weather",
            "files": [path],
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Rome"}}}
            ],
        }]

    def test_http_adapters_keep_text_file_and_tool_calls(self) -> None:
        path = self.make(".png", b"\x89PNG")
        for name in ("openai", "anthropic", "ollama"):
            with self.subTest(provider=name):
                payload = _capture_http_payload(
                    name, ProviderConfig(), messages=self._history(path)
                )
                msg = payload["messages"][0]
                serialized = str(msg)
                with self.subTest(field="text"):
                    self.assertIn("let me check the weather", serialized)
                with self.subTest(field="tool_calls"):
                    self.assertIn("get_weather", serialized)
                with self.subTest(field="file"):
                    if name == "ollama":
                        self.assertTrue(msg.get("images"))
                    else:
                        blocks = msg["content"] if isinstance(msg["content"], list) else []
                        self.assertTrue(
                            any(b.get("type") in ("image", "image_url") for b in blocks)
                        )

    def test_google_keeps_text_file_and_tool_calls(self) -> None:
        from types import SimpleNamespace
        from unified_ai_client.providers.google import GoogleProvider

        provider = GoogleProvider(ProviderConfig(), api_key="fake-key")
        fake_response = MagicMock()
        fake_response.candidates = []
        fake_response.prompt_feedback = None
        fake_response.text = "ok"
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        path = self.make(".png", b"\x89PNG")
        request = AiRequest(
            provider="google", model="gemini-2.5-flash", prompt="ignored",
            messages=self._history(path), timeout=30,
        )
        # _upload_file is patched rather than the raw client: _build_parts_for_files
        # needs a real .uri/.mime_type to build a Part, which a bare MagicMock
        # cannot provide, and the real upload path would poll for ACTIVE state
        # for real seconds against a mock that never satisfies it.
        fake_ref = SimpleNamespace(uri="gs://bucket/photo.png", mime_type="image/png")
        with patch.object(GoogleProvider, "_get_client", return_value=fake_client), \
                patch.object(provider, "_upload_file", return_value=fake_ref):
            provider.call(request)

        contents = fake_client.models.generate_content.call_args.kwargs["contents"]
        history_turn = contents[0]
        serialized = str(history_turn)
        with self.subTest(field="text"):
            self.assertTrue(any(p.text == "let me check the weather" for p in history_turn.parts))
        with self.subTest(field="tool_calls"):
            self.assertIn("get_weather", serialized)
        with self.subTest(field="file"):
            self.assertIn("photo.png", serialized)


class TestProviderRefusalsEndToEnd(FileFixtureCase):
    """The refusal must happen while building content, not at the transport."""

    def test_ollama_refuses_audio(self) -> None:
        """images[] carries images only; audio there is silently ignored."""
        from unified_ai_client.exceptions import UnsupportedFileError

        provider = _provider_class("ollama", "OllamaProvider")(ProviderConfig())
        path = self.make(".mp3")
        with self.assertRaises(UnsupportedFileError):
            provider._process_files_for_message([path], "what do you hear?")

    def test_ollama_still_carries_images_and_text(self) -> None:
        provider = _provider_class("ollama", "OllamaProvider")(ProviderConfig())
        image = self.make(".png", b"\x89PNG")
        notes = self.make(".md", b"green")

        prompt, images = provider._process_files_for_message(
            [image, notes], "describe"
        )
        self.assertEqual(len(images), 1)
        self.assertIn("green", prompt)
        self.assertEqual(prompt.count("describe"), 1)

    def test_anthropic_refuses_audio_instead_of_dropping_it(self) -> None:
        """It used to log a warning and answer as if it had heard the file."""
        from unified_ai_client.exceptions import UnsupportedFileError

        provider = _provider_class("anthropic", "AnthropicProvider")(
            ProviderConfig(), api_key="fake"
        )
        path = self.make(".wav")
        with self.assertRaises(UnsupportedFileError):
            provider._build_user_content("transcribe this", [path])

    def test_compat_providers_refuse_audio_and_pdf(self) -> None:
        from unified_ai_client.exceptions import UnsupportedFileError

        provider = _provider_class("groq", "GroqProvider")(
            ProviderConfig(), api_key="fake"
        )
        for suffix in (".mp3", ".pdf"):
            with self.subTest(suffix=suffix):
                path = self.make(suffix)
                with self.assertRaises(UnsupportedFileError):
                    provider._build_user_content("summarise", [path])

    def test_extensionless_text_files_are_inlined_not_refused(self) -> None:
        """A Dockerfile is among the most common things handed to an LLM.

        It has no extension, so before the name lookup existed it classified as
        unknown and the strict policy refused it.
        """
        provider = _provider_class_by_name("groq")(ProviderConfig(), api_key="fake")
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = os.path.join(directory, "Dockerfile")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("FROM python:3.12-slim")
        self._paths.append(path)

        content = provider._build_user_content("Review this image build.", [path])
        self.assertIsInstance(content, str)
        self.assertIn("python:3.12-slim", content)
        self.assertIn("Dockerfile", content)

    def test_compat_providers_still_inline_text(self) -> None:
        provider = _provider_class("groq", "GroqProvider")(
            ProviderConfig(), api_key="fake"
        )
        path = self.make(".md", b"the colour is blue")
        content = provider._build_user_content("what colour?", [path])

        self.assertIsInstance(content, str)
        self.assertIn("blue", content)
        self.assertIn("what colour?", content)


class TestFileHandlingLive(FileFixtureCase):
    """Real requests confirming the block shapes the offline tests assert.

    These are the checks the documentation could not settle. Each skips when
    its provider is unreachable, so a normal run stays offline.
    """

    def _llamacpp_available(self) -> bool:
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:8080/v1/models", timeout=2)
            return True
        except Exception:
            return False

    def test_llamacpp_live_audio(self) -> None:
        """Confirm this build routes input_audio rather than rejecting it.

        Upstream evidence is contradictory: llama.cpp's server README documents
        the block, while the issue tracking it was closed as not planned. Only
        the running server settles it.
        """
        if not self._llamacpp_available():
            self.skipTest("llama.cpp server not reachable at localhost:8080")

        from unified_ai_client import call_ai

        # A minimal valid WAV: 44-byte header plus one sample of silence.
        header = (
            b"RIFF" + (36 + 2).to_bytes(4, "little") + b"WAVEfmt "
            + (16).to_bytes(4, "little")
            + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
            + (8000).to_bytes(4, "little") + (16000).to_bytes(4, "little")
            + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
            + b"data" + (2).to_bytes(4, "little") + (0).to_bytes(2, "little")
        )
        path = self.make(".wav", header)

        response = call_ai(
            provider="llamacpp",
            model="local",
            prompt="Reply with the single word OK.",
            file_path=path,
            temperature=0.0,
            timeout=60,
            max_retries=1,
        )
        self.assertIsInstance(response.text, str)

    def test_openai_live_pdf(self) -> None:
        """The 'file' block must be accepted where 'input_file' was not."""
        from unified_ai_client.config import load_secrets
        if not load_secrets(os.getcwd()).get("openai_api_key"):
            self.skipTest(
                "openai_api_key not found in secrets.json or environment variables"
            )
        from unified_ai_client import call_ai

        pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 99 99]>>endobj\n"
            b"trailer<</Root 1 0 R>>\n"
        )
        path = self.make(".pdf", pdf)

        response = call_ai(
            provider="openai",
            model="gpt-4o-mini",
            prompt="Reply with exactly the word READ and nothing else.",
            file_path=path,
            temperature=0.0,
            timeout=60,
            max_retries=1,
        )
        self.assertIsInstance(response.text, str)

    def test_cohere_live_image(self) -> None:
        """The one support-table row the documentation left ambiguous."""
        from unified_ai_client.config import load_secrets
        if not load_secrets(os.getcwd()).get("cohere_api_key"):
            self.skipTest(
                "cohere_api_key not found in secrets.json or environment variables"
            )
        from unified_ai_client import call_ai

        # 1x1 transparent PNG.
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYAAAAAM"
            "AASsJTYQAAAAASUVORK5CYII="
        )
        path = self.make(".png", png)

        response = call_ai(
            provider="cohere",
            model="command-a-vision-07-2025",
            prompt="Reply with exactly the word SEEN and nothing else.",
            file_path=path,
            temperature=0.0,
            timeout=60,
            max_retries=1,
        )
        self.assertIsInstance(response.text, str)


# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------

_OK_RESPONSE = {
    "choices": [{"message": {"content": "x"}}],
    "usage": {},
    "content": [{"type": "text", "text": "x"}],
    "message": {"content": "x"},
}


class TestSamplingParameters(unittest.TestCase):
    """What reaches the wire when the caller asks for nothing.

    top_k and top_p used to default to 64 and 0.95 on AiRequest, so the
    `if ... is not None` guards in every adapter were always true and both
    values were sent on every request. OpenAI's Chat Completions API rejects
    top_k as an unknown argument, so the provider was unusable with library
    defaults, and no test looked at the set of payload keys.
    """

    def _openai_payload(self, **request_kwargs) -> dict:
        from unified_ai_client.providers.openai import OpenAiProvider

        provider = OpenAiProvider(ProviderConfig(), api_key="fake-key")
        seen = {}

        def fake_post(self, endpoint, payload, timeout):
            seen["payload"] = payload
            return _OK_RESPONSE

        with patch.object(OpenAiProvider, "_post", fake_post):
            provider.call(AiRequest(
                provider="openai", model="gpt-4o", prompt="hi",
                timeout=30, **request_kwargs,
            ))
        return seen["payload"]

    def _anthropic_payload(self, **request_kwargs) -> dict:
        from unified_ai_client.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(ProviderConfig(), api_key="fake-key")
        seen = {}

        def fake_post(self, payload, timeout):
            seen["payload"] = payload
            return _OK_RESPONSE

        with patch.object(AnthropicProvider, "_post", fake_post):
            provider.call(AiRequest(
                provider="anthropic", model="claude-opus-4-6", prompt="hi",
                timeout=30, **request_kwargs,
            ))
        return seen["payload"]

    def _ollama_options(self, **request_kwargs) -> dict:
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(ProviderConfig())
        seen = {}

        def fake_post(self, endpoint, payload, timeout):
            seen["payload"] = payload
            return _OK_RESPONSE

        with patch.object(OllamaProvider, "_post", fake_post):
            provider.call(AiRequest(
                provider="ollama", model="m", prompt="hi",
                timeout=30, **request_kwargs,
            ))
        return seen["payload"]["options"]

    def test_openai_is_sent_no_top_k(self) -> None:
        """Regression: an unasked-for top_k made every openai call a 400."""
        self.assertNotIn("top_k", self._openai_payload())

    def test_openai_sends_top_k_when_asked(self) -> None:
        """Omitting by default must not mean ignoring an explicit value."""
        self.assertEqual(self._openai_payload(top_k=10)["top_k"], 10)

    def test_config_level_top_p_is_reachable(self) -> None:
        """Regression: `opts.get("top_p")` was dead code.

        request.top_p was never None, so the fallback to a value registered
        via configure_provider() could not be reached.
        """
        from unified_ai_client.providers.openai import OpenAiProvider

        provider = OpenAiProvider(
            ProviderConfig(extra_options={"top_p": 0.1}), api_key="fake-key"
        )
        seen = {}

        def fake_post(self, endpoint, payload, timeout):
            seen["payload"] = payload
            return _OK_RESPONSE

        with patch.object(OpenAiProvider, "_post", fake_post):
            provider.call(AiRequest(
                provider="openai", model="gpt-4o", prompt="hi", timeout=30,
            ))
        self.assertEqual(seen["payload"]["top_p"], 0.1)

    def test_ollama_keeps_its_house_defaults(self) -> None:
        """The local provider's output must not change with this.

        Ollama's own defaults are 40/0.9, so falling through to them would
        quietly alter generation for every existing caller.
        """
        options = self._ollama_options()
        self.assertEqual(options["top_k"], 64)
        self.assertEqual(options["top_p"], 0.95)

    def test_ollama_still_honours_an_explicit_value(self) -> None:
        self.assertEqual(self._ollama_options(top_k=10)["top_k"], 10)

    def _ollama_options_with_config(self, config: ProviderConfig, **request_kwargs) -> dict:
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(config)
        seen = {}

        def fake_post(self, endpoint, payload, timeout):
            seen["payload"] = payload
            return _OK_RESPONSE

        with patch.object(OllamaProvider, "_post", fake_post):
            provider.call(AiRequest(
                provider="ollama", model="m", prompt="hi",
                timeout=30, **request_kwargs,
            ))
        return seen["payload"]["options"]

    def test_ollama_explicit_call_time_value_beats_config_extra_options(self) -> None:
        """Regression: extra_options from configure_provider() won over an
        explicit call-time top_k, the opposite of the documented precedence.

        _fold_options(opts, options) used to run after temperature/top_k/
        top_p were set from the request, unconditionally overwriting them
        with whatever configure_provider() had registered.
        """
        config = ProviderConfig(extra_options={"top_k": 999, "top_p": 0.01})
        options = self._ollama_options_with_config(config, top_k=10, top_p=0.5)
        self.assertEqual(options["top_k"], 10)
        self.assertEqual(options["top_p"], 0.5)

    def test_ollama_config_extra_options_still_apply_when_nothing_else_does(self) -> None:
        """The config-level value must still reach the payload on its own."""
        config = ProviderConfig(extra_options={"top_k": 999})
        options = self._ollama_options_with_config(config)
        self.assertEqual(options["top_k"], 999)

    def test_ollama_call_time_extra_options_beat_config_when_no_param_given(self) -> None:
        """extra_options set at call time override the same key from
        configure_provider(), the ordering docs/configuration.md documents,
        as long as the caller does not also pass the named parameter."""
        config = ProviderConfig(extra_options={"top_k": 999})
        options = self._ollama_options_with_config(
            config, extra_options={"top_k": 5},
        )
        self.assertEqual(options["top_k"], 5)

    def test_ollama_explicit_parameter_beats_call_time_extra_options_too(self) -> None:
        """When a call gives both the named parameter and extra_options for
        the same key, the parameter wins: it is the more specific signal for
        this exact call, and extra_options exists to reach settings the
        named parameters do not cover, not to second-guess the ones given
        alongside it."""
        config = ProviderConfig(extra_options={"top_k": 999})
        options = self._ollama_options_with_config(
            config, top_k=10, extra_options={"top_k": 5},
        )
        self.assertEqual(options["top_k"], 10)

    def test_ollama_config_extra_options_do_not_leak_unrelated_model_params(self) -> None:
        """A sanity check that the reorder did not drop config extras that
        are not temperature/top_k/top_p, such as num_ctx."""
        config = ProviderConfig(extra_options={"num_ctx": 8000})
        options = self._ollama_options_with_config(config)
        self.assertEqual(options["num_ctx"], 8000)

    def test_thinking_strips_what_anthropic_refuses(self) -> None:
        """Regression: thinking=True was a 400 on every Claude model.

        The Messages API rejects top_k and top_p with thinking enabled, and
        accepts temperature only at 1, but all three were sent unconditionally.
        """
        payload = self._anthropic_payload(thinking=True)
        self.assertIn("thinking", payload)
        self.assertNotIn("top_k", payload)
        self.assertNotIn("top_p", payload)
        self.assertEqual(payload["temperature"], 1)

    def test_an_explicit_top_k_is_dropped_too_when_thinking(self) -> None:
        """The API refuses it whoever asked for it."""
        payload = self._anthropic_payload(thinking=True, top_k=10)
        self.assertNotIn("top_k", payload)

    def test_without_thinking_temperature_is_left_alone(self) -> None:
        payload = self._anthropic_payload(temperature=0.3)
        self.assertEqual(payload["temperature"], 0.3)


# ---------------------------------------------------------------------------
# Ollama use_generate: fail fast instead of silently dropping the conversation
# ---------------------------------------------------------------------------

class TestOllamaGenerateModeRejectsChat(unittest.TestCase):
    """/api/generate has no place for tools, history or a system prompt.

    Regression: use_generate=True used to post only the prompt (and any
    images) to /api/generate, silently discarding request.tools,
    request.tool_results, request.messages and request.system_prompt instead
    of refusing. A4/A10-class silent data loss, applied to the whole
    conversation rather than a single field.
    """

    def _call(self, **request_kwargs):
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            ProviderConfig(extra_options={"use_generate": True})
        )

        def fail_if_called(_self, endpoint, payload, timeout):
            raise AssertionError(
                "must not reach the network once the field is unsupported"
            )

        with patch.object(OllamaProvider, "_post", fail_if_called):
            provider.call(AiRequest(
                provider="ollama", model="m", prompt="hi", timeout=30,
                **request_kwargs,
            ))

    def test_rejects_tools(self) -> None:
        from unified_ai_client.exceptions import NonRetryableError
        from unified_ai_client.models import ToolDefinition

        with self.assertRaises(NonRetryableError) as ctx:
            self._call(tools=[ToolDefinition(
                name="f", description="d", parameters={"type": "object"},
            )])
        self.assertIn("tools", str(ctx.exception))

    def test_rejects_tool_results(self) -> None:
        from unified_ai_client.exceptions import NonRetryableError
        from unified_ai_client.models import ToolResult

        with self.assertRaises(NonRetryableError) as ctx:
            self._call(
                messages=[{"role": "user", "content": "hi"}],
                tool_results=[ToolResult(call_id="1", name="f", content="done")],
            )
        self.assertIn("tool_results", str(ctx.exception))

    def test_rejects_message_history(self) -> None:
        from unified_ai_client.exceptions import NonRetryableError

        with self.assertRaises(NonRetryableError) as ctx:
            self._call(messages=[{"role": "user", "content": "earlier"}])
        self.assertIn("messages", str(ctx.exception))

    def test_rejects_system_prompt(self) -> None:
        from unified_ai_client.exceptions import NonRetryableError

        with self.assertRaises(NonRetryableError) as ctx:
            self._call(system_prompt="be terse")
        self.assertIn("system_prompt", str(ctx.exception))

    def test_a_bare_prompt_still_works(self) -> None:
        """The one case use_generate actually supports must be unaffected."""
        from unified_ai_client.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            ProviderConfig(extra_options={"use_generate": True})
        )
        seen = {}

        def fake_post(self, endpoint, payload, timeout):
            seen["endpoint"] = endpoint
            seen["payload"] = payload
            return _OK_RESPONSE

        with patch.object(OllamaProvider, "_post", fake_post):
            provider.call(AiRequest(
                provider="ollama", model="m", prompt="hi", timeout=30,
            ))
        self.assertEqual(seen["endpoint"], "/api/generate")
        self.assertEqual(seen["payload"]["prompt"], "hi")


# ---------------------------------------------------------------------------
# Google call deadline
# ---------------------------------------------------------------------------

class TestGoogleCallTimeout(unittest.TestCase):
    """The timeout has to end the wait, not just describe it."""

    def test_timeout_returns_without_waiting_for_the_call(self) -> None:
        """Regression: the deadline was reported but never enforced.

        The thread pool ran inside a `with` block, and
        ThreadPoolExecutor.__exit__ calls shutdown(wait=True). TimeoutError was
        raised on schedule but the block would not close until the underlying
        call finished, so a request hanging for ten minutes blocked the caller
        for ten minutes and the number passed as `timeout` measured nothing.
        And since TimeoutError is retryable, with_retry paid that cost four
        times over.
        """
        from unified_ai_client.providers.google import GoogleProvider

        provider = GoogleProvider(ProviderConfig(), api_key="fake-key")
        fake_client = MagicMock()
        fake_client.models.generate_content.side_effect = lambda **kw: time.sleep(5)

        request = AiRequest(
            provider="google", model="gemini-2.5-flash", prompt="hi", timeout=1
        )

        started = time.perf_counter()
        with patch.object(GoogleProvider, "_get_client", return_value=fake_client):
            with self.assertRaises(TimeoutError):
                provider.call(request)
        elapsed = time.perf_counter() - started

        # Generous margin: the point is 1s rather than 5s, not the exact figure.
        self.assertLess(
            elapsed, 3.0,
            f"timeout did not end the wait: returned after {elapsed:.2f}s",
        )


# ---------------------------------------------------------------------------
# Google blocked response
# ---------------------------------------------------------------------------

class TestGoogleBlockedResponse(unittest.TestCase):
    """A blocked prompt must not disappear as an unremarkable empty string."""

    def test_a_blocked_prompt_logs_the_reason(self) -> None:
        """Regression: candidates=[] fell into the same except as a genuinely
        malformed response, and the real reason (safety, prohibited content)
        was discarded along with it — the caller got AiResponse(text=""),
        indistinguishable from the model simply saying nothing.
        """
        from unified_ai_client.providers.google import GoogleProvider

        provider = GoogleProvider(ProviderConfig(), api_key="fake-key")
        fake_response = MagicMock()
        fake_response.candidates = []
        fake_response.prompt_feedback.block_reason = "SAFETY"
        fake_response.text = ""
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        request = AiRequest(
            provider="google", model="gemini-2.5-flash", prompt="hi", timeout=30
        )

        with patch.object(GoogleProvider, "_get_client", return_value=fake_client):
            with self.assertLogs(
                "unified_ai_client.providers.google", level="WARNING"
            ) as cm:
                response = provider.call(request)

        self.assertEqual(response.text, "")
        self.assertTrue(any("SAFETY" in line for line in cm.output))

    def test_no_warning_when_nothing_was_blocked(self) -> None:
        """An ordinary empty candidates list, with no prompt_feedback to
        explain it, must not manufacture a warning out of nothing."""
        from unified_ai_client.providers.google import GoogleProvider

        provider = GoogleProvider(ProviderConfig(), api_key="fake-key")
        fake_response = MagicMock()
        fake_response.candidates = []
        fake_response.prompt_feedback = None
        fake_response.text = ""
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        request = AiRequest(
            provider="google", model="gemini-2.5-flash", prompt="hi", timeout=30
        )

        with patch.object(GoogleProvider, "_get_client", return_value=fake_client):
            with self.assertNoLogs(
                "unified_ai_client.providers.google", level="WARNING"
            ):
                provider.call(request)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# 12. HTTP error classification
# ---------------------------------------------------------------------------

def _http_error(
    code: int, body: str, url: str = "http://example.test/v1/chat/completions"
) -> urllib.error.HTTPError:
    """Build the HTTPError urllib would raise for an error response.

    The body is handed over as a stream, exactly as urllib does, so the
    transport has to read it to see anything: a test that pre-decoded it would
    not prove that the read happens at all.
    """
    return urllib.error.HTTPError(
        url, code, "Reason From Status Line", {}, io.BytesIO(body.encode("utf-8"))
    )


class TestHttpErrorClassification(ProviderRegistryIsolation):
    """A 4xx must fail once, with the provider's own message attached.

    Before this, every urllib adapter let urllib.error.HTTPError propagate
    untouched: the response body, which is where an API says what it actually
    rejected, was never read, and a deterministic 401 spent the whole retry
    budget arriving at the same answer.
    """

    # One entry per urllib-backed adapter, with the error shape that API really
    # returns. Driving all three off the shared transport is the point of the
    # refactor, so the same expectations have to hold for each.
    _ADAPTERS = (
        (
            "openai",
            '{"error": {"message": "Unrecognized request argument", "type": "invalid_request_error"}}',
            "Unrecognized request argument",
        ),
        (
            "anthropic",
            '{"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}',
            "invalid x-api-key",
        ),
        (
            "ollama",
            '{"error": "model \'nope\' not found"}',
            "model 'nope' not found",
        ),
    )

    def _send(self, name: str):
        """Fire one request at the named adapter, whatever its _post looks like.

        Anthropic's _post takes no endpoint and ollama's takes no credentials,
        so the call shape differs per adapter even though the transport beneath
        them is now shared.
        """
        cls = _provider_class_by_name(name)
        if name == "ollama":
            return cls(ProviderConfig())._post("/api/chat", {"model": "m"}, 10)
        if name == "anthropic":
            return cls(ProviderConfig(), api_key="k")._post({"model": "m"}, 10)
        return cls(ProviderConfig(), api_key="k")._post(
            "/v1/chat/completions", {"model": "m"}, 10
        )

    def test_a_4xx_carries_the_providers_own_message(self) -> None:
        from unified_ai_client.exceptions import NonRetryableHttpError

        for name, body, expected in self._ADAPTERS:
            with self.subTest(provider=name):
                with patch(
                    "urllib.request.urlopen", side_effect=_http_error(401, body)
                ):
                    with self.assertRaises(NonRetryableHttpError) as ctx:
                        self._send(name)
                self.assertIn(expected, str(ctx.exception))
                self.assertEqual(ctx.exception.code, 401)
                self.assertEqual(ctx.exception.detail, expected)
                self.assertIn(expected, ctx.exception.body)

    def test_existing_httperror_handlers_still_catch_it(self) -> None:
        """The compatibility half of the exception design.

        Consumers written against this library catch urllib.error.HTTPError
        today. Raising a type outside that hierarchy would silently stop their
        handlers from running, which is the failure this inheritance prevents.
        """
        with patch(
            "urllib.request.urlopen",
            side_effect=_http_error(404, '{"error": {"message": "no such model"}}'),
        ):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self._send("openai")
        self.assertEqual(ctx.exception.code, 404)

    def test_the_line_between_retryable_and_not(self) -> None:
        """408, 409, 425 and 429 are the 4xx a later attempt can still win."""
        from unified_ai_client.exceptions import NonRetryableError, ProviderHttpError

        cases = {
            400: False, 401: False, 403: False, 404: False, 422: False,
            408: True, 409: True, 425: True, 429: True,
            500: True, 502: True, 503: True,
        }
        for code, retryable in cases.items():
            with self.subTest(status=code):
                with patch(
                    "urllib.request.urlopen",
                    side_effect=_http_error(code, '{"error": {"message": "x"}}'),
                ):
                    with self.assertRaises(ProviderHttpError) as ctx:
                        self._send("openai")
                self.assertEqual(
                    isinstance(ctx.exception, NonRetryableError),
                    not retryable,
                    f"status {code} landed on the wrong side of the line",
                )

    def test_a_connection_failure_stays_retryable(self) -> None:
        """A local server still starting up must not become a hard failure."""
        from unified_ai_client.exceptions import NonRetryableError

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with self.assertRaises(urllib.error.URLError) as ctx:
                self._send("ollama")
        self.assertNotIsInstance(ctx.exception, NonRetryableError)

    def test_a_4xx_does_not_spend_the_retry_budget(self) -> None:
        """The measurable half of the fix, end to end through call_ai().

        A rejected credential used to cost max_retries+1 attempts and the whole
        exponential backoff before surfacing an error that had been settled by
        the first response.
        """
        from unified_ai_client.client import call_ai
        from unified_ai_client.exceptions import NonRetryableHttpError

        attempts = []

        def fake_urlopen(req, timeout=None):
            attempts.append(req.full_url)
            raise _http_error(
                401, '{"error": {"message": "Invalid API Key"}}', url=req.full_url
            )

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake"}, clear=False):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                with self.assertRaises(NonRetryableHttpError) as ctx:
                    call_ai(
                        "groq",
                        "llama-3.3-70b",
                        "hi",
                        max_retries=3,
                        retry_base_delay=0.01,
                    )

        self.assertEqual(len(attempts), 1, "a deterministic 401 was retried anyway")
        self.assertIn("Invalid API Key", str(ctx.exception))


class TestHttpErrorDetailExtraction(unittest.TestCase):
    """The body shapes the three APIs use, plus the ones they do not."""

    def test_every_shape_the_providers_actually_send(self) -> None:
        from unified_ai_client.http import _extract_detail

        cases = (
            # OpenAI-compatible: error is an object carrying a message.
            ('{"error": {"message": "bad request", "type": "invalid"}}', "bad request"),
            # Anthropic: same shape, inside a typed envelope.
            ('{"type": "error", "error": {"type": "x", "message": "overloaded"}}', "overloaded"),
            # Ollama: error is a bare string.
            ('{"error": "model not found"}', "model not found"),
            # An object with only a type still says more than the status line.
            ('{"error": {"type": "rate_limit_error"}}', "rate_limit_error"),
            # Some gateways put the message at the top level.
            ('{"message": "upstream timeout"}', "upstream timeout"),
        )
        for body, expected in cases:
            with self.subTest(body=body):
                self.assertEqual(_extract_detail(body), expected)

    def test_a_body_that_is_not_json_degrades_to_its_text(self) -> None:
        """A proxy answering with HTML must not turn into a parsing error.

        Replacing the HTTP failure with a JSONDecodeError would hide the status
        the caller needs, which is the same swallow-and-substitute pattern the
        file layer was fixed for.
        """
        from unified_ai_client.http import _extract_detail

        self.assertEqual(
            _extract_detail("<html>502 Bad Gateway</html>"),
            "<html>502 Bad Gateway</html>",
        )
        self.assertEqual(_extract_detail("   "), "")
        self.assertEqual(_extract_detail(""), "")

    def test_a_huge_body_is_truncated(self) -> None:
        """An error page belongs in the log, not inside an exception message."""
        from unified_ai_client.http import _MAX_DETAIL_CHARS, _extract_detail

        detail = _extract_detail("x" * 5000)
        self.assertTrue(detail.endswith("..."))
        self.assertLessEqual(len(detail), _MAX_DETAIL_CHARS + 3)


# ---------------------------------------------------------------------------
# 13. The extra_options namespace
# ---------------------------------------------------------------------------

def _capture_http_payload(name: str, config: ProviderConfig, **request_kwargs) -> dict:
    """Run one call against an http-backed adapter and return the payload sent.

    Anthropic's _post takes no endpoint and ollama's needs no credentials, so
    the interception differs slightly per adapter even though the transport
    beneath them is shared.
    """
    cls = _provider_class_by_name(name)
    provider = cls(config) if name == "ollama" else cls(config, api_key="k")
    captured: dict = {}

    def fake_post(*args):
        # (endpoint, payload, timeout) everywhere except anthropic: (payload, timeout)
        captured.update(args[0] if name == "anthropic" else args[1])
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
            "message": {"content": "ok"},
            "eval_count": 1,
            "prompt_eval_count": 1,
            "content": [{"type": "text", "text": "ok"}],
        }

    fields = {"provider": name, "model": "m", "prompt": "hi", "timeout": 10}
    fields.update(request_kwargs)
    request = AiRequest(**fields)
    with patch.object(provider, "_post", side_effect=fake_post):
        provider.call(request)
    return captured


def _ollama_options(config: ProviderConfig, **request_kwargs) -> dict:
    """The Ollama model `options` block, which is where its keys land."""
    return _capture_http_payload("ollama", config, **request_kwargs)["options"]


class TestLibraryOptionNamespace(ProviderRegistryIsolation):
    """extra_options carries two kinds of key, and only one may reach the wire.

    The library owns the meaning of the keys in `_LIBRARY_OPTION_KEYS`; each
    adapter consumes the ones it owns and must drop the rest. Everything else
    is provider-specific and passes through verbatim. Until 0.5.5 only Ollama
    enforced any of this, so `context_size` -- which config.json.example
    shipped for lmstudio and llamacpp -- travelled straight into the body of
    /v1/chat/completions, and Google's `disable_safety` landed in Ollama's
    model options.
    """

    # The specification, written out rather than derived, on the model of
    # TestFileSupportMatrix._SUPPORT: it mirrors the table in
    # docs/configuration.md and fails when code and document drift apart.
    _NAMESPACE = frozenset({
        "url", "timeout", "sleep_time",
        "temperature", "max_tokens", "max_output_tokens", "top_k", "top_p",
        "context_size", "use_generate", "keep_alive",
        "disable_safety", "upload_poll_timeout",
        "task_type", "output_dimensionality",
    })

    def test_the_namespace_matches_its_specification(self) -> None:
        from unified_ai_client.providers.base import BaseProvider

        self.assertEqual(
            BaseProvider._LIBRARY_OPTION_KEYS,
            self._NAMESPACE,
            "the library option namespace changed without updating this table "
            "or the one in docs/configuration.md",
        )

    def test_every_key_is_owned_by_some_adapter(self) -> None:
        """A key nobody consumes is silently dropped everywhere: dead weight."""
        from unified_ai_client.providers.base import BaseProvider

        infrastructure = {"url", "timeout", "sleep_time"}
        consumed: set[str] = set()
        for name in _PROVIDER_CLASSES:
            consumed |= set(_provider_class_by_name(name)._CONSUMED_OPTION_KEYS)

        self.assertEqual(
            BaseProvider._LIBRARY_OPTION_KEYS - infrastructure - consumed,
            set(),
            "these keys are in the namespace but no adapter reads them",
        )

    def test_no_adapter_claims_a_key_outside_the_namespace(self) -> None:
        from unified_ai_client.providers.base import BaseProvider

        for name in _PROVIDER_CLASSES:
            with self.subTest(provider=name):
                cls = _provider_class_by_name(name)
                self.assertEqual(
                    set(cls._CONSUMED_OPTION_KEYS) - BaseProvider._LIBRARY_OPTION_KEYS,
                    set(),
                    "an adapter declares a key the library does not own",
                )

    def test_a_key_owned_elsewhere_never_reaches_an_http_body(self) -> None:
        """The bug this batch closes, measured on every http adapter."""
        # Exactly what config.json.example shipped, plus a control key only
        # Ollama has ever read.
        config = ProviderConfig(
            extra_options={"context_size": 0, "use_generate": False, "url": "x"}
        )
        for name in ("openai", "groq", "lmstudio", "llamacpp", "anthropic"):
            with self.subTest(provider=name):
                payload = _capture_http_payload(name, config)
                for leaked in ("context_size", "use_generate", "url"):
                    self.assertNotIn(leaked, payload)

    def test_googles_keys_do_not_reach_ollamas_model_options(self) -> None:
        """The same leak in the other direction, which Ollama also had."""
        options = _ollama_options(
            ProviderConfig(extra_options={"disable_safety": True, "task_type": "x"})
        )
        self.assertNotIn("disable_safety", options)
        self.assertNotIn("task_type", options)

    def test_provider_specific_options_still_pass_through(self) -> None:
        """The half that protects against fixing this too greedily.

        `visual_token_budget` reaching Gemma 4 without the library knowing it
        exists is the stated point of extra_options, so a filter that also
        caught unknown keys would be a worse bug than the one it replaced.
        """
        config = ProviderConfig(
            extra_options={"visual_token_budget": 1120, "repeat_penalty": 1.2}
        )
        options = _ollama_options(config)
        self.assertEqual(options["visual_token_budget"], 1120)
        self.assertEqual(options["repeat_penalty"], 1.2)

        for name in ("openai", "anthropic"):
            with self.subTest(provider=name):
                payload = _capture_http_payload(name, config)
                self.assertEqual(payload["visual_token_budget"], 1120)

    def test_the_call_time_escape_hatch_still_wins(self) -> None:
        """0.5.3's precedence must survive the consolidation."""
        options = _ollama_options(
            ProviderConfig(extra_options={"top_k": 999}),
            extra_options={"top_k": 5},
        )
        self.assertEqual(options["top_k"], 5)


class TestTemperatureResolution(ProviderRegistryIsolation):
    """temperature resolves like its sibling levers, or it resolves wrongly.

    It was the one common lever still defaulting to a value rather than None,
    so "not given" was indistinguishable from "given 0.7" and a temperature
    registered through configure_provider() was overwritten on every call
    while top_k, sitting beside it, survived.
    """

    def test_a_configured_temperature_reaches_the_payload(self) -> None:
        config = ProviderConfig(extra_options={"temperature": 0.1})
        for name in ("openai", "anthropic"):
            with self.subTest(provider=name):
                self.assertEqual(
                    _capture_http_payload(name, config)["temperature"], 0.1
                )
        self.assertEqual(_ollama_options(config)["temperature"], 0.1)

    def test_an_explicit_temperature_beats_the_configured_one(self) -> None:
        config = ProviderConfig(extra_options={"temperature": 0.1})
        for name in ("openai", "anthropic"):
            with self.subTest(provider=name):
                payload = _capture_http_payload(name, config, temperature=0.9)
                self.assertEqual(payload["temperature"], 0.9)
        self.assertEqual(_ollama_options(config, temperature=0.9)["temperature"], 0.9)

    def test_configuring_nothing_still_sends_the_old_default(self) -> None:
        """The compatibility half: 0.7 was the signature default for years."""
        config = ProviderConfig()
        for name in ("openai", "anthropic"):
            with self.subTest(provider=name):
                self.assertEqual(
                    _capture_http_payload(name, config)["temperature"], 0.7
                )
        self.assertEqual(_ollama_options(config)["temperature"], 0.7)

    def test_the_script_protocol_never_receives_null(self) -> None:
        """docs/script-protocol.md declares this field `float`, not `float | null`.

        top_k and top_p are documented as nullable and are forwarded raw; this
        one is not, so a script doing arithmetic on it must keep working.
        """
        from unified_ai_client.providers.script import ScriptProvider

        provider = ScriptProvider(ProviderConfig())
        request = AiRequest(provider="script", model="s.py", prompt="hi", timeout=10)
        captured: dict = {}

        def fake_run(cmd, payload, timeout):
            captured.update(payload)
            return {"text": "ok"}

        with patch(
            "unified_ai_client.providers.script._run_script", side_effect=fake_run
        ):
            provider.call(request)

        self.assertIsNotNone(captured["temperature"])
        self.assertEqual(captured["temperature"], 0.7)


# ---------------------------------------------------------------------------
# 14. Invariants shared by every adapter
# ---------------------------------------------------------------------------

class TestToolResultsDoNotReappendThePrompt(ProviderRegistryIsolation):
    """With tool_results present the prompt is already in messages.

    This invariant entered ollama, anthropic and openai_compat in one commit
    and reached google only later, so for a while tool calling on Google
    duplicated the user turn. Nothing asserted it across all four until now:
    the per-adapter tests each check their own, which is exactly the shape of
    coverage that let the fourth drift.
    """

    _PROMPT = "SENTINEL-PROMPT-NOT-TO-BE-REAPPENDED"

    def _tool_results(self):
        from unified_ai_client.models import ToolResult
        return [ToolResult(call_id="1", name="f", content="done")]

    def _history(self):
        return [{"role": "user", "content": self._PROMPT}]

    def test_http_adapters_send_the_prompt_exactly_once(self) -> None:
        for name in ("openai", "anthropic", "ollama"):
            with self.subTest(provider=name):
                payload = _capture_http_payload(
                    name,
                    ProviderConfig(),
                    prompt=self._PROMPT,
                    messages=self._history(),
                    tool_results=self._tool_results(),
                )
                occurrences = sum(
                    1 for m in payload["messages"]
                    if self._PROMPT in str(m.get("content", ""))
                )
                self.assertEqual(
                    occurrences, 1,
                    "the prompt was re-appended on top of the history",
                )

    def test_google_sends_the_prompt_exactly_once(self) -> None:
        from unified_ai_client.providers.google import GoogleProvider

        provider = GoogleProvider(ProviderConfig(), api_key="fake-key")
        fake_response = MagicMock()
        fake_response.candidates = []
        fake_response.prompt_feedback = None
        fake_response.text = "ok"
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        request = AiRequest(
            provider="google", model="gemini-2.5-flash", prompt=self._PROMPT,
            messages=self._history(), tool_results=self._tool_results(), timeout=30,
        )
        with patch.object(GoogleProvider, "_get_client", return_value=fake_client):
            provider.call(request)

        contents = fake_client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(str(contents).count(self._PROMPT), 1)


class TestAttachmentRefusalIsWordedOnce(FileFixtureCase):
    """A provider that promises a file class it cannot build says so one way.

    The three copies of this refusal had already drifted into two wordings,
    "builds no block for them" and "has no way to send them", which is the
    shape of divergence that produced the .mp3 bug in the first place.

    Each case widens SUPPORTED_FILE_TYPES past what the adapter's block builder
    can actually produce, which is the inconsistency the message exists to
    report, and then drives the real code path rather than the shared factory:
    a test that called the factory directly would pass even if no adapter used
    it.
    """

    # (provider, the class it is made to declare, a file of that class)
    _CASES = (
        ("groq", "document", ".pdf"),   # base OpenAiCompat builder: image + audio only
        ("anthropic", "audio", ".mp3"),  # Messages API has no audio block
        ("ollama", "audio", ".mp3"),     # /api/chat carries images[] and nothing else
    )

    def _refusal(self, name: str, declared: str, suffix: str) -> str:
        from unified_ai_client.exceptions import UnsupportedFileError

        cls = _provider_class_by_name(name)
        provider = cls(ProviderConfig()) if name == "ollama" else cls(
            ProviderConfig(), api_key="k"
        )
        widened = cls.SUPPORTED_FILE_TYPES | {declared}
        request = AiRequest(
            provider=name, model="m", prompt="hi", timeout=10,
            file_path=self.make(suffix),
        )
        with patch.object(cls, "SUPPORTED_FILE_TYPES", widened):
            with self.assertRaises(UnsupportedFileError) as ctx:
                provider.call(request)
        return str(ctx.exception)

    def test_the_three_adapters_word_it_identically(self) -> None:
        shapes = set()
        for name, declared, suffix in self._CASES:
            message = self._refusal(name, declared, suffix)
            with self.subTest(provider=name):
                self.assertIn(name, message)
                self.assertIn(declared, message)
            # Strip the parts that legitimately differ: the provider name, the
            # file class and the path. What is left is the wording itself.
            shapes.add(
                message.split("declares support for", 1)[1].split(declared, 1)[1]
                .split(":", 1)[0]
            )
        self.assertEqual(
            len(shapes), 1, f"the refusal is worded {len(shapes)} different ways: {shapes}"
        )


class TestPreloadAndUnloadStayPaired(unittest.TestCase):
    """preload_model and unload_model are two halves of model residency.

    Overriding one without the other leaves a model this library asked for and
    can no longer release, which is precisely what cleanup() exists to prevent.
    Now that preload_model is a concrete no-op, an adapter that overrides it is
    making a claim, and this is what checks the claim.
    """

    def test_only_the_residency_providers_override_preload(self) -> None:
        from unified_ai_client.providers.base import BaseProvider

        for name in _PROVIDER_CLASSES:
            with self.subTest(provider=name):
                cls = _provider_class_by_name(name)
                overrides = cls.preload_model is not BaseProvider.preload_model
                self.assertEqual(
                    overrides, cls.SUPPORTS_UNLOAD,
                    "preload_model and SUPPORTS_UNLOAD disagree about whether "
                    "this provider holds a model resident",
                )


class TestReasoningTokenSplit(unittest.TestCase):
    """The shared estimator must reproduce what the two copies produced."""

    def _split(self, content: str, reasoning: str, total: int) -> int:
        from unified_ai_client.providers.base import BaseProvider
        return BaseProvider._estimate_reasoning_tokens(content, reasoning, total)

    def test_it_matches_the_original_formula(self) -> None:
        for content, reasoning, total in (
            ("answer", "thinking hard", 100),
            ("a" * 900, "b" * 100, 250),
            ("", "only reasoning", 40),
        ):
            with self.subTest(total=total):
                total_chars = len(content) + len(reasoning)
                expected = max(1, round(len(reasoning) / (total_chars / total)))
                self.assertEqual(self._split(content, reasoning, total), expected)

    def test_no_trace_means_no_reasoning_tokens(self) -> None:
        self.assertEqual(self._split("answer", "", 100), 0)

    def test_a_trace_is_never_rounded_down_to_nothing(self) -> None:
        """0 would read as "the model did not think", which is a different claim."""
        self.assertEqual(self._split("a" * 100000, "x", 10), 1)

    def test_nothing_to_divide_is_zero_not_a_crash(self) -> None:
        self.assertEqual(self._split("answer", "trace", 0), 0)
