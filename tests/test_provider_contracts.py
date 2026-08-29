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

# Imported as a top-level module: unittest discovery puts tests/ on sys.path.
from test_providers import ProviderRegistryIsolation  # noqa: E402

from unified_ai_client.models import AiRequest, ProviderConfig  # noqa: E402


def _provider_class(module: str, class_name: str):
    """Import a provider class by module and class name."""
    return getattr(
        importlib.import_module(f"unified_ai_client.providers.{module}"), class_name
    )


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


# ---------------------------------------------------------------------------
# 2. API key handling
# ---------------------------------------------------------------------------

class TestApiKeyHandling(ProviderRegistryIsolation):
    """A missing key must fail with a message, not a type error."""

    _CLOUD = {
        "anthropic": ("anthropic", "AnthropicProvider", "anthropic_api_key"),
        "openai": ("openai", "OpenAiProvider", "openai_api_key"),
        "mistral": ("mistral", "MistralProvider", "mistral_api_key"),
        "cohere": ("cohere", "CohereProvider", "cohere_api_key"),
        "meta": ("meta", "MetaProvider", "meta_api_key"),
        "groq": ("groq", "GroqProvider", "groq_api_key"),
        "xai": ("xai", "XAiProvider", "xai_api_key"),
    }

    _LOCAL = (("lmstudio", "LmStudioProvider"), ("llamacpp", "LlamaCppProvider"))

    def test_cloud_providers_reject_a_missing_key(self) -> None:
        """The error must name the secrets key, so the fix is obvious."""
        for name, (module, class_name, secrets_key) in self._CLOUD.items():
            with self.subTest(provider=name):
                provider = _provider_class(module, class_name)(
                    ProviderConfig(), api_key=""
                )
                with self.assertRaises(ValueError) as ctx:
                    provider._require_api_key()
                self.assertIn(secrets_key, str(ctx.exception))

    def test_local_providers_accept_a_missing_key(self) -> None:
        """A local server without credentials is normal, not an error."""
        for module, class_name in self._LOCAL:
            with self.subTest(provider=module):
                provider = _provider_class(module, class_name)(ProviderConfig())
                self.assertIsNone(provider._require_api_key())

    def test_an_explicit_url_lifts_the_key_requirement(self) -> None:
        """Pointing a cloud adapter elsewhere is supported and needs no key.

        Aiming the OpenAI adapter at a local Ollama or LM Studio server that
        serves /v1/chat/completions is exactly what the url override is for;
        demanding a cloud credential for it would make that impossible.
        """
        for name, (module, class_name, _) in self._CLOUD.items():
            with self.subTest(provider=name):
                provider = _provider_class(module, class_name)(
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

def _make_file(suffix: str, data: bytes = b"x") -> str:
    """Write a temp file with the given extension and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


class FileFixtureCase(unittest.TestCase):
    """Base class that cleans up the temp files a test creates."""

    def setUp(self) -> None:
        self._paths: list[str] = []

    def tearDown(self) -> None:
        for path in self._paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    def make(self, suffix: str, data: bytes = b"x") -> str:
        path = _make_file(suffix, data)
        self._paths.append(path)
        return path


class TestFileSupportMatrix(FileFixtureCase):
    """Every provider declares what it can carry, and refuses the rest.

    The table is the specification: each provider maps to the file classes it
    can transmit natively. 'text' is deliberately absent everywhere, being
    inlined into the prompt rather than carried in a native block.
    """

    _SUPPORT = {
        "google": ("google", "GoogleProvider", {"image", "audio", "document"}),
        "openai": ("openai", "OpenAiProvider", {"image", "audio", "document"}),
        "anthropic": ("anthropic", "AnthropicProvider", {"image", "document"}),
        "llamacpp": ("llamacpp", "LlamaCppProvider", {"image", "audio"}),
        "ollama": ("ollama", "OllamaProvider", {"image"}),
        "mistral": ("mistral", "MistralProvider", {"image"}),
        "cohere": ("cohere", "CohereProvider", {"image"}),
        "meta": ("meta", "MetaProvider", {"image"}),
        "groq": ("groq", "GroqProvider", {"image"}),
        "xai": ("xai", "XAiProvider", {"image"}),
        "lmstudio": ("lmstudio", "LmStudioProvider", {"image"}),
    }

    _SAMPLE = {"image": ".png", "audio": ".mp3", "document": ".pdf"}

    def test_declared_support_matches_the_table(self) -> None:
        for name, (module, class_name, expected) in self._SUPPORT.items():
            with self.subTest(provider=name):
                cls = _provider_class(module, class_name)
                self.assertEqual(set(cls.SUPPORTED_FILE_TYPES), expected)

    def test_unsupported_classes_raise(self) -> None:
        """The refusal must name the file and the provider, not just fail."""
        from unified_ai_client.exceptions import UnsupportedFileError
        from unified_ai_client.file_utils import validate_files

        for name, (_, _, supported) in self._SUPPORT.items():
            for file_type, suffix in self._SAMPLE.items():
                if file_type in supported:
                    continue
                with self.subTest(provider=name, file_type=file_type):
                    path = self.make(suffix)
                    with self.assertRaises(UnsupportedFileError) as ctx:
                        validate_files([path], name, frozenset(supported))
                    message = str(ctx.exception)
                    self.assertIn(path, message)
                    self.assertIn(name, message)

    def test_supported_classes_pass(self) -> None:
        from unified_ai_client.file_utils import validate_files

        for name, (_, _, supported) in self._SUPPORT.items():
            for file_type in supported:
                with self.subTest(provider=name, file_type=file_type):
                    path = self.make(self._SAMPLE[file_type])
                    self.assertIsNone(
                        validate_files([path], name, frozenset(supported))
                    )

    def test_text_is_accepted_everywhere(self) -> None:
        """No provider has a native text block, so inlining must stay open."""
        from unified_ai_client.file_utils import validate_files

        for name, (_, _, supported) in self._SUPPORT.items():
            with self.subTest(provider=name):
                path = self.make(".md", b"# notes")
                self.assertIsNone(
                    validate_files([path], name, frozenset(supported))
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
        cls = _provider_class("script", "ScriptProvider")
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
        import tempfile

        provider = _provider_class("groq", "GroqProvider")(
            ProviderConfig(), api_key="fake"
        )
        directory = tempfile.mkdtemp()
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


if __name__ == "__main__":
    unittest.main()
