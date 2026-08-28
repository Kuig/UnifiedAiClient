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

import importlib
import os
import sys
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


if __name__ == "__main__":
    unittest.main()
