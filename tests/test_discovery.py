"""Tests for list_providers() and list_models().

Fully offline: credentials, the local-server probe, and every HTTP or SDK call
are patched, so none of this depends on which servers happen to be running.
"""

from __future__ import annotations

import importlib
import io
import sys
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_TESTS_DIR = Path(__file__).resolve().parent
for _path in (str(_TESTS_DIR.parent), str(_TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Top-level import, not tests.test_providers: tests/ is not a package and
# discovery puts it on sys.path, so the qualified form would load the module twice.
from test_providers import ProviderRegistryIsolation  # noqa: E402

from unified_ai_client import list_models, list_providers  # noqa: E402
from unified_ai_client.models import ModelInfo, ProviderConfig  # noqa: E402


def _build(module: str, class_name: str, **kwargs):
    cls = getattr(
        importlib.import_module(f"unified_ai_client.providers.{module}"), class_name
    )
    return cls(ProviderConfig(), **kwargs)


class TestListProviders(unittest.TestCase):
    """Only providers whose prerequisite is met are listed."""

    def _list(self, secrets=None, up=(), configs=None, **kwargs):
        """Run list_providers() with credentials and server state faked.

        Args:
            secrets: What load_secrets() returns.
            up: URLs the probe should answer for.
            configs: Per-provider ProviderConfig overrides.
        """
        configs = configs or {}
        probed: list[str] = []

        def fake_reachable(url: str, timeout: float) -> bool:
            probed.append(url)
            return url in up

        with patch("unified_ai_client.config.load_secrets", return_value=secrets or {}), \
                patch(
                    "unified_ai_client.registry._resolve_config",
                    side_effect=lambda n: configs.get(n, ProviderConfig()),
                ), \
                patch("unified_ai_client.registry.is_reachable", fake_reachable):
            result = list_providers(**kwargs)
        self.probed = probed
        return {p.name: p for p in result}

    def test_cloud_provider_is_listed_only_with_a_key(self) -> None:
        listed = self._list(secrets={"groq_api_key": "k"}, probe_local=False)
        self.assertIn("groq", listed)
        self.assertNotIn("openai", listed)
        self.assertNotIn("anthropic", listed)

    def test_explicit_url_stands_in_for_a_key(self) -> None:
        """The same escape hatch _require_api_key() honours."""
        listed = self._list(
            configs={"openai": ProviderConfig(url="http://gateway")},
            probe_local=False,
        )
        self.assertIn("openai", listed)

    def test_local_server_is_listed_only_when_it_answers(self) -> None:
        listed = self._list(up={"http://localhost:11434/api/tags"})
        self.assertIn("ollama", listed)
        self.assertNotIn("lmstudio", listed)
        self.assertNotIn("llamacpp", listed)

    def test_probe_follows_the_configured_url(self) -> None:
        self._list(configs={"lmstudio": ProviderConfig(url="http://box:9999/")})
        self.assertIn("http://box:9999/v1/models", self.probed)

    def test_probe_local_false_makes_no_request(self) -> None:
        listed = self._list(probe_local=False)
        self.assertEqual(self.probed, [])
        self.assertNotIn("ollama", listed)

    def test_script_is_always_listed(self) -> None:
        self.assertIn("script", self._list(probe_local=False))
        self.assertIn("script", self._list())

    def test_info_describes_the_provider(self) -> None:
        listed = self._list(
            secrets={"groq_api_key": "k"}, up={"http://localhost:11434/api/tags"}
        )
        groq, ollama, script = listed["groq"], listed["ollama"], listed["script"]
        self.assertFalse(groq.local)
        self.assertTrue(groq.requires_api_key)
        self.assertEqual(groq.file_types, frozenset({"image"}))
        self.assertTrue(ollama.local)
        self.assertFalse(ollama.requires_api_key)
        self.assertTrue(ollama.supports_unload)
        self.assertEqual(script.file_types, frozenset())

    def test_order_follows_the_registry(self) -> None:
        with patch("unified_ai_client.config.load_secrets", return_value={}), \
                patch("unified_ai_client.registry._resolve_config", return_value=ProviderConfig()), \
                patch("unified_ai_client.registry.is_reachable", return_value=True):
            names = [p.name for p in list_providers()]
        self.assertEqual(names, ["ollama", "lmstudio", "llamacpp", "script"])


class TestIsReachable(unittest.TestCase):
    def test_any_response_counts_as_up(self) -> None:
        from unified_ai_client.http import is_reachable

        with patch("urllib.request.urlopen", return_value=MagicMock()):
            self.assertTrue(is_reachable("http://x", 1))

    def test_an_http_error_still_means_the_server_is_running(self) -> None:
        from unified_ai_client.http import is_reachable

        err = urllib.error.HTTPError("http://x", 404, "nf", {}, io.BytesIO(b""))
        with patch("urllib.request.urlopen", side_effect=err):
            self.assertTrue(is_reachable("http://x", 1))

    def test_connection_failure_means_down(self) -> None:
        from unified_ai_client.http import is_reachable

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            self.assertFalse(is_reachable("http://x", 1))
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            self.assertFalse(is_reachable("http://x", 1))


class TestListModelsOpenAiCompat(unittest.TestCase):
    def test_maps_ids_and_leaves_capabilities_unknown(self) -> None:
        provider = _build("groq", "GroqProvider", api_key="k")
        payload = {"data": [{"id": "llama-3"}, {"id": "mixtral"}, {"object": "x"}]}
        with patch.object(provider, "_get", return_value=payload) as get:
            models = provider.list_models()
        get.assert_called_once_with("/v1/models", provider.config.timeout)
        self.assertEqual([m.id for m in models], ["llama-3", "mixtral"])
        for m in models:
            self.assertEqual(m.provider, "groq")
            self.assertIsNone(m.reasoning)
            self.assertIsNone(m.tool_use)
            self.assertIsNone(m.file_types)

    def test_missing_key_is_an_error_not_an_empty_list(self) -> None:
        provider = _build("groq", "GroqProvider", api_key="")
        with self.assertRaises(ValueError):
            provider.list_models()


class TestListModelsAnthropic(unittest.TestCase):
    def test_reasoning_follows_the_generation(self) -> None:
        provider = _build("anthropic", "AnthropicProvider", api_key="k")
        payload = {"data": [
            {"id": "claude-3-5-haiku-20241022", "display_name": "Haiku 3.5"},
            {"id": "claude-3-7-sonnet-20250219", "display_name": "Sonnet 3.7"},
            {"id": "claude-opus-4-20250514", "display_name": "Opus 4"},
            {"id": "mystery-model"},
        ]}
        with patch.object(provider, "_get", return_value=payload) as get:
            models = {m.id: m for m in provider.list_models()}
        self.assertIn("limit=1000", get.call_args.args[0])
        self.assertIs(models["claude-3-5-haiku-20241022"].reasoning, False)
        self.assertIs(models["claude-3-7-sonnet-20250219"].reasoning, True)
        self.assertIs(models["claude-opus-4-20250514"].reasoning, True)
        self.assertEqual(
            models["claude-3-7-sonnet-20250219"].display_name, "Sonnet 3.7"
        )

    def test_tool_use_and_files_come_from_the_class(self) -> None:
        provider = _build("anthropic", "AnthropicProvider", api_key="k")
        with patch.object(provider, "_get", return_value={"data": [{"id": "claude-opus-4-5"}]}):
            (model,) = provider.list_models()
        self.assertIs(model.tool_use, True)
        self.assertIs(model.embeddings, False)
        self.assertEqual(model.file_types, frozenset({"image", "document"}))


class TestListModelsOllama(unittest.TestCase):
    def _models(self, tags, shows):
        provider = _build("ollama", "OllamaProvider")

        def fake_post(endpoint, payload, timeout):
            result = shows[payload["model"]]
            if isinstance(result, Exception):
                raise result
            return result

        with patch.object(provider, "_get", return_value={"models": tags}), \
                patch.object(provider, "_post", side_effect=fake_post):
            return {m.id: m for m in provider.list_models()}

    def test_capabilities_and_context_come_from_show(self) -> None:
        models = self._models(
            [{"name": "gemma:4b"}, {"name": "bge-m3"}],
            {
                "gemma:4b": {
                    "capabilities": ["completion", "vision", "tools", "thinking"],
                    "model_info": {"gemma3.context_length": 131072, "other": 1},
                },
                "bge-m3": {"capabilities": ["embedding"], "model_info": {}},
            },
        )
        gemma, bge = models["gemma:4b"], models["bge-m3"]
        self.assertIs(gemma.reasoning, True)
        self.assertIs(gemma.tool_use, True)
        self.assertIs(gemma.embeddings, False)
        self.assertEqual(gemma.file_types, frozenset({"image"}))
        self.assertEqual(gemma.context_length, 131072)
        self.assertIs(bge.embeddings, True)
        self.assertIs(bge.tool_use, False)
        self.assertEqual(bge.file_types, frozenset())
        self.assertIsNone(bge.context_length)

    def test_the_audio_capability_is_reported(self) -> None:
        (model,) = self._models(
            [{"name": "gemma4:e2b"}],
            {"gemma4:e2b": {"capabilities": ["completion", "vision", "audio"]}},
        ).values()
        self.assertEqual(model.file_types, frozenset({"image", "audio"}))

    def test_a_failed_show_keeps_the_model_with_unknown_capabilities(self) -> None:
        models = self._models(
            [{"name": "ok"}, {"name": "broken"}],
            {
                "ok": {"capabilities": ["tools"]},
                "broken": urllib.error.URLError("boom"),
            },
        )
        self.assertEqual(set(models), {"ok", "broken"})
        self.assertEqual(
            models["broken"], ModelInfo(id="broken", provider="ollama")
        )

    def test_an_old_server_without_capabilities_leaves_them_unknown(self) -> None:
        (model,) = self._models([{"name": "old"}], {"old": {"details": {}}}).values()
        self.assertIsNone(model.reasoning)
        self.assertIsNone(model.tool_use)
        self.assertIsNone(model.file_types)

    def test_listing_failure_propagates(self) -> None:
        provider = _build("ollama", "OllamaProvider")
        with patch.object(provider, "_get", side_effect=urllib.error.URLError("down")):
            with self.assertRaises(urllib.error.URLError):
                provider.list_models()


class TestListModelsGoogle(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.provider = _build("google", "GoogleProvider", api_key="k")
        except ImportError:
            self.skipTest("google-genai is not installed")

    def test_maps_sdk_fields_and_strips_the_prefix(self) -> None:
        items = [
            SimpleNamespace(
                name="models/gemini-2.5-flash",
                display_name="Gemini 2.5 Flash",
                supported_actions=["generateContent"],
                thinking=True,
                input_token_limit=1048576,
            ),
            SimpleNamespace(
                name="models/gemini-embedding-001",
                display_name=None,
                supported_actions=["embedContent"],
                thinking=None,
                input_token_limit=2048,
            ),
            SimpleNamespace(name="models/bare"),
        ]
        client = MagicMock()
        client.models.list.return_value = iter(items)
        with patch.object(self.provider, "_get_client", return_value=client):
            models = {m.id: m for m in self.provider.list_models()}

        flash = models["gemini-2.5-flash"]
        self.assertIs(flash.reasoning, True)
        self.assertIs(flash.embeddings, False)
        self.assertEqual(
            flash.file_types, frozenset({"image", "audio", "video", "document"})
        )
        self.assertEqual(flash.context_length, 1048576)
        embed = models["gemini-embedding-001"]
        self.assertIs(embed.embeddings, True)
        self.assertIsNone(embed.file_types)
        # An SDK object that reports nothing leaves everything unknown.
        self.assertEqual(models["bare"], ModelInfo(id="bare", provider="google"))


class TestListModelsDispatch(ProviderRegistryIsolation):
    def test_script_cannot_enumerate_models(self) -> None:
        with self.assertRaises(NotImplementedError):
            list_models("script")

    def test_unknown_provider_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            list_models("nope")

    def test_client_function_delegates_to_the_provider(self) -> None:
        provider = MagicMock()
        provider.list_models.return_value = [ModelInfo(id="m", provider="x")]
        with patch("unified_ai_client.client.get_provider", return_value=provider):
            result = list_models("ollama", timeout=5)
        provider.list_models.assert_called_once_with(timeout=5)
        self.assertEqual(result[0].id, "m")


if __name__ == "__main__":
    unittest.main()
