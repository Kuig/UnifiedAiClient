"""Batched embeddings, Ollama's per-call options, and the context-length error.

Everything here runs offline, with the transport intercepted at ``_post`` or
``_run_script``.

Usage:
    python -m unittest tests.test_embeddings
"""
from __future__ import annotations

import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _path in (str(_PROJECT_ROOT), str(_TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Imported as a top-level module: unittest discovery puts tests/ on sys.path.
from test_providers import ProviderRegistryIsolation  # noqa: E402

from unified_ai_client import (  # noqa: E402
    ContextLengthError,
    NonRetryableHttpError,
    configure_provider,
    get_embedding,
    get_embeddings,
    warm_up,
)
from unified_ai_client.models import ProviderConfig  # noqa: E402
from unified_ai_client.providers.llamacpp import LlamaCppProvider  # noqa: E402
from unified_ai_client.providers.ollama import OllamaProvider  # noqa: E402

_EMBED_URL = "http://localhost:11434/api/embed"
_TOO_LONG = "the input length exceeds the context length"


def _vector(text: str) -> list[float]:
    """A vector that says which text it was made from."""
    return [float(len(text)), float(ord(text[0]))]


def _ollama_reply(endpoint: str, payload: dict, timeout: int) -> dict:
    """Answer /api/embed the way Ollama does, one vector per input in order."""
    inputs = payload["input"]
    texts = [inputs] if isinstance(inputs, str) else inputs
    return {"embeddings": [_vector(t) for t in texts]}


def _http_error(detail: str, code: int = 400) -> NonRetryableHttpError:
    return NonRetryableHttpError(_EMBED_URL, code, detail, None, f'{{"error": "{detail}"}}')


class TestBatching(ProviderRegistryIsolation):
    """get_embeddings() splits into batches and keeps the input order."""

    _TEXTS = ["a", "bb", "ccc", "dddd", "eeeee"]

    def test_ollama_batches_keep_order_and_length(self) -> None:
        with patch.object(OllamaProvider, "_post", side_effect=_ollama_reply) as post:
            vectors = get_embeddings("ollama", "m", self._TEXTS, batch_size=2)
        sent = [c.args[1]["input"] for c in post.call_args_list]
        self.assertEqual(sent, [["a", "bb"], ["ccc", "dddd"], "eeeee"])
        self.assertEqual(vectors, [_vector(t) for t in self._TEXTS])

    def test_no_batch_size_sends_one_request(self) -> None:
        with patch.object(OllamaProvider, "_post", side_effect=_ollama_reply) as post:
            vectors = get_embeddings("ollama", "m", self._TEXTS)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.args[1]["input"], self._TEXTS)
        self.assertEqual(len(vectors), len(self._TEXTS))

    def test_a_batch_larger_than_the_list_is_one_request(self) -> None:
        with patch.object(OllamaProvider, "_post", side_effect=_ollama_reply) as post:
            get_embeddings("ollama", "m", self._TEXTS, batch_size=100)
        self.assertEqual(post.call_count, 1)

    def test_openai_compatible_batches_are_reordered_by_index(self) -> None:
        def reply(endpoint: str, payload: dict, timeout: int) -> dict:
            texts = payload["input"] if isinstance(payload["input"], list) else [payload["input"]]
            data = [{"index": i, "embedding": _vector(t)} for i, t in enumerate(texts)]
            return {"data": list(reversed(data))}

        with patch.object(LlamaCppProvider, "_post", side_effect=reply) as post:
            vectors = get_embeddings("llamacpp", "m", self._TEXTS, batch_size=3)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(vectors, [_vector(t) for t in self._TEXTS])

    def test_an_invalid_batch_size_is_refused_before_sending(self) -> None:
        with patch.object(OllamaProvider, "_post", side_effect=AssertionError("sent")):
            for bad in (0, -1, True, 2.5, "2"):
                with self.subTest(batch_size=bad):
                    with self.assertRaises(ValueError):
                        get_embeddings("ollama", "m", self._TEXTS, batch_size=bad)
            with self.assertRaises(ValueError):
                get_embeddings("ollama", "m", [], batch_size=0)

    def test_the_rate_limit_applies_once_per_request(self) -> None:
        with patch("unified_ai_client.client.time.sleep") as sleep, \
             patch.object(OllamaProvider, "_post", side_effect=_ollama_reply):
            get_embeddings("ollama", "m", self._TEXTS, batch_size=2, sleep_time=1)
        self.assertEqual(sleep.call_count, 3)

    def test_a_transient_failure_resends_only_its_own_batch(self) -> None:
        replies = iter([
            {"embeddings": [_vector("a"), _vector("bb")]},
            urllib.error.URLError("down"),
            {"embeddings": [_vector("ccc"), _vector("dddd")]},
            {"embeddings": [_vector("eeeee")]},
        ])

        def reply(endpoint: str, payload: dict, timeout: int) -> dict:
            item = next(replies)
            if isinstance(item, Exception):
                raise item
            return item

        with patch.object(OllamaProvider, "_post", side_effect=reply) as post:
            vectors = get_embeddings(
                "ollama", "m", self._TEXTS, batch_size=2, retry_base_delay=0
            )
        sent = [c.args[1]["input"] for c in post.call_args_list]
        self.assertEqual(sent, [["a", "bb"], ["ccc", "dddd"], ["ccc", "dddd"], "eeeee"])
        self.assertEqual(vectors, [_vector(t) for t in self._TEXTS])


class TestOllamaEmbeddingOptions(ProviderRegistryIsolation):
    """Model options reach /api/embed per call; the configured ones do not."""

    _CPU = {"num_gpu": 0, "keep_alive": "30s", "truncate": False, "context_size": 8192}

    def _payload(self, fn, *args, **kwargs) -> dict:
        with patch.object(OllamaProvider, "_post", side_effect=_ollama_reply) as post:
            fn("ollama", "bge-m3", *args, **kwargs)
        return post.call_args.args[1]

    def test_cpu_options_reach_the_payload_from_both_functions(self) -> None:
        for fn, arg in ((get_embedding, "hi"), (get_embeddings, ["hi", "there"])):
            with self.subTest(fn=fn.__name__):
                payload = self._payload(fn, arg, extra_options=dict(self._CPU))
                self.assertEqual(payload["options"], {"num_gpu": 0, "num_ctx": 8192})
                self.assertEqual(payload["keep_alive"], "30s")
                self.assertIs(payload["truncate"], False)
                self.assertNotIn("num_gpu", payload)

    def test_configured_model_options_stay_with_chat(self) -> None:
        configure_provider("ollama", num_gpu=99, context_size=32768, keep_alive="5m")
        payload = self._payload(get_embedding, "hi")
        self.assertNotIn("options", payload)
        self.assertEqual(payload["keep_alive"], "5m")

    def test_a_call_time_none_is_not_sent(self) -> None:
        payload = self._payload(get_embedding, "hi", extra_options={"num_gpu": None})
        self.assertNotIn("options", payload)

    def test_a_plain_request_is_unchanged(self) -> None:
        payload = self._payload(get_embedding, "hi")
        self.assertEqual(payload, {"model": "bge-m3", "input": "hi"})

    def test_a_chat_only_field_is_not_sent(self) -> None:
        payload = self._payload(get_embedding, "hi", extra_options={"think": True})
        self.assertNotIn("think", payload)
        self.assertNotIn("options", payload)


class TestContextLengthError(ProviderRegistryIsolation):
    """An over-long input with truncation off raises a typed, unretried error."""

    def test_a_single_input_names_itself(self) -> None:
        with patch.object(OllamaProvider, "_post", side_effect=_http_error(_TOO_LONG)) as post:
            with self.assertRaises(ContextLengthError) as ctx:
                get_embedding("ollama", "m", "long", extra_options={"truncate": False})
        self.assertEqual(post.call_count, 1)
        self.assertEqual(ctx.exception.indices, (0,))
        self.assertEqual(ctx.exception.input_index, 0)
        self.assertEqual(ctx.exception.code, 400)
        self.assertIsInstance(ctx.exception, NonRetryableHttpError)
        self.assertIsInstance(ctx.exception, urllib.error.HTTPError)

    def test_a_failing_batch_reports_its_positions_in_the_whole_list(self) -> None:
        replies = [{"embeddings": [[0.1], [0.2]]}, _http_error(_TOO_LONG)]
        with patch.object(OllamaProvider, "_post", side_effect=replies) as post:
            with self.assertRaises(ContextLengthError) as ctx:
                get_embeddings("ollama", "m", list("abcde"), batch_size=2)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(ctx.exception.indices, (2, 3))
        self.assertIsNone(ctx.exception.input_index)
        self.assertIn("inputs 2-3", str(ctx.exception))

    def test_a_failing_batch_of_one_names_the_input(self) -> None:
        replies = [{"embeddings": [[0.1], [0.2]]}] * 2 + [_http_error(_TOO_LONG)]
        with patch.object(OllamaProvider, "_post", side_effect=replies):
            with self.assertRaises(ContextLengthError) as ctx:
                get_embeddings("ollama", "m", list("abcde"), batch_size=2)
        self.assertEqual(ctx.exception.input_index, 4)

    def test_every_wording_of_the_server_is_recognised(self) -> None:
        for detail in (
            _TOO_LONG,
            "input after truncation exceeds maximum context length",
            "input exceeds maximum context length and cannot be truncated further",
        ):
            with self.subTest(detail=detail):
                with patch.object(OllamaProvider, "_post", side_effect=_http_error(detail)):
                    with self.assertRaises(ContextLengthError):
                        get_embedding("ollama", "m", "x")

    def test_another_client_error_keeps_its_own_type(self) -> None:
        with patch.object(OllamaProvider, "_post", side_effect=_http_error("invalid input type")):
            with self.assertRaises(NonRetryableHttpError) as ctx:
                get_embedding("ollama", "m", "x")
        self.assertNotIsInstance(ctx.exception, ContextLengthError)


class TestScriptEmbeddingFallback(ProviderRegistryIsolation):
    """The script protocol has no batch: one run per text, in order."""

    def _run(self, texts: list[str], **kwargs) -> tuple[list, list[dict]]:
        def reply(cmd: list[str], payload: dict, timeout: int) -> dict:
            return {"embedding": _vector(payload["text"])}

        with patch("unified_ai_client.providers.script._run_script", side_effect=reply) as run:
            vectors = get_embeddings("script", "s.py", texts, **kwargs)
        return vectors, [c.args[1] for c in run.call_args_list]

    def test_one_run_per_text_in_order(self) -> None:
        vectors, payloads = self._run(["a", "bb", "ccc"], batch_size=2)
        self.assertEqual([p["text"] for p in payloads], ["a", "bb", "ccc"])
        self.assertEqual(vectors, [_vector(t) for t in ("a", "bb", "ccc")])
        self.assertEqual(payloads[0], {"mode": "embed", "text": "a"})

    def test_extra_options_travel_merged_when_set(self) -> None:
        configure_provider("script", device="cpu")
        _, payloads = self._run(["a", "bb"], extra_options={"num_gpu": 0})
        for payload in payloads:
            self.assertEqual(payload["extra_options"], {"device": "cpu", "num_gpu": 0})


class TestEmbeddingModelPreload(ProviderRegistryIsolation):
    """An embedding-only model is loaded through /api/embed."""

    _NO_CHAT = '"bge-m3" does not support chat'

    def test_the_load_falls_back_to_the_embed_endpoint(self) -> None:
        provider = OllamaProvider(ProviderConfig(extra_options={"repeat_penalty": 1.1}))
        replies = [_http_error(self._NO_CHAT), {"embeddings": []}]
        with patch.object(OllamaProvider, "_post", side_effect=replies) as post:
            provider.preload_model("bge-m3", "30s", extra_options={"num_gpu": 0})
        self.assertEqual([c.args[0] for c in post.call_args_list], ["/api/chat", "/api/embed"])
        self.assertEqual(post.call_args.args[1], {
            "model": "bge-m3", "input": [], "keep_alive": "30s", "options": {"num_gpu": 0},
        })

    def test_a_chat_model_loads_with_one_request(self) -> None:
        provider = OllamaProvider(ProviderConfig())
        with patch.object(OllamaProvider, "_post", return_value={}) as post:
            provider.preload_model("gemma4:e2b")
        self.assertEqual([c.args[0] for c in post.call_args_list], ["/api/chat"])

    def test_another_error_is_not_masked(self) -> None:
        provider = OllamaProvider(ProviderConfig())
        missing = _http_error("model 'nope' not found", code=404)
        with patch.object(OllamaProvider, "_post", side_effect=missing) as post:
            with self.assertRaises(NonRetryableHttpError):
                provider.preload_model("nope")
        self.assertEqual(post.call_count, 1)

    def test_warm_up_succeeds_on_an_embedding_model(self) -> None:
        replies = [_http_error(self._NO_CHAT), {"embeddings": []}]
        with patch.object(OllamaProvider, "_post", side_effect=replies):
            self.assertIs(warm_up("ollama", "bge-m3"), True)


if __name__ == "__main__":
    unittest.main()
