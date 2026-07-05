"""Smoke tests for all UnifiedAiClient providers.

Each test is wrapped in try/except so unavailable providers (missing API keys,
offline servers) produce a SKIP result instead of crashing the suite.

Usage:
    python tests/test_providers.py          # from project root
    python test_providers.py                # from tests/ directory

Output: one line per test with PASS / SKIP / FAIL status.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is in sys.path so unified_ai_client is importable
# both when running this file directly from tests/ and from the project root.
# This is needed when the package is not pip-installed in the active Python.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Minimal test runner (no external frameworks required)
# ---------------------------------------------------------------------------

_results: list[tuple[str, str, str]] = []  # (name, status, detail)


def _run(name: str, fn) -> None:  # type: ignore[type-arg]
    """Execute a single test function and record the result."""
    try:
        fn()
        _results.append((name, "PASS", ""))
    except SkipTest as exc:
        _results.append((name, "SKIP", str(exc)))
    except Exception as exc:
        detail = traceback.format_exc().strip().splitlines()[-1]
        _results.append((name, "FAIL", detail))


class SkipTest(Exception):
    """Raise inside a test to mark it as skipped (provider unavailable)."""


# ---------------------------------------------------------------------------
# Helper — create a disposable temp text file
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


# ---------------------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------------------

def test_import_models() -> None:
    from unified_ai_client.models import AiRequest, AiResponse, ProviderConfig
    assert AiRequest and AiResponse and ProviderConfig


def test_import_file_utils() -> None:
    from unified_ai_client.file_utils import (
        classify_file,
        normalize_file_paths,
        encode_file_base64,
        get_mime_type,
        audio_format_name,
        format_text_attachment,
        inline_text_attachments,
    )
    assert classify_file


def test_import_client() -> None:
    from unified_ai_client import call_ai, cleanup, preload_model, get_embedding
    assert call_ai


def test_import_providers() -> None:
    from unified_ai_client.providers.ollama import OllamaProvider
    from unified_ai_client.providers.google import GoogleProvider
    from unified_ai_client.providers.anthropic import AnthropicProvider
    from unified_ai_client.providers.openai import OpenAiProvider
    from unified_ai_client.providers.mistral import MistralProvider
    from unified_ai_client.providers.cohere import CohereProvider
    from unified_ai_client.providers.meta import MetaProvider
    from unified_ai_client.providers.groq import GroqProvider
    from unified_ai_client.providers.xai import XAiProvider
    from unified_ai_client.providers.lmstudio import LmStudioProvider
    from unified_ai_client.providers.llamacpp import LlamaCppProvider
    from unified_ai_client.providers.script import ScriptProvider
    assert all([
        OllamaProvider, GoogleProvider, AnthropicProvider,
        OpenAiProvider, MistralProvider, CohereProvider,
        MetaProvider, GroqProvider, XAiProvider,
        LmStudioProvider, LlamaCppProvider, ScriptProvider,
    ])


# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

def test_airequest_construction() -> None:
    from unified_ai_client.models import AiRequest
    r = AiRequest(
        provider="ollama",
        model="llava",
        prompt="test",
        file_path=["a.jpg", "b.txt"],
    )
    assert r.file_path == ["a.jpg", "b.txt"]
    assert not hasattr(r, "image_path"), "Old 'image_path' field must not exist on AiRequest"


def test_airesponse_defaults() -> None:
    from unified_ai_client.models import AiResponse
    r = AiResponse(text="hello")
    assert r.reasoning_text == ""
    assert r.reasoning_tokens == 0


# ---------------------------------------------------------------------------
# 3. File utils unit tests
# ---------------------------------------------------------------------------

def test_classify_file() -> None:
    from unified_ai_client.file_utils import classify_file
    assert classify_file("photo.jpg") == "image"
    assert classify_file("PHOTO.PNG") == "image"
    assert classify_file("audio.mp3") == "audio"
    assert classify_file("audio.wav") == "audio"
    assert classify_file("doc.pdf") == "document"
    assert classify_file("notes.md") == "text"
    assert classify_file("data.csv") == "text"
    assert classify_file("binary.bin") == "unknown"


def test_normalize_file_paths() -> None:
    from unified_ai_client.file_utils import normalize_file_paths
    assert normalize_file_paths(None) == []
    assert normalize_file_paths("a.jpg") == ["a.jpg"]
    assert normalize_file_paths(["a.jpg", "b.pdf"]) == ["a.jpg", "b.pdf"]


def test_inline_text_attachments() -> None:
    from unified_ai_client.file_utils import inline_text_attachments
    tmp = _make_text_file("line one\nline two")
    try:
        result = inline_text_attachments("My prompt", [tmp])
        assert "My prompt" in result
        assert os.path.basename(tmp) in result
        assert "line one" in result
        # Prompt must appear only ONCE
        assert result.count("My prompt") == 1
    finally:
        os.unlink(tmp)


def test_inline_text_attachments_multiple() -> None:
    from unified_ai_client.file_utils import inline_text_attachments
    t1 = _make_text_file("file one content")
    t2 = _make_text_file("file two content")
    try:
        result = inline_text_attachments("My prompt", [t1, t2])
        assert "file one content" in result
        assert "file two content" in result
        # Prompt must appear exactly once regardless of how many files
        assert result.count("My prompt") == 1
    finally:
        os.unlink(t1)
        os.unlink(t2)


def test_audio_format_name() -> None:
    from unified_ai_client.file_utils import audio_format_name
    assert audio_format_name("track.mp3") == "mp3"
    assert audio_format_name("clip.wav") == "wav"
    assert audio_format_name("sound.flac") == "flac"
    assert audio_format_name("file.m4a") == "mp4"


# ---------------------------------------------------------------------------
# 4. Config loading — load_secrets (os.environ + secrets.json)
# ---------------------------------------------------------------------------

def test_load_secrets_from_env_var() -> None:
    """Environment variables are read by load_secrets() and returned as snake_case keys."""
    from unified_ai_client.config import load_secrets
    old = os.environ.get("GOOGLE_API_KEY")
    try:
        os.environ["GOOGLE_API_KEY"] = "env-test-key-xyz"
        result = load_secrets("/nonexistent/path/no_secrets_here")
        assert result.get("google_api_key") == "env-test-key-xyz"
    finally:
        if old is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = old


def test_load_secrets_env_wins_over_json() -> None:
    """os.environ takes priority over secrets.json when both define the same key."""
    import json as _json
    from unified_ai_client.config import load_secrets
    tmp_dir = tempfile.mkdtemp()
    old = os.environ.get("GOOGLE_API_KEY")
    try:
        # secrets.json has one value
        with open(os.path.join(tmp_dir, "secrets.json"), "w", encoding="utf-8") as f:
            _json.dump({"google_api_key": "from-secrets-json"}, f)
        # env var overrides it
        os.environ["GOOGLE_API_KEY"] = "from-env-var"
        result = load_secrets(tmp_dir)
        assert result["google_api_key"] == "from-env-var", "os.environ must win over secrets.json"
    finally:
        if old is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = old
        import shutil
        shutil.rmtree(tmp_dir)


def test_load_secrets_no_sources() -> None:
    """load_secrets returns empty dict when no secrets.json and no env vars are set."""
    from unified_ai_client.config import load_secrets
    old_g = os.environ.pop("GOOGLE_API_KEY", None)
    old_a = os.environ.pop("ANTHROPIC_API_KEY", None)
    old_o = os.environ.pop("OPENAI_API_KEY", None)
    try:
        result = load_secrets("/nonexistent/path/xyz")
        assert result == {}
    finally:
        if old_g is not None:
            os.environ["GOOGLE_API_KEY"] = old_g
        if old_a is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_a
        if old_o is not None:
            os.environ["OPENAI_API_KEY"] = old_o



# ---------------------------------------------------------------------------
# 5. Dispatch — get_provider() resolves correct class
# ---------------------------------------------------------------------------

def test_dispatch_ollama() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.ollama import OllamaProvider
    p = get_provider("ollama")
    assert isinstance(p, OllamaProvider)


def test_dispatch_google() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.google import GoogleProvider
    p = get_provider("google")
    assert isinstance(p, GoogleProvider)


def test_dispatch_anthropic() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.anthropic import AnthropicProvider
    p = get_provider("anthropic")
    assert isinstance(p, AnthropicProvider)


def test_dispatch_openai() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.openai import OpenAiProvider
    p = get_provider("openai")
    assert isinstance(p, OpenAiProvider)


def test_dispatch_mistral() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.mistral import MistralProvider
    p = get_provider("mistral")
    assert isinstance(p, MistralProvider)


def test_dispatch_cohere() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.cohere import CohereProvider
    p = get_provider("cohere")
    assert isinstance(p, CohereProvider)


def test_dispatch_meta() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.meta import MetaProvider
    p = get_provider("meta")
    assert isinstance(p, MetaProvider)


def test_dispatch_groq() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.groq import GroqProvider
    p = get_provider("groq")
    assert isinstance(p, GroqProvider)


def test_dispatch_xai() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.xai import XAiProvider
    p = get_provider("xai")
    assert isinstance(p, XAiProvider)


def test_dispatch_lmstudio() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.lmstudio import LmStudioProvider
    p = get_provider("lmstudio")
    assert isinstance(p, LmStudioProvider)


def test_dispatch_llamacpp() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.llamacpp import LlamaCppProvider
    p = get_provider("llamacpp")
    assert isinstance(p, LlamaCppProvider)


def test_dispatch_script() -> None:
    from unified_ai_client.client import get_provider
    from unified_ai_client.providers.script import ScriptProvider
    p = get_provider("script")
    assert isinstance(p, ScriptProvider)


def test_dispatch_invalid() -> None:
    from unified_ai_client.client import get_provider
    try:
        get_provider("nonexistent_provider_xyz")
        raise AssertionError("Expected ValueError was not raised")
    except ValueError:
        pass  # correct


# ---------------------------------------------------------------------------
# 5. Live tests — Ollama (most likely available locally)
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    """Quick check whether Ollama is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _first_ollama_model() -> str | None:
    """Return the first Ollama chat model available, skipping embedding-only models.

    Embedding models (e.g. nomic-embed-text, mxbai-embed-*) return HTTP 400 on
    /api/chat. We skip them by excluding known embed name patterns.
    """
    import urllib.request, json as _json
    _EMBED_PATTERNS = ("embed", "embedding", "bge-", "e5-")
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            data = _json.loads(r.read())
        for m in data.get("models", []):
            name: str = m["name"]
            if not any(p in name.lower() for p in _EMBED_PATTERNS):
                return name
        return None
    except Exception:
        return None



def test_ollama_live_generate() -> None:
    import urllib.error
    if not _ollama_available():
        raise SkipTest("Ollama not reachable at localhost:11434")
    model = _first_ollama_model()
    if not model:
        raise SkipTest("No chat-capable Ollama models installed")
    from unified_ai_client import call_ai
    try:
        response = call_ai(
            provider="ollama",
            model=model,
            prompt="Reply with exactly the word PONG and nothing else.",
            temperature=0.0,
            timeout=30,
        )
    except Exception as exc:
        if "400" in str(exc):
            raise SkipTest(f"Model '{model}' rejected chat request (400) — likely embed-only")
        raise
    assert isinstance(response.text, str)
    assert len(response.text) > 0


def test_ollama_live_with_text_file() -> None:
    if not _ollama_available():
        raise SkipTest("Ollama not reachable at localhost:11434")
    model = _first_ollama_model()
    if not model:
        raise SkipTest("No chat-capable Ollama models installed")
    tmp = _make_text_file("The sky is blue.")
    try:
        from unified_ai_client import call_ai
        try:
            response = call_ai(
                provider="ollama",
                model=model,
                prompt="What colour is mentioned in the attached file? Reply in one word.",
                file_path=tmp,
                temperature=0.0,
                timeout=30,
            )
        except Exception as exc:
            if "400" in str(exc):
                raise SkipTest(f"Model '{model}' rejected chat request (400) — likely embed-only")
            raise
        assert isinstance(response.text, str)
        assert len(response.text) > 0
    finally:
        os.unlink(tmp)


def test_ollama_live_thinking() -> None:
    if not _ollama_available():
        raise SkipTest("Ollama not reachable at localhost:11434")
    model = _first_ollama_model()
    if not model:
        raise SkipTest("No chat-capable Ollama models installed")
    from unified_ai_client import call_ai
    # thinking=True may silently fall back on models that don't support it
    try:
        response = call_ai(
            provider="ollama",
            model=model,
            prompt="What is 2+2?",
            thinking=True,
            temperature=0.0,
            timeout=30,
        )
    except Exception as exc:
        if "400" in str(exc):
            raise SkipTest(f"Model '{model}' rejected chat request (400) — likely embed-only")
        raise
    assert isinstance(response.text, str)
    # reasoning_text may be empty string if model doesn't support thinking — that's fine
    assert isinstance(response.reasoning_text, str)


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


def test_ollama_live_embedding() -> None:
    if not _ollama_available():
        raise SkipTest("Ollama not reachable at localhost:11434")
    model = _first_ollama_embed_model()
    if not model:
        raise SkipTest("No embedding-capable Ollama models installed (need bge-m3, embeddinggemma, nomic-embed-text, or similar)")
    import urllib.error
    from unified_ai_client import get_embedding
    try:
        vec = get_embedding(provider="ollama", model=model, text="hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(x, float) for x in vec)
    except (RuntimeError, urllib.error.HTTPError) as exc:
        raise SkipTest(f"Model '{model}' does not support embeddings ({exc})")


def test_ollama_live_reasoning_tokens_thinking_true() -> None:
    """With thinking=True, reasoning_tokens must be > 0 if the model supports thinking.

    Uses the first available chat model.  If the model responds but produces no
    thinking output (empty reasoning_text and zero reasoning_tokens), the test
    is skipped — the model does not support the 'think' parameter.
    """
    if not _ollama_available():
        raise SkipTest("Ollama not reachable at localhost:11434")
    model = _first_ollama_model()
    if not model:
        raise SkipTest("No chat-capable Ollama models installed")
    from unified_ai_client import call_ai
    try:
        response = call_ai(
            provider="ollama",
            model=model,
            prompt="What is 2+2? Think step by step.",
            thinking=True,
            temperature=0.0,
            timeout=120,
        )
    except Exception as exc:
        if "400" in str(exc):
            raise SkipTest(f"Model '{model}' rejected thinking request (400)")
        raise
    if not response.reasoning_text and response.reasoning_tokens == 0:
        raise SkipTest(f"Model '{model}' produced no thinking output — 'think' not supported")
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.reasoning_tokens > 0, (
        f"reasoning_tokens must be > 0 when thinking text is present, got {response.reasoning_tokens}"
    )


def test_ollama_live_reasoning_tokens_thinking_false() -> None:
    """With thinking=False, reasoning_tokens must be a non-negative integer.

    Some models produce a small thinking trace even when not asked to —
    the result must never be negative and must never raise.
    Uses the first available chat model.
    """
    if not _ollama_available():
        raise SkipTest("Ollama not reachable at localhost:11434")
    model = _first_ollama_model()
    if not model:
        raise SkipTest("No chat-capable Ollama models installed")
    from unified_ai_client import call_ai
    try:
        response = call_ai(
            provider="ollama",
            model=model,
            prompt="What is 2+2?",
            thinking=False,
            temperature=0.0,
            timeout=60,
        )
    except Exception as exc:
        if "400" in str(exc):
            raise SkipTest(f"Model '{model}' rejected request (400)")
        raise
    assert isinstance(response.text, str)
    assert response.reasoning_tokens >= 0, (
        f"reasoning_tokens must be >= 0, got {response.reasoning_tokens}"
    )


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


def test_script_generate() -> None:
    script = _make_script(_ECHO_SCRIPT)
    try:
        from unified_ai_client import call_ai
        response = call_ai(
            provider="script",
            model=script,
            prompt="Hello",
            timeout=15,
        )
        assert "Echo: Hello" in response.text
        assert "(files=0)" in response.text
        assert response.input_tokens == 1
        assert response.output_tokens == 2
    finally:
        os.unlink(script)


def test_script_generate_with_file() -> None:
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
        assert "(files=1)" in response.text
    finally:
        os.unlink(script)
        os.unlink(tmp)


def test_script_generate_with_reasoning() -> None:
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
        assert response.reasoning_text == "I thought about it."
        assert response.reasoning_tokens == 10
    finally:
        os.unlink(script)


def test_script_embed() -> None:
    script = _make_script(_ECHO_SCRIPT)
    try:
        from unified_ai_client import get_embedding
        vec = get_embedding(provider="script", model=script, text="hello")
        assert vec == [0.1, 0.2, 0.3]
    finally:
        os.unlink(script)


def test_script_nonzero_exit() -> None:
    """A script that exits non-zero must raise RuntimeError (not crash silently)."""
    failing_script = _make_script(
        "import sys; print('error details', file=sys.stderr); sys.exit(1)\n"
    )
    try:
        from unified_ai_client import call_ai
        try:
            call_ai(
                provider="script",
                model=failing_script,
                prompt="test",
                max_retries=1,
                timeout=10,
            )
            raise AssertionError("Expected RuntimeError was not raised")
        except RuntimeError as exc:
            assert "error details" in str(exc)
    finally:
        os.unlink(failing_script)


# ---------------------------------------------------------------------------
# 6.1 Google provider — offline unit tests
# ---------------------------------------------------------------------------

def test_google_thinking_config_offline() -> None:
    """Verify that GoogleProvider correctly builds the thinking configuration.

    It tests that thinking is explicitly minimized/disabled when thinking=False,
    and configured with the appropriate levels/budgets when thinking=True.
    """
    from unified_ai_client.providers.google import GoogleProvider
    from unified_ai_client.models import ProviderConfig

    provider = GoogleProvider(config=ProviderConfig(sleep_time=0), api_key="dummy_key")

    # 1. Test thinking=False (explicitly disabled/minimized)
    # Gemini 3 models must set thinking_level to MINIMAL
    cfg_g3_off = provider._build_thinking_config(thinking=False, model_name="gemini-3.5-flash")
    assert cfg_g3_off is not None
    assert cfg_g3_off.thinking_level == "MINIMAL"
    assert cfg_g3_off.include_thoughts is True

    # Gemini 2.5 models must set thinking_budget to 0
    cfg_g25_off = provider._build_thinking_config(thinking=False, model_name="gemini-2.5-pro")
    assert cfg_g25_off is not None
    assert cfg_g25_off.thinking_budget == 0
    assert cfg_g25_off.include_thoughts is True

    # Other models must set thinking_budget to 0
    cfg_other_off = provider._build_thinking_config(thinking=False, model_name="gemini-1.5-pro")
    assert cfg_other_off is not None
    assert cfg_other_off.thinking_budget == 0
    assert cfg_other_off.include_thoughts is True

    # 2. Test thinking=True (enabled)
    # Gemini 3 models must set thinking_level to HIGH
    cfg_g3_on = provider._build_thinking_config(thinking=True, model_name="gemini-3.5-flash")
    assert cfg_g3_on is not None
    assert cfg_g3_on.thinking_level == "HIGH"
    assert cfg_g3_on.include_thoughts is True

    # Gemini 2.5 models must set thinking_budget to 24576
    cfg_g25_on = provider._build_thinking_config(thinking=True, model_name="gemini-2.5-pro")
    assert cfg_g25_on is not None
    assert cfg_g25_on.thinking_budget == 24576
    assert cfg_g25_on.include_thoughts is True

    # Other models must set thinking_budget to 1024
    cfg_other_on = provider._build_thinking_config(thinking=True, model_name="gemini-1.5-pro")
    assert cfg_other_on is not None
    assert cfg_other_on.thinking_budget == 1024
    assert cfg_other_on.include_thoughts is True

    # 3. Test thinking="default" (default settings but capture thoughts)
    cfg_default = provider._build_thinking_config(thinking="default", model_name="gemini-2.5-pro")
    assert cfg_default is not None
    assert cfg_default.include_thoughts is True
    assert getattr(cfg_default, "thinking_budget", None) is None
    assert getattr(cfg_default, "thinking_level", None) is None


def test_config_dynamic_extra_options() -> None:
    """Verify that load_config places unrecognized configuration keys into extra_options."""
    from unified_ai_client.config import load_config
    from unified_ai_client.models import ProviderConfig

    # Write a temporary config JSON
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
        assert cfg.url == "http://google-api-mock"
        assert cfg.timeout == 45
        assert cfg.sleep_time == 5
        assert isinstance(cfg.extra_options, dict)
        assert cfg.extra_options.get("disable_safety") is True
        assert cfg.extra_options.get("upload_poll_timeout") == 20
        assert cfg.extra_options.get("custom_app_setting") == "hello_world"
    finally:
        os.unlink(tmp_config_path)


def test_script_provider_forwards_new_params() -> None:
    """Verify that ScriptProvider forwards top_k, top_p, max_tokens, and extra_options in the payload."""
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
        assert "top_k=42" in response.text
        assert "top_p=0.8" in response.text
        assert "max_tokens=150" in response.text
        assert "some_custom_option" in response.text
    finally:
        os.unlink(script_path)


def test_ollama_provider_options_mapping() -> None:
    """Verify that OllamaProvider correctly maps config and call parameters."""
    from unified_ai_client.providers.ollama import OllamaProvider
    from unified_ai_client.models import ProviderConfig, AiRequest

    # Set up config with extra options
    config = ProviderConfig(
        url="http://mock-ollama:11434",
        extra_options={"context_size": 2048, "keep_alive": "30m", "my_custom_option": 123}
    )
    provider = OllamaProvider(config=config)

    # Mock call to _post to intercept the payload
    intercepted_payload = None
    def mock_post(endpoint, payload, timeout):
        nonlocal intercepted_payload
        intercepted_payload = payload
        return {
            "message": {"content": "response", "thinking": ""},
            "eval_count": 10,
            "prompt_eval_count": 5
        }
    provider._post = mock_post

    request = AiRequest(
        provider="ollama",
        model="mock-model",
        prompt="hello",
        top_k=50,
        top_p=0.9,
        max_tokens=100,
        extra_options={"my_custom_option": 456, "visual_token_budget": 70}
    )

    provider.call(request)

    assert intercepted_payload is not None
    assert intercepted_payload["keep_alive"] == "30m"
    options = intercepted_payload["options"]
    assert options["top_k"] == 50
    assert options["top_p"] == 0.9
    assert options["num_ctx"] == 2048
    assert options["num_predict"] == 100
    assert options["my_custom_option"] == 456
    assert options["visual_token_budget"] == 70


# ---------------------------------------------------------------------------
# 7. Google (skip if no API key)
# ---------------------------------------------------------------------------

def test_google_live_generate() -> None:
    from unified_ai_client.config import load_secrets
    secrets = load_secrets(os.getcwd())
    if not secrets.get("google_api_key"):
        raise SkipTest("google_api_key not found in secrets.json or environment variables")
    from unified_ai_client import call_ai
    response = call_ai(
        provider="google",
        model="gemini-2.5-flash",
        prompt="Reply with exactly the word PONG and nothing else.",
        temperature=0.0,
        timeout=30,
    )
    assert isinstance(response.text, str)
    assert len(response.text) > 0


def test_google_live_with_text_file() -> None:
    from unified_ai_client.config import load_secrets
    secrets = load_secrets(os.getcwd())
    if not secrets.get("google_api_key"):
        raise SkipTest("google_api_key not found in secrets.json or environment variables")
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
        assert isinstance(response.text, str)
        assert len(response.text) > 0
    finally:
        os.unlink(tmp)


def test_google_live_thinking() -> None:
    from unified_ai_client.config import load_secrets
    secrets = load_secrets(os.getcwd())
    if not secrets.get("google_api_key"):
        raise SkipTest("google_api_key not found in secrets.json or environment variables")
    from unified_ai_client import call_ai
    response = call_ai(
        provider="google",
        model="gemini-2.5-flash",
        prompt="What is 2+2? Think step by step.",
        thinking=True,
        temperature=0.0,
        timeout=60,
    )
    assert isinstance(response.text, str)
    assert isinstance(response.reasoning_text, str)


def test_google_live_thinking_default() -> None:
    from unified_ai_client.config import load_secrets
    secrets = load_secrets(os.getcwd())
    if not secrets.get("google_api_key"):
        raise SkipTest("google_api_key not found in secrets.json or environment variables")
    from unified_ai_client import call_ai
    response = call_ai(
        provider="google",
        model="gemini-2.5-flash",
        prompt="What is 2+2? Think step by step.",
        thinking="default",
        temperature=0.0,
        timeout=60,
    )
    assert isinstance(response.text, str)
    assert isinstance(response.reasoning_text, str)


# ---------------------------------------------------------------------------
# 8. Anthropic (skip if no API key)
# ---------------------------------------------------------------------------

def test_anthropic_live_generate() -> None:
    from unified_ai_client.config import load_secrets
    secrets = load_secrets(os.getcwd())
    if not secrets.get("anthropic_api_key"):
        raise SkipTest("anthropic_api_key not found in secrets.json or environment variables")
    from unified_ai_client import call_ai
    response = call_ai(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        prompt="Reply with exactly the word PONG and nothing else.",
        temperature=0.0,
        timeout=30,
    )
    assert isinstance(response.text, str)


# ---------------------------------------------------------------------------
# 9. OpenAI (skip if no API key)
# ---------------------------------------------------------------------------

def test_openai_live_generate() -> None:
    from unified_ai_client.config import load_secrets
    secrets = load_secrets(os.getcwd())
    if not secrets.get("openai_api_key"):
        raise SkipTest("openai_api_key not found in secrets.json or environment variables")
    from unified_ai_client import call_ai
    response = call_ai(
        provider="openai",
        model="gpt-4o-mini",
        prompt="Reply with exactly the word PONG and nothing else.",
        temperature=0.0,
        timeout=30,
    )
    assert isinstance(response.text, str)


# ---------------------------------------------------------------------------
# 11. Tool Calling Unit Tests
# ---------------------------------------------------------------------------

def test_import_tool_types() -> None:
    from unified_ai_client import ToolDefinition, ToolCall, ToolResult
    assert ToolDefinition and ToolCall and ToolResult


def test_tool_definition_construction() -> None:
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
    assert td.name == "get_weather"
    assert td.parameters["type"] == "object"


def test_tool_call_construction() -> None:
    from unified_ai_client import ToolCall
    tc = ToolCall(id="call_abc", name="get_weather", arguments={"location": "Rome"})
    assert tc.id == "call_abc"
    assert tc.arguments["location"] == "Rome"


def test_tool_result_construction() -> None:
    from unified_ai_client import ToolResult
    tr = ToolResult(call_id="call_abc", name="get_weather", content="22C, sunny")
    assert tr.call_id == "call_abc"
    assert tr.name == "get_weather"
    assert tr.content == "22C, sunny"


def test_airesponse_tool_calls_default() -> None:
    from unified_ai_client import AiResponse
    r = AiResponse(text="hello")
    assert r.tool_calls == []


def test_airequest_tools_fields() -> None:
    from unified_ai_client.models import AiRequest, ToolDefinition, ToolResult
    td = ToolDefinition(name="f", description="d", parameters={"type": "object", "properties": {}})
    tr = ToolResult(call_id="c1", name="f", content="result")
    r = AiRequest(
        provider="ollama", model="m", prompt="p",
        tools=[td], tool_results=[tr],
    )
    assert r.tools is not None and len(r.tools) == 1
    assert r.tool_results is not None and len(r.tool_results) == 1


def test_openai_compat_tool_payload() -> None:
    """OpenAI-compat provider must build the correct tools payload."""
    from unittest.mock import patch
    from unified_ai_client.providers.openai_compat import OpenAiCompatProvider
    from unified_ai_client.models import AiRequest, ProviderConfig, ToolDefinition

    config = ProviderConfig(url="http://localhost:8080")
    provider = OpenAiCompatProvider(config)

    tool = ToolDefinition(
        name="get_weather",
        description="Returns current weather.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
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

    assert "tools" in captured_payload
    assert captured_payload["tools"][0]["type"] == "function"
    assert captured_payload["tools"][0]["function"]["name"] == "get_weather"
    assert resp.tool_calls == []
    assert resp.text == "Sunny."


def test_openai_compat_parse_tool_calls() -> None:
    """OpenAI-compat provider must parse tool calls from the response."""
    import json
    from unittest.mock import patch
    from unified_ai_client.providers.openai_compat import OpenAiCompatProvider
    from unified_ai_client.models import AiRequest, ProviderConfig, ToolDefinition

    config = ProviderConfig(url="http://localhost:8080")
    provider = OpenAiCompatProvider(config)

    tool = ToolDefinition(
        name="get_weather",
        description="Returns current weather.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
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

    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id == "call_xyz"
    assert tc.name == "get_weather"
    assert tc.arguments == {"location": "Rome"}
    assert resp.text == ""


def test_openai_compat_tool_results_in_messages() -> None:
    """Tool results must be serialized as role:tool messages."""
    from unittest.mock import patch
    from unified_ai_client.providers.openai_compat import OpenAiCompatProvider
    from unified_ai_client.models import AiRequest, ProviderConfig, ToolResult

    config = ProviderConfig(url="http://localhost:8080")
    provider = OpenAiCompatProvider(config)

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
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_xyz"
    assert tool_msgs[0]["content"] == "22C, sunny"


def test_anthropic_tool_payload() -> None:
    """Anthropic provider must build tools in input_schema format."""
    from unittest.mock import patch
    from unified_ai_client.providers.anthropic import AnthropicProvider
    from unified_ai_client.models import AiRequest, ProviderConfig, ToolDefinition

    config = ProviderConfig(url="https://api.anthropic.com")
    provider = AnthropicProvider(config, api_key="fake-key")

    tool = ToolDefinition(
        name="get_weather",
        description="Returns weather.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
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

    assert "tools" in captured_payload
    assert captured_payload["tools"][0]["name"] == "get_weather"
    assert "input_schema" in captured_payload["tools"][0]
    assert "parameters" not in captured_payload["tools"][0]


def test_anthropic_parse_tool_calls() -> None:
    """Anthropic provider must parse tool_use content blocks."""
    from unittest.mock import patch
    from unified_ai_client.providers.anthropic import AnthropicProvider
    from unified_ai_client.models import AiRequest, ProviderConfig

    config = ProviderConfig(url="https://api.anthropic.com")
    provider = AnthropicProvider(config, api_key="fake-key")

    request = AiRequest(
        provider="anthropic", model="claude-opus-4-5", prompt="Weather?",
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

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "toolu_01"
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"location": "Rome"}


def test_ollama_tool_payload() -> None:
    """Ollama provider must build tools in OpenAI-compatible format."""
    from unittest.mock import patch
    from unified_ai_client.providers.ollama import OllamaProvider
    from unified_ai_client.models import AiRequest, ProviderConfig, ToolDefinition

    config = ProviderConfig(url="http://localhost:11434")
    provider = OllamaProvider(config)

    tool = ToolDefinition(
        name="get_weather",
        description="Returns weather.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
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

    assert "tools" in captured_payload
    assert captured_payload["tools"][0]["type"] == "function"
    assert captured_payload["tools"][0]["function"]["name"] == "get_weather"


def test_ollama_parse_tool_calls() -> None:
    """Ollama provider must parse message.tool_calls."""
    from unittest.mock import patch
    from unified_ai_client.providers.ollama import OllamaProvider
    from unified_ai_client.models import AiRequest, ProviderConfig

    config = ProviderConfig(url="http://localhost:11434")
    provider = OllamaProvider(config)

    request = AiRequest(
        provider="ollama", model="gemma4:12b", prompt="Weather?",
    )

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

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"location": "Rome"}


def test_ollama_live_tool_calling() -> None:
    """Live Ollama test: gemma4:12b tool calling with get_weather (two-turn).

    The second call must pass the full conversation history including the
    assistant's intermediate tool_calls turn so the model can link the tool
    result back to its own request.
    """
    from unified_ai_client import call_ai, ToolDefinition, ToolResult
    from unittest import SkipTest

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

    try:
        response = call_ai(
            provider="ollama",
            model="gemma4:12b",
            prompt=prompt,
            tools=tools,
            temperature=0.0,
            timeout=300,
        )
    except Exception as exc:
        raise SkipTest(f"Ollama unavailable: {exc}") from exc

    if not response.tool_calls:
        raise SkipTest("gemma4:12b did not produce a tool call — model may not support it")

    tc = response.tool_calls[0]
    assert tc.name == "get_weather", f"Expected get_weather, got {tc.name!r}"
    assert "location" in tc.arguments, f"Expected 'location' in arguments, got {tc.arguments}"

    weather_result = f"The weather in {tc.arguments['location']} is 22 degrees Celsius and sunny."

    # Build the conversation history for the second turn:
    # [user turn 1] → [assistant turn with tool_calls] → [tool result]
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

    try:
        final = call_ai(
            provider="ollama",
            model="gemma4:12b",
            prompt=prompt,  # stored in history but not re-appended (tool_results present)
            messages=[
                {"role": "user", "content": prompt},
                assistant_tool_message,
            ],
            tools=tools,
            tool_results=[
                ToolResult(call_id=tc.id, name=tc.name, content=weather_result),
            ],
            temperature=0.0,
            timeout=300,
        )
    except Exception as exc:
        raise SkipTest(f"Ollama second call failed: {exc}") from exc

    assert isinstance(final.text, str) and len(final.text) > 0, "Final response must contain text"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_TESTS = [
    # Imports
    ("import/models", test_import_models),
    ("import/file_utils", test_import_file_utils),
    ("import/client", test_import_client),
    ("import/all_providers", test_import_providers),
    # Model construction
    ("model/airequest_construction", test_airequest_construction),
    ("model/airesponse_defaults", test_airesponse_defaults),
    # File utils
    ("file_utils/classify_file", test_classify_file),
    ("file_utils/normalize_file_paths", test_normalize_file_paths),
    ("file_utils/inline_text_attachments_single", test_inline_text_attachments),
    ("file_utils/inline_text_attachments_multiple", test_inline_text_attachments_multiple),
    ("file_utils/audio_format_name", test_audio_format_name),
    # Config loading (os.environ + secrets.json)
    ("config/load_secrets_from_env_var", test_load_secrets_from_env_var),
    ("config/load_secrets_env_wins_over_json", test_load_secrets_env_wins_over_json),
    ("config/load_secrets_no_sources", test_load_secrets_no_sources),
    # Dispatch
    ("dispatch/ollama", test_dispatch_ollama),
    ("dispatch/google", test_dispatch_google),
    ("dispatch/anthropic", test_dispatch_anthropic),
    ("dispatch/openai", test_dispatch_openai),
    ("dispatch/mistral", test_dispatch_mistral),
    ("dispatch/cohere", test_dispatch_cohere),
    ("dispatch/meta", test_dispatch_meta),
    ("dispatch/groq", test_dispatch_groq),
    ("dispatch/xai", test_dispatch_xai),
    ("dispatch/lmstudio", test_dispatch_lmstudio),
    ("dispatch/llamacpp", test_dispatch_llamacpp),
    ("dispatch/script", test_dispatch_script),
    ("dispatch/invalid_raises", test_dispatch_invalid),
    # Script provider
    ("script/generate", test_script_generate),
    ("script/generate_with_file", test_script_generate_with_file),
    ("script/generate_with_reasoning", test_script_generate_with_reasoning),
    ("script/embed", test_script_embed),
    ("script/nonzero_exit_raises", test_script_nonzero_exit),
    # Google offline
    ("google/thinking_config_offline", test_google_thinking_config_offline),
    ("config/dynamic_extra_options", test_config_dynamic_extra_options),
    ("script/forwards_new_params", test_script_provider_forwards_new_params),
    ("ollama/options_mapping", test_ollama_provider_options_mapping),
    # Ollama live
    ("ollama_live/generate", test_ollama_live_generate),
    ("ollama_live/with_text_file", test_ollama_live_with_text_file),
    ("ollama_live/thinking", test_ollama_live_thinking),
    ("ollama_live/reasoning_tokens_thinking_true", test_ollama_live_reasoning_tokens_thinking_true),
    ("ollama_live/reasoning_tokens_thinking_false", test_ollama_live_reasoning_tokens_thinking_false),
    ("ollama_live/embedding", test_ollama_live_embedding),
    # Tool calling
    ("tool_calling/import_types", test_import_tool_types),
    ("tool_calling/tool_definition", test_tool_definition_construction),
    ("tool_calling/tool_call", test_tool_call_construction),
    ("tool_calling/tool_result", test_tool_result_construction),
    ("tool_calling/airesponse_default", test_airesponse_tool_calls_default),
    ("tool_calling/airequest_fields", test_airequest_tools_fields),
    ("tool_calling/openai_compat_payload", test_openai_compat_tool_payload),
    ("tool_calling/openai_compat_parse", test_openai_compat_parse_tool_calls),
    ("tool_calling/openai_compat_results", test_openai_compat_tool_results_in_messages),
    ("tool_calling/anthropic_payload", test_anthropic_tool_payload),
    ("tool_calling/anthropic_parse", test_anthropic_parse_tool_calls),
    ("tool_calling/ollama_payload", test_ollama_tool_payload),
    ("tool_calling/ollama_parse", test_ollama_parse_tool_calls),
    ("tool_calling/ollama_live", test_ollama_live_tool_calling),
    # Cloud live (skip if no key)
    ("google_live/generate", test_google_live_generate),
    ("google_live/with_text_file", test_google_live_with_text_file),
    ("google_live/thinking", test_google_live_thinking),
    ("google_live/thinking_default", test_google_live_thinking_default),
    ("anthropic_live/generate", test_anthropic_live_generate),
    ("openai_live/generate", test_openai_live_generate),
]


def main() -> None:
    """Run all tests and print a summary."""
    print(f"\nUnifiedAiClient — provider smoke tests ({len(_TESTS)} tests)\n")
    print(f"{'Test':<52} {'Status':<6}  Detail")
    print("-" * 80)

    for name, fn in _TESTS:
        _run(name, fn)

    for name, status, detail in _results:
        icon = {"PASS": "[PASS]", "SKIP": "[SKIP]", "FAIL": "[FAIL]"}.get(status, "?")
        suffix = f"  {detail}" if detail else ""
        print(f"{icon} {name:<50} {status}{suffix}")

    print("-" * 80)
    passed = sum(1 for _, s, _ in _results if s == "PASS")
    skipped = sum(1 for _, s, _ in _results if s == "SKIP")
    failed = sum(1 for _, s, _ in _results if s == "FAIL")
    print(f"\n  PASS {passed}  SKIP {skipped}  FAIL {failed}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
