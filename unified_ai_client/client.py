from __future__ import annotations

import atexit
import os
import sys
import threading
from pathlib import Path
from typing import Any

from unified_ai_client.models import AiRequest, AiResponse, ProviderConfig
from unified_ai_client.providers.base import BaseProvider

# --- Constants ---
_LIB_ROOT = Path(__file__).parent.parent
_DEFAULT_CONFIG_PATH = str(os.path.join(_LIB_ROOT, "config.json"))

# --- Module-level mutable config path (overridable via set_default_config) ---
_effective_config_path: str = _DEFAULT_CONFIG_PATH

# --- Thread-safe global provider cache ---
_PROVIDERS: dict[tuple[str, str], BaseProvider] = {}
_PROVIDERS_LOCK = threading.Lock()

# --- Resource cleanup registration flag ---
_CLEANUP_REGISTERED = False
_CLEANUP_LOCK = threading.Lock()


def set_default_config(path: str) -> None:
    """Override the default config.json path used by all subsequent provider calls.

    Call this once at application startup (e.g., in the entry point) to direct
    UnifiedAiClient to a project-specific config.json instead of the library default.
    Clears the provider cache so that the next call picks up the new configuration.

    Args:
        path: Absolute path to the project's config.json file.
    """
    global _effective_config_path, _PROVIDERS
    _effective_config_path = path
    with _PROVIDERS_LOCK:
        _PROVIDERS.clear()


def get_provider(provider_name: str) -> BaseProvider:
    """Thread-safe factory to resolve, configure, and cache provider instances.

    Uses the currently active config path (set via set_default_config or the
    library default). Provider instances are cached per (provider_name, config_path)
    pair, so different config paths yield independent instances.

    Args:
        provider_name: The lower-case provider name (e.g. 'ollama', 'google').

    Returns:
        An instance of BaseProvider.

    Raises:
        ValueError: If the provider name is not supported.
    """
    provider_name = provider_name.strip().lower()
    cache_key = (provider_name, _effective_config_path)

    if cache_key in _PROVIDERS:
        return _PROVIDERS[cache_key]

    with _PROVIDERS_LOCK:
        if cache_key in _PROVIDERS:
            return _PROVIDERS[cache_key]

        # 1. Load config from the effective path
        from unified_ai_client.config import load_config
        config = load_config(_effective_config_path, ProviderConfig, section=provider_name)

        # 2. Search for secrets.json in the current working directory
        # (= the consuming project's root when the script is launched from there)
        from unified_ai_client.config import load_secrets
        secrets = load_secrets(os.getcwd())

        # 3. Extract API credentials from secrets
        api_key_google = secrets.get("google_api_key")
        api_key_anthropic = secrets.get("anthropic_api_key")
        api_key_openai = secrets.get("openai_api_key")
        api_key_mistral = secrets.get("mistral_api_key")
        api_key_cohere = secrets.get("cohere_api_key")
        api_key_meta = secrets.get("meta_api_key")
        api_key_groq = secrets.get("groq_api_key")
        api_key_xai = secrets.get("xai_api_key")

        # 5. Instantiate provider adapter
        if provider_name == "ollama":
            from unified_ai_client.providers.ollama import OllamaProvider
            provider_instance: BaseProvider = OllamaProvider(config)
        elif provider_name == "google":
            from unified_ai_client.providers.google import GoogleProvider
            provider_instance = GoogleProvider(config, api_key=api_key_google)
        elif provider_name == "anthropic":
            from unified_ai_client.providers.anthropic import AnthropicProvider
            provider_instance = AnthropicProvider(config, api_key=api_key_anthropic)
        elif provider_name == "openai":
            from unified_ai_client.providers.openai import OpenAiProvider
            provider_instance = OpenAiProvider(config, api_key=api_key_openai)
        elif provider_name == "mistral":
            from unified_ai_client.providers.mistral import MistralProvider
            provider_instance = MistralProvider(config, api_key=api_key_mistral)
        elif provider_name == "cohere":
            from unified_ai_client.providers.cohere import CohereProvider
            provider_instance = CohereProvider(config, api_key=api_key_cohere)
        elif provider_name == "meta":
            from unified_ai_client.providers.meta import MetaProvider
            provider_instance = MetaProvider(config, api_key=api_key_meta)
        elif provider_name == "groq":
            from unified_ai_client.providers.groq import GroqProvider
            provider_instance = GroqProvider(config, api_key=api_key_groq)
        elif provider_name == "xai":
            from unified_ai_client.providers.xai import XAiProvider
            provider_instance = XAiProvider(config, api_key=api_key_xai)
        elif provider_name == "lmstudio":
            from unified_ai_client.providers.lmstudio import LmStudioProvider
            provider_instance = LmStudioProvider(config)
        elif provider_name == "llamacpp":
            from unified_ai_client.providers.llamacpp import LlamaCppProvider
            provider_instance = LlamaCppProvider(config)
        elif provider_name == "script":
            from unified_ai_client.providers.script import ScriptProvider
            provider_instance = ScriptProvider(config)
        else:
            raise ValueError(
                f"Unsupported AI provider: '{provider_name}'. "
                f"Supported providers: 'ollama', 'google', 'anthropic', 'openai', "
                f"'mistral', 'cohere', 'meta', 'groq', 'xai', "
                f"'lmstudio', 'llamacpp', 'script'."
            )

        _PROVIDERS[cache_key] = provider_instance
        return provider_instance


def _register_cleanup() -> None:
    """Register the cleanup callback once with atexit."""
    global _CLEANUP_REGISTERED
    if not _CLEANUP_REGISTERED:
        with _CLEANUP_LOCK:
            if not _CLEANUP_REGISTERED:
                atexit.register(cleanup)
                _CLEANUP_REGISTERED = True


def cleanup() -> None:
    """Purge all remote resource caches across all cached providers.

    Specifically deletes uploaded Google AI files to release cloud quota.
    Called automatically at process exit via atexit, and can also be called
    explicitly for eager cleanup (e.g. in finally blocks).
    """
    with _PROVIDERS_LOCK:
        for provider in _PROVIDERS.values():
            try:
                provider.cleanup()
            except Exception:
                pass


def call_ai(
    provider: str,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    messages: list[dict] | None = None,
    file_path: str | list[str] | None = None,
    temperature: float = 0.7,
    thinking: bool | str = "default",
    format_json: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
    retry_base_delay: float = 5.0,
    top_k: int = 64,
    top_p: float = 0.95,
    max_tokens: int | None = None,
    sleep_time: int | None = None,
    extra_options: dict | None = None,
) -> AiResponse:
    """Main routing function for all unified AI text generation requests.

    Args:
        provider: Provider name ('ollama', 'google', 'anthropic', 'openai',
            'mistral', 'cohere', 'meta', 'groq', 'xai',
            'lmstudio', 'llamacpp', or 'script').
        model: Model identifier. For 'script' provider, this is the script
            file path.
        prompt: User prompt text.
        system_prompt: Optional system instructions.
        messages: Optional chat history as a list of role/content dicts.
            Each dict may include an optional 'files' key with a list of
            file paths to attach to that message.
        file_path: Optional local file path or list of paths for multimodal
            input. Supports images, audio, text files, and PDFs. The provider
            handles all encoding and upload internally.
        temperature: Sampling temperature.
        thinking: Enable extended reasoning/thinking mode (True/False) or
            use provider's default behavior ("default").
        format_json: Force JSON-formatted response.
        timeout: Maximum seconds to wait for a response.
        max_retries: Number of retry attempts on failure.
        retry_base_delay: Initial exponential backoff delay in seconds.
        top_k: Sampling parameter top_k (default 64).
        top_p: Sampling parameter top_p (default 0.95).
        max_tokens: Limit on the number of generated tokens.
        sleep_time: Rate limit delay in seconds before calling the API.
        extra_options: Optional dict of arbitrary provider-specific options
            merged into the API payload at call time. Keys and values are
            provider-dependent (e.g. {'visual_token_budget': 1120} for
            Ollama/Gemma4). Override any config-level defaults for the same key.

    Returns:
        AiResponse dataclass containing response text, token metrics, and
        optional reasoning text.
    """
    _register_cleanup()

    prov = get_provider(provider)

    # Resolve centralized sleep rate-limiting delay
    effective_sleep = sleep_time
    if effective_sleep is None:
        effective_sleep = getattr(getattr(prov, "config", None), "sleep_time", 0)

    if effective_sleep > 0:
        import time
        time.sleep(effective_sleep)

    request = AiRequest(
        provider=provider,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
        file_path=file_path,
        temperature=temperature,
        thinking=thinking,
        format_json=format_json,
        timeout=timeout,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
        sleep_time=sleep_time,
        extra_options=extra_options,
    )

    from unified_ai_client.retry import with_retry
    return with_retry(
        prov.call,
        request,
        max_retries=max_retries,
        base_delay=retry_base_delay,
    )


def preload_model(
    provider: str,
    model: str,
    keep_alive: str = "15m",
) -> None:
    """Pre-load a model into resident memory (Ollama only).

    Args:
        provider: Provider name (e.g. 'ollama').
        model: Model identifier.
        keep_alive: How long to keep model loaded (e.g. '15m', '1h').
    """
    prov = get_provider(provider)
    prov.preload_model(model, keep_alive)


def get_embedding(
    provider: str,
    model: str,
    text: str,
) -> list[float]:
    """Generate a text embedding vector.

    Args:
        provider: Provider name.
        model: Embedding model name.
        text: Input text to embed.

    Returns:
        List of floats representing the embedding vector.
    """
    prov = get_provider(provider)
    return prov.get_embedding(model, text)
