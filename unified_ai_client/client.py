from __future__ import annotations

import atexit
import logging
import os
import threading
from pathlib import Path
from typing import Any

from unified_ai_client.models import AiRequest, AiResponse, ProviderConfig, ToolDefinition, ToolResult
from unified_ai_client.providers.base import BaseProvider

_log = logging.getLogger("unified_ai_client.client")

# --- Thread-safe programmatic provider configuration registry ---
# Populated via configure_provider(). Takes priority over file-based config.
_PROVIDER_CONFIGS: dict[str, ProviderConfig] = {}
_PROVIDER_CONFIGS_LOCK = threading.Lock()

# --- Thread-safe global provider instance cache ---
_PROVIDERS: dict[str, BaseProvider] = {}
_PROVIDERS_LOCK = threading.Lock()

# --- Resource cleanup registration flag ---
_CLEANUP_REGISTERED = False
_CLEANUP_LOCK = threading.Lock()

# --- Legacy file-based config path (kept for backward compat with load_config) ---
_LIB_ROOT = Path(__file__).parent.parent
_effective_config_path: str = str(os.path.join(_LIB_ROOT, "config.json"))


def configure_provider(name: str, **kwargs: Any) -> None:
    """Register or update provider-specific configuration programmatically.

    Call this at application startup to set provider settings that do not
    change per-call, such as server URLs, timeouts, or provider-specific
    parameters like ``context_size`` and ``visual_token_budget``.

    Known ``ProviderConfig`` fields (``url``, ``timeout``, ``sleep_time``) are
    stored as typed attributes. All other keyword arguments are collected into
    ``extra_options`` and passed through to the provider as-is (e.g.
    ``context_size``, ``visual_token_budget``, ``keep_alive``,
    ``disable_safety``).

    Subsequent calls are **merge-based**: only the fields explicitly passed are
    updated; previously registered values for other fields are preserved. This
    means ``configure_provider("ollama", url="…")`` followed by
    ``configure_provider("ollama", context_size=8000)`` results in both
    ``url`` and ``context_size`` being active simultaneously.

    Invalidates any cached provider instance for ``name`` so the next call
    picks up the new configuration.

    This function is thread-safe. Concurrent calls for the same provider name
    are serialized via an internal lock. Calling ``configure_provider`` from
    multiple threads for *different* provider names is fully safe. Calling it
    for the *same* name from multiple threads is safe but the last writer wins.

    Args:
        name: Provider name (e.g. ``'ollama'``, ``'google'``).
        **kwargs: Configuration values. Known ProviderConfig fields: ``url``,
            ``timeout``, ``sleep_time``. Everything else goes into
            ``extra_options`` as provider-specific settings.

    Example::

        configure_provider(
            "ollama",
            url="http://192.168.1.5:11434",
            timeout=240,
            context_size=8000,
            visual_token_budget=1120,
        )
        configure_provider("google", sleep_time=3)
    """
    name = name.strip().lower()
    known_fields = {"url", "timeout", "sleep_time"}
    new_known = {k: v for k, v in kwargs.items() if k in known_fields}
    new_extra = {k: v for k, v in kwargs.items() if k not in known_fields}

    with _PROVIDER_CONFIGS_LOCK:
        existing = _PROVIDER_CONFIGS.get(name)
        if existing is not None:
            # Merge: keep existing values, override only explicitly supplied fields
            merged_known: dict[str, Any] = {}
            if existing.url is not None:
                merged_known["url"] = existing.url
            if existing.timeout is not None:
                merged_known["timeout"] = existing.timeout
            if existing.sleep_time is not None:
                merged_known["sleep_time"] = existing.sleep_time
            merged_known.update(new_known)
            merged_extra = dict(existing.extra_options or {})
            merged_extra.update(new_extra)
            config = ProviderConfig(**merged_known, extra_options=merged_extra)
        else:
            config = ProviderConfig(**new_known, extra_options=new_extra)
        _PROVIDER_CONFIGS[name] = config

    # Invalidate cached provider instance so the next call rebuilds with new config
    with _PROVIDERS_LOCK:
        _PROVIDERS.pop(name, None)


def get_provider(provider_name: str) -> BaseProvider:
    """Thread-safe factory to resolve, configure, and cache provider instances.

    Configuration priority (highest to lowest):
    1. Programmatic configuration registered via ``configure_provider()``.
    2. File-based configuration from the library's ``config.json`` (legacy).
    3. Built-in ``ProviderConfig`` defaults.

    Provider instances are cached by name. Calling ``configure_provider()``
    invalidates the cache entry for the affected provider.

    Args:
        provider_name: The lower-case provider name (e.g. ``'ollama'``,
            ``'google'``).

    Returns:
        An instance of ``BaseProvider``.

    Raises:
        ValueError: If the provider name is not supported.
    """
    provider_name = provider_name.strip().lower()

    if provider_name in _PROVIDERS:
        return _PROVIDERS[provider_name]

    with _PROVIDERS_LOCK:
        if provider_name in _PROVIDERS:
            return _PROVIDERS[provider_name]

        # 1. Resolve configuration: programmatic > file > default
        with _PROVIDER_CONFIGS_LOCK:
            programmatic_config = _PROVIDER_CONFIGS.get(provider_name)

        if programmatic_config is not None:
            config = programmatic_config
        else:
            from unified_ai_client.config import load_config
            config = load_config(
                _effective_config_path, ProviderConfig, section=provider_name
            )

        # 2. Search for secrets.json in the current working directory
        # (= the consuming project's root when launched from there)
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

        # 4. Instantiate provider adapter
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

        _PROVIDERS[provider_name] = provider_instance
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
    timeout: int = 300,
    max_retries: int = 3,
    retry_base_delay: float = 5.0,
    top_k: int = 64,
    top_p: float = 0.95,
    max_tokens: int | None = None,
    sleep_time: int | None = None,
    extra_options: dict | None = None,
    tools: list[ToolDefinition] | None = None,
    tool_results: list[ToolResult] | None = None,
) -> AiResponse:
    """Main routing function for all unified AI text generation requests.

    Args:
        provider: Provider name (``'ollama'``, ``'google'``, ``'anthropic'``,
            ``'openai'``, ``'mistral'``, ``'cohere'``, ``'meta'``, ``'groq'``,
            ``'xai'``, ``'lmstudio'``, ``'llamacpp'``, or ``'script'``).
        model: Model identifier. For ``'script'`` provider, this is the script
            file path.
        prompt: User prompt text.
        system_prompt: Optional system instructions.
        messages: Optional chat history as a list of role/content dicts.
            Each dict may include an optional ``'files'`` key with a list of
            file paths to attach to that message.
        file_path: Optional local file path or list of paths for multimodal
            input. Supports images, audio, text files, and PDFs. The provider
            handles all encoding and upload internally.
        temperature: Sampling temperature.
        thinking: Enable extended reasoning/thinking mode (``True``/``False``)
            or use the provider's default behavior (``"default"``).
        format_json: Force JSON-formatted response.
        timeout: Maximum seconds to wait for a response.
        max_retries: Number of retry attempts on failure.
        retry_base_delay: Initial exponential backoff delay in seconds.
        top_k: Sampling parameter top_k (default 64).
        top_p: Sampling parameter top_p (default 0.95).
        max_tokens: Limit on the number of generated tokens.
        sleep_time: Rate limit delay in seconds before calling the API.
            Overrides the value set via ``configure_provider()`` for this call.
        extra_options: Optional dict of provider-specific options merged into
            the API payload at call time. Call-time values override any
            provider-level defaults registered via ``configure_provider()``
            for the same key. Examples: ``{'visual_token_budget': 1120}`` for
            Ollama/Gemma4, ``{'disable_safety': True}`` for Google.
        tools: Optional list of ``ToolDefinition`` objects describing functions
            the model may call. When provided, the model may respond with
            ``AiResponse.tool_calls`` instead of (or in addition to) text.
        tool_results: Optional list of ``ToolResult`` objects containing the
            outputs of previously requested tool calls. Pass these on the
            follow-up call after executing the tools requested by the model.

    Returns:
        ``AiResponse`` dataclass containing response text, token metrics,
        optional reasoning text, and any tool calls requested by the model.
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
        tools=tools,
        tool_results=tool_results,
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
    context_size: int | None = None,
    extra_options: dict | None = None,
) -> None:
    """Pre-load a model into resident memory and register its settings.

    For Ollama, sends a warm-up request that allocates the model in GPU/CPU
    memory with the specified options (e.g. ``context_size`` → ``num_ctx``).
    This avoids a VRAM reallocation on the first ``call_ai()`` call.

    When ``context_size`` or ``extra_options`` are provided, they are
    registered via ``configure_provider()`` so that all subsequent
    ``call_ai()`` calls for this provider automatically use the same settings
    without needing to pass them again per-call.

    ``extra_options`` is merged with any ``extra_options`` previously
    registered via ``configure_provider()``.

    For providers that do not support preloading (Google, Anthropic, OpenAI,
    etc.) this function is a no-op for the warm-up part, but still registers
    any provided settings via ``configure_provider()``.

    Args:
        provider: Provider name (e.g. ``'ollama'``).
        model: Model identifier.
        keep_alive: How long to keep model loaded (e.g. ``'15m'``, ``'1h'``).
            Ollama-specific; ignored by other providers.
        context_size: Context window size in tokens. Ollama maps this to
            ``num_ctx`` in the API payload. If provided, registered via
            ``configure_provider()`` so it persists across all ``call_ai()``
            calls. Passing ``context_size`` here instead of in each
            ``call_ai()`` call prevents Ollama from reloading the model with
            a different context window mid-session.
        extra_options: Optional dict of additional provider-specific settings
            (e.g. ``{'visual_token_budget': 1120}``). Merged with any
            previously registered settings and persisted via
            ``configure_provider()``.
    """
    # Register settings so they persist into all subsequent call_ai() calls
    if context_size is not None or extra_options:
        config_kwargs: dict[str, Any] = {}
        if context_size is not None:
            config_kwargs["context_size"] = context_size
        if extra_options:
            config_kwargs.update(extra_options)
        configure_provider(provider, **config_kwargs)

    # get_provider after configure_provider so it picks up the new config
    prov = get_provider(provider)
    prov.preload_model(model, keep_alive, context_size=context_size, extra_options=extra_options)


def warm_up(
    provider: str,
    model: str,
    file_paths: str | list[str] | None = None,
) -> bool:
    """Pay a provider's one-off costs before the first real call.

    Without this, the setup costs a provider charges once per process (SDK
    import, client construction, DNS + TCP + TLS handshake, model load, remote
    file upload) all land on whichever ``call_ai()`` happens to run first. That
    request then looks slow purely because it went first, which matters when
    the timings are being measured and compared.

    What each provider actually does:

    +-------------------------+------------------------------------------------+
    | ``google``              | Builds the client, issues a free metadata GET, |
    |                         | and uploads ``file_paths`` into the same cache |
    |                         | ``call_ai()`` reads and ``cleanup()`` clears.  |
    | ``ollama``              | Loads the model via Ollama's own warm-up call. |
    | ``lmstudio``,           | Sends a one-token completion, because these    |
    | ``llamacpp``            | servers load the model on first inference.     |
    | ``openai``, ``mistral``,| Free ``GET /v1/models``: opens the connection  |
    | ``cohere``, ``meta``,   | and validates the key without consuming        |
    | ``groq``, ``xai``,      | tokens.                                        |
    | ``anthropic``           |                                                |
    | ``script``              | Sends ``mode: "warm_up"``; scripts that do not |
    |                         | implement it simply report nothing to do.      |
    +-------------------------+------------------------------------------------+

    No provider consumes generation tokens here. The one caveat is
    ``lmstudio`` / ``llamacpp``, where the warm-up is a real (if tiny)
    inference request: free against a local server, billable if those providers
    have been pointed at a paid remote endpoint.

    This function never raises. A failed warm-up is a missed optimisation, not
    an error: the ``call_ai()`` that follows has its own retries and will
    report the real failure. Call ``get_provider(...).warm_up(...)`` directly
    if you want the exception instead.

    Args:
        provider: Provider name (e.g. ``'google'``, ``'ollama'``).
        model: Model identifier to warm up.
        file_paths: Optional path or list of paths to pre-upload. Only used by
            providers that keep a remote file store (currently ``google``) and
            by scripts that choose to act on it.

    Returns:
        True if something was actually warmed up, False if this provider had
        nothing to do or the warm-up failed.
    """
    # Registered here too: a process that only ever calls warm_up() would
    # otherwise leave files uploaded to Google behind on exit.
    _register_cleanup()

    try:
        return get_provider(provider).warm_up(model, file_paths)
    except Exception as exc:
        _log.warning("Warm-up failed for provider '%s' (%s): %s", provider, model, exc)
        return False


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
