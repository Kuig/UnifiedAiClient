from __future__ import annotations

import atexit
import importlib
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unified_ai_client.models import ProviderConfig
from unified_ai_client.providers.base import BaseProvider

_log = logging.getLogger("unified_ai_client.registry")


@dataclass(frozen=True)
class ProviderSpec:
    """Everything get_provider() needs to build one provider, and nothing else.

    Until 0.5.8 this information was duplicated by hand across the
    dispatch chain, the eager api_key_* lookups, the ValueError message and
    two docstrings in client.py alone, plus several more hand-typed tables
    in the test suite. This is the single place it is written down now.

    module/class_name are looked up lazily via importlib, exactly as the
    old if/elif chain's per-branch imports did, so building "ollama" never
    imports Google's SDK. api_key_name is the secrets.json/env-var key this
    provider reads, or None for a provider that takes no credential at all
    (ollama, script) or checks it elsewhere (google, in its lazy client
    getter).
    """

    module: str
    class_name: str
    api_key_name: str | None = None


_PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "ollama": ProviderSpec("unified_ai_client.providers.ollama", "OllamaProvider"),
    "google": ProviderSpec(
        "unified_ai_client.providers.google", "GoogleProvider", "google_api_key"
    ),
    "anthropic": ProviderSpec(
        "unified_ai_client.providers.anthropic", "AnthropicProvider", "anthropic_api_key"
    ),
    "openai": ProviderSpec(
        "unified_ai_client.providers.openai", "OpenAiProvider", "openai_api_key"
    ),
    "mistral": ProviderSpec(
        "unified_ai_client.providers.mistral", "MistralProvider", "mistral_api_key"
    ),
    "cohere": ProviderSpec(
        "unified_ai_client.providers.cohere", "CohereProvider", "cohere_api_key"
    ),
    "meta": ProviderSpec(
        "unified_ai_client.providers.meta", "MetaProvider", "meta_api_key"
    ),
    "groq": ProviderSpec(
        "unified_ai_client.providers.groq", "GroqProvider", "groq_api_key"
    ),
    "xai": ProviderSpec(
        "unified_ai_client.providers.xai", "XAiProvider", "xai_api_key"
    ),
    "lmstudio": ProviderSpec("unified_ai_client.providers.lmstudio", "LmStudioProvider"),
    "llamacpp": ProviderSpec("unified_ai_client.providers.llamacpp", "LlamaCppProvider"),
    "script": ProviderSpec("unified_ai_client.providers.script", "ScriptProvider"),
}

# --- Thread-safe programmatic provider configuration registry ---
# Populated via configure_provider(). Takes priority over file-based config.
_PROVIDER_CONFIGS: dict[str, ProviderConfig] = {}
_PROVIDER_CONFIGS_LOCK = threading.Lock()

# --- Thread-safe global provider instance cache ---
_PROVIDERS: dict[str, BaseProvider] = {}
_PROVIDERS_LOCK = threading.Lock()

# --- Thread-safe registry of models a local provider currently holds resident ---
# (provider_name, model) pairs, populated for providers declaring SUPPORTS_UNLOAD
# and drained by cleanup(). Module-level rather than per-provider on purpose:
# configure_provider() evicts the cached provider instance, so anything tracked
# on the instance would be silently lost the moment someone reconfigures.
_LOADED_MODELS: set[tuple[str, str]] = set()
_LOADED_MODELS_LOCK = threading.Lock()

# --- Names of every provider built in this process ---
# Distinct from _PROVIDERS, which configure_provider() empties: a provider whose
# instance was evicted may still hold module-level resources, and cleanup() has
# to be able to reach it by name to release them.
_BUILT_PROVIDERS: set[str] = set()

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

    The first call for a given provider merges on top of that provider's
    section in the library's ``config.json``, so registering one option does
    not discard the rest of the file. The precedence is per field:
    programmatic > ``config.json`` > ``ProviderConfig`` defaults.

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
        if existing is None:
            # Seed from the file, so the documented "programmatic > config.json
            # > defaults" chain resolves per field rather than per provider.
            # Registering a single extra used to replace the whole section:
            # configure_provider("ollama", context_size=8000) silently dropped
            # the url and timeout config.json had set, with no log to say so.
            # load_config() always returns an instance, falling back to the
            # dataclass defaults when there is no file, so there is no second
            # branch to keep in step.
            from unified_ai_client.config import load_config
            existing = load_config(
                _effective_config_path, ProviderConfig, section=name
            )

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
        _PROVIDER_CONFIGS[name] = config

    _log.info(
        "configure_provider('%s'): url=%s timeout=%s sleep_time=%s extra=%s",
        name, config.url, config.timeout, config.sleep_time, config.extra_options,
    )

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

        spec = _PROVIDER_SPECS.get(provider_name)
        if spec is None:
            raise ValueError(
                f"Unsupported AI provider: '{provider_name}'. Supported "
                f"providers: {', '.join(repr(n) for n in _PROVIDER_SPECS)}."
            )

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

        # 2. Look up the one credential this provider actually needs, if any.
        # Only this key is read, not all eight: until 0.5.8 get_provider()
        # computed every cloud provider's secret on every cache miss,
        # including a request for "ollama", which never uses any of them.
        api_key = ""
        if spec.api_key_name is not None:
            from unified_ai_client.config import load_secrets
            # `or ""` keeps a missing key from arriving as None where the
            # provider signatures declare a str: an unset header value makes
            # urllib fail with a type error instead of a missing-key message.
            api_key = load_secrets(os.getcwd()).get(spec.api_key_name) or ""

        # 3. Instantiate the provider adapter, importing its module lazily so
        # resolving "ollama" never pulls in Google's SDK.
        _log.debug(
            "Building provider '%s' (cache miss): url=%s timeout=%s",
            provider_name, config.url, config.timeout,
        )
        module = importlib.import_module(spec.module)
        provider_class = getattr(module, spec.class_name)
        provider_instance: BaseProvider = (
            provider_class(config, api_key=api_key)
            if spec.api_key_name is not None
            else provider_class(config)
        )

        _PROVIDERS[provider_name] = provider_instance
        _BUILT_PROVIDERS.add(provider_name)
        return provider_instance


def _register_cleanup() -> None:
    """Register the cleanup callback once with atexit."""
    global _CLEANUP_REGISTERED
    if not _CLEANUP_REGISTERED:
        with _CLEANUP_LOCK:
            if not _CLEANUP_REGISTERED:
                atexit.register(cleanup)
                _CLEANUP_REGISTERED = True


def _record_loaded(provider_name: str, provider: BaseProvider, model: str) -> None:
    """Remember that a provider now holds ``model`` resident, if it can.

    A no-op for providers that declare no residency, which is every cloud
    endpoint: they hold nothing on the caller's behalf and have nothing to
    release. For the rest this is what lets ``cleanup()`` free the VRAM at exit,
    including after an unhandled exception or a Ctrl+C.

    Registering a resource and arming its release are the same event, so the
    ``atexit`` hook is installed here rather than at each entry point. Doing it
    per entry point is what let ``preload_model()`` track a model it had no way
    to free: it never called ``_register_cleanup()``, so a process that only
    preloaded left the model resident for good.

    Args:
        provider_name: Registry name the provider was reached by.
        provider: The resolved provider instance.
        model: Model identifier now loaded.
    """
    if not provider.SUPPORTS_UNLOAD:
        return
    _register_cleanup()
    with _LOADED_MODELS_LOCK:
        _LOADED_MODELS.add((provider_name.strip().lower(), model))


def cleanup(*, unload_models: bool = True) -> None:
    """Release what the providers are holding.

    Two kinds of resource: remote ones, such as files uploaded to Google, which
    are deleted to free cloud quota; and local models a provider is holding
    resident, which are unloaded to free VRAM.

    Called automatically at process exit via atexit, so it also covers an
    unhandled exception and Ctrl+C. That is what makes an indefinite
    ``keep_alive=-1`` a safe policy rather than a leak. It can also be called
    explicitly for eager cleanup, for example in a finally block.

    Args:
        unload_models: Whether to also unload local models. Pass False to keep
            them resident, for a caller that runs this mid-session to release
            remote resources and expects the models to stay warm.
    """
    # Snapshot under the lock and work outside it. Everything below is network
    # I/O, and _PROVIDERS_LOCK is not reentrant, so any path reaching
    # get_provider() while it was held would deadlock.
    #
    # Names, not the cached instances: configure_provider() evicts an instance,
    # but the resources it registered live at module level and outlive it. Going
    # through get_provider() rebuilds whatever was evicted, so a provider that
    # uploaded a file and was then reconfigured still gets asked to delete it.
    with _PROVIDERS_LOCK:
        names = sorted(_BUILT_PROVIDERS)

    for provider_name in names:
        try:
            get_provider(provider_name).cleanup()
        except Exception as exc:
            _log.warning("cleanup() failed for provider '%s': %s", provider_name, exc)

    if not unload_models:
        return

    with _LOADED_MODELS_LOCK:
        loaded = sorted(_LOADED_MODELS)

    # Cleared entry by entry, only on success, instead of upfront: an unload
    # that raises must stay in _LOADED_MODELS so the next cleanup() call still
    # knows about it and retries. Clearing before the loop used to lose that
    # tracking the moment a single unload failed, silently turning a retryable
    # failure into a permanent leak for the rest of the process.
    released: list[tuple[str, str]] = []
    for provider_name, model in loaded:
        # Resolved by name for the same reason as above, and safe here only
        # because _PROVIDERS_LOCK was released. One try per model: a provider
        # that fails to release must not stop the others from being released.
        try:
            get_provider(provider_name).unload_model(model)
        except Exception as exc:
            _log.warning(
                "cleanup() failed to unload '%s/%s': %s", provider_name, model, exc
            )
        else:
            _log.info("cleanup() unloaded '%s/%s'", provider_name, model)
            released.append((provider_name, model))

    with _LOADED_MODELS_LOCK:
        _LOADED_MODELS.difference_update(released)


def unload_model(provider: str, model: str, *, timeout: int | None = None) -> None:
    """Release a model a local provider is holding resident, freeing its VRAM.

    The counterpart to ``preload_model()``. Only ``ollama`` and ``script`` have
    a residency concept; for every other provider this is a no-op, because a
    cloud endpoint holds nothing on the caller's behalf.

    The model is dropped from the set ``cleanup()`` drains, so an explicit
    unload is not attempted a second time at exit.

    This function never raises. Failing to free VRAM is not a reason to bring
    down the caller, and the next request will simply reload the model. Call
    ``get_provider(...).unload_model(...)`` directly if you want the exception.

    Args:
        provider: Provider name (e.g. ``'ollama'``).
        model: Model identifier to release.
        timeout: Seconds to wait. Defaults to a deliberately short value rather
            than to the provider's configured timeout: the request queues
            behind any generation already in flight.

    Example::

        warm_up("ollama", "gemma4:12b", keep_alive=-1)
        ...
        unload_model("ollama", "gemma4:12b")
    """
    with _LOADED_MODELS_LOCK:
        _LOADED_MODELS.discard((provider.strip().lower(), model))

    try:
        get_provider(provider).unload_model(model, timeout=timeout)
    except Exception as exc:
        _log.warning("Unload failed for provider '%s' (%s): %s", provider, model, exc)
        return
    _log.info("Unloaded model '%s' from provider '%s'", model, provider)
