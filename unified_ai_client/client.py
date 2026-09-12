from __future__ import annotations

import logging
import time
from typing import Any

from unified_ai_client.models import AiRequest, AiResponse, ToolDefinition, ToolResult
from unified_ai_client.registry import (
    _record_loaded,
    _register_cleanup,
    configure_provider,
    get_provider,
)

_log = logging.getLogger("unified_ai_client.client")


def call_ai(
    provider: str,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    messages: list[dict] | None = None,
    file_path: str | list[str] | None = None,
    temperature: float | None = None,
    thinking: bool | str = "default",
    format_json: bool = False,
    timeout: int | None = None,
    max_retries: int = 3,
    retry_base_delay: float = 5.0,
    top_k: int | None = None,
    top_p: float | None = None,
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
        temperature: Sampling temperature. ``None`` falls back to the value
            registered through ``configure_provider()``, and finally to 0.7.
        thinking: Enable extended reasoning/thinking mode (``True``/``False``)
            or use the provider's default behavior (``"default"``).
        format_json: Force JSON-formatted response.
        timeout: Maximum seconds to wait for a response. Defaults to the
            timeout registered for the provider via ``configure_provider()``,
            or 300 seconds when none was registered. Applies per attempt, so a
            call that keeps timing out costs roughly
            ``(max_retries + 1) x timeout`` plus the backoff.
        max_retries: Number of retry attempts on failure.
        retry_base_delay: Initial exponential backoff delay in seconds.
        top_k: Sampling parameter top_k. ``None`` sends nothing and leaves the
            provider's own default in place, which is the only portable
            answer: OpenAI's Chat Completions API rejects ``top_k`` as an
            unknown argument, while Ollama and Anthropic accept it. Providers
            with a house default of their own document it.
        top_p: Sampling parameter top_p. ``None`` leaves the provider's own
            default in place.
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

    _log.info(
        "call_ai start: provider=%s model=%s prompt_chars=%d thinking=%s tools=%d",
        provider, model, len(prompt), thinking, len(tools or []),
    )

    prov = get_provider(provider)
    # Recorded before the call, not after: a model that loaded and then timed
    # out is resident all the same, and still has to be released at exit.
    _record_loaded(provider, prov, model)

    # Resolve centralized sleep rate-limiting delay
    effective_sleep = sleep_time
    if effective_sleep is None:
        effective_sleep = getattr(getattr(prov, "config", None), "sleep_time", 0)

    if effective_sleep > 0:
        time.sleep(effective_sleep)

    # Resolve the deadline the same way, and here rather than in each adapter:
    # every provider reads request.timeout directly, so a None arriving there
    # would be twelve separate fallbacks to get right. Before this, call_ai()
    # defaulted to a hard 300 that silently outranked configure_provider(),
    # and a configured timeout only ever reached warm-up and embeddings.
    effective_timeout = timeout
    if effective_timeout is None:
        effective_timeout = getattr(getattr(prov, "config", None), "timeout", 300)

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
        timeout=effective_timeout,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
        sleep_time=sleep_time,
        extra_options=extra_options,
        tools=tools,
        tool_results=tool_results,
    )

    if file_path and tool_results:
        # Every adapter treats tool_results as "the consumer already put the
        # user turn in messages", so a file_path passed alongside it is never
        # attached to this request: on the chat providers it is silently
        # skipped, and on Ollama it used to be read and base64-encoded first
        # and only then discarded. Attaching a file on a tool-result
        # continuation is not supported, so this is the caller's one signal
        # that it happened rather than nothing at all.
        _log.warning(
            "call_ai: file_path was given alongside tool_results and will "
            "not be attached to this request; attachments are only sent on "
            "a fresh turn."
        )

    from unified_ai_client.retry import with_retry
    start = time.perf_counter()
    response = with_retry(
        prov.call,
        request,
        max_retries=max_retries,
        base_delay=retry_base_delay,
        label=f"{provider}/{model}",
    )
    _log.info(
        "call_ai success: provider=%s model=%s elapsed=%.2fs input_tokens=%d output_tokens=%d",
        provider, model, time.perf_counter() - start,
        response.input_tokens, response.output_tokens,
    )
    return response


def preload_model(
    provider: str,
    model: str,
    keep_alive: str | int = "15m",
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
        keep_alive: How long to keep the model loaded. A number is seconds,
            ``0`` unloads it once idle and ``-1`` keeps it resident
            indefinitely; a string is a Go duration such as ``'15m'`` or
            ``'1h'``. Honoured by ``ollama`` and ``script``, ignored by every
            other provider. Note the timer is an idle countdown that starts
            when a request *finishes*, not a budget for the request: a model
            serving a call is never evicted mid-wait.

            Unlike ``context_size`` and ``extra_options``, this is not
            persisted via ``configure_provider()``, because it is a property of
            the request rather than of the provider.
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
    _log.info(
        "preload_model('%s', '%s'): keep_alive=%s context_size=%s",
        provider, model, keep_alive, context_size,
    )

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
    _record_loaded(provider, prov, model)


def warm_up(
    provider: str,
    model: str,
    file_paths: str | list[str] | None = None,
    *,
    keep_alive: str | int | None = None,
    timeout: int | None = None,
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
        keep_alive: How long the model should stay resident once loaded, for
            the providers that have a residency concept (``ollama``,
            ``script``). Defaults to whatever was registered via
            ``configure_provider()``, so a warm-up and the calls that follow it
            agree. See ``preload_model()`` for the accepted values.
        timeout: Seconds to wait for the warm-up. Defaults to the timeout
            registered for the provider. Useful when a cold model takes longer
            to load than a normal request takes to answer.

    Returns:
        True if something was actually warmed up, False if this provider had
        nothing to do or the warm-up failed.
    """
    # Registered here too: a process that only ever calls warm_up() would
    # otherwise leave files uploaded to Google behind on exit.
    _register_cleanup()

    try:
        prov = get_provider(provider)
        result = prov.warm_up(
            model, file_paths, keep_alive=keep_alive, timeout=timeout
        )
    except Exception as exc:
        _log.warning("Warm-up failed for provider '%s' (%s): %s", provider, model, exc)
        return False
    if result:
        _record_loaded(provider, prov, model)
        _log.info("Warm-up completed for provider '%s' model '%s'", provider, model)
    return result


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
    # An embedding model occupies VRAM exactly like a chat model does, so it is
    # registered for release the same way, and cleanup() has to be armed for it.
    _register_cleanup()

    prov = get_provider(provider)
    _record_loaded(provider, prov, model)
    return prov.get_embedding(model, text)
