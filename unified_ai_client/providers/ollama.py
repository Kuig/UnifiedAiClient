from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from unified_ai_client.exceptions import NonRetryableError, UnsupportedFileError
from unified_ai_client.file_utils import encode_file_base64, normalize_file_paths
from unified_ai_client.http import get_json, post_json
from unified_ai_client.models import (
    AiRequest,
    AiResponse,
    EmbeddingRequest,
    ModelInfo,
    ProviderConfig,
    ToolCall,
)
from unified_ai_client.providers.base import BaseProvider

_log = logging.getLogger("unified_ai_client.providers.ollama")

# An unload queues behind any generation already running, so it gets its own
# short deadline rather than the provider timeout, which is usually minutes.
_UNLOAD_TIMEOUT: int = 30

_TRANSLATED_OPTIONS: dict[str, str] = {
    "context_size": "num_ctx",
    "max_tokens": "num_predict",
}
"""Library option names that Ollama spells differently."""

_MODEL_PARAM_OPTIONS: frozenset[str] = frozenset({"temperature", "top_k", "top_p"})
"""Library options Ollama accepts under the very same name."""

# Request-level fields, as opposed to the model parameters that live inside
# `options`, per endpoint (ollama api/types.go: ChatRequest, GenerateRequest,
# EmbedRequest). A key in extra_options that names one of these goes into the
# request body itself: inside `options` Ollama would discard it with no more
# than a server-side warning. `system` is absent on purpose, because it is
# filled from system_prompt.
_CHAT_FIELDS: frozenset[str] = frozenset({
    "format", "think", "truncate", "shift", "logprobs", "top_logprobs",
})
_GENERATE_FIELDS: frozenset[str] = _CHAT_FIELDS | {"raw", "suffix", "template", "context"}
_EMBED_FIELDS: frozenset[str] = frozenset({"truncate", "dimensions"})
_TOP_LEVEL_FIELDS: frozenset[str] = _GENERATE_FIELDS | _EMBED_FIELDS


def _fold_options(opts: dict[str, Any], options: dict[str, Any]) -> None:
    """Fold merged options into an Ollama model ``options`` dict, in place.

    Four outcomes, and which one a key gets is the whole of the rule:

    - a name Ollama spells differently is translated (``context_size`` becomes
      ``num_ctx``);
    - a name Ollama shares is kept as it is;
    - a request-level field (``think``, ``raw``, ``suffix``, ...) is left out,
      because it belongs in the body, where ``_request_fields()`` puts it;
    - anything else the library owns is dropped, because this adapter either
      consumes it elsewhere, as it does with ``keep_alive`` and
      ``use_generate``, or it belongs to another adapter entirely, as
      ``disable_safety`` belongs to Google. Neither has any business inside a
      model parameter block.

    Everything else passes through untouched, which is how a model-specific
    knob such as ``visual_token_budget`` or ``repeat_penalty`` reaches the
    model without this library knowing it exists.

    Args:
        opts: Source options, from the provider config or from the request.
        options: The Ollama ``options`` dict to update in place.
    """
    for k, v in opts.items():
        if k in _TRANSLATED_OPTIONS:
            options[_TRANSLATED_OPTIONS[k]] = v
        elif k in _MODEL_PARAM_OPTIONS:
            options[k] = v
        elif k in _TOP_LEVEL_FIELDS or k in BaseProvider._LIBRARY_OPTION_KEYS:
            continue
        else:
            options[k] = v


def _request_fields(opts: dict[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
    """Pick the request-level fields one endpoint accepts out of merged options.

    Args:
        opts: Merged options, config below call-time.
        allowed: The fields the target endpoint defines.

    Returns:
        The fields to merge into the request body, last, so that they win over
        anything the adapter derived itself, such as ``think`` from
        ``thinking`` or ``format`` from ``format_json``.
    """
    return {k: v for k, v in opts.items() if k in allowed}


class OllamaProvider(BaseProvider):
    """Ollama provider adapter utilizing native urllib.request.

    Communicates with a local Ollama server without requiring third-party
    dependencies like the official 'ollama' Python library.

    File handling:
    - Images: encoded as base64 and sent in the 'images' field.
    - Text files: inlined into the prompt using inline_text_attachments().
    - Audio and PDFs: rejected. /api/chat has no field for either. Ollama does
      serve audio-capable models, but reaching them needs its OpenAI-compatible
      endpoint, which this adapter does not use.
    """

    DEFAULT_URL: str = "http://localhost:11434"

    # /api/chat carries attachments in images[] and nothing else. Audio-capable
    # models exist, but reaching them needs the OpenAI-compatible endpoint.
    SUPPORTED_FILE_TYPES: frozenset[str] = frozenset({"image"})

    # Ollama keeps a model in VRAM between requests and can be told to drop it.
    SUPPORTS_UNLOAD: bool = True

    SUPPORTS_EMBEDDINGS: bool = True

    # /api/embed takes an input object with an image and an audio track since
    # Ollama's multimodal embeddings (2026-09). Unlike /api/chat, audio is
    # carried there; which tags can read it is up to the model.
    SUPPORTED_EMBEDDING_FILE_TYPES: frozenset[str] = frozenset({"image", "audio"})

    # Lists the installed models, so it answers "is the server up" for free.
    PROBE_PATH: str | None = "/api/tags"

    # The levers this adapter reads for itself: three folded into `options`
    # under their own name, two translated to Ollama's spelling, and two that
    # steer the adapter without ever reaching the wire as raw keys.
    _CONSUMED_OPTION_KEYS: frozenset[str] = frozenset(
        set(_TRANSLATED_OPTIONS) | _MODEL_PARAM_OPTIONS | {"keep_alive", "use_generate"}
    )

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the Ollama provider.

        Args:
            config: ProviderConfig instance containing connection details.
        """
        self.config = config
        self.base_url = (config.url or self.DEFAULT_URL).rstrip("/")

    def _post(
        self, endpoint: str, payload: dict[str, Any], timeout: int
    ) -> dict[str, Any]:
        """Perform a HTTP POST request to the Ollama server.

        Args:
            endpoint: API endpoint path (e.g. '/api/chat').
            payload: JSON payload dictionary.
            timeout: Request timeout in seconds.

        Returns:
            The parsed JSON response.

        Raises:
            NonRetryableHttpError: On a 4xx other than 408/409/425/429. An
                unknown model is a 404 here, so this is the common case.
            ProviderHttpError: On any other HTTP error response.
            urllib.error.URLError: On network connection issues.
        """
        _log.debug(
            "Ollama POST %s: model=%s timeout=%ss keep_alive=%s options=%s",
            endpoint,
            payload.get("model"),
            timeout,
            payload.get("keep_alive"),
            sorted(payload.get("options") or {}),
        )
        # No auth headers: a local server authenticates nobody.
        return post_json(f"{self.base_url}{endpoint}", payload, timeout)

    def _get(self, endpoint: str, timeout: int) -> dict[str, Any]:
        """Perform a HTTP GET request to the Ollama server.

        Args:
            endpoint: API endpoint path (e.g. '/api/tags').
            timeout: Request timeout in seconds.

        Returns:
            The parsed JSON response.

        Raises:
            NonRetryableHttpError: On a 4xx other than 408/409/425/429.
            ProviderHttpError: On any other HTTP error response.
            urllib.error.URLError: On network connection issues.
        """
        # No auth headers: a local server authenticates nobody.
        return get_json(f"{self.base_url}{endpoint}", timeout)

    def list_models(self, *, timeout: int | None = None) -> list[ModelInfo]:
        """List the installed models, with the capabilities Ollama reports.

        ``GET /api/tags`` names the models and ``POST /api/show`` describes each
        one: a ``capabilities`` array (``vision``, ``tools``, ``thinking``,
        ``embedding``) and a context length. That is one extra request per
        model, so a failure on a single model leaves its capability fields
        ``None`` and keeps it in the list: a model that is installed is still
        installed when the detail lookup fails.

        Args:
            timeout: Seconds to wait for each request. Defaults to the
                configured timeout.

        Returns:
            One ``ModelInfo`` per installed model.

        Raises:
            NonRetryableHttpError: On a 4xx other than 408/409/425/429.
            ProviderHttpError: On any other HTTP error response.
            urllib.error.URLError: On network connection issues, listing only.
        """
        effective_timeout = self._resolve_timeout(timeout)
        tags = self._get("/api/tags", effective_timeout)

        models: list[ModelInfo] = []
        for entry in tags.get("models", []):
            name = entry.get("model") or entry.get("name")
            if not name:
                continue
            try:
                details = self._post("/api/show", {"model": name}, effective_timeout)
            except (OSError, ValueError) as exc:
                # URLError, HTTP errors and a JSON decode failure all land here.
                _log.debug("Ollama /api/show failed for '%s': %s", name, exc)
                models.append(ModelInfo(id=name, provider=self.provider_name))
                continue
            models.append(self._model_info(name, details))
        return models

    def _model_info(self, name: str, details: dict[str, Any]) -> ModelInfo:
        """Build a ``ModelInfo`` from one ``/api/show`` response.

        Args:
            name: The model identifier.
            details: Parsed ``/api/show`` response.

        Returns:
            The model with each field set only if the response said enough to
            decide it. A response without a ``capabilities`` array (an older
            server) leaves those fields unknown rather than all ``False``.
        """
        info = ModelInfo(id=name, provider=self.provider_name)
        capabilities = details.get("capabilities")
        if isinstance(capabilities, list):
            info = replace(
                info,
                reasoning="thinking" in capabilities,
                tool_use="tools" in capabilities,
                embeddings="embedding" in capabilities,
                file_types=(
                    frozenset({"image"}) if "vision" in capabilities else frozenset()
                ),
            )

        # The context length is keyed by architecture: "llama.context_length",
        # "gemma3.context_length", and so on.
        model_info = details.get("model_info") or {}
        for key, value in model_info.items():
            if key.endswith(".context_length") and isinstance(value, int):
                info = replace(info, context_length=value)
                break
        return info

    def _process_files_for_message(
        self, file_paths: list[str], prompt: str
    ) -> tuple[str, list[str]]:
        """Process a list of file paths for an Ollama message.

        Images are base64-encoded into the 'images' field; text files are
        inlined into the prompt. Anything else is rejected: /api/chat carries no
        other kind of attachment, so a file placed in images[] that is not an
        image is read as a corrupt image or ignored outright, and the model
        answers as though nothing had been attached.

        Args:
            file_paths: List of file paths to process.
            prompt: The message text prompt to augment.

        Returns:
            A tuple of (modified_prompt, list_of_base64_strings).

        Raises:
            FileNotFoundError: If an attachment does not exist.
            UnsupportedFileError: If an attachment is neither an image nor text.
        """
        effective_prompt, native = self._partition_attachments(file_paths, prompt)

        multimodal_data: list[str] = []
        for fp, ft in native:
            if ft != "image":
                raise self._unsupported_native_error(fp, ft)
            _log.debug("Ollama: encoding '%s' as base64 image data", fp)
            multimodal_data.append(encode_file_base64(fp))

        return effective_prompt, multimodal_data

    def call(self, request: AiRequest) -> AiResponse:
        """Execute an AI inference call to Ollama.

        Token counting notes:
        - ``output_tokens`` maps to ``eval_count``, which Ollama reports as the
          total generated tokens (thinking trace + final response combined).
        - ``reasoning_tokens`` is estimated when ``message.thinking`` is present
          in the response. The estimation uses the exact chars-per-token ratio
          of this response (``eval_count / (len(content) + len(thinking))``),
          which is self-consistent and avoids a fixed global ratio.
        - Both ``reasoning_tokens`` and ``raw_thinking`` are always processed
          — a model may produce thinking output even when not explicitly requested.

        Args:
            request: The request containing parameters.

        Returns:
            Standardized response.
        """
        messages: list[dict[str, Any]] = []

        # 1. System Prompt
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # 2. Chat History (with optional file attachments per message)
        if request.messages:
            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                msg_files = normalize_file_paths(msg.get("files"))
                multimodal: list[str] = []
                if msg_files:
                    content, multimodal = self._process_files_for_message(
                        msg_files, content
                    )

                entry: dict[str, Any] = {"role": role, "content": content}
                if multimodal:
                    entry["images"] = multimodal
                # Preserve tool_calls on assistant messages so Ollama can
                # correctly link subsequent tool results back to the call,
                # regardless of whether this same message also carries a file:
                # the two used to be on an if/else, so a message with both
                # silently lost its tool_calls.
                if role == "assistant" and msg.get("tool_calls"):
                    entry["tool_calls"] = msg["tool_calls"]
                messages.append(entry)

        # 3. Resolve options early: the file-processing step below needs to
        # know use_generate before it can decide whether its result is used.
        opts = self._merge_options(request)
        use_generate = opts.get("use_generate", False)

        # /api/generate has no place for any of these: no message history, no
        # tool schema, no way to append a tool result. Until this check
        # existed, use_generate silently dropped them instead of refusing, the
        # same class of silent data loss the file-attachment and
        # history-message invariants elsewhere in this adapter already guard
        # against. A system prompt is not on the list: /api/generate carries
        # it in its own `system` field.
        if use_generate:
            unsupported = [
                name for name, value in (
                    ("tools", request.tools),
                    ("tool_results", request.tool_results),
                    ("messages", request.messages),
                ) if value
            ]
            if unsupported:
                raise NonRetryableError(
                    "Ollama's use_generate routes to /api/generate, which has "
                    f"no place for {', '.join(unsupported)}. Turn "
                    "use_generate off, or drop the field(s), to use "
                    "/api/chat instead."
                )
        else:
            # The reverse case: a completion-only field asked for on the chat
            # endpoint would be dropped, and a `suffix` silently ignored is a
            # wrong answer, not a degraded one.
            generate_only = sorted(
                k for k in opts if k in _GENERATE_FIELDS - _CHAT_FIELDS
            )
            if generate_only:
                raise NonRetryableError(
                    f"Ollama's /api/chat has no place for {', '.join(generate_only)}; "
                    "only /api/generate accepts it. Set use_generate=True."
                )

        endpoint_fields = _GENERATE_FIELDS if use_generate else _CHAT_FIELDS
        self._report_dropped_options(
            self._foreign_option_keys(opts)
            | {k for k in opts if k in _TOP_LEVEL_FIELDS - endpoint_fields},
            "owned by another provider, or not a field of this endpoint",
        )

        # 4. Current User Message — only add when NOT in a tool result continuation.
        # When tool_results are provided, the consumer has already placed the user
        # message in `messages` (history) and the prompt should not be re-appended.
        #
        # File processing runs whenever its result is actually used: always
        # for /api/generate below, whose payload has no other place to carry
        # the prompt or attachments, and only for a fresh /api/chat turn
        # otherwise — encoding an attachment a tool-result continuation will
        # never send used to run unconditionally, which is wasted work.
        if use_generate or not request.tool_results:
            file_paths = normalize_file_paths(request.file_path)
            effective_prompt, multimodal_data = self._process_files_for_message(
                file_paths, request.prompt
            )
        else:
            effective_prompt, multimodal_data = request.prompt, []

        if not request.tool_results:
            user_message: dict[str, Any] = {
                "role": "user",
                "content": effective_prompt,
            }
            if multimodal_data:
                user_message["images"] = multimodal_data
            messages.append(user_message)

        # 5. Build the model options.
        #
        # Fold extra_options first, config below call-time so the latter
        # overrides the former for the same key (docs/configuration.md), then
        # let an explicit named parameter win over both — it is the more
        # specific signal the caller gave for this exact call. Only when
        # neither an explicit parameter nor any extra_options set a value
        # does Ollama's own default apply, and it applies because the key is
        # left out: the Modelfile's PARAMETER lines and the server defaults
        # are what the model's author tuned, and this library has no better
        # value to substitute for them.
        #
        # This used to run in the opposite order: temperature/top_k/top_p
        # were set from the request first, and _fold_options(opts, options)
        # ran last, so a top_k registered via configure_provider() silently
        # overrode an explicit call_ai(top_k=...) every time.
        options: dict[str, Any] = {}
        _fold_options(self.config.extra_options or {}, options)
        _fold_options(request.extra_options or {}, options)

        for key, value in (
            ("temperature", request.temperature),
            ("top_k", request.top_k),
            ("top_p", request.top_p),
        ):
            resolved = self._prefer_request(value, opts, key)
            if resolved is not None:
                options[key] = resolved

        # Call-time max_tokens takes precedence
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        keep_alive = opts.get("keep_alive")

        if use_generate:
            # /api/generate takes one prompt, its attachments, a system prompt
            # and the same think flag /api/chat does; history and tools were
            # refused above.
            payload: dict[str, Any] = {
                "model": request.model,
                "prompt": effective_prompt,
                "stream": False,
                "options": options,
            }
            if request.system_prompt:
                payload["system"] = request.system_prompt
            if keep_alive is not None:
                payload["keep_alive"] = keep_alive
            if multimodal_data:
                payload["images"] = multimodal_data

            if request.format_json:
                payload["format"] = "json"

            if request.thinking is True:
                payload["think"] = True
            elif request.thinking is False:
                payload["think"] = False

            payload.update(_request_fields(opts, _GENERATE_FIELDS))

            resp = self._post("/api/generate", payload, request.timeout)

            content_text = resp.get("response", "") or ""
            eval_count = resp.get("eval_count", 0)
            # The trace comes back in its own top-level field, separate from
            # the answer, exactly as message.thinking does on /api/chat.
            raw_thinking = resp.get("thinking", "") or ""
            reasoning_text = raw_thinking
            tool_calls: list[ToolCall] = []  # refused above
        else:
            payload: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
                "stream": False,
                "options": options,
            }
            if keep_alive is not None:
                payload["keep_alive"] = keep_alive

            if request.format_json:
                payload["format"] = "json"

            # Tool definitions — Ollama uses OpenAI-compatible format
            if request.tools:
                payload["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        },
                    }
                    for t in request.tools
                ]

            # Tool results — appended as role:"tool" messages
            if request.tool_results:
                for tr in request.tool_results:
                    messages.append({
                        "role": "tool",
                        "content": tr.content,
                    })

            payload["messages"] = messages

            if request.thinking is True:
                payload["think"] = True
            elif request.thinking is False:
                payload["think"] = False

            # Request-level fields from extra_options go last, so a `think`
            # level such as "high" or a JSON-schema `format` overrides what
            # the tri-state flags above derived.
            payload.update(_request_fields(opts, _CHAT_FIELDS))

            # 5. Make request
            resp = self._post("/api/chat", payload, request.timeout)

            # 6. Parse output
            message = resp.get("message", {})
            content_text = message.get("content", "") or ""
            eval_count = resp.get("eval_count", 0)

            raw_thinking = message.get("thinking", "") or ""
            reasoning_text = raw_thinking

            # Parse tool calls
            raw_tool_calls = message.get("tool_calls") or []
            tool_calls: list[ToolCall] = []
            for tc in raw_tool_calls:
                fn = tc.get("function", {})
                tc_id = tc.get("id") or f"{fn.get('name', 'tool')}_{len(tool_calls)}"
                tool_calls.append(ToolCall(
                    id=tc_id,
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments") or {},
                ))

        # Ollama reports one aggregate eval_count, so the split between the
        # thinking trace and the answer has to be estimated.
        reasoning_tokens = self._estimate_reasoning_tokens(
            content_text, raw_thinking, eval_count
        )

        return AiResponse(
            text=content_text,
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=max(0, eval_count - reasoning_tokens),
            reasoning_tokens=reasoning_tokens,
            reasoning_text=reasoning_text,
            tool_calls=tool_calls,
        )

    def preload_model(
        self,
        model: str,
        keep_alive: str | int | None = None,
        context_size: int | None = None,
        extra_options: dict | None = None,
        *,
        timeout: int | None = None,
    ) -> None:
        """Pre-load an Ollama model into memory with the specified options.

        Includes any provider-level ``extra_options`` (registered via
        ``configure_provider()``) and call-time overrides in the preload
        payload so Ollama allocates the model with the correct options
        (e.g. context window size) from the start.

        Args:
            model: Model identifier.
            keep_alive: How long to keep the model loaded. A number is seconds,
                ``0`` unloads it once idle and ``-1`` keeps it resident
                indefinitely; a string is a Go duration such as ``'15m'``.
                ``None`` falls back to the value registered via
                ``configure_provider()``, then to the server's own default.
            context_size: Context window size in tokens. Mapped to ``num_ctx``.
                Takes priority over ``context_size`` in ``extra_options``.
            extra_options: Additional provider-specific options merged into the
                preload request options dict.
            timeout: Seconds to wait for the request. Defaults to the timeout
                registered for this provider.
        """
        options: dict[str, Any] = {}

        # 1. Config-level extra_options (from configure_provider / previous preload)
        _fold_options(self.config.extra_options or {}, options)

        # 2. Call-time extra_options (override config)
        _fold_options(extra_options or {}, options)

        # 3. Explicit context_size parameter (highest priority)
        if context_size is not None:
            options["num_ctx"] = context_size

        payload: dict[str, Any] = {"model": model, "messages": []}
        effective_keep_alive = self._resolve_keep_alive(keep_alive)
        if effective_keep_alive is not None:
            payload["keep_alive"] = effective_keep_alive
        if options:
            payload["options"] = options

        self._post("/api/chat", payload, self._resolve_timeout(timeout))

    def warm_up(
        self,
        model: str,
        file_paths: str | list[str] | None = None,
        *,
        keep_alive: str | int | None = None,
        timeout: int | None = None,
    ) -> bool:
        """Load the model into memory via Ollama's own warm-up request.

        Delegates to ``preload_model()``, which posts an empty message list to
        ``/api/chat``: Ollama's official way to allocate a model without
        generating anything.

        Args:
            model: Model identifier to load.
            file_paths: Ignored. Ollama inlines attachments into the request
                and keeps no remote file store to populate.
            keep_alive: How long the model should stay resident. Defaults to the
                value registered via ``configure_provider()``, so a warm-up and
                the calls that follow it agree on the residency.
            timeout: Seconds to wait for the load. Defaults to the timeout
                registered for this provider.

        Returns:
            Always True: loading the model is always real work.
        """
        effective_keep_alive = self._resolve_keep_alive(keep_alive)
        _log.debug(
            "Ollama warm-up: model=%s keep_alive=%s timeout=%ss",
            model, effective_keep_alive, self._resolve_timeout(timeout),
        )
        self.preload_model(model, effective_keep_alive, timeout=timeout)
        return True

    def unload_model(self, model: str, *, timeout: int | None = None) -> None:
        """Release the model from memory, freeing its VRAM.

        Sends the request Ollama documents for this, and the same one ``ollama
        stop`` sends: an empty chat with ``keep_alive`` set to zero, which
        expires the model's idle timer immediately.

        Args:
            model: Model identifier to release.
            timeout: Seconds to wait. Defaults to 30 rather than to the
                provider's configured timeout, which is usually minutes: this
                request queues behind any generation already in flight, and a
                failure to free VRAM must not stall the caller for that long.
        """
        _log.debug("Ollama unload: model=%s", model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [],
            "keep_alive": 0,
        }
        self._post(
            "/api/chat",
            payload,
            timeout if timeout is not None else _UNLOAD_TIMEOUT,
        )

    def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        """Embed text, a batch of texts, or one input with media, via /api/embed.

        The embedding model is subject to the same residency policy as a chat
        model, so that a ``keep_alive`` registered via ``configure_provider()``
        governs both and ``cleanup()`` can release either.

        Media travel as an input object, ``{"text", "image", "audio"}`` with the
        files in base64, which Ollama embeds into one vector. Each key holds one
        file, so a second image or a second recording is refused rather than
        silently dropped. Whether the model can read the medium is up to the
        tag: a model without the encoder answers with an error of its own.

        Args:
            request: The embedding request.

        Returns:
            One vector per input.

        Raises:
            UnsupportedFileError: If a file is neither text nor image or audio,
                or a second file of the same class is attached.
            EmptyResponseError: If the server returns no vector for an input.
        """
        opts = self._merge_options(request)

        if request.file_path:
            text, native = self._partition_attachments(
                request.file_path, request.inputs[0], self.SUPPORTED_EMBEDDING_FILE_TYPES
            )
            item: dict[str, str] = {"text": text} if text else {}
            for fp, ft in native:
                if ft in item:
                    raise UnsupportedFileError(
                        f"Ollama's /api/embed takes one {ft} per input: '{fp}' "
                        "would be a second one. Embed it separately."
                    )
                item[ft] = encode_file_base64(fp)
            inputs: Any = [item]
        elif len(request.inputs) == 1:
            inputs = request.inputs[0]
        else:
            inputs = list(request.inputs)

        payload: dict[str, Any] = {"model": request.model, "input": inputs}
        keep_alive = opts.get("keep_alive")
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        payload.update(_request_fields(opts, _EMBED_FIELDS))
        self._report_dropped_options(
            set(opts) - _EMBED_FIELDS - {"keep_alive"},
            "/api/embed takes no model options",
        )
        if request.dimensions is not None:
            payload["dimensions"] = request.dimensions

        resp = self._post("/api/embed", payload, request.timeout)
        vectors = [[float(x) for x in v] for v in resp.get("embeddings") or []]
        return self._check_vector_count(vectors, request)
