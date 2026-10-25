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

# Request-body fields per endpoint, copied from ChatRequest, GenerateRequest and
# EmbedRequest in ollama's api/types.go. Inside `options` Ollama would discard
# them with only a server-side warning. `system` is left out: system_prompt fills it.
_CHAT_FIELDS: frozenset[str] = frozenset({
    "format", "think", "truncate", "shift", "logprobs", "top_logprobs",
})
_GENERATE_FIELDS: frozenset[str] = _CHAT_FIELDS | {"raw", "suffix", "template", "context"}
_EMBED_FIELDS: frozenset[str] = frozenset({"truncate", "dimensions"})
_TOP_LEVEL_FIELDS: frozenset[str] = _GENERATE_FIELDS | _EMBED_FIELDS

_MESSAGE_FIELDS: tuple[str, ...] = (
    "role", "content", "thinking", "images", "tool_calls", "tool_name", "tool_call_id",
)
"""The fields of ``Message`` in ollama's api/types.go, the ones a replayed reply keeps."""


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


def _tool_link(name: str | None, call_id: str | None) -> dict[str, str]:
    """The fields that tie a ``role: "tool"`` message to the call it answers."""
    link: dict[str, str] = {}
    if name:
        link["tool_name"] = name
    if call_id:
        link["tool_call_id"] = call_id
    return link


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

    # /api/chat carries attachments in images[] and nothing else.
    SUPPORTED_FILE_TYPES: frozenset[str] = frozenset({"image"})

    # Ollama keeps a model in VRAM between requests and can be told to drop it.
    SUPPORTS_UNLOAD: bool = True

    SUPPORTS_EMBEDDINGS: bool = True

    # /api/embed takes an input object with an image and an audio track; which
    # tags can read them is up to the model.
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

        Routes to ``/api/chat``, or to ``/api/generate`` when ``use_generate``
        is set. ``output_tokens`` and ``reasoning_tokens`` split Ollama's single
        ``eval_count``, see ``_estimate_reasoning_tokens()``; the thinking trace
        is harvested whether or not it was asked for, since a model may produce
        one anyway.

        Args:
            request: The request containing parameters.

        Returns:
            Standardized response.

        Raises:
            NonRetryableError: If the request needs a field the chosen endpoint
                does not have.
        """
        opts = self._merge_options(request)
        use_generate = bool(opts.get("use_generate", False))
        self._check_endpoint_fields(request, opts, use_generate)
        if use_generate:
            return self._generate(request, opts)
        return self._chat(request, opts)

    def _check_endpoint_fields(
        self, request: AiRequest, opts: dict[str, Any], use_generate: bool
    ) -> None:
        """Refuse what the chosen endpoint cannot carry, and log what it drops.

        Raises:
            NonRetryableError: If ``/api/generate`` is asked for history, tools
                or tool results, or ``/api/chat`` for a completion-only field.
                Dropping either would change the answer rather than degrade it.
        """
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

    def _build_options(self, request: AiRequest, opts: dict[str, Any]) -> dict[str, Any]:
        """Build the model ``options`` block.

        ``extra_options`` are folded first, config below call-time, and an
        explicit request argument wins over both. A lever nobody set is left
        out, so the Modelfile's ``PARAMETER`` lines and the server defaults stay
        in charge.

        Args:
            request: The request whose arguments take precedence.
            opts: Merged options from ``_merge_options()``.

        Returns:
            The ``options`` dict for the payload.
        """
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

        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        return options

    def _base_payload(self, request: AiRequest, opts: dict[str, Any]) -> dict[str, Any]:
        """The fields ``/api/chat`` and ``/api/generate`` share."""
        payload: dict[str, Any] = {
            "model": request.model,
            "stream": False,
            "options": self._build_options(request, opts),
        }
        keep_alive = opts.get("keep_alive")
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if request.format_json:
            payload["format"] = "json"
        if request.thinking is True:
            payload["think"] = True
        elif request.thinking is False:
            payload["think"] = False
        return payload

    def _build_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Convert one history entry, keeping its text, files and tool calls.

        Args:
            msg: Source message dict. May contain ``files`` and, on an assistant
                message, ``tool_calls``; a ``role: "tool"`` entry may carry
                ``name`` and ``tool_call_id``. An assistant entry carrying this
                provider's ``provider_state`` is replayed from it, thinking
                trace included.

        Returns:
            Ollama-format message dict.
        """
        native = self._native_message(msg)
        if isinstance(native, dict):
            return {k: native[k] for k in _MESSAGE_FIELDS if k in native}

        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_files = normalize_file_paths(msg.get("files"))
        images: list[str] = []
        if msg_files:
            content, images = self._process_files_for_message(msg_files, content)

        entry: dict[str, Any] = {"role": role, "content": content}
        if images:
            entry["images"] = images
        if role == "assistant" and msg.get("tool_calls"):
            entry["tool_calls"] = []
            for tc in msg["tool_calls"]:
                call_id, name, arguments = self._tool_call_fields(tc)
                call: dict[str, Any] = {"function": {"name": name, "arguments": arguments}}
                if call_id:
                    call["id"] = call_id
                entry["tool_calls"].append(call)
        if role == "tool":
            entry.update(_tool_link(msg.get("name"), msg.get("tool_call_id")))
        return entry

    def _chat(self, request: AiRequest, opts: dict[str, Any]) -> AiResponse:
        """Send the request to ``/api/chat`` and parse the reply."""
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for msg in request.messages or []:
            messages.append(self._build_message(msg))

        # Skipped turns are not encoded either: a tool-result continuation's
        # attachments would be read for nothing.
        if self._wants_current_turn(request):
            prompt, images = self._process_files_for_message(
                normalize_file_paths(request.file_path), request.prompt
            )
            user_message: dict[str, Any] = {"role": "user", "content": prompt}
            if images:
                user_message["images"] = images
            messages.append(user_message)

        for tr in request.tool_results or []:
            messages.append(
                {"role": "tool", "content": tr.content, **_tool_link(tr.name, tr.call_id)}
            )

        payload = self._base_payload(request, opts)
        payload["messages"] = messages
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
        # Last, so a `think` level such as "high" or a JSON-schema `format`
        # overrides what the tri-state flags derived.
        payload.update(_request_fields(opts, _CHAT_FIELDS))

        resp = self._post("/api/chat", payload, request.timeout)
        message = resp.get("message", {})
        return self._build_response(
            resp,
            message.get("content", "") or "",
            message.get("thinking", "") or "",
            self._parse_tool_calls(message),
            native={"role": "assistant", **message},
        )

    def _generate(self, request: AiRequest, opts: dict[str, Any]) -> AiResponse:
        """Send the request to ``/api/generate`` and parse the reply.

        The endpoint takes one prompt with its attachments, a system prompt and
        the same ``think`` flag as ``/api/chat``; history and tools were refused
        by ``_check_endpoint_fields()``.
        """
        prompt, images = self._process_files_for_message(
            normalize_file_paths(request.file_path), request.prompt
        )
        payload = self._base_payload(request, opts)
        payload["prompt"] = prompt
        if request.system_prompt:
            payload["system"] = request.system_prompt
        if images:
            payload["images"] = images
        payload.update(_request_fields(opts, _GENERATE_FIELDS))

        resp = self._post("/api/generate", payload, request.timeout)
        return self._build_response(
            resp, resp.get("response", "") or "", resp.get("thinking", "") or "", []
        )

    @staticmethod
    def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
        """Read the tool calls of a chat reply, giving each an id.

        Ollama does not always return one, so a missing id is derived from the
        tool name and position.
        """
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            tc_id = tc.get("id") or f"{fn.get('name', 'tool')}_{len(tool_calls)}"
            tool_calls.append(ToolCall(
                id=tc_id,
                name=fn.get("name", ""),
                arguments=fn.get("arguments") or {},
            ))
        return tool_calls

    def _build_response(
        self,
        resp: dict[str, Any],
        text: str,
        thinking: str,
        tool_calls: list[ToolCall],
        native: dict[str, Any] | None = None,
    ) -> AiResponse:
        """Assemble the ``AiResponse``, splitting ``eval_count`` between trace and answer.

        ``native`` is the reply message ``/api/chat`` returned, kept for replay;
        ``/api/generate`` has no message to keep.
        """
        eval_count = resp.get("eval_count", 0)
        reasoning_tokens = self._estimate_reasoning_tokens(text, thinking, eval_count)
        return AiResponse(
            text=text,
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=max(0, eval_count - reasoning_tokens),
            reasoning_tokens=reasoning_tokens,
            reasoning_text=thinking,
            tool_calls=tool_calls,
            message=self._assistant_message(text, tool_calls, native),
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
        _fold_options(self.config.extra_options or {}, options)
        _fold_options(extra_options or {}, options)
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
