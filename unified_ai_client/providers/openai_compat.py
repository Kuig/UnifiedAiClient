from __future__ import annotations

import json
import logging
import os
from typing import Any

from unified_ai_client.exceptions import UnsupportedFileError, UnsupportedOperationError
from unified_ai_client.file_utils import (
    audio_format_name,
    encode_file_base64,
    get_mime_type,
    normalize_file_paths,
)
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

_log = logging.getLogger("unified_ai_client.providers.openai_compat")


class OpenAiCompatProvider(BaseProvider):
    """Base class for providers using the OpenAI-compatible /v1/chat/completions API.

    Shared by OpenAiProvider, MistralProvider, CohereProvider, MetaProvider,
    GroqProvider, XAiProvider, LmStudioProvider, and LlamaCppProvider.
    Subclasses override DEFAULT_URL and may override _build_native_block()
    to add native support for audio or PDF file types.

    File handling (default behaviour inherited by all subclasses):
    - Images: base64-encoded as image_url content blocks.
    - Text: inlined into the prompt via inline_text_attachments().
    - Anything else: rejected by validate_files() before the request is built.

    Reasoning text: populated from the 'reasoning_content' field in the
    response message choice, whenever the endpoint returns one. llama-server
    does by default, LM Studio when it is set to separate the trace, and the
    cloud endpoints for their reasoning models.
    """

    DEFAULT_URL: str = "http://localhost:8080"

    API_PREFIX: str = "/v1"
    """Path segment between the base URL and the endpoint names.

    Every request path is built from it: ``/v1/chat/completions``,
    ``/v1/models``, ``/v1/embeddings``. A provider whose OpenAI-compatible API
    lives under another prefix (Z.ai serves it at ``/api/paas/v4``) sets this
    instead of overriding the methods that post. ``PROBE_PATH`` is separate and
    read off the class by ``list_providers()``, so a local subclass that
    changes the prefix sets both.
    """

    REASONING_PARAM: str | None = None
    """Payload field that controls reasoning, or None when there is none.

    Sent only when ``thinking`` is True or False: a non-reasoning model rejects
    the parameter outright, so ``"default"`` must leave it out.
    """

    # image_url is the only block every one of these endpoints understands.
    SUPPORTED_FILE_TYPES: frozenset[str] = frozenset({"image"})

    # The rest of the library namespace is dropped, not forwarded: a strict
    # endpoint answers an unknown field such as context_size with a 400.
    _CONSUMED_OPTION_KEYS: frozenset[str] = frozenset(
        {"temperature", "max_tokens", "top_k", "top_p"}
    )

    SUPPORTS_EMBEDDINGS: bool = True

    EMBEDDING_DIMENSIONS_PARAM: str | None = None
    """Request field that shortens an embedding vector, or None if there is none.

    Only OpenAI's ``dimensions`` is documented across this family. Where it is
    None, a ``dimensions`` argument is refused rather than silently ignored: a
    full-size vector where a short one was asked for would break whatever
    index it is written to.
    """

    # Probed only on the two local subclasses; cloud ones are judged by credential.
    PROBE_PATH: str | None = "/v1/models"

    LAZY_MODEL_LOAD: bool = False
    """Whether this endpoint defers loading the model until first inference.

    True only for the local servers, LM Studio and llama.cpp, where a metadata
    GET returns instantly and leaves the load cost for the first real call, so
    warming up has to send a one-token completion instead. That completion is
    billable inference in principle; it stays acceptable only because these
    servers are local, which is why the default is False and a paid endpoint
    must never set it.
    """

    def __init__(self, config: ProviderConfig, api_key: str = "") -> None:
        """Initialize the provider.

        Args:
            config: ProviderConfig with connection settings.
            api_key: Optional Bearer token for Authorization header.
        """
        self.config = config
        self.api_key = api_key or ""
        self.base_url = (config.url or self.DEFAULT_URL).rstrip("/")

    def _auth_headers(self) -> dict[str, str]:
        """Headers that identify this caller to the endpoint.

        Returns:
            A Bearer authorization header, or an empty dict for a local server
            that authenticates nobody.
        """
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _post(self, endpoint: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        """HTTP POST to the OpenAI-compatible endpoint.

        Args:
            endpoint: API path (e.g. '/v1/chat/completions').
            payload: Request payload dict.
            timeout: Seconds to wait.

        Returns:
            Parsed JSON response dict.

        Raises:
            NonRetryableHttpError: On a 4xx other than 408/409/425/429.
            ProviderHttpError: On any other HTTP error response.
            urllib.error.URLError: On connection errors.
            json.JSONDecodeError: On unparseable response.
            ValueError: If this provider requires an API key and none is set.
        """
        self._require_api_key()
        return post_json(
            f"{self.base_url}{endpoint}", payload, timeout, self._auth_headers()
        )

    def _get(self, endpoint: str, timeout: int) -> dict[str, Any]:
        """HTTP GET against the OpenAI-compatible endpoint.

        Args:
            endpoint: API path (e.g. '/v1/models').
            timeout: Seconds to wait.

        Returns:
            Parsed JSON response dict.

        Raises:
            NonRetryableHttpError: On a 4xx other than 408/409/425/429.
            ProviderHttpError: On any other HTTP error response.
            urllib.error.URLError: On connection errors.
            json.JSONDecodeError: On unparseable response.
            ValueError: If this provider requires an API key and none is set.
        """
        self._require_api_key()
        return get_json(f"{self.base_url}{endpoint}", timeout, self._auth_headers())

    def _warm_up_completion(self, model: str, timeout: int | None = None) -> None:
        """Send a one-token completion to force a lazy server to load the model.

        Only meaningful for local servers that defer the model load until the
        first inference. Do not call this against a paid remote endpoint: the
        request is small but it is still billable inference.

        Args:
            model: Model identifier to load.
            timeout: Seconds to wait. Defaults to the configured timeout.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        self._post(
            f"{self.API_PREFIX}/chat/completions", payload, self._resolve_timeout(timeout)
        )

    def warm_up(
        self,
        model: str,
        file_paths: str | list[str] | None = None,
        *,
        keep_alive: str | int | None = None,
        timeout: int | None = None,
    ) -> bool:
        """Open the connection with a free metadata request.

        ``GET /v1/models`` costs nothing and consumes no tokens, so it is safe
        on paid endpoints. It pays the DNS + TCP + TLS handshake and validates
        the API key, but it does not load the model. Providers that declare
        ``LAZY_MODEL_LOAD`` send a one-token completion instead, because on
        those servers the metadata GET would return before any load happened.

        Args:
            model: The model to load, used only when ``LAZY_MODEL_LOAD`` is set.
                Otherwise unused: listing models warms the channel regardless.
            file_paths: Ignored. These providers inline attachments into the
                request and keep no remote file store.
            keep_alive: Ignored. A cloud endpoint holds no model on the
                caller's behalf, so it has no residency to control.
            timeout: Seconds to wait. Defaults to the configured timeout.

        Returns:
            Always True: the connection is always worth opening early.
        """
        if self.LAZY_MODEL_LOAD:
            self._warm_up_completion(model, timeout)
        else:
            self._get(f"{self.API_PREFIX}/models", self._resolve_timeout(timeout))
        return True

    def _reasoning_payload(self, thinking: bool | str) -> dict[str, Any]:
        """Map the tri-state ``thinking`` onto this endpoint's request fields.

        The hook a subclass overrides when its reasoning control is not a single
        string field. Only called for an explicit True/False: "default" (the
        call_ai default) sends nothing, which keeps plain non-reasoning models
        working, since they reject the parameter outright.

        A dict value returned here is merged with the caller's own dict under the
        same key, the caller's keys winning, so a hook can contribute to a field
        such as ``chat_template_kwargs`` without erasing what the caller put in it.

        Args:
            thinking: True or False.

        Returns:
            The fields to add to the request body, empty for no control.
        """
        if not self.REASONING_PARAM:
            return {}
        return {self.REASONING_PARAM: "high" if thinking else "none"}

    def _build_native_block(self, file_path: str, file_type: str) -> dict[str, Any]:
        """Build one provider-native content block for a non-text attachment.

        Covers the two block shapes shared across OpenAI-compatible endpoints.
        Subclasses override to add their own, calling super() for the rest.

        Args:
            file_path: Path to the attachment.
            file_type: Its class from ``classify_file()``.

        Returns:
            A single content block dict.

        Raises:
            UnsupportedFileError: If this class has no block shape here. Reaching
                this means SUPPORTED_FILE_TYPES claims a class the builder cannot
                actually produce.
        """
        name = os.path.basename(file_path)

        if file_type == "image":
            mime = get_mime_type(file_path)
            b64 = encode_file_base64(file_path)
            _log.debug("File '%s' → image_url block", name)
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }

        if file_type == "audio":
            b64 = encode_file_base64(file_path)
            fmt = audio_format_name(file_path)
            _log.debug("File '%s' → input_audio block (format=%s)", name, fmt)
            return {
                "type": "input_audio",
                "input_audio": {"data": b64, "format": fmt},
            }

        if file_type == "video":
            # llama-server's own extension, not an OpenAI block: only a
            # provider declaring 'video' reaches this.
            _log.debug("File '%s' → input_video block", name)
            return {
                "type": "input_video",
                "input_video": {"data": encode_file_base64(file_path)},
            }

        raise self._unsupported_native_error(file_path, file_type)

    def _build_user_content(
        self, prompt: str, file_paths: list[str]
    ) -> str | list[dict[str, Any]]:
        """Build the user message 'content' field.

        Returns a plain string if there are no file attachments, or a list of
        content blocks (text + images) if there are files.

        Args:
            prompt: The user prompt text.
            file_paths: List of file paths to attach.

        Returns:
            String or list of content block dicts.

        Raises:
            FileNotFoundError: If an attachment does not exist.
            UnsupportedFileError: If an attachment is neither text nor a class
                this provider declares in SUPPORTED_FILE_TYPES.
        """
        if not file_paths:
            return prompt

        effective_prompt, native = self._partition_attachments(file_paths, prompt)
        native_blocks = [self._build_native_block(fp, ft) for fp, ft in native]

        if native_blocks:
            content: list[dict[str, Any]] = [{"type": "text", "text": effective_prompt}]
            content.extend(native_blocks)
            return content

        return effective_prompt

    def _build_message(
        self, msg: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a single message dict, processing optional 'files' key.

        Args:
            msg: Source message dict with at least 'role' and 'content' keys.
                 May also contain 'files': list[str], 'tool_calls' (for
                 assistant messages that requested tool execution), or
                 'tool_call_id' (required on a 'tool' message, to link it
                 back to the call it answers).

        Returns:
            OpenAI-format message dict.
        """
        role = msg.get("role", "user")
        text = msg.get("content", "")
        file_paths = normalize_file_paths(msg.get("files"))
        content = self._build_user_content(text, file_paths)
        entry: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and msg.get("tool_calls"):
            entry["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "type": "function",
                    "function": tc.get("function", tc),
                }
                for i, tc in enumerate(msg["tool_calls"])
            ]
        # Read from the entry, never invented: a fallback id would link the
        # result to the wrong call as readily as to the right one.
        if role == "tool" and msg.get("tool_call_id"):
            entry["tool_call_id"] = msg["tool_call_id"]
        return entry

    def call(self, request: AiRequest) -> AiResponse:
        """Execute an inference call to the OpenAI-compatible endpoint.

        Args:
            request: The structured request.

        Returns:
            Standardized AiResponse.
        """
        opts = self._merge_options(request)
        payload = self._build_payload(request, opts, self._build_messages(request))
        resp = self._post(f"{self.API_PREFIX}/chat/completions", payload, request.timeout)
        return self._parse_response(resp)

    def _build_messages(self, request: AiRequest) -> list[dict[str, Any]]:
        """System prompt, history, the current turn and any tool results, in order.

        A tool-result continuation has its user turn in the history already, so
        the prompt is not appended a second time.
        """
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for msg in request.messages or []:
            messages.append(self._build_message(msg))

        if not request.tool_results:
            file_paths = normalize_file_paths(request.file_path)
            user_content = self._build_user_content(request.prompt, file_paths)
            messages.append({"role": "user", "content": user_content})

        for tr in request.tool_results or []:
            messages.append({
                "role": "tool",
                "tool_call_id": tr.call_id,
                "content": tr.content,
            })
        return messages

    def _build_payload(
        self,
        request: AiRequest,
        opts: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the request body around the built messages.

        A sampling lever that resolves to None is left out, so the server's own
        default applies: the reasoning models reject any temperature but their
        own, and a local server's preset is what the model's author tuned.
        """
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
        }
        if request.format_json:
            payload["response_format"] = {"type": "json_object"}

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

        max_tokens = self._prefer_request(request.max_tokens, opts, "max_tokens")
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens

        for key, value in (
            ("temperature", request.temperature),
            ("top_k", request.top_k),
            ("top_p", request.top_p),
        ):
            resolved = self._prefer_request(value, opts, key)
            if resolved is not None:
                payload[key] = resolved

        reasoning: dict[str, Any] = {}
        if request.thinking != "default" and request.thinking is not None:
            reasoning = self._reasoning_payload(request.thinking)

        # Provider-specific options go last so a caller can override the mapping
        # above; where both carry a dict under the same key the two are merged,
        # the caller's keys winning.
        passthrough = self._passthrough_options(opts)
        for key, value in reasoning.items():
            if isinstance(value, dict) and isinstance(passthrough.get(key), dict):
                passthrough[key] = {**value, **passthrough[key]}
        payload.update(reasoning)
        payload.update(passthrough)
        return payload

    def _parse_response(self, resp: dict[str, Any]) -> AiResponse:
        """Read text, tool calls, the reasoning trace and token counts.

        ``reasoning_content`` is read even when thinking was not requested,
        since a model may produce it anyway.
        """
        choice = resp["choices"][0]["message"]
        usage = resp.get("usage", {})

        tool_calls: list[ToolCall] = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"].get("arguments", "{}")),
            )
            for tc in choice.get("tool_calls") or []
        ]

        reasoning_tokens = (
            usage.get("completion_tokens_details", {})
            .get("reasoning_tokens", 0)
        )
        total_output = usage.get("completion_tokens", 0)

        return AiResponse(
            text=choice.get("content") or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=max(0, total_output - reasoning_tokens),
            reasoning_tokens=reasoning_tokens,
            reasoning_text=choice.get("reasoning_content") or "",
            tool_calls=tool_calls,
        )

    def list_models(self, *, timeout: int | None = None) -> list[ModelInfo]:
        """List models through ``GET /v1/models``.

        The OpenAI-compatible listing carries an id and little else, so every
        capability field is left ``None``: the endpoint does not say which models
        reason, call tools or accept images, and guessing from the name would
        age as fast as the catalogue does.

        Args:
            timeout: Seconds to wait. Defaults to the configured timeout.

        Returns:
            One ``ModelInfo`` per entry in the response's ``data`` array.

        Raises:
            NonRetryableHttpError: On a 4xx other than 408/409/425/429.
            ProviderHttpError: On any other HTTP error response.
            urllib.error.URLError: On connection errors.
            ValueError: If this provider requires an API key and none is set.
        """
        resp = self._get(f"{self.API_PREFIX}/models", self._resolve_timeout(timeout))
        return [
            ModelInfo(id=item["id"], provider=self.provider_name)
            for item in resp.get("data", [])
            if item.get("id")
        ]

    def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        """Embed text, a batch of texts, or one input with media, via /embeddings.

        Text, single or batched, goes in the standard OpenAI shape. Media do
        not exist in that shape: a provider that declares
        ``SUPPORTED_EMBEDDING_FILE_TYPES`` (llama-server) receives an input
        object ``{"content": [text, *blocks]}``, the blocks built by
        ``_build_native_block()`` exactly as for a chat message, and returns one
        vector for it.

        Only the call-time ``extra_options`` are forwarded here, not the
        configured ones: a provider configuration is written for chat requests,
        and a key such as ``reasoning_effort`` registered there would be a
        malformed embeddings request.

        Args:
            request: The embedding request.

        Returns:
            One vector per input.

        Raises:
            UnsupportedOperationError: If this provider serves no embeddings
                endpoint, or cannot shorten vectors and ``dimensions`` was set.
                Both are raised before anything is sent.
            UnsupportedFileError: If a file is neither text nor a class this
                provider can embed.
            EmptyResponseError: If the response holds no vector for an input.
        """
        if not self.SUPPORTS_EMBEDDINGS:
            raise UnsupportedOperationError(
                f"Provider '{self.provider_name}' serves no embeddings endpoint. "
                "Use another provider for embeddings."
            )
        if request.dimensions is not None and not self.EMBEDDING_DIMENSIONS_PARAM:
            raise UnsupportedOperationError(
                f"Provider '{self.provider_name}' has no way to request a vector "
                "size. Leave dimensions unset, then truncate the vector and "
                "renormalise it yourself."
            )

        inputs: Any
        if request.file_path:
            text, native = self._partition_attachments(
                request.file_path, request.inputs[0], self.SUPPORTED_EMBEDDING_FILE_TYPES
            )
            if native:
                content = [{"type": "text", "text": text}] if text else []
                content.extend(self._build_native_block(fp, ft) for fp, ft in native)
                inputs = [{"content": content}]
            else:
                inputs = text
        elif len(request.inputs) == 1:
            inputs = request.inputs[0]
        else:
            inputs = list(request.inputs)

        payload: dict[str, Any] = {"model": request.model, "input": inputs}
        if request.dimensions is not None:
            payload[self.EMBEDDING_DIMENSIONS_PARAM] = request.dimensions
        payload.update(self._passthrough_options(request.extra_options or {}))

        resp = self._post(f"{self.API_PREFIX}/embeddings", payload, request.timeout)
        data = sorted(resp.get("data") or [], key=lambda item: item.get("index", 0))
        vectors = [[float(x) for x in item["embedding"]] for item in data]
        return self._check_vector_count(vectors, request)
