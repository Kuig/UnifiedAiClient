from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from unified_ai_client.exceptions import ModelRefusalError, NonRetryableError
from unified_ai_client.file_utils import (
    encode_file_base64,
    get_mime_type,
    normalize_file_paths,
)
from unified_ai_client.http import get_json, post_json
from unified_ai_client.models import (
    AiRequest,
    AiResponse,
    ModelInfo,
    ProviderConfig,
    ToolCall,
)
from unified_ai_client.providers.base import BaseProvider

_log = logging.getLogger("unified_ai_client.providers.anthropic")

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider.

    Uses the /v1/messages endpoint directly via urllib, with no SDK.

    File handling:
    - Images: base64 image content block.
    - PDFs: base64 document content block.
    - Text files: inlined via inline_text_attachments().
    - Audio: rejected. The Messages API has no audio block.

    Thinking: the request shape depends on the model generation, see
    _build_thinking_payload(). Reasoning text is extracted from
    'type: "thinking"' content blocks in the response, and from Claude 4 on it
    is a summary, flagged through ``reasoning_is_summary``.

    A policy refusal (``stop_reason: "refusal"``) raises ``ModelRefusalError``
    rather than coming back as an empty or truncated answer.
    """

    DEFAULT_URL: str = "https://api.anthropic.com"

    REQUIRES_API_KEY: bool = True
    SECRETS_KEY: str = "anthropic_api_key"

    SUPPORTED_FILE_TYPES: frozenset[str] = frozenset({"image", "document"})

    # Mandatory on /v1/messages, so used when nobody set max_tokens. Also sizes
    # the thinking budget on models older than 4.6.
    DEFAULT_MAX_TOKENS: int = 8192

    # The rest of the library namespace is dropped, not forwarded:
    # /v1/messages rejects an unknown top-level field.
    _CONSUMED_OPTION_KEYS: frozenset[str] = frozenset(
        {"temperature", "max_tokens", "top_k", "top_p"}
    )

    # Below 4.6 only the fixed-budget form is accepted; from 4.7 only adaptive.
    _ADAPTIVE_SINCE: tuple[int, int] = (4, 6)
    # Claude 3.7 is where extended thinking was introduced.
    _THINKING_SINCE: tuple[int, int] = (3, 7)
    # Anthropic's documented floor for budget_tokens.
    _MIN_THINKING_BUDGET: int = 1024
    # From Claude 4 on the API never returns the raw chain of thought, only a
    # summary of it; 3.7 returned the full trace.
    _SUMMARISED_SINCE: tuple[int, int] = (4, 0)
    # From 4.7 on thinking.display defaults to "omitted", which returns thinking
    # blocks with empty text; "summarized" has to be asked for.
    _DISPLAY_SINCE: tuple[int, int] = (4, 7)
    # Family to first version where {"type": "disabled"} is a 400, so
    # thinking=False lowers the effort instead.
    _NO_DISABLE_SINCE: dict[str, tuple[int, int]] = {
        "opus": (5, 5), "fable": (0, 0), "mythos": (0, 0),
    }
    # Models whose off switch is {"type": "between_tools"}, "disabled" being a
    # 400 there too.
    _BETWEEN_TOOLS_SINCE: dict[str, tuple[int, int]] = {"sonnet": (5, 5)}
    # What thinking=False becomes on a model in _NO_DISABLE_SINCE: the lowest
    # effort the API offers, the closest it allows to "off".
    _MINIMAL_EFFORT: str = "low"

    def __init__(self, config: ProviderConfig, api_key: str = "") -> None:
        """Initialize the Anthropic provider.

        Args:
            config: ProviderConfig with connection settings.
            api_key: Anthropic API key from secrets.json as 'anthropic_api_key'.
        """
        self.config = config
        self.api_key = api_key or ""
        self.base_url = (config.url or self.DEFAULT_URL).rstrip("/")

    @staticmethod
    def _model_version(model: str) -> tuple[int, int] | None:
        """Extract the (major, minor) generation from a Claude model name.

        The version sits in different places depending on the naming era:
        'claude-3-5-haiku-latest' puts it before the role, 'claude-opus-4-5'
        after it, and 'claude-opus-5' carries no minor at all. Taking the first
        numeric group and its optional companion handles all three.

        The minor is capped at two digits and must not be followed by another
        one, which is what keeps a dated id apart from a versioned one.
        'claude-opus-4-20250514' carries no minor and would otherwise read as
        (4, 20250514), clearing every future threshold and picking the wrong
        thinking form for the whole Claude 4.0 line.

        Args:
            model: Model identifier as passed to call_ai().

        Returns:
            A (major, minor) tuple, or None if no version could be read.
        """
        match = re.search(r"(\d+)(?:-(\d{1,2}))?(?!\d)", model)
        if not match:
            return None
        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0
        return major, minor

    @staticmethod
    def _model_family(model: str) -> str | None:
        """Extract the model line ('opus', 'sonnet', ...) from a model name.

        Args:
            model: Model identifier as passed to call_ai().

        Returns:
            The family name, or None if the name carries none of the known ones.
        """
        match = re.search(r"(opus|sonnet|haiku|fable|mythos)", model)
        return match.group(1) if match else None

    def _matches(self, model: str, table: dict[str, tuple[int, int]]) -> bool:
        """Whether a model falls under a family-keyed version rule.

        Args:
            model: Model identifier as passed to call_ai().
            table: Family name to the first version the rule applies to.

        Returns:
            True if the model's family is in the table and its version, when
            readable, is at or past the threshold. An unreadable version counts
            as past it, for the same reason ``_uses_adaptive_thinking()``
            guesses towards the newer form.
        """
        since = table.get(self._model_family(model) or "")
        if since is None:
            return False
        version = self._model_version(model)
        return version is None or version >= since

    def _uses_adaptive_thinking(self, model: str) -> bool:
        """Whether this model takes the adaptive thinking form.

        Sending the wrong form is a hard 400 in either direction, so the
        distinction matters: models below 4.6 reject 'adaptive', and models
        from 4.7 on reject 'budget_tokens'.

        An unrecognisable name is treated as adaptive. Guessing towards the
        newer form breaks only models on their way out, where guessing towards
        the older one would break every current model.

        Args:
            model: Model identifier as passed to call_ai().

        Returns:
            True for the adaptive form, False for the fixed-budget form.
        """
        version = self._model_version(model)
        if version is None:
            return True
        return version >= self._ADAPTIVE_SINCE

    def _build_thinking_payload(
        self, thinking: bool | str, model: str, max_tokens: int
    ) -> dict[str, Any] | None:
        """Build the 'thinking' field for the request, or None to omit it.

        Args:
            thinking: True, False, or "default" as passed to call_ai().
            model: Model identifier, which decides the accepted form.
            max_tokens: Resolved output cap. Anthropic requires
                ``budget_tokens`` to stay below it.

        Returns:
            The dict to send as ``thinking``, or None to leave it out entirely.
        """
        if thinking == "default" or thinking is None:
            return None

        adaptive = self._uses_adaptive_thinking(model)

        if thinking:
            if adaptive:
                payload: dict[str, Any] = {"type": "adaptive"}
                # From 4.7 the trace comes back empty unless asked for; this
                # changes what is returned, not what is computed or billed.
                version = self._model_version(model)
                if version is None or version >= self._DISPLAY_SINCE:
                    payload["display"] = "summarized"
                return payload
            budget = max(self._MIN_THINKING_BUDGET, max_tokens // 2)
            return {"type": "enabled", "budget_tokens": budget}

        # Older models have no 'disabled' form: their thinking is off unless enabled.
        if not adaptive:
            return None
        # These cannot switch it off; _minimal_effort() lowers the effort instead.
        if self._matches(model, self._NO_DISABLE_SINCE):
            return None
        if self._matches(model, self._BETWEEN_TOOLS_SINCE):
            return {"type": "between_tools"}
        return {"type": "disabled"}

    def _minimal_effort(self, thinking: bool | str, model: str) -> str | None:
        """The effort to send for thinking=False on a model that cannot disable it.

        On those models ``{"type": "disabled"}`` is a 400, so "off" cannot be
        expressed. The lowest effort is the closest the API allows, the same
        compromise Google's Gemini 3 forces with ``thinking_level=MINIMAL``.

        Args:
            thinking: True, False, or "default" as passed to call_ai().
            model: Model identifier.

        Returns:
            The effort level to send, or None when there is nothing to lower.
        """
        if thinking is False and self._matches(model, self._NO_DISABLE_SINCE):
            return self._MINIMAL_EFFORT
        return None

    @staticmethod
    def _apply_thinking_constraints(payload: dict[str, Any]) -> None:
        """Strip the sampling controls extended thinking refuses, in place.

        With thinking enabled the Messages API rejects ``top_k`` and ``top_p``
        outright, and accepts ``temperature`` only at its default of 1, so
        sending any of them turns ``thinking=True`` into a 400. Dropping them is
        the only option that keeps the call alive, and thinking is what the
        caller asked for in this very call. Since the library no longer sends a
        sampling value nobody chose, anything present here was set by the
        caller or the config, so dropping it is announced with a warning.

        Args:
            payload: The request body, modified in place.
        """
        if payload.get("thinking", {}).get("type") in ("disabled", "between_tools"):
            return
        dropped = [k for k in ("temperature", "top_k", "top_p") if k in payload]
        for key in dropped:
            payload.pop(key)
        if dropped:
            _log.warning(
                "Anthropic: dropping %s, which the Messages API refuses with "
                "thinking enabled.",
                ", ".join(dropped),
            )

    def _auth_headers(self) -> dict[str, str]:
        """Headers that identify this caller to the Anthropic API.

        Returns:
            The x-api-key credential plus the pinned API version, which the
            Messages API requires on every request.
        """
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _post(self, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        """HTTP POST to the Anthropic /v1/messages endpoint.

        Takes no endpoint argument, unlike its siblings: this provider posts to
        exactly one path.

        Args:
            payload: Request payload dict.
            timeout: Seconds to wait.

        Returns:
            Parsed JSON response dict.

        Raises:
            NonRetryableHttpError: On a 4xx other than 408/409/425/429.
            ProviderHttpError: On any other HTTP error response.
            urllib.error.URLError: On connection errors.
            ValueError: If no API key is set.
        """
        self._require_api_key()
        return post_json(
            f"{self.base_url}/v1/messages", payload, timeout, self._auth_headers()
        )

    def _get(self, endpoint: str, timeout: int) -> dict[str, Any]:
        """HTTP GET against the Anthropic API.

        Args:
            endpoint: API path (e.g. '/v1/models').
            timeout: Seconds to wait.

        Returns:
            Parsed JSON response dict.

        Raises:
            NonRetryableHttpError: On a 4xx other than 408/409/425/429.
            ProviderHttpError: On any other HTTP error response.
            urllib.error.URLError: On connection errors.
            ValueError: If no API key is set.
        """
        self._require_api_key()
        return get_json(f"{self.base_url}{endpoint}", timeout, self._auth_headers())

    def list_models(self, *, timeout: int | None = None) -> list[ModelInfo]:
        """List models through ``GET /v1/models``.

        The listing carries an id and a display name. What the rest of the
        fields hold is this library's knowledge of the Claude line, not the
        endpoint's: every Claude model calls tools and takes images and PDFs,
        and extended thinking arrived with 3.7, read off the model name by
        ``_model_version()``. A name that does not parse leaves ``reasoning``
        unknown.

        Args:
            timeout: Seconds to wait. Defaults to the configured timeout.

        Returns:
            One ``ModelInfo`` per model, newest first as the API orders them.

        Raises:
            NonRetryableHttpError: On a 4xx other than 408/409/425/429.
            ProviderHttpError: On any other HTTP error response.
            urllib.error.URLError: On connection errors.
            ValueError: If no API key is set.
        """
        # The endpoint pages at 20 by default; 1000 is its documented ceiling.
        resp = self._get("/v1/models?limit=1000", self._resolve_timeout(timeout))
        models: list[ModelInfo] = []
        for item in resp.get("data", []):
            model_id = item.get("id")
            if not model_id:
                continue
            version = self._model_version(model_id)
            models.append(
                ModelInfo(
                    id=model_id,
                    provider=self.provider_name,
                    display_name=item.get("display_name"),
                    reasoning=None if version is None else version >= self._THINKING_SINCE,
                    tool_use=True,
                    file_types=self.SUPPORTED_FILE_TYPES,
                    embeddings=False,
                )
            )
        return models

    def warm_up(
        self,
        model: str,
        file_paths: str | list[str] | None = None,
        *,
        keep_alive: str | int | None = None,
        timeout: int | None = None,
    ) -> bool:
        """Open the connection with a free metadata request.

        ``GET /v1/models`` consumes no tokens. It pays the DNS + TCP + TLS
        handshake and validates the API key.

        Args:
            model: Unused. Listing models warms the channel regardless.
            file_paths: Ignored. Anthropic takes attachments inline in the
                request and keeps no remote file store.
            keep_alive: Ignored. Anthropic holds no model on the caller's
                behalf, so it has no residency to control.
            timeout: Seconds to wait. Defaults to the configured timeout.

        Returns:
            Always True.
        """
        self._get("/v1/models", self._resolve_timeout(timeout))
        return True

    def _build_content_blocks(
        self, prompt: str, file_paths: list[str], *, pad_if_empty: bool = True
    ) -> list[dict[str, Any]]:
        """Build the native-attachment and text blocks any message is made of.

        Shared by a plain user turn and a history message that also carries
        tool_calls, so text and attachments are handled identically either
        way: a caller does not lose its file or its prompt just because
        tool_calls happen to be present in the same message.

        Args:
            prompt: The message text.
            file_paths: List of file paths to attach.
            pad_if_empty: When true and neither an attachment nor a non-empty
                prompt produced a block, still emit one empty text block,
                because the Messages API rejects an empty content array. A
                caller that is about to append its own blocks afterwards
                (tool_use blocks, currently the only such caller) passes
                False: those guarantee the array is non-empty on their own,
                so padding here would only waste a block.

        Returns:
            Content blocks: one native block per attachment, followed by a
            text block when the prompt has anything to say (or when nothing
            else does and padding was requested).

        Raises:
            FileNotFoundError: If an attachment does not exist.
            UnsupportedFileError: If an attachment is neither text nor a class
                the Messages API accepts. Audio lands here: dropping it silently
                let the model answer as though it had heard the recording.
        """
        effective_prompt, native = self._partition_attachments(file_paths, prompt)

        content: list[dict[str, Any]] = []
        for fp, ft in native:
            if ft not in ("image", "document"):
                raise self._unsupported_native_error(fp, ft)
            _log.debug("Anthropic: '%s' → %s base64 block", os.path.basename(fp), ft)
            content.append({
                "type": ft,
                "source": {
                    "type": "base64",
                    "media_type": (
                        "application/pdf" if ft == "document" else get_mime_type(fp)
                    ),
                    "data": encode_file_base64(fp),
                },
            })

        # An empty text block is a 400, and prompt="" with only a file attached
        # is a legitimate "describe this" call.
        if effective_prompt or (pad_if_empty and not content):
            content.append({"type": "text", "text": effective_prompt})
        return content

    def _build_user_content(
        self, prompt: str, file_paths: list[str]
    ) -> list[dict[str, Any]]:
        """Build Anthropic-style content blocks for a user message.

        Args:
            prompt: The user prompt text.
            file_paths: List of file paths to attach.

        Returns:
            List of content block dicts in Anthropic format.

        Raises:
            FileNotFoundError: If an attachment does not exist.
            UnsupportedFileError: If an attachment is neither text nor a class
                the Messages API accepts. Audio lands here: dropping it silently
                let the model answer as though it had heard the recording.
        """
        return self._build_content_blocks(prompt, file_paths)

    def _build_message(
        self, msg: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a single message dict, processing optional 'files' key.

        Args:
            msg: Source message dict. May contain 'files' (list of file paths)
                 or 'tool_calls' (for assistant messages that requested tool
                 execution, converted to Anthropic ``tool_use`` blocks). A
                 ``role: "tool"`` entry becomes a user message with one
                 ``tool_result`` block. An assistant entry carrying this
                 provider's ``provider_state`` is replayed from it, signed
                 thinking blocks included.

        Returns:
            Anthropic-format message dict.

        Raises:
            NonRetryableError: If a ``role: "tool"`` entry has no
                ``tool_call_id``, which is the only link to its ``tool_use``.
        """
        native = self._native_message(msg)
        if isinstance(native, list):
            return {"role": "assistant", "content": native}

        role = msg.get("role", "user")
        text = msg.get("content", "")
        file_paths = normalize_file_paths(msg.get("files"))

        if role == "tool":
            if not msg.get("tool_call_id"):
                raise NonRetryableError(
                    "A history entry with role 'tool' needs the 'tool_call_id' of "
                    "the call it answers; ToolResult.to_message() includes it."
                )
            return {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": text,
            }]}

        if role == "assistant" and msg.get("tool_calls"):
            # Same builder as any other message, so a file on this entry is
            # kept; the tool_use blocks below make padding unnecessary.
            content = self._build_content_blocks(text, file_paths, pad_if_empty=False)
            for i, tc in enumerate(msg["tool_calls"]):
                call_id, name, arguments = self._tool_call_fields(tc)
                content.append({
                    "type": "tool_use",
                    "id": call_id or f"tool_{i}",
                    "name": name,
                    "input": arguments,
                })
            return {"role": role, "content": content}

        if file_paths or role == "user":
            content = self._build_user_content(text, file_paths)
        else:
            content = [{"type": "text", "text": text}]
        return {"role": role, "content": content}

    def call(self, request: AiRequest) -> AiResponse:
        """Execute an inference call to the Anthropic Messages API.

        Args:
            request: The structured request.

        Returns:
            Standardized AiResponse.

        Raises:
            NonRetryableError: If ``format_json`` is set without a schema.
            ModelRefusalError: If the model declined on policy grounds.
        """
        opts = self._merge_options(request)

        # No schema-less JSON mode exists: structured output takes a schema, and
        # the prefill trick is a 400 from Claude 4.6 on. Refusing beats ignoring.
        if request.format_json and "format" not in (opts.get("output_config") or {}):
            raise NonRetryableError(
                "Anthropic has no schema-less JSON mode, so format_json=True "
                "cannot be honoured on its own. Pass a schema instead: "
                'extra_options={"output_config": {"format": {"type": '
                '"json_schema", "schema": {...}}}}.'
            )

        payload = self._build_payload(request, opts, self._build_messages(request))
        resp = self._post(payload, request.timeout)
        return self._parse_response(resp, request.model)

    def _build_messages(self, request: AiRequest) -> list[dict[str, Any]]:
        """History, the current turn and any tool results, in order.

        The current turn follows ``_wants_current_turn()``. Tool results travel
        as ``tool_result`` blocks in the user message right after the assistant
        message holding the matching ``tool_use`` blocks. Consecutive user
        messages are merged into one, so several ``role: "tool"`` entries, or a
        tool result followed by a new prompt, form a single valid turn.
        """
        messages: list[dict[str, Any]] = []
        for msg in request.messages or []:
            self._append_merged(messages, self._build_message(msg))

        if self._wants_current_turn(request):
            file_paths = normalize_file_paths(request.file_path)
            self._append_merged(messages, {
                "role": "user",
                "content": self._build_user_content(request.prompt, file_paths),
            })
        if request.tool_results:
            self._append_merged(messages, {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.call_id,
                        "content": tr.content,
                    }
                    for tr in request.tool_results
                ],
            })
        return messages

    @staticmethod
    def _append_merged(messages: list[dict[str, Any]], message: dict[str, Any]) -> None:
        """Append a message, folding it into the previous one when both are user turns."""
        if messages and message["role"] == "user" and messages[-1]["role"] == "user":
            messages[-1] = {
                "role": "user",
                "content": list(messages[-1]["content"]) + list(message["content"]),
            }
        else:
            messages.append(message)

    def _build_payload(
        self,
        request: AiRequest,
        opts: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the request body around the built messages.

        ``max_tokens`` is the one value this adapter still supplies when nobody
        set it, because the Messages API rejects a request without it.
        """
        max_tokens = self._prefer_request(
            request.max_tokens, opts, "max_tokens", self.DEFAULT_MAX_TOKENS
        )
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        for key, value in (
            ("temperature", request.temperature),
            ("top_k", request.top_k),
            ("top_p", request.top_p),
        ):
            resolved = self._prefer_request(value, opts, key)
            if resolved is not None:
                payload[key] = resolved

        payload.update(self._passthrough_options(opts))

        if request.system_prompt:
            payload["system"] = request.system_prompt

        if request.tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in request.tools
            ]

        thinking_payload = self._build_thinking_payload(
            request.thinking, request.model, max_tokens
        )
        if thinking_payload is not None:
            payload["thinking"] = thinking_payload
            self._apply_thinking_constraints(payload)

        effort = self._minimal_effort(request.thinking, request.model)
        if effort is not None:
            # Copied, not updated in place: an output_config from extra_options
            # is the caller's own dict, and an effort set there wins.
            output_config = dict(payload.get("output_config") or {})
            output_config.setdefault("effort", effort)
            payload["output_config"] = output_config
        return payload

    def _parse_response(self, resp: dict[str, Any], model: str) -> AiResponse:
        """Read text, thinking and tool calls out of the content blocks.

        Thinking blocks are read even when thinking was not requested.
        ``output_tokens`` covers trace and answer together, so the split is
        estimated from the character ratio. From Claude 4 on the trace is a
        summary, never the chain of thought itself; an unreadable model name is
        taken to be a current model.

        Raises:
            ModelRefusalError: On ``stop_reason: "refusal"``, which arrives as a
                200 with content empty or cut off where a classifier stopped it.
        """
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.get("content", []):
            if block.get("type") == "thinking":
                reasoning_parts.append(block.get("thinking", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                ))
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        raw_reasoning = "".join(reasoning_parts)
        response_text = "".join(text_parts)

        if resp.get("stop_reason") == "refusal":
            details = resp.get("stop_details") or {}
            raise ModelRefusalError(
                self.provider_name,
                category=details.get("category"),
                explanation=details.get("explanation") or "",
                partial_text=response_text,
            )

        usage = resp.get("usage", {})
        output_tokens = usage.get("output_tokens", 0)
        reasoning_tokens = self._estimate_reasoning_tokens(
            response_text, raw_reasoning, output_tokens
        )
        version = self._model_version(model)
        summarised = version is None or version >= self._SUMMARISED_SINCE
        native = resp.get("content", [])

        return AiResponse(
            text=response_text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=max(0, output_tokens - reasoning_tokens),
            reasoning_tokens=reasoning_tokens,
            reasoning_text=raw_reasoning,
            reasoning_is_summary=bool(raw_reasoning) and summarised,
            tool_calls=tool_calls,
            message=self._assistant_message(response_text, tool_calls, native),
        )
