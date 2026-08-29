from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from unified_ai_client.file_utils import (
    classify_file,
    encode_file_base64,
    get_mime_type,
    inline_text_attachments,
    normalize_file_paths,
    validate_files,
)
from unified_ai_client.models import AiRequest, AiResponse, ProviderConfig, ToolCall
from unified_ai_client.providers.base import BaseProvider

_log = logging.getLogger("unified_ai_client.providers.anthropic")

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider.

    Uses the /v1/messages endpoint directly via urllib — no SDK required.

    File handling:
    - Images: base64 image content block.
    - PDFs: base64 document content block.
    - Text files: inlined via inline_text_attachments().
    - Audio: rejected. The Messages API has no audio block.

    Thinking: the request shape depends on the model generation, see
    _build_thinking_payload(). Reasoning text is extracted from
    'type: "thinking"' content blocks in the response.
    """

    DEFAULT_URL: str = "https://api.anthropic.com"

    REQUIRES_API_KEY: bool = True
    SECRETS_KEY: str = "anthropic_api_key"

    SUPPORTED_FILE_TYPES: frozenset[str] = frozenset({"image", "document"})

    # Claude 4.6 is where adaptive thinking was introduced. Below it only the
    # fixed-budget form is accepted; from 4.7 on only the adaptive one is.
    _ADAPTIVE_SINCE: tuple[int, int] = (4, 6)
    # Anthropic's documented floor for budget_tokens.
    _MIN_THINKING_BUDGET: int = 1024

    def __init__(self, config: ProviderConfig, api_key: str = "") -> None:
        """Initialize the Anthropic provider.

        Args:
            config: ProviderConfig with connection settings.
            api_key: Anthropic API key from secrets.json as 'anthropic_api_key'.
        """
        self.config = config
        self.api_key = api_key or ""
        self.base_url = (config.url or self.DEFAULT_URL).rstrip("/")

    def _require_api_key(self) -> None:
        """Fail early and clearly when the API key is missing.

        Only enforced when talking to Anthropic's own endpoint. A caller who
        set an explicit url has pointed the adapter at something else, such as
        a local proxy, which may well need no credentials.

        Checked at request time rather than in ``__init__`` because
        ``get_provider()`` instantiates providers eagerly, before it is known
        whether the caller will ever send a request.

        Raises:
            ValueError: If no API key was supplied.
        """
        if not self.api_key and self.config.url is None:
            raise ValueError(
                f"Missing API key for provider 'anthropic'. "
                f"Add '{self.SECRETS_KEY}' to secrets.json "
                f"or set {self.SECRETS_KEY.upper()}."
            )

    @staticmethod
    def _model_version(model: str) -> tuple[int, int] | None:
        """Extract the (major, minor) generation from a Claude model name.

        The version sits in different places depending on the naming era:
        'claude-3-5-haiku-latest' puts it before the role, 'claude-opus-4-5'
        after it, and 'claude-opus-5' carries no minor at all. Taking the first
        numeric group and its optional companion handles all three.

        Args:
            model: Model identifier as passed to call_ai().

        Returns:
            A (major, minor) tuple, or None if no version could be read.
        """
        match = re.search(r"(\d+)(?:-(\d+))?", model)
        if not match:
            return None
        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0
        return major, minor

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
                return {"type": "adaptive"}
            # budget_tokens must stay under max_tokens, and above the floor.
            budget = max(self._MIN_THINKING_BUDGET, max_tokens // 2)
            return {"type": "enabled", "budget_tokens": budget}

        # thinking=False. Older models have no 'disabled' form: for them
        # thinking is off unless explicitly enabled, so omitting is correct.
        return {"type": "disabled"} if adaptive else None

    def _post(self, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        """HTTP POST to the Anthropic /v1/messages endpoint.

        Args:
            payload: Request payload dict.
            timeout: Seconds to wait.

        Returns:
            Parsed JSON response dict.

        Raises:
            urllib.error.HTTPError: On HTTP error responses.
            urllib.error.URLError: On connection errors.
            ValueError: If no API key is set.
        """
        self._require_api_key()
        url = f"{self.base_url}/v1/messages"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, endpoint: str, timeout: int) -> dict[str, Any]:
        """HTTP GET against the Anthropic API via urllib.

        Kept separate from ``_post``, which hardcodes the /v1/messages path and
        sends a JSON body.

        Args:
            endpoint: API path (e.g. '/v1/models').
            timeout: Seconds to wait.

        Returns:
            Parsed JSON response dict.

        Raises:
            urllib.error.HTTPError: On HTTP error responses.
            urllib.error.URLError: On connection errors.
            ValueError: If no API key is set.
        """
        self._require_api_key()
        url = f"{self.base_url}{endpoint}"
        headers: dict[str, str] = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def warm_up(
        self,
        model: str,
        file_paths: str | list[str] | None = None,
    ) -> bool:
        """Open the connection with a free metadata request.

        ``GET /v1/models`` consumes no tokens. It pays the DNS + TCP + TLS
        handshake and validates the API key.

        Args:
            model: Unused. Listing models warms the channel regardless.
            file_paths: Ignored. Anthropic takes attachments inline in the
                request and keeps no remote file store.

        Returns:
            Always True.
        """
        self._get("/v1/models", self.config.timeout)
        return True

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
        validate_files(file_paths, self.provider_name, self.SUPPORTED_FILE_TYPES)

        content: list[dict[str, Any]] = []
        text_files: list[str] = []

        for fp in file_paths:
            ft = classify_file(fp)

            if ft == "image":
                b64 = encode_file_base64(fp)
                mime = get_mime_type(fp)
                _log.info("Anthropic: '%s' → image base64 block", os.path.basename(fp))
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64,
                    },
                })

            elif ft == "document":
                b64 = encode_file_base64(fp)
                _log.info("Anthropic: '%s' → document base64 block", os.path.basename(fp))
                content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": b64,
                    },
                })

            else:
                text_files.append(fp)

        effective_prompt = inline_text_attachments(prompt, text_files)
        content.append({"type": "text", "text": effective_prompt})
        return content

    def _build_message(
        self, msg: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a single message dict, processing optional 'files' key.

        Args:
            msg: Source message dict. May contain 'files' (list of file paths)
                 or 'tool_calls' (for assistant messages that requested tool
                 execution — converted to Anthropic ``tool_use`` blocks).

        Returns:
            Anthropic-format message dict.
        """
        role = msg.get("role", "user")
        text = msg.get("content", "")
        file_paths = normalize_file_paths(msg.get("files"))

        if role == "assistant" and msg.get("tool_calls"):
            # Convert consumer's tool_calls into Anthropic tool_use blocks
            # so the API can link subsequent tool_result messages back.
            content: list[dict[str, Any]] = []
            if text:
                content.append({"type": "text", "text": text})
            for i, tc in enumerate(msg["tool_calls"]):
                fn = tc.get("function", tc)
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"tool_{i}"),
                    "name": fn.get("name", ""),
                    "input": fn.get("arguments", {}),
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
        """
        # Merge options
        opts = {}
        if self.config.extra_options:
            opts.update(self.config.extra_options)
        if request.extra_options:
            opts.update(request.extra_options)

        messages: list[dict[str, Any]] = []

        if request.messages:
            for msg in request.messages:
                messages.append(self._build_message(msg))

        # Current user message — only add when NOT in a tool result continuation.
        if not request.tool_results:
            file_paths = normalize_file_paths(request.file_path)
            user_content = self._build_user_content(request.prompt, file_paths)
            messages.append({"role": "user", "content": user_content})

        max_tokens = request.max_tokens if request.max_tokens is not None else opts.get("max_tokens", 8192)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": request.temperature,
        }

        top_k = request.top_k if request.top_k is not None else opts.get("top_k")
        if top_k is not None:
            payload["top_k"] = top_k

        top_p = request.top_p if request.top_p is not None else opts.get("top_p")
        if top_p is not None:
            payload["top_p"] = top_p

        # Populate payload from other opts
        for k, v in opts.items():
            if k not in ("max_tokens", "temperature", "top_k", "top_p", "timeout", "sleep_time", "keep_alive"):
                payload[k] = v

        if request.system_prompt:
            payload["system"] = request.system_prompt

        # Tool definitions
        if request.tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in request.tools
            ]

        # Tool results — Anthropic expects them as a user message with
        # tool_result content blocks, immediately following the assistant
        # message that contained the tool_use blocks.
        if request.tool_results:
            messages.append({
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

        thinking_payload = self._build_thinking_payload(
            request.thinking, request.model, max_tokens
        )
        if thinking_payload is not None:
            payload["thinking"] = thinking_payload

        resp = self._post(payload, request.timeout)

        # Parse content blocks — always collect text, thinking, and tool_use.
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

        usage = resp.get("usage", {})
        output_tokens = usage.get("output_tokens", 0)

        # Anthropic's output_tokens aggregates thinking + response tokens.
        # Estimate reasoning_tokens from the char-ratio of the combined output.
        reasoning_tokens = 0
        if raw_reasoning:
            total_chars = len(response_text) + len(raw_reasoning)
            if total_chars > 0 and output_tokens > 0:
                chars_per_token = total_chars / output_tokens
                reasoning_tokens = max(1, round(len(raw_reasoning) / chars_per_token))

        return AiResponse(
            text=response_text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=max(0, output_tokens - reasoning_tokens),
            reasoning_tokens=reasoning_tokens,
            reasoning_text=raw_reasoning,
            tool_calls=tool_calls,
        )

    def preload_model(
        self,
        model: str,
        keep_alive: str = "15m",
        context_size: int | None = None,
        extra_options: dict | None = None,
    ) -> None:
        """Model preloading is not supported by the Anthropic API.

        Args:
            model: Unused.
            keep_alive: Unused.
            context_size: Unused.
            extra_options: Unused.
        """
        pass

    def get_embedding(self, model: str, text: str) -> list[float]:
        """Anthropic does not support text embeddings.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Anthropic offers no embeddings API of its own and recommends a "
            "third-party provider. Use another provider for embeddings."
        )
