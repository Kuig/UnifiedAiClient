from __future__ import annotations

import json
import logging
import os
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
    - Audio: not natively supported; fallback to text inline with warning.
    - Text files: inlined via inline_text_attachments().

    Thinking: enabled via 'thinking: {type: "adaptive"}' in the request
    payload. Reasoning text is extracted from 'type: "thinking"' content
    blocks in the response.
    """

    DEFAULT_URL: str = "https://api.anthropic.com"

    def __init__(self, config: ProviderConfig, api_key: str = "") -> None:
        """Initialize the Anthropic provider.

        Args:
            config: ProviderConfig with connection settings.
            api_key: Anthropic API key from secrets.json as 'anthropic_api_key'.
        """
        self.config = config
        self.api_key = api_key
        if self.config.url == "http://localhost:11434":
            self.config.url = self.DEFAULT_URL

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
        """
        url = f"{self.config.url.rstrip('/')}/v1/messages"
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
        """
        url = f"{self.config.url.rstrip('/')}{endpoint}"
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
        """
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

            elif ft == "audio":
                _log.warning(
                    "Anthropic: audio not natively supported, skipping '%s'",
                    os.path.basename(fp),
                )

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

        if request.thinking is True:
            payload["thinking"] = {"type": "adaptive"}

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
            "Anthropic does not support text embeddings via the Messages API."
        )
