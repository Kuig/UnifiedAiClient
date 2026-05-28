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
from unified_ai_client.models import AiRequest, AiResponse, ProviderConfig
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
            msg: Source message dict.

        Returns:
            Anthropic-format message dict.
        """
        role = msg.get("role", "user")
        text = msg.get("content", "")
        file_paths = normalize_file_paths(msg.get("files"))
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
        if self.config.sleep_time > 0:
            time.sleep(self.config.sleep_time)

        messages: list[dict[str, Any]] = []

        if request.messages:
            for msg in request.messages:
                messages.append(self._build_message(msg))

        file_paths = normalize_file_paths(request.file_path)
        user_content = self._build_user_content(request.prompt, file_paths)
        messages.append({"role": "user", "content": user_content})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": request.temperature,
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        if request.thinking:
            payload["thinking"] = {"type": "adaptive"}

        resp = self._post(payload, request.timeout)

        # Parse content blocks
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for block in resp.get("content", []):
            if block.get("type") == "thinking":
                reasoning_parts.append(block.get("thinking", ""))
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        usage = resp.get("usage", {})
        return AiResponse(
            text="".join(text_parts),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=0,
            reasoning_text="".join(reasoning_parts) if request.include_reasoning else "",
        )

    def preload_model(self, model: str, keep_alive: str = "15m") -> None:
        """Model preloading is not supported by the Anthropic API.

        Args:
            model: Unused.
            keep_alive: Unused.
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
