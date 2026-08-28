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

_log = logging.getLogger("unified_ai_client.providers.openai_compat")


class OpenAiCompatProvider(BaseProvider):
    """Base class for providers using the OpenAI-compatible /v1/chat/completions API.

    Shared by OpenAiProvider, LmStudioProvider, and LlamaCppProvider.
    Subclasses override DEFAULT_URL and may override _build_file_content_blocks()
    to add native support for audio or PDF file types.

    File handling (default behaviour inherited by all subclasses):
    - Images: base64-encoded as image_url content blocks.
    - Audio, text, PDF, unknown: inlined into the prompt via
      inline_text_attachments().

    Reasoning text: populated from the 'reasoning_content' field in the
    response message choice, if present (available on OpenAI o3/o4 models).
    Local providers (LM Studio, llama.cpp) always return an empty string.
    """

    DEFAULT_URL: str = "http://localhost:8080"

    def __init__(self, config: ProviderConfig, api_key: str = "") -> None:
        """Initialize the provider.

        Args:
            config: ProviderConfig with connection settings.
            api_key: Optional Bearer token for Authorization header.
        """
        self.config = config
        self.api_key = api_key
        # Apply default URL if the config still holds the Ollama placeholder
        if self.config.url == "http://localhost:11434":
            self.config.url = self.DEFAULT_URL

    def _post(self, endpoint: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        """HTTP POST to the OpenAI-compatible endpoint via urllib.

        Args:
            endpoint: API path (e.g. '/v1/chat/completions').
            payload: Request payload dict.
            timeout: Seconds to wait.

        Returns:
            Parsed JSON response dict.

        Raises:
            urllib.error.HTTPError: On HTTP error responses.
            urllib.error.URLError: On connection errors.
            json.JSONDecodeError: On unparseable response.
        """
        url = f"{self.config.url.rstrip('/')}{endpoint}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, endpoint: str, timeout: int) -> dict[str, Any]:
        """HTTP GET against the OpenAI-compatible endpoint via urllib.

        Args:
            endpoint: API path (e.g. '/v1/models').
            timeout: Seconds to wait.

        Returns:
            Parsed JSON response dict.

        Raises:
            urllib.error.HTTPError: On HTTP error responses.
            urllib.error.URLError: On connection errors.
            json.JSONDecodeError: On unparseable response.
        """
        url = f"{self.config.url.rstrip('/')}{endpoint}"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _warm_up_completion(self, model: str) -> None:
        """Send a one-token completion to force a lazy server to load the model.

        Only meaningful for local servers that defer the model load until the
        first inference. Do not call this against a paid remote endpoint: the
        request is small but it is still billable inference.

        Args:
            model: Model identifier to load.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        self._post("/v1/chat/completions", payload, self.config.timeout)

    def warm_up(
        self,
        model: str,
        file_paths: str | list[str] | None = None,
    ) -> bool:
        """Open the connection with a free metadata request.

        ``GET /v1/models`` costs nothing and consumes no tokens, so it is safe
        on paid endpoints. It pays the DNS + TCP + TLS handshake and validates
        the API key. It does not load the model: subclasses backed by a local
        server that loads lazily override this with a minimal completion.

        Args:
            model: Unused. Listing models warms the channel regardless.
            file_paths: Ignored. These providers inline attachments into the
                request and keep no remote file store.

        Returns:
            Always True: the connection is always worth opening early.
        """
        self._get("/v1/models", self.config.timeout)
        return True

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
        """
        if not file_paths:
            return prompt

        # Separate text-fallback files from natively-supported ones
        native_blocks: list[dict[str, Any]] = []
        text_files: list[str] = []

        for fp in file_paths:
            ft = classify_file(fp)
            if ft == "image":
                mime = get_mime_type(fp)
                b64 = encode_file_base64(fp)
                _log.info("File '%s' → image_url block", os.path.basename(fp))
                native_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            else:
                text_files.append(fp)

        effective_prompt = inline_text_attachments(prompt, text_files)

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
                 May also contain 'files': list[str] or 'tool_calls' (for
                 assistant messages that requested tool execution).

        Returns:
            OpenAI-format message dict.
        """
        role = msg.get("role", "user")
        text = msg.get("content", "")
        file_paths = normalize_file_paths(msg.get("files"))
        content = self._build_user_content(text, file_paths)
        entry: dict[str, Any] = {"role": role, "content": content}
        # Preserve tool_calls on assistant messages so the API can link
        # subsequent tool results back to the model's original request.
        if role == "assistant" and msg.get("tool_calls"):
            entry["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "type": "function",
                    "function": tc.get("function", tc),
                }
                for i, tc in enumerate(msg["tool_calls"])
            ]
        return entry

    def call(self, request: AiRequest) -> AiResponse:
        """Execute an inference call to the OpenAI-compatible endpoint.

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

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        if request.messages:
            for msg in request.messages:
                messages.append(self._build_message(msg))

        # Current user message — only add when NOT in a tool result continuation.
        # When tool_results are provided, the consumer has already placed the user
        # message in `messages` and the prompt should not be re-appended.
        if not request.tool_results:
            file_paths = normalize_file_paths(request.file_path)
            user_content = self._build_user_content(request.prompt, file_paths)
            messages.append({"role": "user", "content": user_content})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.format_json:
            payload["response_format"] = {"type": "json_object"}

        # Tool definitions
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

        # Tool results — appended as role:"tool" messages after the last user message
        if request.tool_results:
            for tr in request.tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr.call_id,
                    "content": tr.content,
                })

        max_tokens = request.max_tokens if request.max_tokens is not None else opts.get("max_tokens")
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens

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

        resp = self._post("/v1/chat/completions", payload, request.timeout)
        choice = resp["choices"][0]["message"]
        usage = resp.get("usage", {})

        # Parse tool calls if present
        raw_tool_calls = choice.get("tool_calls") or []
        tool_calls: list[ToolCall] = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"].get("arguments", "{}")),
            )
            for tc in raw_tool_calls
        ]

        # Always read reasoning_content from the response: the model may
        # produce reasoning output even when not explicitly requested.
        raw_reasoning = choice.get("reasoning_content") or ""

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
        """Model preloading is not supported by OpenAI-compatible endpoints.

        Args:
            model: Unused.
            keep_alive: Unused.
            context_size: Unused.
            extra_options: Unused.
        """
        pass  # No-op: not supported

    def get_embedding(self, model: str, text: str) -> list[float]:
        """Generate a text embedding vector.

        Args:
            model: Embedding model name.
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        payload = {"model": model, "input": text}
        resp = self._post("/v1/embeddings", payload, self.config.timeout)
        return [float(x) for x in resp["data"][0]["embedding"]]
