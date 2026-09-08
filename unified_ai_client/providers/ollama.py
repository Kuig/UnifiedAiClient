from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from unified_ai_client.exceptions import UnsupportedFileError
from unified_ai_client.file_utils import (
    encode_file_base64,
    inline_text_attachments,
    normalize_file_paths,
)
from unified_ai_client.models import AiRequest, AiResponse, ProviderConfig, ToolCall
from unified_ai_client.providers.base import BaseProvider

_log = logging.getLogger("unified_ai_client.providers.ollama")

_DEFAULT_KEEP_ALIVE: str | int = "15m"

# An unload queues behind any generation already running, so it gets its own
# short deadline rather than the provider timeout, which is usually minutes.
_UNLOAD_TIMEOUT: int = 30

_NON_MODEL_OPTIONS: frozenset[str] = frozenset(
    {"keep_alive", "timeout", "sleep_time", "use_generate", "url"}
)
"""Keys this library reads itself, which must never reach Ollama's ``options``.

``extra_options`` carries two unrelated kinds of key: Ollama model parameters,
which are forwarded verbatim so a caller can reach any of them without this
library knowing their names, and UAC-level controls that steer the adapter.
Only the second kind is listed here. Anything absent from this set is assumed
to be a model parameter and passed through untouched.
"""


def _fold_options(opts: dict[str, Any], options: dict[str, Any]) -> None:
    """Fold UAC-level options into an Ollama model ``options`` dict, in place.

    Two names are translated to Ollama's own spelling, the control keys in
    ``_NON_MODEL_OPTIONS`` are dropped, and everything else passes through.

    Args:
        opts: Source options, from the provider config or from the request.
        options: The Ollama ``options`` dict to update in place.
    """
    for k, v in opts.items():
        if k in _NON_MODEL_OPTIONS:
            continue
        if k == "context_size":
            options["num_ctx"] = v
        elif k == "max_tokens":
            options["num_predict"] = v
        else:
            options[k] = v


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
            urllib.error.HTTPError: On HTTP errors.
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
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

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
        multimodal_data: list[str] = []
        text_files: list[str] = []

        for fp, ft in self._validate_files(file_paths):
            if ft == "image":
                _log.debug("Ollama: encoding '%s' as base64 image data", fp)
                multimodal_data.append(encode_file_base64(fp))
            elif ft == "text":
                text_files.append(fp)
            else:
                raise UnsupportedFileError(
                    f"Provider '{self.provider_name}' declares support for {ft} "
                    f"files but has no way to send them: '{fp}'."
                )

        effective_prompt = inline_text_attachments(prompt, text_files)
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
                if msg_files:
                    content, multimodal = self._process_files_for_message(
                        msg_files, content
                    )
                    entry: dict[str, Any] = {"role": role, "content": content}
                    if multimodal:
                        entry["images"] = multimodal
                    messages.append(entry)
                else:
                    entry = {"role": role, "content": content}
                    # Preserve tool_calls on assistant messages so Ollama can
                    # correctly link subsequent tool results back to the call.
                    if role == "assistant" and msg.get("tool_calls"):
                        entry["tool_calls"] = msg["tool_calls"]
                    messages.append(entry)

        # 3. Current User Message — only add when NOT in a tool result continuation.
        # When tool_results are provided, the consumer has already placed the user
        # message in `messages` (history) and the prompt should not be re-appended.
        file_paths = normalize_file_paths(request.file_path)
        effective_prompt, multimodal_data = self._process_files_for_message(
            file_paths, request.prompt
        )

        if not request.tool_results:
            user_message: dict[str, Any] = {
                "role": "user",
                "content": effective_prompt,
            }
            if multimodal_data:
                user_message["images"] = multimodal_data
            messages.append(user_message)

        # 4. Build payload
        opts = {}
        if self.config.extra_options:
            opts.update(self.config.extra_options)
        if request.extra_options:
            opts.update(request.extra_options)

        options: dict[str, Any] = {"temperature": request.temperature}

        if request.top_k is not None:
            options["top_k"] = request.top_k
        if request.top_p is not None:
            options["top_p"] = request.top_p

        _fold_options(opts, options)

        # Call-time max_tokens takes precedence
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        keep_alive = opts.get("keep_alive", _DEFAULT_KEEP_ALIVE)
        use_generate = opts.get("use_generate", False)

        if use_generate:
            # Map back to /api/generate format
            # We assume effective_prompt is the main prompt and multimodal_data are the images
            payload: dict[str, Any] = {
                "model": request.model,
                "prompt": effective_prompt,
                "stream": False,
                "options": options,
                "keep_alive": keep_alive,
            }
            if multimodal_data:
                payload["images"] = multimodal_data
            
            if request.format_json:
                payload["format"] = "json"
                
            resp = self._post("/api/generate", payload, request.timeout)
            
            content_text = resp.get("response", "") or ""
            eval_count = resp.get("eval_count", 0)
            raw_thinking = ""
            reasoning_text = ""  # /api/generate doesn't cleanly separate thinking tokens yet
            tool_calls: list[ToolCall] = []  # tool calling not supported in /api/generate
        else:
            payload: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
                "stream": False,
                "options": options,
                "keep_alive": keep_alive,
            }

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

        reasoning_tokens = 0
        if raw_thinking:
            total_chars = len(content_text) + len(raw_thinking)
            if total_chars > 0 and eval_count > 0:
                chars_per_token = total_chars / eval_count
                reasoning_tokens = max(1, round(len(raw_thinking) / chars_per_token))

        return AiResponse(
            text=content_text,
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=max(0, eval_count - reasoning_tokens),
            reasoning_tokens=reasoning_tokens,
            reasoning_text=reasoning_text,
            tool_calls=tool_calls,
        )

    def _resolve_keep_alive(self, keep_alive: str | int | None) -> str | int:
        """Resolve the residency to request, explicit argument first.

        Precedence is the explicit argument, then the ``keep_alive`` registered
        via ``configure_provider()``, then the library default. The middle step
        is the one that matters: ``call()`` reads that same config key, so a
        warm-up that skipped it would load the model with a different residency
        than every request that follows it.

        Args:
            keep_alive: The caller's explicit value, or None to fall back.

        Returns:
            The value to put on the wire.
        """
        if keep_alive is not None:
            return keep_alive
        opts = self.config.extra_options or {}
        return opts.get("keep_alive", _DEFAULT_KEEP_ALIVE)

    def _resolve_timeout(self, timeout: int | None) -> int:
        """Resolve the socket timeout, explicit argument over configured one.

        Args:
            timeout: The caller's explicit value, or None to fall back.

        Returns:
            Seconds to wait.
        """
        return timeout if timeout is not None else self.config.timeout

    def preload_model(
        self,
        model: str,
        keep_alive: str | int = _DEFAULT_KEEP_ALIVE,
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

        payload: dict[str, Any] = {
            "model": model,
            "messages": [],
            "keep_alive": keep_alive,
        }
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

    def get_embedding(self, model: str, text: str) -> list[float]:
        """Generate a text embedding vector using Ollama.

        The embedding model is subject to the same residency policy as a chat
        model, so that a ``keep_alive`` registered via ``configure_provider()``
        governs both and ``cleanup()`` can release either.

        Args:
            model: Embedding model name.
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            RuntimeError: If the server returns no embeddings.
        """
        payload = {
            "model": model,
            "input": text,
            "keep_alive": self._resolve_keep_alive(None),
        }
        resp = self._post("/api/embed", payload, self.config.timeout)
        embeddings = resp.get("embeddings", [])
        if embeddings and len(embeddings) > 0:
            return [float(x) for x in embeddings[0]]
        raise RuntimeError("No embeddings returned by Ollama server.")
