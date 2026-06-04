from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from unified_ai_client.file_utils import (
    classify_file,
    encode_file_base64,
    inline_text_attachments,
    normalize_file_paths,
)
from unified_ai_client.models import AiRequest, AiResponse, ProviderConfig
from unified_ai_client.providers.base import BaseProvider

_log = logging.getLogger("unified_ai_client.providers.ollama")


class OllamaProvider(BaseProvider):
    """Ollama provider adapter utilizing native urllib.request.

    Communicates with a local Ollama server without requiring third-party
    dependencies like the official 'ollama' Python library.

    File handling:
    - Images and audio: encoded as base64 and sent in the 'images' field.
      Ollama uses this single field for all multimodal data (images and audio
      for models like Gemma4).
    - Text files: inlined into the prompt using inline_text_attachments().
    - PDFs: treated as text (inline attempt). A warning is logged.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the Ollama provider.

        Args:
            config: ProviderConfig instance containing connection details.
        """
        self.config = config

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
        url = f"{self.config.url.rstrip('/')}{endpoint}"
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

        Separates files by type:
        - Images and audio: base64-encoded for the 'images' field.
        - Text/document/unknown: inlined into the prompt.

        Args:
            file_paths: List of file paths to process.
            prompt: The message text prompt to augment.

        Returns:
            A tuple of (modified_prompt, list_of_base64_strings).
        """
        multimodal_data: list[str] = []
        text_files: list[str] = []

        for fp in file_paths:
            ft = classify_file(fp)
            if ft in ("image", "audio"):
                _log.info(
                    "Ollama: encoding '%s' (%s) as base64 multimodal data",
                    fp,
                    ft,
                )
                multimodal_data.append(encode_file_base64(fp))
            elif ft == "document":
                _log.warning(
                    "Ollama: PDF '%s' not natively supported — attempting text inline",
                    fp,
                )
                text_files.append(fp)
            else:
                text_files.append(fp)

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
                    messages.append({"role": role, "content": content})

        # 3. Current User Message
        file_paths = normalize_file_paths(request.file_path)
        effective_prompt, multimodal_data = self._process_files_for_message(
            file_paths, request.prompt
        )

        user_message: dict[str, Any] = {
            "role": "user",
            "content": effective_prompt,
        }
        if multimodal_data:
            user_message["images"] = multimodal_data
        messages.append(user_message)

        # 4. Build payload
        options: dict[str, Any] = {"temperature": request.temperature}
        if self.config.context_size > 0:
            options["num_ctx"] = self.config.context_size

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": self.config.keep_alive,
        }

        if request.format_json:
            payload["format"] = "json"

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

        # Both reasoning_tokens and raw_thinking are always processed.
        # Ollama does not expose a dedicated thinking-token counter — eval_count
        # aggregates both thinking and response tokens.  When thinking text is
        # present, estimate reasoning_tokens using the exact chars-per-token
        # ratio of this response: eval_count / (len(content) + len(thinking)).
        # This avoids a fixed global ratio and is self-consistent per response.
        raw_thinking = message.get("thinking", "") or ""
        reasoning_text = raw_thinking

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
        )

    def preload_model(self, model: str, keep_alive: str = "15m") -> None:
        """Pre-load an Ollama model into memory.

        Args:
            model: Model identifier.
            keep_alive: Duration to keep model loaded.
        """
        payload = {
            "model": model,
            "messages": [],
            "keep_alive": keep_alive,
        }
        self._post("/api/chat", payload, self.config.timeout)

    def get_embedding(self, model: str, text: str) -> list[float]:
        """Generate a text embedding vector using Ollama.

        Args:
            model: Embedding model name.
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            RuntimeError: If the server returns no embeddings.
        """
        payload = {"model": model, "input": text}
        resp = self._post("/api/embed", payload, self.config.timeout)
        embeddings = resp.get("embeddings", [])
        if embeddings and len(embeddings) > 0:
            return [float(x) for x in embeddings[0]]
        raise RuntimeError("No embeddings returned by Ollama server.")
