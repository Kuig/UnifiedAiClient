from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from google import genai
from google.genai import types

from unified_ai_client.file_utils import normalize_file_paths
from unified_ai_client.models import AiRequest, AiResponse, ProviderConfig
from unified_ai_client.providers.base import BaseProvider

_log = logging.getLogger("unified_ai_client.providers.google")


class GoogleProvider(BaseProvider):
    """Google AI (Gemini) provider adapter utilizing the google.genai SDK.

    Manages connection caching, cloud file uploads with ACTIVE polling,
    upload caching to avoid re-uploading the same local file within a
    session, rate-limit delaying, safety filters, thinking configuration
    variants, and rigorous file cleanup on termination.
    """

    def __init__(self, config: ProviderConfig, api_key: str | None = None) -> None:
        """Initialize the Google provider.

        Args:
            config: ProviderConfig instance with settings.
            api_key: Google AI API key, read from secrets.json as 'google_api_key'.
        """
        self.config = config
        self.api_key = api_key
        # dict[abs_path, FileRef] — avoids re-uploading the same file
        self._uploaded_files: dict[str, Any] = {}
        self._client: genai.Client | None = None
        self._client_lock = threading.Lock()

    def _get_client(self) -> genai.Client:
        """Lazy thread-safe initialization of the GenAI client.

        Returns:
            The genai.Client instance.

        Raises:
            ValueError: If api_key is missing.
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    if not self.api_key:
                        raise ValueError(
                            "Missing Google AI API key. Add 'google_api_key' to secrets.json."
                        )
                    self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _upload_file(self, file_path: str, upload_poll_timeout: int = 15) -> Any:
        """Upload a file to Google Cloud or return the cached reference.

        If the same local file path has already been uploaded in this
        session, the existing remote reference is returned immediately
        without a network round-trip.

        Polls at most `upload_poll_timeout` times (one second apart)
        waiting for the file to become ACTIVE.

        Args:
            file_path: Path to the local file to upload.
            upload_poll_timeout: Timeout in seconds for ACTIVE polling.

        Returns:
            The uploaded file reference from google.genai.

        Raises:
            RuntimeError: On upload failure or polling timeout.
        """
        abs_path = os.path.abspath(file_path)
        if abs_path in self._uploaded_files:
            _log.info(
                "File '%s' already uploaded, reusing cached reference",
                os.path.basename(file_path),
            )
            return self._uploaded_files[abs_path]

        client = self._get_client()
        from pathlib import Path
        try:
            _log.info("Uploading file '%s' to Google AI...", os.path.basename(file_path))
            file_ref = client.files.upload(file=Path(file_path))
            for elapsed in range(upload_poll_timeout):
                file_info = client.files.get(name=file_ref.name)
                state_str = str(file_info.state).upper()
                if "ACTIVE" in state_str:
                    _log.info(
                        "File '%s' is ACTIVE after %ds",
                        os.path.basename(file_path),
                        elapsed + 1,
                    )
                    self._uploaded_files[abs_path] = file_ref
                    return file_ref
                elif "FAILED" in state_str:
                    raise RuntimeError("Google file processing failed.")
                elif "PROCESSING" in state_str:
                    time.sleep(1)
                else:
                    self._uploaded_files[abs_path] = file_ref
                    return file_ref
            raise TimeoutError(
                f"Google file upload polling timed out after {upload_poll_timeout}s"
            )
        except Exception as exc:
            raise RuntimeError(f"Google file upload failed: {exc}") from exc

    def _build_safety_settings(
        self, disable: bool
    ) -> list[types.SafetySetting] | None:
        """Build safety settings list.

        Args:
            disable: If True, set all safety categories to BLOCK_NONE.

        Returns:
            List of SafetySetting objects, or None to use defaults.
        """
        if not disable:
            return None
        return [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
            ),
        ]

    def _build_thinking_config(
        self,
        thinking: bool,
        model_name: str,
    ) -> types.ThinkingConfig | None:
        """Build thinking configuration for the model.

        Adapts to model architecture: Gemini 3.x uses thinking_level,
        Gemini 2.5 uses thinking_budget, others use a small budget default.
        When thinking is True, adds include_thoughts=True so the
        SDK returns thought parts in the response candidates.

        Args:
            thinking: Whether thinking mode is requested.
            model_name: The target model identifier.

        Returns:
            ThinkingConfig instance, or None if thinking is disabled.
        """
        model_lower = model_name.lower()
        kwargs: dict[str, Any] = {}

        if thinking == "default":
            return types.ThinkingConfig(include_thoughts=True)

        if not thinking:
            if "gemini-3" in model_lower:
                try:
                    kwargs["thinking_level"] = getattr(
                        types.ThinkingLevel, "MINIMAL", "MINIMAL"
                    )
                except AttributeError:
                    kwargs["thinking_level"] = "MINIMAL"
            else:
                kwargs["thinking_budget"] = 0
        else:
            if "gemini-3" in model_lower:
                try:
                    kwargs["thinking_level"] = getattr(
                        types.ThinkingLevel, "HIGH", "HIGH"
                    )
                except AttributeError:
                    kwargs["thinking_level"] = "HIGH"
            elif "gemini-2.5" in model_lower:
                kwargs["thinking_budget"] = 24576
            else:
                kwargs["thinking_budget"] = 1024

        kwargs["include_thoughts"] = True
        return types.ThinkingConfig(**kwargs)

    def _build_parts_for_files(self, file_paths: list[str], upload_poll_timeout: int = 15) -> list[Any]:
        """Upload files and return a list of Part objects.

        Args:
            file_paths: List of local file paths to attach.
            upload_poll_timeout: Timeout in seconds for ACTIVE polling.

        Returns:
            List of types.Part objects referencing the uploaded files.
        """
        parts = []
        for fp in file_paths:
            ref = self._upload_file(fp, upload_poll_timeout)
            parts.append(
                types.Part.from_uri(file_uri=ref.uri, mime_type=ref.mime_type)
            )
        return parts

    def call(self, request: AiRequest) -> AiResponse:
        """Execute a text generation call to Google AI.

        Args:
            request: The AI request containing parameters.

        Returns:
            Standardized AiResponse.
        """
        client = self._get_client()

        # Merge options
        opts = {}
        if self.config.extra_options:
            opts.update(self.config.extra_options)
        if request.extra_options:
            opts.update(request.extra_options)

        upload_poll_timeout = opts.get("upload_poll_timeout", 15)

        # --- Build contents list ---
        contents: list[Any] = []

        # Previous conversation history (with optional file attachments)
        if request.messages:
            for msg in request.messages:
                role = msg.get("role", "user")
                if role == "assistant":
                    role = "model"
                parts: list[Any] = []
                # Attach files from message history
                msg_files = normalize_file_paths(msg.get("files"))
                if msg_files:
                    parts.extend(self._build_parts_for_files(msg_files, upload_poll_timeout))
                parts.append(
                    types.Part.from_text(text=msg.get("content", ""))
                )
                contents.append(types.Content(role=role, parts=parts))

        # Current turn: files + prompt
        current_parts: list[Any] = []
        file_paths = normalize_file_paths(request.file_path)
        if file_paths:
            current_parts.extend(self._build_parts_for_files(file_paths, upload_poll_timeout))
        current_parts.append(types.Part.from_text(text=request.prompt))
        contents.append(types.Content(role="user", parts=current_parts))

        # --- Build generation config ---
        config_kwargs = {
            "temperature": request.temperature,
            "system_instruction": request.system_prompt,
            "safety_settings": self._build_safety_settings(opts.get("disable_safety", False)),
            "thinking_config": self._build_thinking_config(
                request.thinking, request.model
            ),
            "response_mime_type": "application/json" if request.format_json else None,
        }

        top_k = request.top_k if request.top_k is not None else opts.get("top_k")
        if top_k is not None:
            config_kwargs["top_k"] = top_k

        top_p = request.top_p if request.top_p is not None else opts.get("top_p")
        if top_p is not None:
            config_kwargs["top_p"] = top_p

        max_output_tokens = request.max_tokens if request.max_tokens is not None else (opts.get("max_output_tokens") or opts.get("max_tokens"))
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens

        # Pass other recognized fields if present in opts
        for field_name in ("response_schema", "stop_sequences", "presence_penalty", "frequency_penalty", "seed"):
            if field_name in opts:
                config_kwargs[field_name] = opts[field_name]

        gen_config = types.GenerateContentConfig(**config_kwargs)

        def _do_call() -> Any:
            return client.models.generate_content(
                model=request.model,
                contents=contents,
                config=gen_config,
            )

        # Execute with timeout via thread pool
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_call)
            try:
                response = future.result(timeout=request.timeout)
            except FutureTimeout:
                raise TimeoutError(
                    f"Google AI call timed out after {request.timeout}s"
                )

        # --- Parse token usage ---
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            reasoning_tokens = getattr(usage, "thoughts_token_count", 0) or 0

        # --- Extract text and reasoning ---
        # Always collect thought parts regardless of request.thinking: a model
        # may produce thinking output even when thinking was not explicitly
        # requested (e.g., thinking=False only minimises thinking, not prevents
        # it entirely).
        response_text = ""
        reasoning_text = ""
        try:
            for part in response.candidates[0].content.parts:
                if getattr(part, "thought", False):
                    reasoning_text += part.text or ""
                else:
                    response_text += part.text or ""
        except (IndexError, AttributeError):
            response_text = getattr(response, "text", "") or ""

        return AiResponse(
            text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            reasoning_text=reasoning_text,
        )

    def preload_model(self, model: str, keep_alive: str = "15m") -> None:
        """Model preloading is not supported on the Google provider.

        Args:
            model: Model identifier. Unused.
            keep_alive: Duration. Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Google provider does not support model pre-loading.")

    def get_embedding(self, model: str, text: str) -> list[float]:
        """Text embeddings are not supported on the Google provider.

        Args:
            model: Model identifier. Unused.
            text: Input text. Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Google provider does not support text embeddings.")

    def cleanup(self) -> None:
        """Delete all uploaded files from Google remote cloud cache."""
        if not self._uploaded_files:
            return
        try:
            client = self._get_client()
        except Exception:
            return
        for ref in list(self._uploaded_files.values()):
            try:
                client.files.delete(name=ref.name)
            except Exception:
                pass
        self._uploaded_files.clear()
        _log.info("Google: remote file cache cleared.")
