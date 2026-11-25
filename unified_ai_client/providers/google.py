from __future__ import annotations

import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types

from unified_ai_client.exceptions import (
    NonRetryableError,
    UnsupportedFileError,
    UnsupportedOperationError,
)
from unified_ai_client.file_utils import get_mime_type, normalize_file_paths
from unified_ai_client.http import is_settled_status
from unified_ai_client.models import (
    AiRequest,
    AiResponse,
    EmbeddingRequest,
    Grounding,
    ModelInfo,
    ProviderConfig,
    ToolCall,
)
from unified_ai_client.providers.base import BaseProvider

_log = logging.getLogger("unified_ai_client.providers.google")

_UPLOADED_FILES: dict[str, Any] = {}
"""Remote file references, keyed by absolute local path.

Module-level rather than per-instance because ``configure_provider('google')``
evicts the cached provider instance. Held on the instance, this dict went with
it: the uploaded files were left on the remote quota with nothing holding a
reference to delete them, and the next call re-uploaded every one of them.

Assumes a single Google identity per process, which is what ``load_secrets()``
provides. The key is the path alone, so a second API key in the same process
would reuse references it cannot read.
"""

_UPLOADED_FILES_LOCK = threading.Lock()
"""Guards reads and writes of ``_UPLOADED_FILES``, not the upload itself.

Two threads asking for the same new file can still upload it twice, which is the
behaviour this cache has always had. Serialising the upload window would mean
holding a lock across tens of seconds of network I/O.
"""

_DEFAULT_UPLOAD_POLL_TIMEOUT = 15
"""Seconds to wait for an upload to turn ACTIVE when ``upload_poll_timeout`` is unset."""

_DEVELOPER_API_VERSION = "v1beta"
"""API version of the Gemini Developer API client, pinned rather than inherited.

google-genai 3.0 moved the SDK default from ``v1beta`` to ``v1``. There, verified
live on gemini-2.5-flash, thinking, JSON mode, function calling and search are each
a 400 asking for ``v1beta``, and since every request carries a ``ThinkingConfig``
no call got through at all; the ``-latest`` aliases and some preview models do not
exist on ``v1`` either. Not applied on Vertex, whose version is ``v1beta1``, nor
when ``GOOGLE_GENAI_API_VERSION`` is set, which is the caller's own choice.
"""

_FORWARDED_CONFIG_FIELDS: tuple[str, ...] = (
    "response_schema", "stop_sequences", "presence_penalty", "frequency_penalty", "seed",
)
"""``GenerateContentConfig`` fields taken from ``extra_options`` as they are.

The config is a typed model that rejects unknown fields, so this adapter forwards a
named list instead of everything outside the library namespace.
"""

_GENERATION_ONLY_KEYS: frozenset[str] = frozenset(
    {"disable_safety", "temperature", "max_tokens", "max_output_tokens", "top_k", "top_p",
     "upload_poll_timeout"}
    | set(_FORWARDED_CONFIG_FIELDS)
)
"""Keys ``call()`` reads that an embedding request has no use for."""


class _SettledClientError(NonRetryableError, errors.ClientError):
    """A 4xx from the SDK that will come back identically on every attempt.

    Still a ``ClientError``, so a consumer catching the SDK's type keeps
    catching it; the ``NonRetryableError`` half makes ``with_retry()`` give up
    on the first attempt, as ``NonRetryableHttpError`` does for the HTTP
    adapters. Defined here rather than in ``exceptions.py`` because this is the
    only module allowed to import ``google.genai``.
    """


class _SettledServerError(NonRetryableError, errors.ServerError):
    """A settled 5xx from the SDK (501), the server-side twin of the above."""


def _settle(exc: BaseException) -> BaseException:
    """The non-retryable form of a settled SDK error, or ``exc`` itself.

    Until 0.9.2 every SDK error was retried, so a 400 that no attempt could
    change cost four attempts and the whole backoff before reaching the caller.
    The rule on status codes is ``http.is_settled_status()``, the one the HTTP
    adapters follow.
    """
    if isinstance(exc, NonRetryableError) or not isinstance(exc, errors.APIError):
        return exc
    if not isinstance(exc.code, int) or not is_settled_status(exc.code):
        return exc
    if isinstance(exc, errors.ClientError):
        return _SettledClientError(exc.code, exc.details, exc.response)
    if isinstance(exc, errors.ServerError):
        return _SettledServerError(exc.code, exc.details, exc.response)
    return exc


class GoogleProvider(BaseProvider):
    """Google AI (Gemini) provider adapter utilizing the google.genai SDK.

    Manages connection caching, cloud file uploads with ACTIVE polling,
    upload caching to avoid re-uploading the same local file within a
    session, rate-limit delaying, safety filters, thinking configuration
    variants, and rigorous file cleanup on termination.
    """

    # No verbatim passthrough here: anything else reaches GenerateContentConfig
    # only through _FORWARDED_CONFIG_FIELDS.
    _CONSUMED_OPTION_KEYS: frozenset[str] = frozenset({
        "temperature", "max_tokens", "max_output_tokens", "top_k", "top_p",
        "disable_safety", "upload_poll_timeout",
        "task_type", "output_dimensionality",
    })

    # The Files API takes every class the library classifies, so only a file
    # classify_file() cannot place at all is refused before upload.
    SUPPORTED_FILE_TYPES: frozenset[str] = frozenset({
        "image", "audio", "video", "document",
    })

    SUPPORTS_EMBEDDINGS: bool = True

    # Grounding with Google Search, the google_search tool of Gemini 2.0 on.
    SUPPORTS_WEB_SEARCH: bool = True

    # FunctionCallingConfig(mode="NONE"). Honoured by gemini-2.5-flash, but
    # after a tool history gemini-3.5-flash still returns a function_call when
    # it wants another tool and the closing prompt does not tell it to answer
    # (seen live), with or without the declarations.
    TOOL_CHOICE_FIELD: bool = True

    # gemini-embedding-2 takes them all, inline only (embed() says why).
    # gemini-embedding-001 would ignore them silently, so embed() refuses
    # media on an embedding model older than 2.
    SUPPORTED_EMBEDDING_FILE_TYPES: frozenset[str] = frozenset({
        "image", "audio", "video", "document",
    })

    def __init__(self, config: ProviderConfig, api_key: str | None = None) -> None:
        """Initialize the Google provider.

        Args:
            config: ProviderConfig instance with settings.
            api_key: Google AI API key, read from secrets.json as 'google_api_key'.
        """
        self.config = config
        self.api_key = api_key
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
                    self._client = self._build_client()
        return self._client

    def _build_client(self) -> genai.Client:
        """Build the client on ``_DEVELOPER_API_VERSION`` unless told otherwise.

        Vertex is detected through the client's own ``vertexai`` property
        rather than by reading the environment here, since the SDK decides it
        from more than one variable (``GOOGLE_GENAI_USE_VERTEXAI``,
        ``GOOGLE_GENAI_USE_ENTERPRISE``). Building a client opens no
        connection, so the rebuild costs nothing on the network.

        Returns:
            A new genai.Client.
        """
        if os.environ.get("GOOGLE_GENAI_API_VERSION"):
            return genai.Client(api_key=self.api_key)
        client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(api_version=_DEVELOPER_API_VERSION),
        )
        if client.vertexai:
            client.close()
            client = genai.Client(api_key=self.api_key)
        return client

    def _upload_file(
        self, file_path: str, upload_poll_timeout: int = _DEFAULT_UPLOAD_POLL_TIMEOUT
    ) -> Any:
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
            NonRetryableError: If the SDK answers with a settled status (a 4xx
                other than 408, 409, 425 and 429), also the SDK's own error type.
            RuntimeError: On any other upload failure or polling timeout.
        """
        abs_path = os.path.abspath(file_path)
        with _UPLOADED_FILES_LOCK:
            cached = _UPLOADED_FILES.get(abs_path)
        if cached is not None:
            _log.info(
                "File '%s' already uploaded, reusing cached reference",
                os.path.basename(file_path),
            )
            return cached

        client = self._get_client()
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
                    with _UPLOADED_FILES_LOCK:
                        _UPLOADED_FILES[abs_path] = file_ref
                    return file_ref
                elif "FAILED" in state_str:
                    raise RuntimeError("Google file processing failed.")
                elif "PROCESSING" in state_str:
                    time.sleep(1)
                else:
                    with _UPLOADED_FILES_LOCK:
                        _UPLOADED_FILES[abs_path] = file_ref
                    return file_ref
            raise TimeoutError(
                f"Google file upload polling timed out after {upload_poll_timeout}s"
            )
        except Exception as exc:
            settled = _settle(exc)
            if isinstance(settled, NonRetryableError):
                raise settled from exc
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
        thinking: bool | str,
        model_name: str,
    ) -> types.ThinkingConfig | None:
        """Build thinking configuration for the model.

        Adapts to model architecture: Gemini 3 and newer use thinking_level,
        Gemini 2.5 uses thinking_budget, others (a name without a version
        included) use a small budget default. When thinking is True, adds
        include_thoughts=True so the SDK returns thought parts in the response
        candidates.

        Args:
            thinking: True, False, or "default" as passed to call_ai().
            model_name: The target model identifier.

        Returns:
            ThinkingConfig instance, or None if thinking is disabled.
        """
        version = self._gemini_version(model_name)
        uses_level = version is not None and version >= (3, 0)
        kwargs: dict[str, Any] = {}

        if thinking == "default":
            return types.ThinkingConfig(include_thoughts=True)

        if not thinking:
            if uses_level:
                try:
                    kwargs["thinking_level"] = getattr(
                        types.ThinkingLevel, "MINIMAL", "MINIMAL"
                    )
                except AttributeError:
                    kwargs["thinking_level"] = "MINIMAL"
            else:
                kwargs["thinking_budget"] = 0
        else:
            if uses_level:
                try:
                    kwargs["thinking_level"] = getattr(
                        types.ThinkingLevel, "HIGH", "HIGH"
                    )
                except AttributeError:
                    kwargs["thinking_level"] = "HIGH"
            elif version == (2, 5):
                kwargs["thinking_budget"] = 24576
            else:
                kwargs["thinking_budget"] = 1024

        kwargs["include_thoughts"] = True
        return types.ThinkingConfig(**kwargs)

    @staticmethod
    def _gemini_version(model: str) -> tuple[int, int] | None:
        """Extract the (major, minor) generation from a Gemini model name.

        The one rule in this adapter for what a model generation can do, so the
        thinking form and the search-with-tools check cannot drift apart.

        Args:
            model: Model identifier as passed to call_ai(), with or without
                the ``models/`` prefix.

        Returns:
            ``(3, 5)`` for 'gemini-3.5-flash', ``(4, 0)`` for 'gemini-4-x', or
            None for a name carrying no version ('gemini-flash-latest',
            'gemini-embedding-2'), which each caller resolves on its own.
        """
        match = re.search(r"gemini-(\d+)(?:\.(\d+))?", model.lower())
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2) or 0)

    @staticmethod
    def _embedding_version(model: str) -> int | None:
        """Extract the generation from a Gemini embedding model name.

        Args:
            model: Model identifier, with or without the ``models/`` prefix.

        Returns:
            ``1`` for 'gemini-embedding-001', ``2`` for 'gemini-embedding-2' and
            its '-preview', or None for a name without one.
        """
        match = re.search(r"gemini-embedding-(\d+)", model.lower())
        return int(match.group(1)) if match else None

    @staticmethod
    def _with_deadline(fn: Any, timeout: int | None) -> Any:
        """Run an SDK call under a real deadline.

        The SDK takes no per-call timeout, so the call runs in a worker thread
        that is abandoned when the deadline passes.

        Deliberately not a `with` block. ThreadPoolExecutor.__exit__ calls
        shutdown(wait=True), so raising inside the block does not end it: the
        context manager keeps waiting for the call it just gave up on, and the
        timeout ends up measuring nothing. shutdown(wait=False) hands the
        thread back to the interpreter instead, so the deadline is real.

        Args:
            fn: Zero-argument callable making the SDK call.
            timeout: Seconds to wait.

        Returns:
            Whatever ``fn`` returns.

        Raises:
            TimeoutError: If the deadline passes first.
            NonRetryableError: If the SDK answers with a settled status, as the
                SDK's own error type too (see ``_settle()``).
        """
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(fn)
            try:
                return future.result(timeout=timeout)
            except FutureTimeout:
                raise TimeoutError(f"Google AI call timed out after {timeout}s")
            except errors.APIError as exc:
                settled = _settle(exc)
                if settled is exc:
                    raise
                raise settled from exc
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _build_parts_for_files(
        self,
        file_paths: list[str],
        upload_poll_timeout: int = _DEFAULT_UPLOAD_POLL_TIMEOUT,
    ) -> list[Any]:
        """Upload files and return a list of Part objects.

        Generation only: ``embed_content`` refuses a Files API reference, so
        ``embed()`` sends its media inline instead.

        Args:
            file_paths: List of local file paths to attach.
            upload_poll_timeout: Timeout in seconds for ACTIVE polling.

        Returns:
            List of types.Part objects referencing the uploaded files.

        Raises:
            FileNotFoundError: If an attachment does not exist. Checked up front
                so a bad path fails before earlier files are uploaded and have
                to be cleaned up again.
            UnsupportedFileError: If a path classifies as 'unknown'. Every class
                the library recognises is uploadable; one it cannot place at all
                is not.
        """
        parts = []
        for fp, _ in self._validate_files(file_paths):
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
        opts = self._merge_options(request)
        upload_poll_timeout = opts.get("upload_poll_timeout", _DEFAULT_UPLOAD_POLL_TIMEOUT)

        # The config first: it can refuse the request, and the contents upload.
        gen_config = self._build_config(request, opts)
        contents = self._build_contents(request, upload_poll_timeout)
        response = self._with_deadline(
            lambda: client.models.generate_content(
                model=request.model,
                contents=contents,
                config=gen_config,
            ),
            request.timeout,
        )
        return self._parse_response(response, request)

    def _build_message(self, msg: dict[str, Any], upload_poll_timeout: int) -> types.Content:
        """Convert one history entry, keeping its files, tool calls and text.

        The three are independent: a message carrying several of them keeps
        all. An entry with none of them still gets an empty text Part, since
        ``Content(parts=[])`` is not a valid turn.

        Args:
            msg: Source message dict. May contain ``files`` and, on an assistant
                message, ``tool_calls``. An assistant entry carrying this
                provider's ``provider_state`` is replayed from it, thought
                signatures included.
            upload_poll_timeout: Seconds to wait for each upload to turn ACTIVE.

        Returns:
            The ``types.Content`` for this turn.
        """
        native = self._native_message(msg)
        if isinstance(native, dict):
            return types.Content.model_validate(native)

        role = msg.get("role", "user")
        if role == "assistant":
            role = "model"
        parts: list[Any] = []
        msg_files = normalize_file_paths(msg.get("files"))
        if msg_files:
            parts.extend(self._build_parts_for_files(msg_files, upload_poll_timeout))

        # Google links a function_response to the function_call Part before it.
        if role == "model" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                _, name, arguments = self._tool_call_fields(tc)
                parts.append(types.Part.from_function_call(name=name, args=arguments))
        content_text = msg.get("content", "")
        if content_text or not parts:
            parts.append(types.Part.from_text(text=content_text))
        return types.Content(role=role, parts=parts)

    def _build_contents(self, request: AiRequest, upload_poll_timeout: int) -> list[Any]:
        """History, the current turn and any tool results, in order.

        The current turn follows ``_wants_current_turn()``; an empty text Part
        is skipped when a file already fills it, since ``prompt=""`` with only
        an image is a legitimate "describe this" call. A ``role: "tool"`` entry
        becomes a ``function_response`` Part, named after the entry or, failing
        that, after the call its ``tool_call_id`` points to. Consecutive
        responses share one user turn, as parallel calls require.

        Raises:
            NonRetryableError: If a tool entry's function name cannot be found,
                which Gemini needs to route the response.
        """
        contents: list[Any] = []
        call_names: dict[str, str] = {}
        for msg in request.messages or []:
            for tc in msg.get("tool_calls") or []:
                call_id, name, _ = self._tool_call_fields(tc)
                if call_id:
                    call_names[call_id] = name
            if msg.get("role") == "tool":
                name = msg.get("name") or call_names.get(msg.get("tool_call_id") or "")
                if not name:
                    raise NonRetryableError(
                        "A history entry with role 'tool' needs the 'name' of the "
                        "function it answers, or a 'tool_call_id' matching an "
                        "earlier tool call; ToolResult.to_message() includes both."
                    )
                self._append_function_responses(
                    contents, [self._function_response_part(name, msg.get("content", ""))]
                )
                continue
            contents.append(self._build_message(msg, upload_poll_timeout))

        if self._wants_current_turn(request):
            current_parts: list[Any] = []
            file_paths = normalize_file_paths(request.file_path)
            if file_paths:
                current_parts.extend(self._build_parts_for_files(file_paths, upload_poll_timeout))
            if request.prompt or not current_parts:
                current_parts.append(types.Part.from_text(text=request.prompt))
            contents.append(types.Content(role="user", parts=current_parts))
        if request.tool_results:
            self._append_function_responses(contents, [
                self._function_response_part(tr.name, tr.content)
                for tr in request.tool_results
            ])
        return contents

    @staticmethod
    def _function_response_part(name: str, content: str) -> Any:
        """One tool result as a ``function_response`` Part.

        Matched to its call by name, not by id: when Gemini returns a call
        without an id the ``ToolCall`` carries a generated one, and sending that
        back would not match anything.
        """
        return types.Part.from_function_response(name=name, response={"result": content})

    @staticmethod
    def _append_function_responses(contents: list[Any], parts: list[Any]) -> None:
        """Add response Parts to the previous turn if it holds only responses."""
        last = contents[-1] if contents else None
        if (
            last is not None
            and last.role == "user"
            and last.parts
            and all(getattr(p, "function_response", None) is not None for p in last.parts)
        ):
            last.parts.extend(parts)
        else:
            contents.append(types.Content(role="user", parts=parts))

    def _build_config(
        self, request: AiRequest, opts: dict[str, Any]
    ) -> types.GenerateContentConfig:
        """Build the generation config, logging the options it cannot carry.

        A None value is dropped by the SDK's own serialisation, which leaves the
        model's default in charge.

        Raises:
            UnsupportedOperationError: If ``web_search`` and ``tools`` are both
                set on a model older than Gemini 3.
        """
        version = self._gemini_version(request.model)
        if request.tools and request.web_search and version is not None and version < (3, 0):
            # Verified live on gemini-2.5-flash, -flash-lite and -pro: "Built-in
            # tools and Function Calling cannot be combined" without the
            # server-side invocations, "Tool call context circulation is not
            # enabled" with them. A name without a version is assumed recent.
            raise UnsupportedOperationError(
                f"Model '{request.model}' cannot combine web_search with function "
                "tools; Gemini 3 and newer can. Drop one of the two, or use a "
                "Gemini 3 model."
            )
        config_kwargs: dict[str, Any] = {
            "temperature": self._prefer_request(request.temperature, opts, "temperature"),
            "system_instruction": request.system_prompt,
            "safety_settings": self._build_safety_settings(opts.get("disable_safety", False)),
            "thinking_config": self._build_thinking_config(
                request.thinking, request.model
            ),
            "response_mime_type": "application/json" if request.format_json else None,
        }

        top_k = self._prefer_request(request.top_k, opts, "top_k")
        if top_k is not None:
            config_kwargs["top_k"] = top_k

        top_p = self._prefer_request(request.top_p, opts, "top_p")
        if top_p is not None:
            config_kwargs["top_p"] = top_p

        max_output_tokens = self._prefer_request(
            request.max_tokens,
            opts,
            "max_output_tokens",
            opts.get("max_tokens"),
        )
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens

        for field_name in _FORWARDED_CONFIG_FIELDS:
            if field_name in opts:
                config_kwargs[field_name] = opts[field_name]
        self._report_dropped_options(
            self._foreign_option_keys(opts), "owned by another provider"
        )
        self._report_dropped_options(
            {
                k for k in opts
                if k not in self._LIBRARY_OPTION_KEYS and k not in _FORWARDED_CONFIG_FIELDS
            },
            "GenerateContentConfig has no such field",
            logging.WARNING,
        )

        tools: list[Any] = []
        if request.tools:
            declarations = [
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
                for t in request.tools
            ]
            tools.append(types.Tool(function_declarations=declarations))
        tool_config: dict[str, Any] = {}
        if request.web_search:
            # A Tool of its own: one Tool carries one kind.
            tools.append(types.Tool(google_search=types.GoogleSearch()))
            if request.tools:
                # Gemini 3 answers 400 to a built-in tool beside function
                # declarations unless the server-side invocations are returned
                # too (verified live on gemini-3.5-flash). They come back as
                # tool_call/tool_response Parts, kept in provider_state for replay.
                tool_config["include_server_side_tool_invocations"] = True
        if request.tools and request.tool_choice == "none":
            tool_config["function_calling_config"] = types.FunctionCallingConfig(
                mode="NONE"
            )
        if tool_config:
            config_kwargs["tool_config"] = types.ToolConfig(**tool_config)
        if tools:
            config_kwargs["tools"] = tools

        return types.GenerateContentConfig(**config_kwargs)

    def _parse_response(self, response: Any, request: AiRequest) -> AiResponse:
        """Read text, thought parts, tool calls, grounding and token counts.

        Thought parts are read whatever ``thinking`` was, since a model may
        think anyway (on Gemini 3, ``thinking=False`` only minimises it). They
        are always a summary the model wrote of its own reasoning, never the
        chain of thought, hence ``reasoning_is_summary``.
        """
        model = request.model
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            reasoning_tokens = getattr(usage, "thoughts_token_count", 0) or 0

        response_text = ""
        reasoning_text = ""
        tool_calls: list[ToolCall] = []
        native: dict[str, Any] | None = None
        try:
            candidate_content = response.candidates[0].content
            dumped = candidate_content.model_dump(mode="json", exclude_none=True)
            native = dumped if isinstance(dumped, dict) else None
            for part in candidate_content.parts:
                if getattr(part, "thought", False):
                    reasoning_text += part.text or ""
                elif getattr(part, "function_call", None) is not None:
                    fc = part.function_call
                    call_id = getattr(fc, "id", None) or f"{fc.name}_{len(tool_calls)}"
                    tool_calls.append(ToolCall(
                        id=call_id,
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    ))
                else:
                    response_text += part.text or ""
        except (IndexError, AttributeError):
            # No candidate at all means the prompt itself was blocked. The empty
            # text that follows reads as "the model said nothing", so the block
            # reason is logged rather than lost.
            block_reason = getattr(
                getattr(response, "prompt_feedback", None), "block_reason", None
            )
            if block_reason:
                _log.warning(
                    "Google blocked the prompt for model '%s': %s",
                    model, block_reason,
                )
            response_text = getattr(response, "text", "") or ""

        return AiResponse(
            text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            reasoning_text=reasoning_text,
            reasoning_is_summary=bool(reasoning_text),
            tool_calls=tool_calls,
            message=self._assistant_message(response_text, tool_calls, native),
            grounding=self._grounding(response, request),
        )

    def _grounding(self, response: Any, request: AiRequest) -> Grounding | None:
        """Read the first candidate's search, wherever this mode puts it.

        Search alone reports it in ``grounding_metadata``. Beside function
        declarations, with server-side invocations on, the same search arrives
        as a ``tool_call`` Part (its ``queries``) and a ``tool_response`` Part
        (the suggestions), and the metadata may be absent (verified live on
        gemini-3.5-flash). Both are read.

        The chunk URIs are Google's redirect links and the titles the sites'
        domains; they are reported as they come, since resolving each link
        would cost a request of its own.
        """
        if not request.web_search:
            return None
        try:
            candidate = response.candidates[0]
        except (IndexError, AttributeError, TypeError):
            candidate = None
        queries: list[str] = []
        sources: list[tuple[str | None, str | None]] = []
        suggestions: str | None = None

        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        for part in parts:
            tool_call = getattr(part, "tool_call", None)
            if tool_call is not None and "SEARCH" in str(getattr(tool_call, "tool_type", "")):
                queries.extend((getattr(tool_call, "args", None) or {}).get("queries") or [])
            tool_response = getattr(part, "tool_response", None)
            if tool_response is not None:
                body = getattr(tool_response, "response", None) or {}
                suggestions = body.get("search_suggestions") or suggestions

        meta = getattr(candidate, "grounding_metadata", None)
        if meta is not None:
            queries.extend(
                q for q in getattr(meta, "web_search_queries", None) or [] if q not in queries
            )
            for chunk in getattr(meta, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                if web is not None:
                    sources.append((getattr(web, "uri", None), getattr(web, "title", None)))
            entry_point = getattr(meta, "search_entry_point", None)
            suggestions = getattr(entry_point, "rendered_content", None) or suggestions
        return self._build_grounding(request, queries, sources, suggestions)

    def warm_up(
        self,
        model: str,
        file_paths: str | list[str] | None = None,
        *,
        keep_alive: str | int | None = None,
        timeout: int | None = None,
    ) -> bool:
        """Build the client, open the connection, and pre-upload any files.

        This is where the Google provider's one-off costs actually live: the
        lazy ``google.genai`` import, the client construction, the DNS + TCP +
        TLS handshake, and above all the file upload with its ACTIVE polling,
        which can take tens of seconds per file.

        ``models.get()`` is a plain metadata GET: it consumes no tokens, and it
        validates the API key and the model name as a side effect. Uploaded
        files land in the same cache ``call()`` reads and ``cleanup()`` clears,
        so warming up does not change the resource lifecycle.

        Args:
            model: Model identifier to validate and warm the connection with.
            file_paths: Optional path or list of paths to upload ahead of time.
            keep_alive: Ignored. Google holds no model on the caller's
                behalf, so it has no residency to control.
            timeout: Ignored. This warm-up goes through the genai SDK,
                which manages its own deadlines; the upload polling has
                its own ``upload_poll_timeout`` setting instead.

        Returns:
            Always True: this provider always has something to warm up.
        """
        opts = self.config.extra_options or {}
        upload_poll_timeout = opts.get("upload_poll_timeout", _DEFAULT_UPLOAD_POLL_TIMEOUT)

        client = self._get_client()
        client.models.get(model=model)

        # Validated first: an unusable file would otherwise be uploaded and held
        # against quota until cleanup(), for a request that refuses it anyway.
        for path, _ in self._validate_files(normalize_file_paths(file_paths)):
            self._upload_file(path, upload_poll_timeout)

        return True

    def list_models(self, *, timeout: int | None = None) -> list[ModelInfo]:
        """List models through the SDK's ``models.list()``.

        Each entry reports the actions it supports and, on recent SDK versions,
        whether it thinks and how large its input window is. Fields the SDK
        object lacks stay ``None``. The ``models/`` prefix is stripped from the
        id, since that is the form ``call_ai()`` is given.

        Args:
            timeout: Ignored. The call goes through the genai SDK, which manages
                its own deadlines.

        Returns:
            One ``ModelInfo`` per model the key can see.

        Raises:
            ValueError: If no API key is set.
        """
        models: list[ModelInfo] = []
        for item in self._get_client().models.list():
            name = (getattr(item, "name", None) or "").removeprefix("models/")
            if not name:
                continue
            actions = getattr(item, "supported_actions", None)
            generates = None if actions is None else "generateContent" in actions
            models.append(
                ModelInfo(
                    id=name,
                    provider=self.provider_name,
                    display_name=getattr(item, "display_name", None),
                    reasoning=getattr(item, "thinking", None),
                    file_types=self.SUPPORTED_FILE_TYPES if generates else None,
                    embeddings=None if actions is None else "embedContent" in actions,
                    context_length=getattr(item, "input_token_limit", None),
                )
            )
        return models

    def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        """Embed text, a batch of texts, or one input with media, via embed_content.

        ``task_type`` and ``output_dimensionality`` come from
        ``configure_provider()`` or call-time ``extra_options``; a
        ``dimensions`` argument wins over ``output_dimensionality``. Absent,
        Google applies its own defaults. Note that ``gemini-embedding-2``
        rejects ``task_type``: the task goes into the text as a prefix instead.

        Media are sent inline, as parts of one ``Content`` together with the
        text, which Google embeds into a single aggregated vector; a text file
        is inlined into the text, as on every other adapter. Not uploaded:
        ``embed_content`` answers 400 "Request contains an invalid argument"
        to a Files API reference, for every medium (verified live on
        gemini-embedding-2 with PNG, WAV, PDF and MP4, all accepted inline).
        Until 0.9.2 they were uploaded, so a media embedding never worked here.

        Args:
            request: The embedding request.

        Returns:
            One vector per input.

        Raises:
            UnsupportedFileError: If a file classifies as 'unknown'.
            EmptyResponseError: If the API returns no vector for an input.
            TimeoutError: If the deadline passes first.
        """
        opts = self._merge_options(request)
        config_kwargs: dict[str, Any] = {}
        if "task_type" in opts:
            config_kwargs["task_type"] = opts["task_type"]
        dimensions = self._prefer_request(request.dimensions, opts, "output_dimensionality")
        if dimensions is not None:
            config_kwargs["output_dimensionality"] = dimensions
        self._report_dropped_options(
            {k for k in opts if k not in self._CONSUMED_OPTION_KEYS or k in _GENERATION_ONLY_KEYS},
            "not read by embed_content",
        )

        contents: Any
        if request.file_path:
            text, native = self._partition_attachments(
                request.file_path, request.inputs[0], self.SUPPORTED_EMBEDDING_FILE_TYPES
            )
            version = self._embedding_version(request.model)
            if native and version is not None and version < 2:
                # Inline, gemini-embedding-001 takes an image without an error and
                # returns the vector of the text alone (verified live): a wrong
                # answer with nothing to say so. A name without a version is
                # assumed recent, as _gemini_version() callers do.
                raise UnsupportedFileError(
                    f"Model '{request.model}' embeds text only and would ignore "
                    f"'{native[0][0]}' without an error. Use gemini-embedding-2 or "
                    "newer for images, audio, video and PDFs."
                )
            parts: list[Any] = []
            if text:
                parts.append(types.Part.from_text(text=text))
            for path, _ in native:
                parts.append(types.Part.from_bytes(
                    data=Path(path).read_bytes(), mime_type=get_mime_type(path)
                ))
            contents = [types.Content(role="user", parts=parts)]
        elif len(request.inputs) == 1:
            contents = request.inputs[0]
        else:
            # One Content per text. A plain list of strings is not a batch on
            # every model: gemini-embedding-2 treats it as the parts of one
            # input and returns a single aggregated vector.
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text=t)])
                for t in request.inputs
            ]

        client = self._get_client()
        response = self._with_deadline(
            lambda: client.models.embed_content(
                model=request.model,
                contents=contents,
                config=types.EmbedContentConfig(**config_kwargs) if config_kwargs else None,
            ),
            request.timeout,
        )

        embeddings = getattr(response, "embeddings", None) or []
        vectors = [[float(x) for x in e.values] for e in embeddings]
        return self._check_vector_count(vectors, request)

    def cleanup(self) -> None:
        """Delete every uploaded file from Google's remote store.

        Clears the module-level cache, so files uploaded by an earlier instance
        of this provider are released too. The client is built from the current
        credentials; if those have changed since the upload the delete fails and
        is swallowed, which is the same outcome as any other transient failure
        here.
        """
        with _UPLOADED_FILES_LOCK:
            refs = list(_UPLOADED_FILES.values())
        if not refs:
            return
        try:
            client = self._get_client()
        except Exception:
            return
        for ref in refs:
            try:
                client.files.delete(name=ref.name)
            except Exception as exc:
                # Swallowed so the other deletes still run, but a file left on
                # the quota is the leak cleanup() exists to prevent.
                _log.debug("Failed to delete uploaded file '%s': %s", ref.name, exc)
        with _UPLOADED_FILES_LOCK:
            _UPLOADED_FILES.clear()
        _log.info("Google: remote file cache cleared.")
