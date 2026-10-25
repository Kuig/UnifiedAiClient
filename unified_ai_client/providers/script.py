from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from unified_ai_client.exceptions import UnsupportedOperationError
from unified_ai_client.file_utils import normalize_file_paths
from unified_ai_client.models import (
    AiRequest,
    AiResponse,
    EmbeddingRequest,
    ProviderConfig,
    ToolCall,
)
from unified_ai_client.providers.base import BaseProvider

_log = logging.getLogger("unified_ai_client.providers.script")

_UNSUPPORTED_MODE = "unsupported_mode"
"""The ``error.type`` a script writes to stdout when it declines a mode."""


def _resolve_interpreter(script_path: str) -> list[str]:
    """Resolve the command used to invoke a script.

    For Python scripts, checks whether the script's directory contains a
    .venv and uses its interpreter if found. Falls back to sys.executable
    if no local .venv is present. Non-Python files are invoked directly.

    Args:
        script_path: Absolute or relative path to the script file.

    Returns:
        A list representing the command prefix, ready for subprocess.run().
    """
    if not script_path.endswith(".py"):
        return [script_path]

    script_dir = Path(script_path).parent
    candidates = [
        script_dir / ".venv" / "Scripts" / "python.exe",  # Windows
        script_dir / ".venv" / "bin" / "python",           # Unix / macOS
    ]
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate), script_path]
    return [sys.executable, script_path]


def _run_script(cmd: list[str], payload: dict, timeout: int) -> dict:
    """Spawn the script subprocess, send the payload, and return parsed output.

    Args:
        cmd: The full command list (interpreter + script path, or just path).
        payload: The request dict to serialize and send on stdin.
        timeout: Maximum seconds to wait before killing the process.

    Returns:
        The parsed JSON dict from the script's stdout.

    Raises:
        subprocess.TimeoutExpired: If the script exceeds timeout seconds.
        UnsupportedOperationError: If the script exits non-zero and declares
            the mode unsupported on stdout. Never retried.
        RuntimeError: If the script exits with a non-zero return code for any
            other reason, including a script that predates that declaration.
        json.JSONDecodeError: If stdout is not valid JSON.
    """
    result = subprocess.run(
        cmd,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        if _declares_unsupported_mode(result.stdout):
            raise UnsupportedOperationError(
                f"Script '{cmd[-1]}' does not implement mode '{payload.get('mode')}'."
            )
        stderr_detail = result.stderr.strip() or "(no stderr output)"
        raise RuntimeError(
            f"Script '{cmd[-1]}' exited with code {result.returncode}. "
            f"Stderr: {stderr_detail}"
        )
    return json.loads(result.stdout)


def _declares_unsupported_mode(stdout: str) -> bool:
    """Whether a failed script's stdout is the "mode not implemented" marker."""
    try:
        data = json.loads(stdout)
    except ValueError:
        return False
    error = data.get("error") if isinstance(data, dict) else None
    return isinstance(error, dict) and error.get("type") == _UNSUPPORTED_MODE


class ScriptProvider(BaseProvider):
    """Provider adapter that delegates inference to an external script.

    The script must implement the stdin/stdout JSON protocol defined in
    docs/script-protocol.md. The 'model' field of the request is interpreted
    as the path to the script file. Both sides of the pipe are UTF-8.

    Interpreter resolution for Python scripts (.py extension):
      1. If a .venv directory exists in the script's own directory, its
         Python interpreter is used.
      2. If no local .venv is found, sys.executable is used.

    Each call_ai() invocation spawns a fresh subprocess (stateless model).

    Attachments are not validated here. The script receives raw paths and is the
    only party that knows what it can open, so refusing a type on its behalf
    would be guesswork.
    """

    # Only the script knows whether it holds a model or embeds text, so these
    # modes are offered and a script without them declines. There is no
    # list_models: the "model" is the script's own path.
    SUPPORTS_UNLOAD: bool = True
    SUPPORTS_EMBEDDINGS: bool = True

    # Empty on purpose: the whole merged extra_options dict reaches the script,
    # library keys included, since only the script knows what it understands.
    # The sampling levers are read out of it but never removed.
    _CONSUMED_OPTION_KEYS: frozenset[str] = frozenset()

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the ScriptProvider.

        Args:
            config: ProviderConfig. Supplies the timeout and the options
                registered via ``configure_provider()``.
        """
        self.config = config

    def call(self, request: AiRequest) -> AiResponse:
        """Execute an inference call by spawning the target script.

        Args:
            request: The structured request. request.model is the script path.

        Returns:
            Standardized response with text and token counts.

        Raises:
            FileNotFoundError: If the script path does not exist.
            subprocess.TimeoutExpired: If the script exceeds request.timeout.
            RuntimeError: If the script exits with a non-zero return code.
            json.JSONDecodeError: If stdout is not valid JSON.
            KeyError: If the response JSON is missing the required 'text' field.
        """
        cmd = _resolve_interpreter(request.model)

        opts = self._merge_options(request)

        payload = {
            "mode": "generate",
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "messages": request.messages,
            "file_path": normalize_file_paths(request.file_path),
            # null means "use your own default" in the protocol.
            "temperature": self._prefer_request(request.temperature, opts, "temperature"),
            "thinking": request.thinking,
            "format_json": request.format_json,
            "timeout": request.timeout,
            "top_k": self._prefer_request(request.top_k, opts, "top_k"),
            "top_p": self._prefer_request(request.top_p, opts, "top_p"),
            "max_tokens": self._prefer_request(request.max_tokens, opts, "max_tokens"),
            "extra_options": opts or None,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in request.tools
            ] if request.tools else None,
            "tool_results": [
                {
                    "call_id": tr.call_id,
                    "name": tr.name,
                    "content": tr.content,
                }
                for tr in request.tool_results
            ] if request.tool_results else None,
        }

        data = _run_script(cmd, payload, request.timeout)

        raw_tool_calls = data.get("tool_calls") or []
        tool_calls: list[ToolCall] = [
            ToolCall(
                id=tc.get("id", f"{tc.get('name', 'tool')}_{i}"),
                name=tc["name"],
                arguments=tc.get("arguments", {}),
            )
            for i, tc in enumerate(raw_tool_calls)
        ]

        return AiResponse(
            text=data["text"],
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
            reasoning_text=data.get("reasoning_text", ""),
            tool_calls=tool_calls,
            message=self._assistant_message(data["text"], tool_calls),
        )

    def preload_model(
        self,
        model: str,
        keep_alive: str | int | None = None,
        context_size: int | None = None,
        extra_options: dict | None = None,
    ) -> None:
        """Ask the script to preload itself via the 'preload' protocol mode.

        A script that declares the mode unsupported is logged at debug level.
        Any other non-zero exit is a warning: it may be a real failure, or a
        script that declines without the declaration.

        Args:
            model: Path to the script file.
            keep_alive: Forwarded to the script, which decides what it means.
                A number is seconds, ``0`` means unload once idle and ``-1``
                means keep resident; a string is a Go duration. ``None`` falls
                back to the value registered via ``configure_provider()``, and
                travels as null when there is none.
            context_size: Context window size in tokens. ``None`` falls back
                to the registered value, then to null.
            extra_options: Additional script-specific settings, merged over
                the ones registered via ``configure_provider()``.
        """
        cmd = _resolve_interpreter(model)
        registered = self.config.extra_options or {}
        merged_options = {**registered, **(extra_options or {})}
        payload = {
            "mode": "preload",
            "keep_alive": self._resolve_keep_alive(keep_alive),
            "context_size": (
                context_size if context_size is not None
                else registered.get("context_size")
            ),
            "extra_options": merged_options or None,
            "timeout": self.config.timeout,
        }
        self._run_optional_mode(cmd, payload, self.config.timeout)

    def warm_up(
        self,
        model: str,
        file_paths: str | list[str] | None = None,
        *,
        keep_alive: str | int | None = None,
        timeout: int | None = None,
    ) -> bool:
        """Ask the script to warm itself up via the 'warm_up' protocol mode.

        A script that declines the mode, with or without declaring it, has
        nothing to warm up: that is False, not an error, as the warm-up
        contract asks.

        Args:
            model: Path to the script file.
            file_paths: Optional path or list of paths the script may want to
                read or index ahead of the first call.
            keep_alive: Forwarded to the script, which decides what it means.
                ``None`` falls back to the value registered via
                ``configure_provider()``; the field is always present, and is
                null when neither source set one.
            timeout: Seconds to allow the script. Defaults to the timeout
                registered for this provider.

        Returns:
            True if the script reported that it warmed up, False if it does not
            implement the mode or reported that it had nothing to do.
        """
        cmd = _resolve_interpreter(model)
        effective_timeout = self._resolve_timeout(timeout)
        payload = {
            "mode": "warm_up",
            "file_path": normalize_file_paths(file_paths),
            "keep_alive": self._resolve_keep_alive(keep_alive),
            "timeout": effective_timeout,
        }
        data = self._run_optional_mode(cmd, payload, effective_timeout)
        if data is None:
            return False
        return bool(data.get("warmed_up", True))

    def unload_model(self, model: str, *, timeout: int | None = None) -> None:
        """Ask the script to release what it holds, via the 'unload' mode.

        Declining is the normal case, not a misconfiguration: most scripts hold
        nothing between calls. Logged and otherwise ignored, as in
        ``preload_model()``.

        Args:
            model: Path to the script file.
            timeout: Seconds to allow the script. Defaults to the timeout
                registered for this provider.
        """
        cmd = _resolve_interpreter(model)
        effective_timeout = self._resolve_timeout(timeout)
        payload = {
            "mode": "unload",
            "timeout": effective_timeout,
        }
        self._run_optional_mode(cmd, payload, effective_timeout)

    @staticmethod
    def _run_optional_mode(cmd: list[str], payload: dict, timeout: int) -> dict | None:
        """Run a mode a script may decline, logging a decline instead of raising.

        A declared decline is debug-level. A bare non-zero exit stays a
        warning, because without the declaration it cannot be told from a
        crash. A timeout, unparseable output or a missing script still
        propagates: none of them means "not implemented".

        Returns:
            The script's output, or None when it declined.
        """
        mode = payload["mode"]
        try:
            return _run_script(cmd, payload, timeout)
        except UnsupportedOperationError:
            _log.debug("Script '%s' does not implement mode '%s'", cmd[-1], mode)
        except RuntimeError as exc:
            _log.warning(
                "Script '%s' does not implement mode '%s', or failed while "
                "trying: %s", cmd[-1], mode, exc,
            )
        return None
    def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        """Generate embeddings by spawning the target script in embed mode.

        One subprocess per input: the protocol's ``embed`` payload carries a
        single ``text``, and a batch field a script did not know about would be
        ignored, returning one vector where several were expected. Files and a
        vector size travel as the optional ``file_path`` and ``dimensions``
        fields, sent only when set, so the payload of a plain text request is
        exactly what it always was.

        Args:
            request: The embedding request. ``request.model`` is the script path.

        Returns:
            One vector per input.

        Raises:
            subprocess.TimeoutExpired: If the script exceeds the timeout.
            UnsupportedOperationError: If the script declares ``embed``
                unsupported. Raised on the first attempt, never retried.
            RuntimeError: If the script exits with a non-zero return code.
            json.JSONDecodeError: If stdout is not valid JSON.
            KeyError: If the response JSON is missing the 'embedding' field.
        """
        cmd = _resolve_interpreter(request.model)
        files = normalize_file_paths(request.file_path)
        vectors: list[list[float]] = []
        for text in request.inputs:
            payload: dict[str, Any] = {"mode": "embed", "text": text}
            if files:
                payload["file_path"] = files
            if request.dimensions is not None:
                payload["dimensions"] = request.dimensions
            data = _run_script(cmd, payload, request.timeout)
            vectors.append([float(x) for x in data["embedding"]])
        return self._check_vector_count(vectors, request)
