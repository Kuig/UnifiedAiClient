from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

from unified_ai_client.file_utils import normalize_file_paths
from unified_ai_client.models import AiRequest, AiResponse, ProviderConfig, ToolCall
from unified_ai_client.providers.base import BaseProvider


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
        RuntimeError: If the script exits with a non-zero return code.
        json.JSONDecodeError: If stdout is not valid JSON.
    """
    result = subprocess.run(
        cmd,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr_detail = result.stderr.strip() or "(no stderr output)"
        raise RuntimeError(
            f"Script '{cmd[-1]}' exited with code {result.returncode}. "
            f"Stderr: {stderr_detail}"
        )
    return json.loads(result.stdout)


class ScriptProvider(BaseProvider):
    """Provider adapter that delegates inference to an external script.

    The script must implement the stdin/stdout JSON protocol defined in
    LLM_BEHAVIOUR_REQUIREMENTS.md. The 'model' field of the request is
    interpreted as the path to the script file.

    Interpreter resolution for Python scripts (.py extension):
      1. If a .venv directory exists in the script's own directory, its
         Python interpreter is used.
      2. If no local .venv is found, sys.executable is used.

    Each call_ai() invocation spawns a fresh subprocess (stateless model).
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the ScriptProvider.

        Args:
            config: ProviderConfig. Not used directly but kept for API
                consistency with other providers.
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

        payload = {
            "mode": "generate",
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "messages": request.messages,
            "file_path": normalize_file_paths(request.file_path),
            "temperature": request.temperature,
            "thinking": request.thinking,
            "format_json": request.format_json,
            "timeout": request.timeout,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "extra_options": request.extra_options,
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

        # Parse tool calls if returned by the script
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
        )

    def preload_model(
        self,
        model: str,
        keep_alive: str = "15m",
        context_size: int | None = None,
        extra_options: dict | None = None,
    ) -> None:
        """Not supported by ScriptProvider. Emits a warning and returns.

        Args:
            model: Script path. Unused.
            keep_alive: Unused.
            context_size: Unused.
            extra_options: Unused.
        """
        warnings.warn(
            f"ScriptProvider does not support model preloading. "
            f"preload_model('{model}') call ignored.",
            UserWarning,
            stacklevel=2,
        )

    def get_embedding(self, model: str, text: str) -> list[float]:
        """Generate a text embedding by spawning the target script in embed mode.

        Args:
            model: Path to the script file.
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            subprocess.TimeoutExpired: If the script exceeds the default timeout.
            RuntimeError: If the script exits with a non-zero return code.
            json.JSONDecodeError: If stdout is not valid JSON.
            KeyError: If the response JSON is missing the 'embedding' field.
        """
        cmd = _resolve_interpreter(model)
        payload = {"mode": "embed", "text": text}
        data = _run_script(cmd, payload, timeout=120)
        return data["embedding"]
