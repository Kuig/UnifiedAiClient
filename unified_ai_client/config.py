from __future__ import annotations
import json
import os
from typing import Any


# Mapping from environment variable names to the snake_case keys used in
# secrets.json and returned by load_secrets().  Only sensitive credentials
# belong here — server URLs are configuration and live in config.json.
_ENV_VAR_MAP: dict[str, str] = {
    "GOOGLE_API_KEY": "google_api_key",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "OPENAI_API_KEY": "openai_api_key",
}


def load_secrets(project_root: str, filename: str = "secrets.json") -> dict[str, str]:
    """Load API credentials from environment variables and/or a secrets.json file.

    Both sources are optional and may coexist. When the same key is present in
    both, **environment variables take priority** (standard 12-factor convention —
    deployment-level config beats file-level config).

    Sources (highest → lowest priority):

    1. Environment variables (``os.environ``) — set by shell, Docker, CI/CD, etc.
    2. ``secrets.json`` at ``project_root`` — for local development.

    Supported keys:

    +------------------------+-------------------------+
    | ``secrets.json`` key   | Environment variable    |
    +========================+=========================+
    | ``google_api_key``     | ``GOOGLE_API_KEY``      |
    | ``anthropic_api_key``  | ``ANTHROPIC_API_KEY``   |
    | ``openai_api_key``     | ``OPENAI_API_KEY``      |
    +------------------------+-------------------------+

    Server URLs for local providers (Ollama, LM Studio, llama.cpp) are
    **not** credentials — they belong in ``config.json`` and are never
    read from environment variables by this function.

    Args:
        project_root: Absolute path to the consuming project's root directory.
        filename: Name of the JSON secrets file. Default: ``"secrets.json"``.

    Returns:
        A merged dictionary of ``snake_case`` key → string value pairs.
        Returns an empty dict if neither source provides any values.
    """
    # Start with secrets.json (lower priority)
    merged: dict[str, str] = {}
    json_path = os.path.join(project_root, filename)
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    merged[str(k)] = str(v)
        except Exception:
            pass

    # Override with environment variables (higher priority)
    for env_key, secret_key in _ENV_VAR_MAP.items():
        value = os.environ.get(env_key, "").strip()
        if value:
            merged[secret_key] = value

    return merged


def load_config(
    config_path: str,
    dataclass_type: type[Any],
    section: str | None = None,
) -> Any:
    """Load a JSON config file into a dataclass with fallback defaults.

    Supports both flat JSON files and nested JSON files (via the section argument).
    If the file is missing, malformed, or doesn't match the dataclass fields,
    returns a dataclass instance populated with all default values.

    Args:
        config_path: Absolute path to the JSON config file.
        dataclass_type: The target dataclass type (must have defaults for all fields).
        section: Optional key to extract from a nested JSON before parsing.

    Returns:
        An instance of the specified dataclass type.
    """
    if not os.path.exists(config_path):
        return dataclass_type()

    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return dataclass_type()

        # If a section is specified, extract it first
        if section is not None:
            data = data.get(section, {})
            if not isinstance(data, dict):
                return dataclass_type()

        # Extract only the keys that match the dataclass fields
        fields = dataclass_type.__dataclass_fields__
        filtered_data = {k: v for k, v in data.items() if k in fields}
        if "extra_options" in fields:
            extra = {k: v for k, v in data.items() if k not in fields}
            if "extra_options" in data and isinstance(data["extra_options"], dict):
                extra.update(data["extra_options"])
            filtered_data["extra_options"] = extra
        return dataclass_type(**filtered_data)
    except Exception:
        return dataclass_type()
