from __future__ import annotations

import base64
import logging
import mimetypes
import os

from unified_ai_client.exceptions import (
    FileDecodeError,
    MissingFileError,
    UnsupportedFileError,
)

_log = logging.getLogger("unified_ai_client.file_utils")

# ---------------------------------------------------------------------------
# File type classification
# ---------------------------------------------------------------------------

_IMAGE_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
})
_AUDIO_EXTS: frozenset[str] = frozenset({
    ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus",
})
_TEXT_EXTS: frozenset[str] = frozenset({
    ".txt", ".md", ".markdown", ".mdx", ".csv", ".tsv", ".json",
    ".jsonl", ".ndjson", ".html", ".htm", ".xml", ".svg",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".log",
    ".env", ".properties",
    ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".vue", ".svelte", ".css", ".scss", ".sass", ".less",
    ".sql", ".sh", ".bat", ".ps1", ".pl", ".r", ".jl",
    ".java", ".kt", ".scala", ".groovy", ".gradle",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hh", ".hxx",
    ".cs", ".rs", ".go", ".rb", ".php", ".swift", ".dart", ".lua",
    ".proto", ".graphql", ".gql", ".tf", ".cmake",
    ".patch", ".diff",
    ".tex", ".bib", ".rst", ".adoc",
})
_DOCUMENT_EXTS: frozenset[str] = frozenset({".pdf"})

# Files whose name *is* their type. Matched before the extension lookup because
# os.path.splitext() finds nothing usable in either 'Dockerfile' or '.gitignore':
# the first has no extension, the second is read as one with an empty name.
_TEXT_FILENAMES: frozenset[str] = frozenset({
    "dockerfile", "containerfile", "makefile", "gnumakefile",
    "rakefile", "gemfile", "procfile", "vagrantfile", "justfile",
    "license", "readme", "changelog", "authors", "notice",
    ".gitignore", ".gitattributes", ".dockerignore", ".editorconfig", ".env",
})


def classify_file(path: str) -> str:
    """Classify a file by its name and extension.

    The classification decides how a file reaches the model: images, audio and
    PDFs become native content blocks where the provider supports them, text is
    inlined into the prompt, and 'unknown' is refused rather than guessed at.

    Well-known extensionless names ('Dockerfile', 'Makefile', '.gitignore') are
    recognised by basename, case-insensitively. A suffixed variant such as
    'Dockerfile.dev' is not: it classifies as unknown.

    Args:
        path: Path to the file.

    Returns:
        One of: 'image', 'audio', 'text', 'document', 'unknown'.
    """
    if os.path.basename(path).lower() in _TEXT_FILENAMES:
        return "text"

    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _TEXT_EXTS:
        return "text"
    if ext in _DOCUMENT_EXTS:
        return "document"
    return "unknown"


def normalize_file_paths(file_path: str | list[str] | None) -> list[str]:
    """Normalize file_path to a flat list of strings.

    Args:
        file_path: A single path, a list of paths, or None.

    Returns:
        A list of path strings. Empty list if input is None.
    """
    if file_path is None:
        return []
    if isinstance(file_path, str):
        return [file_path]
    return list(file_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_files(
    paths: list[str],
    provider_name: str,
    supported: frozenset[str],
) -> None:
    """Reject attachments a provider cannot transmit, before the request is built.

    Every path is checked before any of them is read or encoded, so a rejected
    attachment costs nothing even when it is listed after large ones.

    ``'text'`` is always accepted: no provider exposes a native block for a
    ``.md`` or ``.csv``, so those are inlined into the prompt instead (Google
    uploads them, which reaches the same place by a different route). Every
    other class must be one the provider declares, otherwise the file would be
    silently dropped or fed through as unreadable text, and the model would
    answer as though it had received it.

    Args:
        paths: Local file paths to check.
        provider_name: Provider name, used in the error message.
        supported: File classes this provider can transmit natively, from
            ``BaseProvider.SUPPORTED_FILE_TYPES``.

    Raises:
        MissingFileError: If a path does not exist or is not a regular file.
            A ``FileNotFoundError`` subclass, so existing handlers still catch it.
        UnsupportedFileError: If a file's class is neither ``'text'`` nor one
            the provider supports.
    """
    for path in paths:
        if not os.path.isfile(path):
            raise MissingFileError(f"Attachment not found: '{path}'")

        file_type = classify_file(path)
        if file_type == "text" or file_type in supported:
            continue

        accepted = ", ".join(sorted(supported | {"text"})) or "text"
        if file_type == "unknown":
            detail = (
                f"Unrecognised file type: '{path}'. Provider "
                f"'{provider_name}' accepts: {accepted}."
            )
        else:
            detail = (
                f"Provider '{provider_name}' cannot accept {file_type} files: "
                f"'{path}'. Accepted: {accepted}."
            )
        raise UnsupportedFileError(detail)


# ---------------------------------------------------------------------------
# File reading and encoding
# ---------------------------------------------------------------------------


def read_text_file(path: str) -> str:
    """Read a text file and return its content.

    Args:
        path: Path to the text file.

    Returns:
        The file content as a string.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def encode_file_base64(path: str) -> str:
    """Read a file and return its base64-encoded content.

    Args:
        path: Path to the file.

    Returns:
        Base64-encoded file content as a string.
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(path: str) -> str:
    """Guess the MIME type for a file path.

    Args:
        path: Path to the file.

    Returns:
        MIME type string. Falls back to 'application/octet-stream' if unknown.
    """
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def audio_format_name(path: str) -> str:
    """Return the audio codec name for a file path.

    Used by OpenAI's input_audio format field, which expects the codec name
    (e.g. 'wav', 'mp3') rather than a MIME type.

    Args:
        path: Path to the audio file.

    Returns:
        Lowercase codec name (e.g. 'wav', 'mp3', 'flac', 'opus').
    """
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    # Map extension aliases to canonical codec names
    _aliases: dict[str, str] = {
        "m4a": "mp4",
        "aac": "aac",
        "ogg": "ogg",
    }
    return _aliases.get(ext, ext)


# ---------------------------------------------------------------------------
# Text file inlining
# ---------------------------------------------------------------------------


def format_text_attachment(file_path: str) -> str:
    """Format a single text file as a self-contained attachment block.

    Produces a delimited block containing the file's content, without
    appending or modifying any prompt. Use inline_text_attachments() to
    combine one or more of these blocks with a prompt.

    Args:
        file_path: Path to the text file.

    Returns:
        A formatted string block with file header, content, and footer.

    Raises:
        FileNotFoundError: If the file does not exist.
        FileDecodeError: If the file is not valid UTF-8. Substituting
            placeholder text here would reach the model as though it were the
            file's real content.
    """
    filename = os.path.basename(file_path)
    try:
        content = read_text_file(file_path)
    except UnicodeDecodeError as exc:
        raise FileDecodeError(
            f"Text attachment '{file_path}' is not valid UTF-8. "
            f"Its extension suggests text, but its content is not."
        ) from exc
    _log.info(
        "File '%s' formatted as text attachment (%d chars)",
        filename,
        len(content),
    )
    return (
        f"=== Attached file: {filename} ===\n"
        f"{content}\n"
        f"==================================="
    )


def inline_text_attachments(prompt: str, file_paths: list[str]) -> str:
    """Prepend one or more text file attachments to a prompt.

    Builds all attachment blocks first, then appends the user prompt once.
    This avoids prompt nesting or duplication when multiple files are attached.

    Args:
        prompt: The original user prompt.
        file_paths: List of paths to text files to attach.

    Returns:
        A string with all attachment blocks followed by the prompt.
    """
    if not file_paths:
        return prompt
    blocks = [format_text_attachment(fp) for fp in file_paths]
    return "\n\n".join(blocks) + "\n\n" + prompt
