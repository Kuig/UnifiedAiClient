from __future__ import annotations

import base64
import logging
import mimetypes
import os

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
    ".txt", ".md", ".csv", ".json", ".py", ".html", ".htm",
    ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log",
    ".js", ".ts", ".css", ".sql", ".sh", ".bat", ".ps1",
    ".r", ".java", ".c", ".cpp", ".h", ".rs", ".go",
    ".rb", ".php", ".swift", ".kt", ".scala", ".lua",
    ".tex", ".rst", ".adoc", ".tsv",
})
_DOCUMENT_EXTS: frozenset[str] = frozenset({".pdf"})


def classify_file(path: str) -> str:
    """Classify a file by its extension.

    Args:
        path: Path to the file.

    Returns:
        One of: 'image', 'audio', 'text', 'document', 'unknown'.
    """
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
    """
    filename = os.path.basename(file_path)
    try:
        content = read_text_file(file_path)
    except Exception:
        content = "[File could not be read as text]"
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
