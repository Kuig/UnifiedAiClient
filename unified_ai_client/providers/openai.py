from __future__ import annotations

import logging
import os
from typing import Any

from unified_ai_client.file_utils import (
    audio_format_name,
    classify_file,
    encode_file_base64,
    get_mime_type,
    inline_text_attachments,
    normalize_file_paths,
)
from unified_ai_client.providers.openai_compat import OpenAiCompatProvider

_log = logging.getLogger("unified_ai_client.providers.openai")


class OpenAiProvider(OpenAiCompatProvider):
    """OpenAI API provider (api.openai.com).

    Extends the base OpenAI-compatible provider with native support for:
    - Audio files: sent as 'input_audio' content blocks (base64, codec format).
    - PDF files: sent as 'input_file' content blocks (base64 data URL).

    Reasoning text is populated from the 'reasoning_content' field in the
    response message, available on o3/o4 class models.
    """

    DEFAULT_URL: str = "https://api.openai.com"

    def _build_user_content(
        self, prompt: str, file_paths: list[str]
    ) -> str | list[dict[str, Any]]:
        """Build user content with native audio and PDF support.

        Args:
            prompt: The user prompt text.
            file_paths: List of file paths to attach.

        Returns:
            String or list of content block dicts.
        """
        if not file_paths:
            return prompt

        native_blocks: list[dict[str, Any]] = []
        text_files: list[str] = []

        for fp in file_paths:
            ft = classify_file(fp)

            if ft == "image":
                mime = get_mime_type(fp)
                b64 = encode_file_base64(fp)
                _log.info("OpenAI: '%s' → image_url block", os.path.basename(fp))
                native_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })

            elif ft == "audio":
                b64 = encode_file_base64(fp)
                fmt = audio_format_name(fp)
                _log.info(
                    "OpenAI: '%s' → input_audio block (format=%s)",
                    os.path.basename(fp),
                    fmt,
                )
                native_blocks.append({
                    "type": "input_audio",
                    "input_audio": {"data": b64, "format": fmt},
                })

            elif ft == "document":
                b64 = encode_file_base64(fp)
                _log.info("OpenAI: '%s' → input_file block (PDF)", os.path.basename(fp))
                native_blocks.append({
                    "type": "input_file",
                    "file": {
                        "filename": os.path.basename(fp),
                        "file_data": f"data:application/pdf;base64,{b64}",
                    },
                })

            else:
                _log.info(
                    "OpenAI: '%s' (%s) → text inline fallback",
                    os.path.basename(fp),
                    ft,
                )
                text_files.append(fp)

        effective_prompt = inline_text_attachments(prompt, text_files)

        if native_blocks:
            content: list[dict[str, Any]] = [
                {"type": "text", "text": effective_prompt}
            ]
            content.extend(native_blocks)
            return content

        return effective_prompt
