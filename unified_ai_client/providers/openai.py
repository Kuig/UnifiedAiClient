from __future__ import annotations

import logging
import os
from typing import Any

from unified_ai_client.file_utils import encode_file_base64
from unified_ai_client.providers.openai_compat import OpenAiCompatProvider

_log = logging.getLogger("unified_ai_client.providers.openai")


class OpenAiProvider(OpenAiCompatProvider):
    """OpenAI API provider (api.openai.com).

    Extends the base OpenAI-compatible provider with native PDF support: a
    'file' content block carrying a base64 data URL. Images and audio use the
    shared blocks from OpenAiCompatProvider.

    Reasoning text is populated from the 'reasoning_content' field in the
    response message, available on o3/o4 class models.
    """

    DEFAULT_URL: str = "https://api.openai.com"

    REQUIRES_API_KEY: bool = True
    SECRETS_KEY: str = "openai_api_key"
    REASONING_PARAM: str | None = "reasoning_effort"

    SUPPORTED_FILE_TYPES: frozenset[str] = frozenset({"image", "audio", "document"})

    def _build_native_block(self, file_path: str, file_type: str) -> dict[str, Any]:
        """Add PDF blocks to the shared OpenAI-compatible set.

        The Chat Completions content part for a document is ``file``. The name
        ``input_file`` belongs to the Responses API and is rejected here.

        Args:
            file_path: Path to the attachment.
            file_type: Its class from ``classify_file()``.

        Returns:
            A single content block dict.
        """
        if file_type == "document":
            b64 = encode_file_base64(file_path)
            _log.info("OpenAI: '%s' → file block (PDF)", os.path.basename(file_path))
            return {
                "type": "file",
                "file": {
                    "filename": os.path.basename(file_path),
                    "file_data": f"data:application/pdf;base64,{b64}",
                },
            }

        return super()._build_native_block(file_path, file_type)
