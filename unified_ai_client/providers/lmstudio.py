from __future__ import annotations

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class LmStudioProvider(OpenAiCompatProvider):
    """LM Studio local provider.

    Connects to a locally running LM Studio server using the OpenAI-compatible
    /v1/chat/completions API. Default URL: http://localhost:1234.

    All file handling and reasoning text extraction is inherited from
    OpenAiCompatProvider. LM Studio does not expose native reasoning tokens;
    reasoning_text will always be an empty string.
    """

    DEFAULT_URL: str = "http://localhost:1234"

    def warm_up(
        self,
        model: str,
        file_paths: str | list[str] | None = None,
    ) -> bool:
        """Load the model with a one-token completion.

        LM Studio loads a model on first inference, so the inherited
        ``GET /v1/models`` would return instantly and leave the load cost for
        the first real call. The completion is billable inference in principle,
        but the server is local, so in practice it is free.

        Args:
            model: Model identifier to load.
            file_paths: Ignored. LM Studio inlines attachments into the request.

        Returns:
            Always True.
        """
        self._warm_up_completion(model)
        return True
