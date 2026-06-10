from __future__ import annotations

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class MistralProvider(OpenAiCompatProvider):
    """Mistral API provider.

    Connects to Mistral's API endpoint using the OpenAI-compatible
    /v1/chat/completions API. Default URL: https://api.mistral.ai.
    """

    DEFAULT_URL: str = "https://api.mistral.ai"
