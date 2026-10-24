from __future__ import annotations

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class MetaProvider(OpenAiCompatProvider):
    """Meta Model API provider.

    Connects to Meta's OpenAI-compatible Model API (models such as
    ``muse-spark-1.3``). Default URL: https://api.meta.ai.

    Until 0.6.0 this pointed at https://api.llama-api.com, a third-party
    domain rather than Meta's own; Meta's Llama API preview, which it stood in
    for, has since closed.
    """

    DEFAULT_URL: str = "https://api.meta.ai"

    REQUIRES_API_KEY: bool = True
    SECRETS_KEY: str = "meta_api_key"

    # The Model API documents chat completions only.
    SUPPORTS_EMBEDDINGS: bool = False
