from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AiResponse:
    """Standardized response from any AI provider.

    Attributes:
        text: The generated text response.
        input_tokens: Number of input/prompt tokens consumed.
        output_tokens: Number of output/completion tokens generated.
        reasoning_tokens: Number of tokens used for internal reasoning/thinking.
            Zero if the provider does not report this separately.
        reasoning_text: Full reasoning/thinking text returned by the model.
            Empty string if reasoning was not requested or not supported by the
            provider.
    """
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_text: str = ""


@dataclass
class AiRequest:
    """Internal request object passed to provider adapters.

    Constructed by client.py from call_ai() arguments. Not part of the
    public API — consumers never create this directly.

    Attributes:
        provider: Provider name (e.g., 'ollama', 'google', 'anthropic').
        model: Model identifier or script path (for 'script' provider).
        prompt: User prompt text.
        system_prompt: Optional system instruction.
        messages: Optional chat history as a list of dicts with 'role' and
            'content' keys. Each dict may also contain an optional 'files'
            key with a list of file paths to attach to that message.
        file_path: Optional local file path or list of file paths for
            multimodal input. Accepts images, audio, text files, and PDFs.
            The provider handles all encoding, upload, and fallback internally.
        temperature: Sampling temperature.
        thinking: Whether to enable extended thinking/reasoning mode.
        format_json: Whether to force JSON output format.
        timeout: Maximum seconds to wait for a response.
    """
    provider: str
    model: str
    prompt: str
    system_prompt: str | None = None
    messages: list[dict] | None = None
    file_path: str | list[str] | None = None
    temperature: float = 0.7
    thinking: bool = False
    format_json: bool = False
    timeout: int = 120


@dataclass
class ProviderConfig:
    """Provider-specific configuration loaded from config.json.

    This is a flat dataclass holding the superset of all provider settings.
    Each provider reads only the fields it needs and ignores the rest.

    Attributes:
        url: Base URL for the provider's API endpoint.
        timeout: Maximum seconds to wait for a response.
        keep_alive: Duration to keep a model loaded in memory (Ollama).
        sleep_time: Seconds to sleep before each API call. Used for cloud
            provider rate limiting. Defaults to 0 (no sleep).
        disable_safety: Whether to disable safety filters (Google only).
        upload_poll_timeout: Maximum polling iterations when waiting for a
            Google-uploaded file to become ACTIVE. Each iteration sleeps 1s.
        max_tokens: Maximum number of tokens in the response. Used by
            Anthropic and OpenAI.
        context_size: Context window size override. Used by Ollama (num_ctx)
            and local providers. 0 means use the model default.
    """
    url: str = "http://localhost:11434"
    timeout: int = 120
    keep_alive: str = "15m"
    sleep_time: int = 0
    disable_safety: bool = False
    upload_poll_timeout: int = 15
    max_tokens: int = 8192
    context_size: int = 0
