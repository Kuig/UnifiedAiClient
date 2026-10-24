from __future__ import annotations

from typing import Any

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class LlamaCppProvider(OpenAiCompatProvider):
    """llama.cpp server provider.

    Connects to a locally running llama-server using the OpenAI-compatible
    /v1/chat/completions API. Default URL: http://localhost:8080.

    All file handling is inherited from OpenAiCompatProvider. The thinking
    trace comes back in 'reasoning_content', which llama-server fills by
    default (--reasoning-format auto); started with --reasoning-format none, it
    leaves the trace inside the answer and reasoning_text stays empty.
    """

    DEFAULT_URL: str = "http://localhost:8080"

    # llama-server routes an input_audio block through miniaudio (mp3, wav,
    # flac), so this is the one local provider that takes audio directly.
    SUPPORTED_FILE_TYPES: frozenset[str] = frozenset({"image", "audio"})

    # Started in multi-model mode, llama-server loads on first inference, so
    # warming up has to send a completion rather than the inherited metadata GET.
    LAZY_MODEL_LOAD: bool = True

    def _reasoning_payload(self, thinking: bool | str) -> dict[str, Any]:
        """Turn thinking on or off through both of llama-server's controls.

        ``reasoning_effort: "none"`` is the server's own off switch; any other
        value is only handed to the chat template. Templates in the Qwen style,
        MiniCPM-V 4.5's among them, read ``enable_thinking`` from
        ``chat_template_kwargs`` instead, and ignore the effort. Sending both
        reaches either kind of template, and each ignores the variable it does
        not use.

        Args:
            thinking: True or False.

        Returns:
            Both fields, merged with any ``chat_template_kwargs`` the caller set.
        """
        return {
            "reasoning_effort": "high" if thinking else "none",
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        }
