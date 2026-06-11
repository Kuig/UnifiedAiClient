from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unified_ai_client.models import AiRequest, AiResponse


class BaseProvider(ABC):
    """Abstract base class for AI provider adapters.

    Every provider (Ollama, Gemini, etc.) must implement this interface.
    The client.py router dispatches requests to providers through this interface,
    ensuring complete provider agnosticism.
    """

    @abstractmethod
    def call(self, request: AiRequest) -> AiResponse:
        """Execute an AI inference call.

        Args:
            request: The structured request containing all parameters.

        Returns:
            Standardized response with text and token counts.
        """
        ...

    @abstractmethod
    def preload_model(
        self,
        model: str,
        keep_alive: str = "15m",
        context_size: int | None = None,
        extra_options: dict | None = None,
    ) -> None:
        """Pre-load a model into memory for faster first inference.

        Args:
            model: Model identifier to preload.
            keep_alive: How long to keep the model in memory.
            context_size: Context window size in tokens (provider-specific).
                Ollama maps this to ``num_ctx``. Ignored by providers that
                do not support preloading.
            extra_options: Additional provider-specific options to include in
                the preload request. Ignored by providers that do not support
                preloading.

        Raises:
            NotImplementedError: If the provider does not support preloading.
        """
        ...

    @abstractmethod
    def get_embedding(self, model: str, text: str) -> list[float]:
        """Generate a text embedding vector.

        Args:
            model: Embedding model identifier.
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            NotImplementedError: If the provider does not support embeddings.
        """
        ...

    def cleanup(self) -> None:
        """Release any remote resources held by this provider.

        Called on process termination via atexit handler and explicitly
        by consuming projects. Default implementation is a no-op.
        Override in providers that upload files or hold remote resources.
        """
        pass
