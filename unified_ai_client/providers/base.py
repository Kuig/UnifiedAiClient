from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from unified_ai_client.file_utils import validate_files

if TYPE_CHECKING:
    from unified_ai_client.models import AiRequest, AiResponse


class BaseProvider(ABC):
    """Abstract base class for AI provider adapters.

    Every provider (Ollama, Gemini, etc.) must implement this interface.
    The client.py router dispatches requests to providers through this interface,
    ensuring complete provider agnosticism.
    """

    SUPPORTED_FILE_TYPES: frozenset[str] = frozenset()
    """File classes this provider can transmit natively.

    Values come from ``classify_file()``: ``'image'``, ``'audio'``,
    ``'document'``. ``'text'`` is deliberately absent, and never declared: text
    files are inlined into the prompt by every provider rather than carried in a
    native block, so ``validate_files()`` accepts them unconditionally.

    The empty default means "nothing but text", which is the safe answer for a
    provider whose capabilities have not been established.
    """

    @property
    def provider_name(self) -> str:
        """The registry name this adapter is reached by.

        Derived from the class name so error messages can name the provider the
        caller actually passed to ``call_ai()`` without every subclass having to
        repeat it: ``LmStudioProvider`` becomes ``'lmstudio'``.

        Returns:
            The lower-case provider name.
        """
        return type(self).__name__.removesuffix("Provider").lower()

    def _validate_files(self, paths: list[str]) -> list[tuple[str, str]]:
        """Check attachments against this provider's declared capabilities.

        The seam every adapter routes its attachments through, so the policy is
        declared once here rather than re-assembled from ``provider_name`` and
        ``SUPPORTED_FILE_TYPES`` at each call site. ``script`` is the deliberate
        exception and never calls it: only the script knows what it can open.

        Args:
            paths: Local file paths attached to the current turn.

        Returns:
            One ``(path, file_type)`` pair per path, for the caller to build its
            content blocks from without classifying a second time.

        Raises:
            MissingFileError: If a path does not exist.
            UnsupportedFileError: If a file is neither text nor a class this
                provider declares.
        """
        return validate_files(paths, self.provider_name, self.SUPPORTED_FILE_TYPES)

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

    def warm_up(
        self,
        model: str,
        file_paths: str | list[str] | None = None,
    ) -> bool:
        """Pay this provider's one-off costs ahead of the first real call.

        Brings the provider into the state where a subsequent ``call()`` no
        longer has to pay setup costs it would otherwise charge to whichever
        request happens to come first: SDK import, client construction, DNS +
        TCP + TLS handshake, model load, remote file upload.

        Implementations must not consume generation tokens. Where a provider
        offers no free way to warm up, leaving this default in place is the
        correct answer, not a gap.

        Args:
            model: Model identifier to warm up.
            file_paths: Optional path or list of paths to pre-upload, for
                providers that keep a remote file store. Ignored by providers
                that inline attachments into the request.

        Returns:
            True if something was actually warmed up, False if this provider
            has nothing to do. Never raises for the "nothing to warm up" case.
        """
        return False

    def cleanup(self) -> None:
        """Release any remote resources held by this provider.

        Called on process termination via atexit handler and explicitly
        by consuming projects. Default implementation is a no-op.
        Override in providers that upload files or hold remote resources.
        """
        pass
