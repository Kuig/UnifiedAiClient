from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from unified_ai_client.exceptions import UnsupportedFileError
from unified_ai_client.file_utils import inline_text_attachments, validate_files

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

    SUPPORTS_UNLOAD: bool = False
    """Whether this provider has a model residency concept at all.

    True only for providers that keep a model loaded between requests and can
    be told to release it: ``ollama`` and ``script``. A cloud endpoint holds
    nothing on the caller's behalf, so there is nothing to unload.

    ``client.cleanup()`` reads this to decide which models to track and release
    at exit, which is why it is a class flag rather than an ``isinstance``
    check. The ``False`` default means "nothing to release", the safe answer for
    a provider whose behaviour has not been established.
    """

    REQUIRES_API_KEY: bool = False
    """Whether a missing credential is an error for this provider.

    False is the safe default: a local server, a subprocess and any adapter
    whose credential story has not been established all run fine without one.
    Cloud adapters set it True and name their key in ``SECRETS_KEY``.
    """

    SECRETS_KEY: str = ""
    """Key name in secrets.json, used to build a useful error message."""

    _LIBRARY_OPTION_KEYS: frozenset[str] = frozenset({
        # Infrastructure. Resolved by client.py; an adapter only ever sees these
        # in extra_options when a caller puts them there by hand.
        "url", "timeout", "sleep_time",
        # The common levers. Every adapter maps these onto its own payload
        # field, so none of them may also travel as a raw key.
        "temperature", "max_tokens", "max_output_tokens", "top_k", "top_p",
        # Provider-scoped controls the library owns. docs/configuration.md names
        # which adapter reads each one; the others must drop it.
        "context_size", "use_generate", "keep_alive",
        "disable_safety", "upload_poll_timeout",
        "task_type", "output_dimensionality",
    })
    """Every option key this library assigns a meaning to.

    ``extra_options`` carries two unrelated kinds of key. The ones listed here
    belong to the library: some adapter reads each of them and maps it onto a
    native field. Everything absent from this set is provider-specific and is
    forwarded verbatim, which is how a knob such as ``visual_token_budget``
    reaches a model without this library knowing it exists.

    Membership here means "never travels raw". An adapter that owns a key
    consumes it explicitly and declares it in ``_CONSUMED_OPTION_KEYS``; an
    adapter that does not own it drops it. Both halves matter: until 0.5.5 only
    Ollama filtered anything, so ``configure_provider("groq", use_generate=False)``
    put ``use_generate`` in the body of /v1/chat/completions, and the
    ``context_size`` that config.json.example shipped for lmstudio and llamacpp
    went the same way.
    """

    DEFAULT_TEMPERATURE: float = 0.7
    """The temperature used when neither the caller nor the config set one.

    ``AiRequest.temperature`` defaults to None so that "not given" stays
    distinguishable from "given 0.7", which is what lets a temperature
    registered through ``configure_provider()` actually reach the payload. The
    value that used to be the signature default lives here instead, so the
    behaviour a caller sees when configuring nothing is unchanged.
    """

    _CONSUMED_OPTION_KEYS: frozenset[str] = frozenset()
    """The subset of ``_LIBRARY_OPTION_KEYS`` this adapter reads for itself.

    Declarative, and read by a contract test rather than at runtime: it is what
    keeps the per-adapter behaviour and the table in docs/configuration.md from
    drifting apart again. Empty is right for an adapter that consumes none, such
    as ``script``, which forwards the whole dict onward by contract.
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

    def _require_api_key(self) -> None:
        """Fail early and clearly when a required credential is missing.

        The one place the rule lives, so the eight cloud adapters cannot drift
        apart on the wording of an error the user is meant to act on.

        Two things it deliberately does not do. It is not called from
        ``__init__``, because ``get_provider()`` instantiates providers eagerly,
        long before anyone knows whether a request will follow;
        ``GoogleProvider`` defers the same check to its lazy client getter for
        the same reason. And it stays silent when ``config.url`` is set: a
        caller who pointed a cloud adapter somewhere else has said "this is a
        proxy or a gateway", which may legitimately need no credentials, and
        aiming the OpenAI adapter at a local server that serves
        /v1/chat/completions is exactly what the url override is for.

        Returns immediately for providers that need no key, before touching
        ``api_key`` or ``config``, which is what lets ``ollama`` and ``script``
        inherit it without declaring either.

        Raises:
            ValueError: If this provider needs a key and none was supplied.
        """
        if not self.REQUIRES_API_KEY:
            return
        if not self.api_key and self.config.url is None:
            raise ValueError(
                f"Missing API key for provider '{self.provider_name}'. "
                f"Add '{self.SECRETS_KEY}' to secrets.json "
                f"or set {self.SECRETS_KEY.upper()}."
            )

    def _merge_options(self, request: AiRequest) -> dict[str, Any]:
        """Collapse config-level and call-level ``extra_options`` into one dict.

        Call-level wins for the same key: it is the more specific signal, and
        docs/configuration.md documents it as the escape hatch that overrides
        anything registered through ``configure_provider()``.

        Args:
            request: The request whose ``extra_options`` take precedence.

        Returns:
            The merged options, safe to mutate.
        """
        opts: dict[str, Any] = {}
        if self.config.extra_options:
            opts.update(self.config.extra_options)
        if request.extra_options:
            opts.update(request.extra_options)
        return opts

    @staticmethod
    def _prefer_request(
        value: Any, opts: dict[str, Any], key: str, default: Any = None
    ) -> Any:
        """Resolve one common lever across its three possible sources.

        The precedence is the same for every one of them: an explicit call-time
        argument, then ``extra_options``, then the provider's own default.
        ``None`` is the "not given" signal, which is why the sampling parameters
        on ``AiRequest`` default to it rather than to a value.

        Args:
            value: The named argument off the request, or None if unset.
            opts: Merged options from ``_merge_options()``.
            key: The option key to look up when the argument was not given.
            default: Value to use when neither source has one.

        Returns:
            The resolved value, possibly ``default``.
        """
        if value is not None:
            return value
        resolved = opts.get(key)
        return default if resolved is None else resolved

    def _passthrough_options(self, opts: dict[str, Any]) -> dict[str, Any]:
        """The options this library assigns no meaning to.

        Everything in the library's own namespace has already been consumed by
        the adapter that owns it, so what is left is provider-specific and is
        forwarded to the endpoint untouched.

        Args:
            opts: Merged options from ``_merge_options()``.

        Returns:
            A new dict holding only the keys outside ``_LIBRARY_OPTION_KEYS``.
        """
        return {k: v for k, v in opts.items() if k not in self._LIBRARY_OPTION_KEYS}

    def _resolve_timeout(self, timeout: int | None) -> int:
        """Resolve an optional timeout against this provider's configuration.

        Args:
            timeout: Seconds to wait, or None to use the configured value.

        Returns:
            The number of seconds to wait.
        """
        return timeout if timeout is not None else self.config.timeout

    @staticmethod
    def _estimate_reasoning_tokens(
        content: str, reasoning: str, total_tokens: int
    ) -> int:
        """Split a combined output token count across reasoning and content.

        Ollama and Anthropic both report one aggregate figure covering the
        thinking trace and the answer together, so the division has to be
        estimated from the character ratio of the two.

        It is an estimate of a *measurement*, not transport glue: consumers
        compare trace length and composition, so if the formula is ever refined
        it must be refined in one place.

        Args:
            content: The answer text.
            reasoning: The thinking trace, empty when there was none.
            total_tokens: The provider's aggregate output token count.

        Returns:
            The tokens attributable to the trace, 0 when there is no trace or
            no count to divide. Never 0 for a non-empty trace, which would read
            as "the model did not think".
        """
        if not reasoning:
            return 0
        total_chars = len(content) + len(reasoning)
        if total_chars <= 0 or total_tokens <= 0:
            return 0
        chars_per_token = total_chars / total_tokens
        return max(1, round(len(reasoning) / chars_per_token))

    def _partition_attachments(
        self, paths: list[str], prompt: str
    ) -> tuple[str, list[tuple[str, str]]]:
        """Validate attachments and split them into inlined text and native.

        The seam the whole attachment policy runs through, so it is expressed
        once rather than re-derived per adapter. Text is always accepted and
        always inlined, because no endpoint has a native block for a .md or a
        .csv; everything else must become a provider-native block, and
        ``_validate_files()`` has already refused anything this provider does
        not declare.

        ``google`` does not use this: it uploads every attachment through the
        Files API and has never inlined anything.

        Args:
            paths: Local file paths attached to the current turn.
            prompt: The prompt text the inlined attachments are appended to.

        Returns:
            ``(effective_prompt, native)``, where ``native`` holds one
            ``(path, file_type)`` pair per attachment needing a native block.

        Raises:
            MissingFileError: If a path does not exist.
            UnsupportedFileError: If a file is neither text nor a class this
                provider declares.
        """
        native: list[tuple[str, str]] = []
        text_files: list[str] = []

        for file_path, file_type in self._validate_files(paths):
            if file_type == "text":
                text_files.append(file_path)
            else:
                native.append((file_path, file_type))

        return inline_text_attachments(prompt, text_files), native

    def _unsupported_native_error(
        self, file_path: str, file_type: str
    ) -> UnsupportedFileError:
        """Build the error for a class this provider declares but cannot build.

        The inconsistency it reports is a bug in the adapter, not in the
        caller's input: ``SUPPORTED_FILE_TYPES`` promised something the block
        builder does not deliver. Kept in one place because the three adapters
        had already drifted into two different wordings for it.

        Args:
            file_path: The attachment that could not be turned into a block.
            file_type: Its class, as returned by ``classify_file()``.

        Returns:
            The exception to raise.
        """
        return UnsupportedFileError(
            f"Provider '{self.provider_name}' declares support for {file_type} "
            f"files but builds no block for them: '{file_path}'."
        )

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

    def preload_model(
        self,
        model: str,
        keep_alive: str | int = "15m",
        context_size: int | None = None,
        extra_options: dict | None = None,
    ) -> None:
        """Pre-load a model into memory for faster first inference.

        Concrete and a no-op by default, for the same reason ``warm_up()``,
        ``unload_model()`` and ``cleanup()`` are: "there is no model to load"
        is a valid answer for a cloud endpoint, not a gap, and an adapter
        written before this hook existed must keep instantiating. The three
        providers that overrode it only to write "Unused." four times are the
        cost the abstract version charged.

        Providers that override this hold a model resident and therefore also
        declare ``SUPPORTS_UNLOAD``: preload and unload are a pair, and a
        contract test asserts the two sets match.

        Args:
            model: Model identifier to preload.
            keep_alive: How long to keep the model in memory. A number is
                seconds, ``0`` unloads the model once idle and ``-1`` keeps
                it resident indefinitely; a string is a Go duration such as
                ``'15m'``. Ignored by providers with no residency concept.
            context_size: Context window size in tokens (provider-specific).
                Ollama maps this to ``num_ctx``. Ignored by providers that
                do not support preloading.
            extra_options: Additional provider-specific options to include in
                the preload request. Ignored by providers that do not support
                preloading.
        """
        return None

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
        *,
        keep_alive: str | int | None = None,
        timeout: int | None = None,
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
            keep_alive: How long the model should stay resident once loaded.
                ``None`` leaves the provider's own resolution in place. Accepted
                and ignored by providers with no residency concept.
            timeout: Seconds to wait for the warm-up request. ``None`` falls
                back to the provider's configured timeout.

        Returns:
            True if something was actually warmed up, False if this provider
            has nothing to do. Never raises for the "nothing to warm up" case.
        """
        return False

    def unload_model(self, model: str, *, timeout: int | None = None) -> None:
        """Release a model this provider is holding resident.

        The counterpart to ``preload_model()``. Concrete rather than abstract,
        for the same reason ``warm_up()`` is: "there is nothing to release" is a
        valid answer, and a provider written before this hook existed must keep
        working unchanged.

        Providers that override this declare ``SUPPORTS_UNLOAD = True``.

        Args:
            model: Model identifier to release.
            timeout: Seconds to wait for the request. ``None`` lets the provider
                choose, which is deliberately short: an unload queues behind
                whatever generation is already running.
        """
        return None

    def cleanup(self) -> None:
        """Release any remote resources held by this provider.

        Called on process termination via atexit handler and explicitly
        by consuming projects. Default implementation is a no-op.
        Override in providers that upload files or hold remote resources.
        """
        pass
