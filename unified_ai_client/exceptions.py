from __future__ import annotations

import urllib.error
from typing import Any


class NonRetryableError(Exception):
    """Base class for failures that retrying cannot fix.

    ``with_retry()`` re-raises these immediately instead of spending its retry
    budget on them. A malformed request, an unsupported attachment or a missing
    file will fail identically on every attempt, so backing off and trying again
    only delays the error the caller needs to see.
    """


class UnsupportedFileError(NonRetryableError, ValueError):
    """A provider was handed a file it has no way to transmit.

    Raised before the request leaves the process, rather than letting the file
    be silently dropped or inlined as unreadable text. Subclasses ``ValueError``
    so existing handlers around ``call_ai()`` keep catching it.
    """


class MissingFileError(NonRetryableError, FileNotFoundError):
    """An attachment path does not exist.

    Subclasses ``FileNotFoundError`` so ordinary handlers still catch it, while
    the ``NonRetryableError`` half keeps the retry budget from being spent on a
    path that will be just as absent on the next attempt.
    """


class FileDecodeError(NonRetryableError, ValueError):
    """A text attachment could not be decoded as UTF-8.

    Only text files are inlined into the prompt, so a decode failure means the
    file is not what its extension claims. Raised instead of substituting
    placeholder text, which the model cannot distinguish from real content.
    """


class UnsupportedOperationError(NonRetryableError, NotImplementedError):
    """A provider was asked for something it does not offer at all.

    Embeddings on a provider with no embeddings endpoint, or a reduced vector
    size on one that cannot produce it. Raised before anything is sent.
    Subclasses ``NotImplementedError`` so handlers written against the earlier
    behaviour keep catching it; the ``NonRetryableError`` half is what stops
    ``with_retry()`` from spending its backoff budget on an answer that cannot
    change.
    """


class EmptyResponseError(NonRetryableError, RuntimeError):
    """A provider answered successfully but returned none of what was asked for.

    An embeddings response with no vector, typically because the model is not
    an embedding model. Subclasses ``RuntimeError``, which is what was raised
    before, so existing handlers keep catching it.
    """


class ModelRefusalError(NonRetryableError):
    """The provider declined to answer, on policy grounds.

    The request succeeded at the HTTP level and the response is well formed,
    but what it carries is a refusal rather than an answer: empty, or cut off
    where a safety classifier stopped it. Returning that text as an ordinary
    ``AiResponse`` would hand the caller a non-answer that looks like one, so it
    is raised instead. Not retried: the same request is declined the same way.

    Attributes:
        provider: Registry name of the provider that declined.
        category: The provider's own classification of the refusal, such as
            ``"cyber"``, or None when it gave none.
        explanation: The provider's explanation, or an empty string.
        partial_text: Any text generated before the refusal, for a caller that
            wants it despite the refusal.
    """

    def __init__(
        self,
        provider: str,
        category: str | None = None,
        explanation: str = "",
        partial_text: str = "",
    ) -> None:
        """Build the error.

        Args:
            provider: Registry name of the provider that declined.
            category: The provider's classification of the refusal, if any.
            explanation: The provider's explanation, if any.
            partial_text: Text generated before the refusal, if any.
        """
        message = f"Provider '{provider}' declined the request"
        if category:
            message += f" (category: {category})"
        if explanation:
            message += f": {explanation}"
        super().__init__(message)
        self.provider = provider
        self.category = category
        self.explanation = explanation
        self.partial_text = partial_text


class ProviderHttpError(urllib.error.HTTPError):
    """An HTTP error response from a provider endpoint, with its body read.

    ``urllib`` raises ``HTTPError`` with the status line only, so the reason the
    provider actually gave -- which lives in the response body -- is lost the
    moment the exception propagates. This subclass reads that body once, at the
    point of failure, and carries the message it found.

    Subclasses ``urllib.error.HTTPError`` so a handler written before this
    existed keeps catching it, in the same way ``UnsupportedFileError``
    subclasses ``ValueError``.

    Attributes:
        code: The HTTP status code, inherited from ``HTTPError``.
        detail: The provider's own error message, extracted from the body and
            truncated to a readable length. Falls back to the HTTP reason
            phrase when the body carried nothing usable.
        body: The raw response body as text, for a caller that needs more than
            ``detail`` carries.
    """

    def __init__(
        self,
        url: str,
        code: int,
        detail: str,
        hdrs: Any = None,
        body: str = "",
    ) -> None:
        """Build the error.

        Args:
            url: The URL that was requested.
            code: HTTP status code of the response.
            detail: Message to report, already extracted from the body. Becomes
                ``msg``, so ``str(exc)`` reads "HTTP Error 401: <detail>".
            hdrs: Response headers, as handed over by ``urllib``.
            body: Raw response body as text.
        """
        # fp=None is explicitly supported by HTTPError: the body is already
        # read, and leaving the original stream attached would hand callers a
        # file object that is spent.
        super().__init__(url, code, detail, hdrs, None)
        # HTTPError substitutes an empty BytesIO for a None fp and wraps it in
        # a tempfile wrapper, which emits a ResourceWarning when the exception
        # is finally collected. There is nothing to read from it -- the body
        # lives in ``body`` -- so close it now rather than leave every raised
        # error warning in the consumer's own logs.
        self.close()
        self.detail = detail
        self.body = body


class NonRetryableHttpError(NonRetryableError, ProviderHttpError):
    """A 4xx response that will fail identically on every attempt.

    A malformed payload, an unknown model or a rejected credential is settled
    the instant the server answers: retrying spends the whole backoff budget to
    arrive at the same status. ``with_retry()`` re-raises these immediately.

    The transient 4xx codes -- 408, 409, 425 and 429 -- are deliberately not
    raised as this class: a later attempt genuinely can succeed. The one 5xx
    that is, 501 Not Implemented, says the server does not do what was asked,
    which no retry changes.
    """
