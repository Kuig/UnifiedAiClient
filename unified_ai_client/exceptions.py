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
    raised as this class: a later attempt genuinely can succeed.
    """
