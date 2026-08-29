from __future__ import annotations


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
