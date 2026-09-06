from __future__ import annotations
import logging
import time
from typing import Any, Callable

from unified_ai_client.exceptions import NonRetryableError

_log = logging.getLogger("unified_ai_client.retry")


def with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 5.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    label: str = "",
    **kwargs: Any,
) -> Any:
    """Execute fn with exponential backoff on failure.

    Args:
        fn: The callable/function to run.
        *args: Positional arguments passed to fn.
        max_retries: Number of retry attempts before giving up.
        base_delay: Initial delay in seconds before the first retry.
        backoff_factor: Multiplier applied to delay on subsequent attempts.
        retryable_exceptions: Tuple of exceptions that trigger a retry.
        label: Optional description of the call for log messages (e.g.
            ``"ollama/gemma4"``). Defaults to ``fn``'s name when omitted.
        **kwargs: Keyword arguments passed to fn.

    Returns:
        The return value of the function invocation.

    Raises:
        NonRetryableError: Immediately, without consuming an attempt. These
            failures are deterministic, so every retry would reproduce them.
        BaseException: The last raised exception once retry budget is exhausted,
            or any exception not listed in retryable_exceptions raised immediately.
    """
    tag = label or getattr(fn, "__name__", repr(fn))
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except NonRetryableError as e:
            _log.error("%s: non-retryable error, giving up immediately: %s", tag, e)
            raise
        except retryable_exceptions as e:
            attempt += 1
            if attempt > max_retries:
                _log.error("%s: giving up after %d attempt(s): %s", tag, attempt, e)
                raise e
            delay = base_delay * (backoff_factor ** (attempt - 1))
            _log.warning(
                "%s: attempt %d/%d failed (%s), retrying in %.1fs",
                tag, attempt, max_retries, e, delay,
            )
            time.sleep(delay)
