from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, NoReturn

from unified_ai_client.exceptions import NonRetryableHttpError, ProviderHttpError

_log = logging.getLogger("unified_ai_client.http")

_TRANSIENT_CLIENT_STATUSES: frozenset[int] = frozenset({408, 409, 425, 429})
"""4xx codes a later attempt can still succeed on.

Every other 4xx is settled the moment the server answers: a malformed payload,
an unknown model or a rejected credential produces the same status however many
times it is sent. These four do not. 408 and 425 are the server asking for the
request again, 409 is a state conflict that another request may clear, and 429
is a rate limit that the backoff exists precisely to wait out.
"""

_MAX_DETAIL_CHARS: int = 500
"""Cap on the message carried in the exception.

An endpoint behind a misconfigured proxy answers with an HTML page, not JSON.
That whole page inside an exception message makes the traceback unreadable
without adding anything, so it is truncated here and logged in full at DEBUG.
"""


def _extract_detail(body: str) -> str:
    """Pull the provider's own error message out of a response body.

    The three transports this module serves report errors in three shapes, and
    all three are read here so that no adapter has to know about the others:

    - OpenAI-compatible: ``{"error": {"message": ..., "type": ...}}``
    - Anthropic: ``{"type": "error", "error": {"type": ..., "message": ...}}``
    - Ollama: ``{"error": "model 'x' not found"}``, a string rather than an
      object.

    Every step is guarded. A body that is not JSON, or is JSON of an
    unanticipated shape, degrades to the raw text: replacing an HTTP failure
    with a parsing failure would hide the very thing the caller needs to see.

    Args:
        body: The response body, already decoded to text.

    Returns:
        The extracted message, truncated to a readable length, or an empty
        string when the body carried nothing at all.
    """
    detail = ""
    stripped = body.strip()
    if not stripped:
        return ""

    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            detail = str(error.get("message") or error.get("type") or "")
        elif isinstance(error, str):
            detail = error
        if not detail:
            message = parsed.get("message") or parsed.get("detail")
            if isinstance(message, str):
                detail = message

    if not detail:
        detail = stripped

    if len(detail) > _MAX_DETAIL_CHARS:
        _log.debug("Full error body (%d chars): %s", len(body), body)
        detail = detail[:_MAX_DETAIL_CHARS] + "..."
    return detail


def _raise_for_http_error(exc: urllib.error.HTTPError, url: str) -> NoReturn:
    """Re-raise an HTTPError as the typed error the retry wrapper understands.

    Reading the body is the whole point: without it the caller sees
    "HTTP Error 400: Bad Request" and has no way to learn which field the
    provider rejected.

    Args:
        exc: The exception urllib raised.
        url: The URL that was requested, for the error message.

    Raises:
        NonRetryableHttpError: For a 4xx other than the transient ones.
        ProviderHttpError: For every other status.
    """
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - a body we cannot read must not mask the status
        body = ""
    finally:
        # The body has been consumed, so release the underlying connection here
        # rather than leaving it to the garbage collector.
        try:
            exc.close()
        except Exception:  # noqa: BLE001 - closing is best effort
            pass

    detail = _extract_detail(body) or (exc.reason or "")
    if 400 <= exc.code < 500 and exc.code not in _TRANSIENT_CLIENT_STATUSES:
        raise NonRetryableHttpError(url, exc.code, detail, exc.hdrs, body) from exc
    raise ProviderHttpError(url, exc.code, detail, exc.hdrs, body) from exc


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST a JSON payload and return the parsed JSON response.

    The single transport every urllib-backed adapter goes through, so error
    classification is written once instead of once per provider.

    Args:
        url: Absolute URL to post to.
        payload: Request payload, serialised as JSON.
        timeout: Seconds to wait for the response.
        headers: Extra headers, typically authentication. ``Content-Type`` is
            always set and cannot be overridden.

    Returns:
        The parsed JSON response.

    Raises:
        NonRetryableHttpError: On a 4xx other than 408/409/425/429.
        ProviderHttpError: On any other HTTP error response.
        urllib.error.URLError: On a connection failure, which stays retryable.
        json.JSONDecodeError: If the response body is not JSON.
    """
    request_headers: dict[str, str] = dict(headers or {})
    request_headers["Content-Type"] = "application/json"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        _raise_for_http_error(exc, url)


def get_json(
    url: str,
    timeout: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """GET a URL and return the parsed JSON response.

    Args:
        url: Absolute URL to fetch.
        timeout: Seconds to wait for the response.
        headers: Extra headers, typically authentication.

    Returns:
        The parsed JSON response.

    Raises:
        NonRetryableHttpError: On a 4xx other than 408/409/425/429.
        ProviderHttpError: On any other HTTP error response.
        urllib.error.URLError: On a connection failure, which stays retryable.
        json.JSONDecodeError: If the response body is not JSON.
    """
    req = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        _raise_for_http_error(exc, url)
