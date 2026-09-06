from __future__ import annotations
import logging
import sys

# Console verbosity control for unified_ai_client's own internal logging.
#
# Scoped entirely to the "unified_ai_client" logger and its children — never
# the root logger, matching the same rule silence_sdks() follows for
# third-party loggers. A caller that never calls set_verbosity() sees no
# change: nothing here runs at import time.

_LEVELS: dict[str, int | None] = {
    "silent": None,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "debug": logging.DEBUG,
}

_PACKAGE_LOGGER_NAME = "unified_ai_client"
_PREFIX_FORMAT = "UAC :: %(levelname)-8s %(name)s: %(message)s"

_handler: logging.Handler | None = None


def set_verbosity(level: str) -> None:
    """Control how much of unified_ai_client's own activity is printed to the console.

    Attaches (or removes) a ``StreamHandler`` on the ``"unified_ai_client"``
    logger only — never the root logger, so this cannot interfere with a
    host application's own logging configuration or duplicate output onto
    its handlers. Every level, including ``"silent"``, disables propagation
    to the root logger, so calling this at least once takes full ownership
    of whether unified_ai_client's diagnostics reach the console at all.

    This is independent of :func:`silence_sdks`, which controls third-party
    SDK loggers instead: call both if you want first-party and third-party
    debug output together.

    Calling this again replaces the previously attached handler rather than
    stacking a second one.

    Args:
        level: One of ``"silent"`` (nothing printed, ever), ``"error"``
            (``ERROR``/``CRITICAL`` only), ``"warning"`` (``WARNING`` and
            above), or ``"debug"`` (everything, including fine-grained
            per-file/per-block detail). Case-insensitive.

    Raises:
        ValueError: If ``level`` is not one of the accepted values.

    Example::

        set_verbosity("debug")   # see everything during development
        set_verbosity("silent")  # back to full silence
    """
    global _handler

    level = level.strip().lower()
    if level not in _LEVELS:
        raise ValueError(
            f"Unknown verbosity level {level!r}. Expected one of: "
            f"{', '.join(sorted(_LEVELS))}."
        )

    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)

    if _handler is not None:
        logger.removeHandler(_handler)
        _handler = None

    logger.propagate = False

    if level == "silent":
        logger.setLevel(logging.CRITICAL + 1)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_PREFIX_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(_LEVELS[level])
    _handler = handler
