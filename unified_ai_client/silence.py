from __future__ import annotations
import logging


def silence_sdks() -> None:
    """Silence noisy third-party SDK loggers.

    Sets these loggers to WARNING level to suppress verbose INFO/DEBUG output.
    Should be called once at application startup by the consuming project.
    """
    noisy_loggers = [
        "google_genai",
        "google.genai",
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "anthropic",
    ]
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
