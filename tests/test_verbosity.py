"""Tests for unified_ai_client.verbosity.set_verbosity() and the retry logging it exposes.

Usage:
    python -m unittest discover -s tests     # from project root
    python -m unittest tests.test_verbosity
"""
from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is in sys.path so unified_ai_client is importable
# both when running this file directly from tests/ and from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from unified_ai_client import set_verbosity
from unified_ai_client.exceptions import NonRetryableError
from unified_ai_client.retry import with_retry


class VerbosityIsolation(unittest.TestCase):
    """Restores the 'unified_ai_client' logger to its pre-test state.

    Mirrors ProviderRegistryIsolation in test_providers.py: set_verbosity()
    mutates global logger state (level, propagate, handlers), and tests can
    run in any order.
    """

    def setUp(self) -> None:
        super().setUp()
        self._logger = logging.getLogger("unified_ai_client")
        self._saved_level = self._logger.level
        self._saved_propagate = self._logger.propagate
        self._saved_handlers = list(self._logger.handlers)

    def tearDown(self) -> None:
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)
        for h in self._saved_handlers:
            self._logger.addHandler(h)
        self._logger.setLevel(self._saved_level)
        self._logger.propagate = self._saved_propagate
        import unified_ai_client.verbosity as _verbosity
        _verbosity._handler = None
        super().tearDown()


class TestSetVerbosity(VerbosityIsolation):
    def test_unknown_level_raises(self) -> None:
        with self.assertRaises(ValueError):
            set_verbosity("chatty")

    def test_silent_disables_output(self) -> None:
        set_verbosity("silent")
        logger = logging.getLogger("unified_ai_client")
        self.assertGreater(logger.level, logging.CRITICAL)
        self.assertFalse(logger.propagate)
        self.assertEqual(logger.handlers, [])

    def test_error_level_attaches_handler(self) -> None:
        set_verbosity("error")
        logger = logging.getLogger("unified_ai_client")
        self.assertEqual(logger.level, logging.ERROR)
        self.assertEqual(len(logger.handlers), 1)
        self.assertFalse(logger.propagate)

    def test_warning_level(self) -> None:
        set_verbosity("warning")
        logger = logging.getLogger("unified_ai_client")
        self.assertEqual(logger.level, logging.WARNING)

    def test_debug_level_prefixes_output(self) -> None:
        set_verbosity("debug")
        logger = logging.getLogger("unified_ai_client")
        self.assertEqual(logger.level, logging.DEBUG)
        handler = logger.handlers[0]
        formatted = handler.formatter.format(
            logging.LogRecord(
                "unified_ai_client.client", logging.INFO, __file__, 1, "hello", None, None,
            )
        )
        self.assertTrue(formatted.startswith("UAC :: "))

    def test_case_insensitive_and_stripped(self) -> None:
        set_verbosity("  DEBUG  ")
        self.assertEqual(logging.getLogger("unified_ai_client").level, logging.DEBUG)

    def test_repeated_calls_do_not_stack_handlers(self) -> None:
        set_verbosity("debug")
        set_verbosity("warning")
        set_verbosity("debug")
        logger = logging.getLogger("unified_ai_client")
        self.assertEqual(len(logger.handlers), 1)


class TestRetryLogging(VerbosityIsolation):
    def test_retryable_failure_logs_warning_then_succeeds(self) -> None:
        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("transient")
            return "ok"

        with self.assertLogs("unified_ai_client.retry", level="WARNING") as cm:
            result = with_retry(flaky, max_retries=3, base_delay=0, label="test/flaky")
        self.assertEqual(result, "ok")
        self.assertTrue(any("test/flaky" in line for line in cm.output))

    def test_exhausted_retries_logs_error(self) -> None:
        def always_fails() -> None:
            raise RuntimeError("permanent")

        with self.assertLogs("unified_ai_client.retry", level="ERROR") as cm:
            with self.assertRaises(RuntimeError):
                with_retry(always_fails, max_retries=1, base_delay=0, label="test/always_fails")
        self.assertTrue(any("giving up" in line for line in cm.output))

    def test_non_retryable_error_logs_error_immediately(self) -> None:
        def bad_input() -> None:
            raise NonRetryableError("deterministic failure")

        with self.assertLogs("unified_ai_client.retry", level="ERROR") as cm:
            with self.assertRaises(NonRetryableError):
                with_retry(bad_input, max_retries=3, base_delay=0, label="test/bad_input")
        self.assertTrue(any("non-retryable" in line for line in cm.output))


if __name__ == "__main__":
    unittest.main()
