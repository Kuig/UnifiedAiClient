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
from unittest.mock import patch

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


class TestSetVerbosityIsThreadSafe(VerbosityIsolation):
    """Concurrent set_verbosity() calls must not race on the shared handler.

    Regression: the module-level `_handler` was read, then mutated, with no
    lock. Two threads could interleave their remove-old/add-new sequence and
    leave a handler attached to the logger that `_handler` no longer
    references, which no later call would ever find to remove — an orphan
    that silently duplicates every line this library logs.
    """

    def test_a_second_call_waits_for_the_first_to_leave_the_critical_section(self) -> None:
        """Forces the interleaving instead of hoping the GIL produces it.

        A plain "fire N threads and see if it broke" test is not reliable here:
        the critical section is a handful of attribute accesses with no I/O, so
        under CPython's GIL it usually completes in one scheduling slice even
        with no lock at all, and the race almost never shows up experimentally.
        This patches addHandler so the first call parks *inside* the section
        it is supposed to hold exclusively, then checks whether a second call
        reached that same point before being let go — which it must not, with
        the lock in place.
        """
        import threading

        entered = threading.Event()
        release = threading.Event()
        reached_add_handler: list[int] = []
        real_add_handler = logging.Logger.addHandler

        def parking_add_handler(self_logger, handler):
            reached_add_handler.append(1)
            entered.set()
            release.wait(timeout=2)
            return real_add_handler(self_logger, handler)

        with patch.object(logging.Logger, "addHandler", parking_add_handler):
            first = threading.Thread(target=set_verbosity, args=("debug",))
            first.start()
            self.assertTrue(
                entered.wait(timeout=2), "the first call never reached addHandler"
            )

            second = threading.Thread(target=set_verbosity, args=("warning",))
            second.start()
            # A generous window for the second call to race in if the critical
            # section were unprotected — it needs only a few bytecode steps to
            # reach its own addHandler, nowhere near this long.
            second.join(timeout=0.3)

            self.assertEqual(
                len(reached_add_handler), 1,
                "a second call reached addHandler while the first was still "
                "inside the critical section",
            )

            release.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertEqual(len(self._logger.handlers), 1)


if __name__ == "__main__":
    unittest.main()
