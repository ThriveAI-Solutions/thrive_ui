"""
utils.quick_logger
Vendored and adapted from https://github.com/JoeEberle/quick_logger

Provides step-level traceability, performance tracking, and a unified
logging interface (``pvlog``) that wraps Python's standard ``logging``
module.  All existing handlers (console, file rotation, Discord) are
preserved because this module delegates to ``logging.getLogger()``.

Adaptations from upstream quick_logger:
- Removed ``talking_code`` dependency (text-to-speech disabled by default).
- Removed ``logging.basicConfig()`` call to avoid conflicting with the
  existing dict-based config in ``logging_config.py``.
- ``pvlog()`` routes through a named logger instead of the root logger.
- Added ``get_logger()`` factory so each module keeps its own logger name
  for filtering and test-patching.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

# Re-export stdlib level constants so callers don't need ``import logging``
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# ── Text-to-speech toggle (disabled; no talking_code dependency) ────────
_speaking_log = False
_speaking_steps = False


def set_speaking_log(on_off: bool = False) -> None:
    global _speaking_log
    _speaking_log = on_off


def get_speaking_log() -> bool:
    return _speaking_log


def set_speaking_steps(on_off: bool = False) -> None:
    global _speaking_steps
    _speaking_steps = on_off


def get_speaking_steps() -> bool:
    return _speaking_steps


# ── Logger factory ──────────────────────────────────────────────────────


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib ``Logger`` for *name*.

    This is intentionally a thin wrapper so callers can still do
    ``logger.debug(...)`` when ``pvlog`` is not needed.
    """
    return logging.getLogger(name)


# ── Core logging function ──────────────────────────────────────────────

# Map level strings accepted by pvlog to stdlib level constants
_LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "exception": logging.ERROR,
}

# Default module-level logger used when no explicit logger is supplied
_default_logger = logging.getLogger("quick_logger")


def pvlog(
    log_level: str,
    log_string: str,
    *,
    logger: logging.Logger | None = None,
    step: int | None = None,
) -> None:
    """Print *and* log a message through the stdlib logging hierarchy.

    Parameters
    ----------
    log_level : str
        One of ``debug``, ``info``, ``warn``/``warning``, ``error``,
        ``critical``, or ``exception``.
    log_string : str
        The message to log.
    logger : logging.Logger, optional
        Logger instance to route through.  Falls back to the module-level
        ``quick_logger`` logger when *None*.
    step : int, optional
        If provided, the message is prefixed with ``Step <n> - ``.
    """
    _logger = logger or _default_logger

    if step is not None:
        log_string = f"Step {step} - {log_string}"

    level = _LEVEL_MAP.get(log_level.lower(), logging.INFO)
    _logger.log(level, log_string)


# ── Performance tracking ───────────────────────────────────────────────


def set_start_time() -> float:
    """Capture and return the current wall-clock time."""
    return time.time()


def calculate_process_performance(
    solution_name: str,
    process_start_time: float,
    *,
    logger: logging.Logger | None = None,
) -> str:
    """Log elapsed time with a human-readable duration classification.

    Returns a one-line status string.
    """
    _logger = logger or _default_logger

    stop_time = time.time()
    duration = stop_time - process_start_time
    stop_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    _logger.info("PERFORMANCE %s total duration: %.2f s", solution_name, duration)
    _logger.info("PERFORMANCE %s stop time: %s", solution_name, stop_stamp)

    if duration > 600.0:
        classification = "LONG"
        _logger.info(
            "PERFORMANCE %s LONG process duration > 10 min: %.2f s — optimization required",
            solution_name,
            duration,
        )
    elif duration > 120.0:
        classification = "Medium"
        _logger.info(
            "PERFORMANCE %s Medium process duration > 2 min: %.2f s — optimization optional",
            solution_name,
            duration,
        )
    elif duration > 3.0:
        classification = "Low"
        _logger.info(
            "PERFORMANCE %s Low process duration < 3 min: %.2f s — optimization optional",
            solution_name,
            duration,
        )
    else:
        classification = "Short"
        _logger.info(
            "PERFORMANCE %s Short process duration < 3 s: %.2f s — optimization not recommended",
            solution_name,
            duration,
        )

    status = f"END {solution_name} duration={duration:.2f}s classification={classification}"
    _logger.info("END %s %s", solution_name, "=" * 45)
    return status
