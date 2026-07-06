"""
utils.logging_config
Standard-library logging configured *once* for the whole app,
now using quick_logger for unified traceability.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

from .discord_logging import add_discord_handler_if_configured
from .quick_logger import pvlog, set_speaking_log

LOG_DIR = Path(__file__).with_name("logs")
LOG_DIR.mkdir(exist_ok=True)

_configured = False

# Libraries whose DEBUG/INFO chatter dominates the file log when the root
# logger is opened up (PIL logs one line per PNG chunk, httpx per request …).
_NOISY_THIRD_PARTY_LOGGERS = (
    "PIL",
    "httpx",
    "httpcore",
    "openai",
    "matplotlib",
    "watchdog",
    "fsevents",
    "chromadb",
)


def _dict_config(debug: bool = False) -> dict:
    """Return a logging.config-compatible dict."""
    level = "DEBUG" if debug else "INFO"

    return {
        "version": 1,
        "disable_existing_loggers": False,  # capture libs that configured themselves
        # ───────────── formatters ─────────────
        "formatters": {
            "console": {
                "format": "{asctime} {levelname:<8} {name}: {message}",
                "style": "{",
                "datefmt": "%H:%M:%S",
            },
            "file": {
                "format": "{asctime} [{process:05d}] {levelname:<8} {name}: {message}",
                "style": "{",
            },
        },
        # ───────────── handlers ─────────────
        "handlers": {
            # human-readable console, coloured by the terminal itself
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "console",
                "stream": "ext://sys.stderr",
            },
            # daily rolling log (14-day retention)
            "file.daily": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": level,
                "formatter": "file",
                "filename": str(LOG_DIR / "app.log"),
                "when": "midnight",
                "backupCount": 14,
                "encoding": "utf-8",
            },
            # separate error log (10 MB rotation)
            "file.error": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "file",
                "filename": str(LOG_DIR / "error.log"),
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        # ───────────── loggers ─────────────
        "loggers": {
            # root logger catches *everything*
            "": {
                "level": level,
                "handlers": ["console", "file.daily", "file.error"],
            },
            # Silence noisy third-party loggers
            "urllib3.connectionpool": {
                "level": "WARNING",
                "handlers": [],
                "propagate": False,
            },
            **{name: {"level": "WARNING"} for name in _NOISY_THIRD_PARTY_LOGGERS},
        },
    }


def _get_streamlit_logging_section() -> dict | None:
    """Return the ``[logging]`` section from ``st.secrets``.

    Module-level helper so tests can patch it. Raises if streamlit is
    unavailable; ``_resolve_debug`` catches that and falls back to defaults.
    """
    import streamlit as st  # local: streamlit may not be importable in unit tests

    return dict(st.secrets.get("logging", {}))


def _resolve_debug(debug: bool | None) -> bool:
    if debug is not None:
        return bool(debug)
    try:
        section = _get_streamlit_logging_section() or {}
        return bool(section.get("debug", False))
    except Exception:
        return False


def _reset_for_tests() -> None:
    """Allow ``setup_logging`` to run again. Test helper; not for app use."""
    global _configured
    _configured = False


# Public API -------------------------------------------------------------
def setup_logging(*, debug: bool | None = None) -> None:
    """
    Configure logging once per process; later calls are no-ops (Streamlit
    re-executes app.py on every rerun). Level is INFO unless *debug* is
    passed or ``[logging].debug`` is set in secrets.
    Initializes quick_logger with text-to-speech disabled.
    """
    global _configured
    if _configured:
        return

    resolved_debug = _resolve_debug(debug)
    logging.config.dictConfig(_dict_config(resolved_debug))
    _configured = True

    # Disable text-to-speech — not needed in production
    set_speaking_log(False)

    # Discord handler will be added later when Streamlit secrets are available
    pvlog("debug", "Basic logging configuration complete - Discord handler will be added when secrets are available")

    pvlog("debug", f"Logging configured (debug={resolved_debug})")
