"""
utils.logging_config
Standard-library logging configured *once* for the whole app.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

from .discord_logging import add_discord_handler_if_configured

LOG_DIR = Path(__file__).with_name("logs")
LOG_DIR.mkdir(exist_ok=True)


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
                "level": "DEBUG",
                "formatter": "file",
                "filename": str(LOG_DIR / "app.log"),
                "when": "midnight",
                "backupCount": 14,
                "encoding": "utf-8",
            },
            # separate error log (10 MB rotation)
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
                "level": "DEBUG",
                "handlers": ["console", "file.daily", "file.error"],
            },
            # Silence noisy third-party loggers
            "urllib3.connectionpool": {
                "level": "WARNING",
                "handlers": [],
                "propagate": False,
            },
        },
    }


# Public API -------------------------------------------------------------
def setup_logging(*, debug: bool = False) -> None:
    """
    Configure logging.  Call exactly once, early in the main process.
    """
    logging.config.dictConfig(_dict_config(debug))

    # Discord handler will be added later when Streamlit secrets are available
    print("Basic logging configuration complete - Discord handler will be added when secrets are available")

    logging.getLogger(__name__).debug("Logging configured (debug=%s)", debug)
