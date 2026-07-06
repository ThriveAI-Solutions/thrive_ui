"""Tests for utils.logging_config (once-per-process setup, INFO default, noisy-lib pinning)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from utils.logging_config import _dict_config, _reset_for_tests, setup_logging


@pytest.fixture(autouse=True)
def _reset_configured_flag():
    _reset_for_tests()
    yield
    _reset_for_tests()


# ── once-per-process ──────────────────────────────────────────────────────


class TestSetupLoggingIdempotence:
    def test_second_call_is_noop(self):
        with patch("logging.config.dictConfig") as mock_dict_config:
            setup_logging()
            setup_logging()
        assert mock_dict_config.call_count == 1

    def test_reset_allows_reconfiguration(self):
        with patch("logging.config.dictConfig") as mock_dict_config:
            setup_logging()
            _reset_for_tests()
            setup_logging()
        assert mock_dict_config.call_count == 2


# ── level resolution ──────────────────────────────────────────────────────


class TestLevelResolution:
    def test_default_is_info(self):
        with patch("logging.config.dictConfig") as mock_dict_config:
            with patch("utils.logging_config._get_streamlit_logging_section", return_value=None):
                setup_logging()
        config = mock_dict_config.call_args[0][0]
        assert config["handlers"]["console"]["level"] == "INFO"
        assert config["handlers"]["file.daily"]["level"] == "INFO"
        assert config["loggers"][""]["level"] == "INFO"

    def test_explicit_debug_param_wins(self):
        with patch("logging.config.dictConfig") as mock_dict_config:
            with patch("utils.logging_config._get_streamlit_logging_section", return_value={"debug": False}):
                setup_logging(debug=True)
        config = mock_dict_config.call_args[0][0]
        assert config["handlers"]["console"]["level"] == "DEBUG"
        assert config["handlers"]["file.daily"]["level"] == "DEBUG"

    def test_secrets_debug_flag_enables_debug(self):
        with patch("logging.config.dictConfig") as mock_dict_config:
            with patch("utils.logging_config._get_streamlit_logging_section", return_value={"debug": True}):
                setup_logging()
        config = mock_dict_config.call_args[0][0]
        assert config["handlers"]["console"]["level"] == "DEBUG"

    def test_unreadable_secrets_fall_back_to_info(self):
        with patch("logging.config.dictConfig") as mock_dict_config:
            with patch(
                "utils.logging_config._get_streamlit_logging_section",
                side_effect=RuntimeError("no streamlit"),
            ):
                setup_logging()
        config = mock_dict_config.call_args[0][0]
        assert config["handlers"]["console"]["level"] == "INFO"


# ── noisy third-party loggers ─────────────────────────────────────────────


class TestNoisyLoggerPinning:
    NOISY = ("PIL", "httpx", "httpcore", "openai", "matplotlib", "watchdog", "fsevents", "chromadb")

    @pytest.mark.parametrize("name", NOISY)
    def test_pinned_to_warning_by_default(self, name):
        config = _dict_config(debug=False)
        assert config["loggers"][name]["level"] == "WARNING"

    @pytest.mark.parametrize("name", NOISY)
    def test_pinned_to_warning_even_in_debug(self, name):
        config = _dict_config(debug=True)
        assert config["loggers"][name]["level"] == "WARNING"

    def test_error_file_handler_stays_at_error(self):
        config = _dict_config(debug=False)
        assert config["handlers"]["file.error"]["level"] == "ERROR"
