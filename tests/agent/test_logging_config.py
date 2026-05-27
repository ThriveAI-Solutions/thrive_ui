# tests/agent/test_logging_config.py
import json
from agent.logging_config import AgentLoggingConfig, cap_json


def test_config_defaults_when_no_secrets():
    cfg = AgentLoggingConfig.from_secrets({})
    assert cfg.mode == "full"
    assert cfg.max_logged_result_bytes == 5_000_000
    assert cfg.max_logged_event_bytes == 5_000_000
    assert cfg.retention_days == 0
    assert cfg.enabled is True


def test_config_reads_values_and_disabled_flag():
    cfg = AgentLoggingConfig.from_secrets({"mode": "disabled", "max_logged_result_bytes": 10, "retention_days": 30})
    assert cfg.mode == "disabled"
    assert cfg.enabled is False
    assert cfg.max_logged_result_bytes == 10


def test_config_rejects_unknown_mode():
    cfg = AgentLoggingConfig.from_secrets({"mode": "bogus"})
    assert cfg.mode == "full"  # falls back to safe default


def test_cap_json_under_limit_passes_through():
    text, truncated, nbytes, digest = cap_json({"a": 1}, max_bytes=1000)
    assert json.loads(text) == {"a": 1}
    assert truncated is False
    assert nbytes == len(text.encode("utf-8"))
    assert len(digest) == 64


def test_cap_json_over_limit_truncates_and_flags():
    big = {"rows": ["x" * 100 for _ in range(100)]}
    text, truncated, nbytes, digest = cap_json(big, max_bytes=200)
    assert truncated is True
    assert nbytes > 200  # original size recorded
    assert len(text.encode("utf-8")) <= 200 + 64  # stored text is bounded
    assert len(digest) == 64  # hash of ORIGINAL payload


def test_cap_json_handles_none():
    text, truncated, nbytes, digest = cap_json(None, max_bytes=1000)
    assert text is None
    assert truncated is False
    assert nbytes == 0
    assert digest is None
