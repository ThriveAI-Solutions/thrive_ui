# agent/logging_config.py
"""Configuration + payload-capping helpers for agentic run logging.

Reads the [agent_logging] section of secrets.toml:

    [agent_logging]
    mode = "full"                       # "full" | "scrubbed" | "disabled"
    max_logged_result_bytes = 5000000
    max_logged_event_bytes = 5000000
    retention_days = 0                  # 0 = keep indefinitely
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

_VALID_MODES = ("full", "scrubbed", "disabled")
_DEFAULT_MAX_BYTES = 5_000_000


@dataclass(frozen=True)
class AgentLoggingConfig:
    mode: str = "full"
    max_logged_result_bytes: int = _DEFAULT_MAX_BYTES
    max_logged_event_bytes: int = _DEFAULT_MAX_BYTES
    retention_days: int = 0

    @property
    def enabled(self) -> bool:
        return self.mode in ("full", "scrubbed")

    @classmethod
    def from_secrets(cls, section: dict | None) -> "AgentLoggingConfig":
        section = dict(section or {})
        mode = section.get("mode", "full")
        if mode not in _VALID_MODES:
            mode = "full"
        return cls(
            mode=mode,
            max_logged_result_bytes=int(section.get("max_logged_result_bytes", _DEFAULT_MAX_BYTES)),
            max_logged_event_bytes=int(section.get("max_logged_event_bytes", _DEFAULT_MAX_BYTES)),
            retention_days=int(section.get("retention_days", 0)),
        )

    @classmethod
    def from_streamlit(cls) -> "AgentLoggingConfig":
        try:
            import streamlit as st

            return cls.from_secrets(dict(st.secrets.get("agent_logging", {})))
        except Exception:
            return cls()


def cap_json(
    obj: Any,
    max_bytes: int,
) -> tuple[Optional[str], bool, int, Optional[str]]:
    """Serialize `obj` to JSON, capping stored size.

    Returns (stored_text, truncated, original_byte_count, sha256_of_original).
    - None obj -> (None, False, 0, None).
    - Under cap -> full JSON, truncated=False.
    - Over cap -> text truncated to max_bytes (UTF-8 safe), truncated=True,
      original byte count + hash of the FULL payload preserved for integrity.
    """
    if obj is None:
        return None, False, 0, None
    full = json.dumps(obj, default=str)
    raw = full.encode("utf-8")
    nbytes = len(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if nbytes <= max_bytes:
        return full, False, nbytes, digest
    truncated_text = raw[:max_bytes].decode("utf-8", errors="ignore")
    return truncated_text, True, nbytes, digest
