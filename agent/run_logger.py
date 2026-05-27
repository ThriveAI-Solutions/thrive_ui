# agent/run_logger.py
"""AgentRunLogger — full-fidelity logging for agentic runs.

Promoted from agent.audit.AuditLogger. Writes to the app SQLite DB:
- thrive_agent_run         (one row per question; rollup)
- thrive_agent_run_event   (append-only ordered timeline; source of truth)
- thrive_tool_call         (enriched per-tool rows)
- thrive_agent_patient_access (queryable patient touches)

Mode-aware via AgentLoggingConfig:
- "full":     verbatim payloads (byte-capped + hashed)
- "scrubbed": reuses agent.audit.scrub_arguments_json / summarize_result
- "disabled": logger is never constructed (deps_builder returns None)

Every write is wrapped so logging NEVER breaks a run, and commits per call
so the SQLite write lock is not held while the agent yields more events.
"""

from __future__ import annotations

from typing import Any, Optional


def extract_source_ids(payload: Any) -> list[tuple[str, Optional[str]]]:
    """Walk a tool-result payload and return [(source_id, display_name?)].

    Deduped by source_id, order-preserving. Looks for dict keys named
    'source_id'; pairs with a sibling 'display_name' when present.
    """
    found: dict[str, Optional[str]] = {}

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            sid = obj.get("source_id")
            if isinstance(sid, str) and sid:
                name = obj.get("display_name")
                if sid not in found or (found[sid] is None and name):
                    found[sid] = name if isinstance(name, str) else None
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)

    _walk(payload)
    return list(found.items())
