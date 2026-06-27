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

from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from agent.audit import scrub_arguments_json, summarize_result
from agent.logging_config import AgentLoggingConfig, cap_json
from orm.models import AgentPatientAccess, AgentRun, AgentRunEvent, ToolCall


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


@dataclass
class AgentRunLogger:
    """Owns one run's logging lifecycle. Constructed per-run in deps_builder.

    `seq` and `call_index` are monotonic counters held for the run's duration.
    All public methods are guarded so logging never breaks the agent run.
    """

    session: Session
    config: AgentLoggingConfig
    run_id: str
    session_id: str
    user_id: int
    user_role: int
    group_id: Optional[str] = None

    _seq: int = field(default=0, init=False)
    _call_index: int = field(default=0, init=False)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _safe_commit(self) -> None:
        try:
            self.session.commit()
        except Exception:
            try:
                self.session.rollback()
            except Exception:
                pass

    def _append_event(
        self,
        event_type: str,
        *,
        payload: Any = None,
        payload_summary: Optional[str] = None,
        turn_index: Optional[int] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        elapsed_ms: Optional[int] = None,
    ) -> int:
        seq = self._next_seq()
        try:
            text, truncated, nbytes, digest = (None, False, 0, None)
            if payload is not None and self.config.mode == "full":
                text, truncated, nbytes, digest = cap_json(payload, self.config.max_logged_event_bytes)
            elif payload is not None:
                # scrubbed: store only a short summary, never the full payload
                payload_summary = payload_summary or (str(payload)[:200])
            row = AgentRunEvent(
                run_id=self.run_id,
                seq=seq,
                event_type=event_type,
                turn_index=turn_index,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                payload_json=text,
                payload_summary=payload_summary,
                payload_truncated=truncated,
                payload_bytes=nbytes or None,
                payload_hash=digest,
                elapsed_ms=elapsed_ms,
            )
            self.session.add(row)
            self._safe_commit()
        except Exception:
            pass
        return seq

    # ---- public API -------------------------------------------------

    def start_run(
        self,
        *,
        question: str,
        llm_provider: Optional[str],
        llm_model: Optional[str],
        selected_patient: Optional[dict],
        group_id: Optional[str],
        parent_run_id: Optional[str] = None,
        resume_reason: Optional[str] = None,
        user_message_id: Optional[int] = None,
        system_prompt_hash: Optional[str] = None,
        tool_schema_hash: Optional[str] = None,
        message_history: Any = None,
        app_git_sha: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> None:
        self.group_id = group_id
        try:
            history_text = None
            if message_history is not None and self.config.mode == "full":
                history_text, *_ = cap_json(message_history, self.config.max_logged_event_bytes)
            sp = selected_patient or {}
            run = AgentRun(
                run_id=self.run_id,
                session_id=self.session_id,
                group_id=group_id,
                parent_run_id=parent_run_id,
                resume_reason=resume_reason,
                user_message_id=user_message_id,
                user_id=self.user_id,
                user_role=self.user_role,
                question=question,
                selected_patient_source_id=sp.get("source_id"),
                selected_patient_display_name=sp.get("display_name"),
                selected_patient_dob=sp.get("dob"),
                selected_patient_selection_origin=sp.get("selection_origin"),
                llm_provider=llm_provider,
                llm_model=llm_model,
                system_prompt_hash=system_prompt_hash,
                tool_schema_hash=tool_schema_hash,
                message_history_json=history_text,
                status="open",
                success=False,
                logging_mode=self.config.mode,
                schema_version=1,
                app_git_sha=app_git_sha,
                environment=environment,
            )
            self.session.add(run)
            self._safe_commit()
        except Exception:
            pass
        self._append_event("run_started", payload={"question": question})
        if selected_patient and selected_patient.get("source_id"):
            self._record_access(
                source_id=selected_patient["source_id"],
                display_name=selected_patient.get("display_name"),
                access_type="pinned_at_run_start",
                access_origin="run_context",
            )

    def log_event(
        self,
        event_type: str,
        *,
        payload: Any = None,
        turn_index: Optional[int] = None,
        elapsed_ms: Optional[int] = None,
        payload_summary: Optional[str] = None,
    ) -> int:
        return self._append_event(
            event_type, payload=payload, turn_index=turn_index, elapsed_ms=elapsed_ms, payload_summary=payload_summary
        )

    def log_tool_started(
        self, *, tool_name: str, tool_call_id: Optional[str], turn_index: Optional[int], arguments: dict
    ) -> int:
        return self._append_event(
            "tool_call_started",
            payload={"arguments": arguments},
            turn_index=turn_index,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    def log_tool_completed(
        self,
        *,
        tool_name: str,
        tool_call_id: Optional[str],
        turn_index: Optional[int],
        arguments: dict,
        result_obj: Any,
        sql_executed: list,
        elapsed_ms: int,
        success: bool,
        error: Optional[str],
        selected_patient_source_id: Optional[str],
        started_event_seq: Optional[int] = None,
    ) -> None:
        self._call_index += 1
        call_index = self._call_index
        scrubbed = self.config.mode == "scrubbed"
        try:
            # Arguments: verbatim in full, scrubbed (hashed literals) otherwise.
            if scrubbed:
                arguments_json = scrub_arguments_json(tool_name, arguments)
            else:
                arguments_json, *_ = cap_json(arguments, self.config.max_logged_result_bytes)

            # Summary always cheap + PHI-safe.
            try:
                result_summary = summarize_result(tool_name, result_obj)
            except Exception:
                result_summary = f"result_type={type(result_obj).__name__}"

            # Full result + SQL: only stored in full mode.
            if scrubbed:
                result_json, r_trunc, r_bytes, r_hash = (None, False, None, None)
                sql_json, s_trunc, s_bytes, s_hash = (None, False, None, None)
            else:
                result_json, r_trunc, r_bytes, r_hash = cap_json(result_obj, self.config.max_logged_result_bytes)
                sql_json, s_trunc, s_bytes, s_hash = cap_json(sql_executed, self.config.max_logged_result_bytes)

            completed_seq = self._append_event(
                "tool_call_completed",
                payload={"result": result_obj, "sql_executed": sql_executed} if not scrubbed else None,
                payload_summary=result_summary,
                turn_index=turn_index,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                elapsed_ms=elapsed_ms,
            )

            row = ToolCall(
                session_id=self.session_id,
                user_id=self.user_id,
                user_role=self.user_role,
                selected_patient_source_id=selected_patient_source_id,
                tool_name=tool_name,
                arguments_json=arguments_json or "{}",
                result_summary=result_summary,
                elapsed_ms=elapsed_ms,
                success=success,
                error=error,
                run_id=self.run_id,
                tool_call_id=tool_call_id,
                call_index=call_index,
                turn_index=turn_index,
                attempt_index=1,
                started_event_seq=started_event_seq,
                completed_event_seq=completed_seq,
                result_json=result_json,
                result_truncated=r_trunc,
                result_bytes=r_bytes,
                result_hash=r_hash,
                sql_executed_json=sql_json,
                sql_executed_truncated=s_trunc,
                sql_executed_bytes=s_bytes,
                sql_executed_hash=s_hash,
            )
            self.session.add(row)
            self._safe_commit()

            # Patient access derived from full result payloads only.
            if not scrubbed:
                for sid, name in extract_source_ids(result_obj):
                    self._record_access(
                        source_id=sid,
                        display_name=name,
                        access_type="tool_result",
                        access_origin="tool",
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        event_seq=completed_seq,
                    )
        except Exception:
            pass

    def log_chooser_candidates(self, payload: dict) -> None:
        if self.config.mode == "scrubbed":
            return
        for sid, name in extract_source_ids(payload):
            self._record_access(
                source_id=sid, display_name=name, access_type="selection_shown", access_origin="agent_disambiguation"
            )

    def finalize_run(
        self,
        *,
        status: str,
        final_answer_text: Optional[str],
        usage: Optional[dict],
        total_elapsed_ms: Optional[int],
        cap_reached: Optional[str],
        final_message_id: Optional[int] = None,
        error_type: Optional[str] = None,
        error: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> None:
        if status == "cap_reached":
            self._append_event("cap_reached", payload={"reason": cap_reached})
        if status == "failed":
            self._append_event("run_failed", payload={"error_type": error_type, "error": error})
        try:
            run = self.session.query(AgentRun).filter_by(run_id=self.run_id).first()
            if run is not None:
                usage = usage or {}
                run.status = status
                run.success = status == "success"
                run.final_answer_text = final_answer_text
                run.final_message_id = final_message_id
                run.input_tokens = usage.get("input_tokens")
                run.output_tokens = usage.get("output_tokens")
                run.total_tokens = usage.get("total_tokens")
                run.tool_call_count = self._call_index
                run.event_count = self._seq + 1  # +1 for the terminal event below
                run.total_elapsed_ms = total_elapsed_ms
                run.cap_reached = cap_reached
                run.error_type = error_type
                run.error = error
                run.stack_trace = stack_trace if self.config.mode == "full" else None
                run.completed_at = func.now()
                self.session.add(run)
                self._safe_commit()
        except Exception:
            pass
        # Always emit the universal terminal marker last. `cap_reached` and
        # `run_failed` are detail events; `run_completed` closes every timeline.
        self._append_event("run_completed", payload={"status": status})

    def _record_access(
        self,
        *,
        source_id: str,
        display_name: Optional[str],
        access_type: str,
        access_origin: str,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        event_seq: Optional[int] = None,
    ) -> None:
        try:
            self.session.add(
                AgentPatientAccess(
                    run_id=self.run_id,
                    tool_call_id=tool_call_id,
                    event_seq=event_seq,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    source_id=source_id,
                    display_name=display_name,
                    access_type=access_type,
                    access_origin=access_origin,
                    tool_name=tool_name,
                )
            )
            self._safe_commit()
        except Exception:
            pass


def mark_run_fallback_invoked(
    session: Session,
    *,
    run_id: str,
    fallback_sql: Optional[str],
) -> None:
    """Mark an already-finalized agent run as ``fallback_invoked`` and
    persist the Vanna SQL that fired (Epic #228 / #233).

    Called from ``agent.runtime._maybe_invoke_vanna_fallback`` on the
    Streamlit script thread, using a fresh ``SessionLocal`` — the
    original ``AgentRunLogger.session`` is owned by the loop thread and
    may already be closed.

    Silently no-ops when ``run_id`` does not resolve to a row and on any
    DB error. An audit gap is preferable to losing the user's Vanna
    answer to a logging failure.
    """
    try:
        run = session.query(AgentRun).filter_by(run_id=run_id).first()
        if run is None:
            return
        run.status = "fallback_invoked"
        run.fallback_sql = fallback_sql
        session.add(run)

        # Append a timeline event so the inspector reflects the transition.
        # `seq` is MAX(seq)+1 for this run — the run's original logger is
        # gone, so we recompute from the existing rows.
        next_seq_row = session.query(func.max(AgentRunEvent.seq)).filter(AgentRunEvent.run_id == run_id).scalar()
        next_seq = (next_seq_row or 0) + 1
        session.add(
            AgentRunEvent(
                run_id=run_id,
                seq=next_seq,
                event_type="fallback_invoked",
                payload_summary=f"fallback_sql_chars={len(fallback_sql or '')}",
            )
        )
        session.commit()
    except Exception:
        try:
            session.rollback()
        except Exception:
            pass
