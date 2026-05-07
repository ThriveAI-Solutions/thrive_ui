"""selection_instructions() injects the currently-selected patient into
the agent's per-run instructions so the model knows to skip find_patient.

This replaces the [Regression context] prompt-prefix hack with the
pydantic-ai-idiomatic dynamic instructions pattern.
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

from agent.deps import SelectedPatient
from agent.instructions import selection_instructions


def _ctx(selected_patient):
    """Minimal stand-in for RunContext — selection_instructions only
    reads ctx.deps.selected_patient."""
    return SimpleNamespace(deps=SimpleNamespace(selected_patient=selected_patient))


def test_returns_empty_when_no_patient_selected():
    assert selection_instructions(_ctx(None)) == ""


def test_includes_source_id_when_patient_selected():
    sp = SelectedPatient(
        source_id="src-john-1962",
        display_name="John Smith",
        dob=date(1962, 5, 1),
        selected_at=datetime(2026, 5, 6, 21, 32),
        selection_origin="user_click",
    )
    out = selection_instructions(_ctx(sp))
    assert "src-john-1962" in out


def test_instructs_agent_not_to_call_find_patient():
    sp = SelectedPatient(
        source_id="src-john-1962",
        display_name="John Smith",
        dob=date(1962, 5, 1),
        selected_at=datetime(2026, 5, 6),
        selection_origin="user_click",
    )
    out = selection_instructions(_ctx(sp))
    # Must explicitly steer away from find_patient — that's the whole point.
    assert "find_patient" in out.lower()


def test_includes_display_name_and_dob_for_grounding():
    sp = SelectedPatient(
        source_id="src-x",
        display_name="Jane Doe",
        dob=date(1980, 7, 14),
        selected_at=datetime(2026, 5, 6),
        selection_origin="user_click",
    )
    out = selection_instructions(_ctx(sp))
    assert "Jane Doe" in out
    assert "1980-07-14" in out


def test_handles_missing_dob_gracefully():
    sp = SelectedPatient(
        source_id="src-y",
        display_name="No DOB",
        dob=None,
        selected_at=datetime(2026, 5, 6),
        selection_origin="user_click",
    )
    # Must not raise; should still emit source_id.
    out = selection_instructions(_ctx(sp))
    assert "src-y" in out
