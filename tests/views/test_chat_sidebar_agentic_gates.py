"""Tests for Epic #171 / Feature #172.

Covers:
  - ``_render_reset_agent_sidebar`` early-returns when ``agentic_mode``
    is off, renders normally when on or absent (default True).
  - ``_settings_dialog_body`` renders the LLM / Agentic / Display
    sections in the order LLM → Agentic → Display (with matching
    subheader labels in the same order).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_st_stub(*, agentic_mode=None, checkbox_state=None):
    """MagicMock-based stub for ``views.chat_bot.st``.

    ``session_state`` is a real dict (so ``get`` / ``setdefault`` work),
    everything else is a MagicMock so any attribute access (e.g.
    ``st.sidebar.container``, ``st.button``, ``st.caption`` …) yields a
    spy we can introspect later.

    ``agentic_mode`` seeds the persisted-preference key
    (``st.session_state["agentic_mode"]``). ``checkbox_state`` seeds the
    dialog checkbox widget-state key
    (``st.session_state["agentic_mode_dialog_checkbox"]``) which the
    sidebar gate prefers when present.
    """
    stub = MagicMock()
    session = {}
    if agentic_mode is not None:
        session["agentic_mode"] = agentic_mode
    if checkbox_state is not None:
        session["agentic_mode_dialog_checkbox"] = checkbox_state
    stub.session_state = session
    # Make ``with st.sidebar.container():`` work as a context manager.
    stub.sidebar.container.return_value.__enter__ = MagicMock(return_value=MagicMock())
    stub.sidebar.container.return_value.__exit__ = MagicMock(return_value=False)
    # Same for ``st.container()`` (used in unrelated paths but safe to wire).
    stub.container.return_value.__enter__ = MagicMock(return_value=MagicMock())
    stub.container.return_value.__exit__ = MagicMock(return_value=False)
    stub.button = MagicMock(return_value=False)
    return stub


# ---------------------------------------------------------------------------
# Reset Agent sidebar visibility tests
# ---------------------------------------------------------------------------


def test_reset_agent_hidden_when_agentic_mode_off():
    """``agentic_mode = False`` must short-circuit before touching
    ``st.sidebar.container``."""
    from views import chat_bot

    stub = _make_st_stub(agentic_mode=False)
    with patch.object(chat_bot, "st", stub):
        chat_bot._render_reset_agent_sidebar()

    stub.sidebar.container.assert_not_called()


def test_reset_agent_visible_when_agentic_mode_on():
    """``agentic_mode = True`` must reach ``st.sidebar.container``."""
    from views import chat_bot

    stub = _make_st_stub(agentic_mode=True)
    with (
        patch.object(chat_bot, "st", stub),
        patch.object(chat_bot.time, "time", return_value=1000.0),
    ):
        chat_bot._render_reset_agent_sidebar()

    stub.sidebar.container.assert_called_once()


def test_reset_agent_visible_when_agentic_mode_absent():
    """Fresh session with no ``agentic_mode`` preference defaults to
    True so the button still renders."""
    from views import chat_bot

    stub = _make_st_stub(agentic_mode=None)
    with (
        patch.object(chat_bot, "st", stub),
        patch.object(chat_bot.time, "time", return_value=1000.0),
    ):
        chat_bot._render_reset_agent_sidebar()

    stub.sidebar.container.assert_called_once()


# ---------------------------------------------------------------------------
# Live-update tests — the sidebar reads the dialog checkbox's widget
# state so toggling reflects on the same rerun (no page refresh needed).
# ---------------------------------------------------------------------------


def test_sidebar_prefers_checkbox_widget_state_when_set():
    """Even if ``agentic_mode`` persists as True, an unchecked dialog
    checkbox (``agentic_mode_dialog_checkbox = False``) must hide the
    sidebar on the same rerun. This is the bug fix: toggling the
    checkbox in an open dialog updates the sidebar live."""
    from views import chat_bot

    stub = _make_st_stub(agentic_mode=True, checkbox_state=False)
    with patch.object(chat_bot, "st", stub):
        chat_bot._render_reset_agent_sidebar()

    stub.sidebar.container.assert_not_called()


def test_sidebar_prefers_checkbox_widget_state_when_on():
    """``agentic_mode_dialog_checkbox = True`` shows the sidebar even
    when ``agentic_mode`` mirror hasn't been refreshed yet (e.g., the
    DB hasn't been re-read this rerun)."""
    from views import chat_bot

    stub = _make_st_stub(agentic_mode=False, checkbox_state=True)
    with (
        patch.object(chat_bot, "st", stub),
        patch.object(chat_bot.time, "time", return_value=1000.0),
    ):
        chat_bot._render_reset_agent_sidebar()

    stub.sidebar.container.assert_called_once()


def test_sidebar_falls_back_to_agentic_mode_when_checkbox_absent():
    """Before the Settings dialog has been opened in this session, the
    checkbox widget key doesn't exist. Fall back to the persisted
    ``agentic_mode`` value."""
    from views import chat_bot

    stub = _make_st_stub(agentic_mode=False, checkbox_state=None)
    with patch.object(chat_bot, "st", stub):
        chat_bot._render_reset_agent_sidebar()

    stub.sidebar.container.assert_not_called()


# ---------------------------------------------------------------------------
# Settings dialog section order test
# ---------------------------------------------------------------------------


def test_settings_dialog_section_order_is_llm_then_agentic_then_display():
    """``_settings_dialog_body`` renders LLM → Agentic → Display in that
    order. We invoke the body directly (not the ``@st.dialog``-decorated
    wrapper) to avoid Streamlit's dialog-open machinery at import time."""
    from views import chat_bot

    call_order: list[str] = []

    def _record_llm():
        call_order.append("llm")

    def _record_agentic():
        call_order.append("agentic")

    def _record_display():
        call_order.append("display")

    st_stub = MagicMock()
    st_stub.button = MagicMock(return_value=False)

    with (
        patch.object(chat_bot, "st", st_stub),
        patch.object(chat_bot, "get_vn", return_value=None),
        patch.object(chat_bot, "_render_llm_section", side_effect=_record_llm),
        patch.object(chat_bot, "_render_agentic_section", side_effect=_record_agentic),
        patch.object(chat_bot, "_render_display_form", side_effect=_record_display),
    ):
        chat_bot._settings_dialog_body()

    assert call_order == ["llm", "agentic", "display"], (
        f"Settings dialog body must render sections in LLM → Agentic → Display order; got {call_order}"
    )

    subheader_labels = [c.args[0] for c in st_stub.subheader.call_args_list if c.args]
    assert subheader_labels == ["LLM", "Agentic", "Display"], (
        f"Settings dialog subheaders must render in LLM → Agentic → Display order; got {subheader_labels}"
    )
