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


def _make_st_stub(*, agentic_mode):
    """MagicMock-based stub for ``views.chat_bot.st``.

    ``session_state`` is a real dict (so ``get`` / ``setdefault`` work),
    everything else is a MagicMock so any attribute access (e.g.
    ``st.sidebar.container``, ``st.button``, ``st.caption`` …) yields a
    spy we can introspect later.

    ``agentic_mode`` seeds the persisted-preference key
    (``st.session_state["agentic_mode"]``) which the sidebar gate reads
    directly. Commit-on-close semantics: the gate only reads this key
    (NOT the dialog checkbox's widget-state key), so the sidebar
    doesn't update live while the dialog is open — it picks up the
    new value on the next rerun (which fires when the dialog closes
    and ``set_user_preferences_in_session_state`` re-reads from DB).
    """
    stub = MagicMock()
    stub.session_state = {} if agentic_mode is None else {"agentic_mode": agentic_mode}
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
# Commit-on-close guarantee — the sidebar ignores the dialog checkbox's
# widget-state key. It reads only ``agentic_mode``, which gets refreshed
# from the DB on the next rerun (which fires when the dialog closes).
# ---------------------------------------------------------------------------


def test_sidebar_does_not_read_dialog_checkbox_widget_state():
    """If the dialog checkbox's widget state disagrees with the
    persisted ``agentic_mode`` (e.g., user toggled checkbox in an open
    dialog but the rerun hasn't yet propagated through DB), the
    sidebar must reflect the PERSISTED value, not the in-flight
    checkbox state. This enforces commit-on-close: sidebar updates
    only after the dialog closes and ``set_user_preferences_in_session_state``
    refreshes ``agentic_mode`` from the DB."""
    from views import chat_bot

    # Persisted preference says True (sidebar should be visible). The
    # in-flight checkbox state says False (user toggled but hasn't
    # closed the dialog yet).
    stub = _make_st_stub(agentic_mode=True)
    stub.session_state["agentic_mode_dialog_checkbox"] = False
    with (
        patch.object(chat_bot, "st", stub),
        patch.object(chat_bot.time, "time", return_value=1000.0),
    ):
        chat_bot._render_reset_agent_sidebar()

    # Sidebar still renders because the persisted value is True, even
    # though the in-flight checkbox state disagrees.
    stub.sidebar.container.assert_called_once()


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
