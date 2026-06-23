"""Unit tests for the gated transient thinking-bubble (Epic #222 / Feature #226).

Covers the agentic render path's `ThinkingDeltaEvent` consumer in
``agent.runtime._render_event``. The persistent ``MessageType.THINKING``
message added on ``ThinkingCompletedEvent`` renders via
``utils.chat_bot_helper._render_thinking`` and is gated there (Feature
#225); this file does not re-cover the persistent-render branch.
"""

from unittest.mock import MagicMock, patch

from agent.state import ThinkingCompletedEvent, ThinkingDeltaEvent


def _fake_st(show_thinking: bool | None) -> MagicMock:
    """Build a MagicMock standing in for ``streamlit`` inside agent.runtime.

    ``show_thinking`` controls what ``st.session_state.get("show_thinking_process",
    False)`` returns; pass ``None`` to simulate a session where the key is
    missing entirely (default-off per Epic #222 acceptance criteria).
    ``st.empty()``, ``st.container()``, and ``st.chat_message()`` are
    configured as context managers so the ``with`` blocks in the production
    code don't crash when entered.
    """
    fake_st = MagicMock()
    overrides: dict = {} if show_thinking is None else {"show_thinking_process": show_thinking}
    fake_st.session_state.get = MagicMock(side_effect=lambda key, default=None: overrides.get(key, default))

    outer = MagicMock()
    container = MagicMock()
    container.__enter__ = MagicMock(return_value=None)
    container.__exit__ = MagicMock(return_value=None)
    outer.container = MagicMock(return_value=container)
    fake_st.empty = MagicMock(return_value=outer)

    chat_message = MagicMock()
    chat_message.__enter__ = MagicMock(return_value=None)
    chat_message.__exit__ = MagicMock(return_value=None)
    fake_st.chat_message = MagicMock(return_value=chat_message)
    return fake_st


def test_thinking_delta_off_renders_static_caption_once(monkeypatch):
    """Toggle off → one static caption, no streaming header in markdown."""
    import agent.runtime as runtime

    fake_st = _fake_st(show_thinking=False)
    monkeypatch.setattr(runtime, "st", fake_st)

    state: dict = {"thinking": {}, "text": {}}
    runtime._render_event(ThinkingDeltaEvent(delta="thinking content", turn_index=1), state)

    fake_st.caption.assert_called_once_with("🤔 Thinking…")
    # The "🤔 **Thinking...**" streaming header is rendered via st.markdown
    # in _write_slot — when off, that path must NOT be taken.
    streaming_markdown_calls = [
        c for c in fake_st.markdown.call_args_list if "Thinking..." in (c.args[0] if c.args else "")
    ]
    assert streaming_markdown_calls == [], (
        f"Expected no streaming-header markdown calls when toggle is off; saw {streaming_markdown_calls}"
    )
    slot = state["thinking"][1]
    assert slot["static_rendered"] is True
    assert slot["buf"] == "thinking content", "Buffer must still accumulate for parity with on-path."


def test_thinking_delta_off_is_idempotent_on_repeats(monkeypatch):
    """Toggle off + multiple deltas for same turn_index → caption rendered ONCE."""
    import agent.runtime as runtime

    fake_st = _fake_st(show_thinking=False)
    monkeypatch.setattr(runtime, "st", fake_st)

    state: dict = {"thinking": {}, "text": {}}
    for i, chunk in enumerate(["chunk-1", "chunk-2", "chunk-3"]):
        runtime._render_event(ThinkingDeltaEvent(delta=chunk, turn_index=1), state)

    fake_st.caption.assert_called_once_with("🤔 Thinking…")
    slot = state["thinking"][1]
    assert slot["static_rendered"] is True
    assert slot["buf"] == "chunk-1chunk-2chunk-3"


def test_thinking_delta_on_renders_streaming_header(monkeypatch):
    """Toggle on → existing _write_slot behavior (streaming header markdown)."""
    import agent.runtime as runtime

    fake_st = _fake_st(show_thinking=True)
    monkeypatch.setattr(runtime, "st", fake_st)

    state: dict = {"thinking": {}, "text": {}}
    runtime._render_event(ThinkingDeltaEvent(delta="streamed chunk", turn_index=1), state)

    streaming_calls = [c for c in fake_st.markdown.call_args_list if "Thinking..." in (c.args[0] if c.args else "")]
    assert len(streaming_calls) == 1, f"Expected one streaming markdown when toggle is on, saw {streaming_calls}"
    assert "streamed chunk" in streaming_calls[0].args[0], "Streamed delta must appear in the markdown body."
    # The static off-path caption must NOT have been called.
    fake_st.caption.assert_not_called()


def test_thinking_delta_off_defaults_when_pref_missing(monkeypatch):
    """No `show_thinking_process` key in session state → behaves as off (default per Epic)."""
    import agent.runtime as runtime

    fake_st = _fake_st(show_thinking=None)
    monkeypatch.setattr(runtime, "st", fake_st)

    state: dict = {"thinking": {}, "text": {}}
    runtime._render_event(ThinkingDeltaEvent(delta="content", turn_index=1), state)

    fake_st.caption.assert_called_once_with("🤔 Thinking…")


def test_thinking_completed_empties_slot_and_persists_regardless_of_toggle(monkeypatch):
    """ThinkingCompletedEvent unchanged: always empties the slot AND adds
    a persistent MessageType.THINKING message. Toggle state does not gate
    persistence — only rendering of the persisted message (handled by #225)."""
    import agent.runtime as runtime

    for show in (False, True):
        fake_st = _fake_st(show_thinking=show)
        monkeypatch.setattr(runtime, "st", fake_st)

        state: dict = {"thinking": {}, "text": {}}
        # Seed a prior delta so the slot exists.
        runtime._render_event(ThinkingDeltaEvent(delta="content", turn_index=1), state)
        outer = state["thinking"][1]["outer"]

        with patch("utils.chat_bot_helper.add_message") as add_message_mock:
            runtime._render_event(
                ThinkingCompletedEvent(text="content", elapsed_ms=500, turn_index=1),
                state,
            )

        outer.empty.assert_called_once()
        assert 1 not in state["thinking"], "Completed turn must be removed from the state dict."
        add_message_mock.assert_called_once()
        persisted_msg = add_message_mock.call_args.args[0]
        from utils.enums import MessageType

        assert persisted_msg.type == MessageType.THINKING.value, (
            f"Persistent message must be MessageType.THINKING regardless of toggle, got {persisted_msg.type}"
        )
