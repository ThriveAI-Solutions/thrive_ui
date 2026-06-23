"""Unit tests for the gated `_render_thinking` renderer (Epic #222 / Feature #225).

Covers the Vanna single-shot render path. When ``show_thinking_process`` is
off the renderer collapses to a compact placeholder; when on it renders the
full expander with the message content. The live-streaming block at
``chat_bot_helper.py:~1450`` is exercised via manual smoke per the spec
(Ollama runtime required) — only the deterministic renderer path is
unit-tested here.
"""

from decimal import Decimal
from unittest.mock import MagicMock

from orm.models import Message
from utils.enums import MessageType, RoleType


def _make_thinking_message(content: str = "internal monologue", elapsed: float | None = 0.42) -> Message:
    msg = Message(
        role=RoleType.ASSISTANT,
        content=content,
        type=MessageType.THINKING,
        query="",
        question="",
        elapsed_time=Decimal(str(elapsed)) if elapsed is not None else None,
    )
    return msg


def _fake_st(session_state_overrides: dict | None = None) -> MagicMock:
    """A MagicMock standing in for the ``streamlit`` module inside chat_bot_helper.

    ``st.session_state.get`` returns values from ``session_state_overrides``;
    keys absent from the dict fall through to the default the caller passed.
    ``st.expander`` is configured as a context manager so the `with` block in
    ``_render_thinking`` doesn't crash when entered.
    """
    overrides = session_state_overrides or {}
    fake_st = MagicMock()
    fake_st.session_state.get = MagicMock(side_effect=lambda key, default=None: overrides.get(key, default))
    fake_st.expander.return_value.__enter__ = MagicMock(return_value=None)
    fake_st.expander.return_value.__exit__ = MagicMock(return_value=None)
    return fake_st


def test_render_thinking_off_renders_caption_only(monkeypatch):
    """When the user has the toggle off, the renderer must show a single
    compact caption and skip the expander entirely."""
    import utils.chat_bot_helper as helper

    fake_st = _fake_st({"show_thinking_process": False})
    monkeypatch.setattr(helper, "st", fake_st)

    helper._render_thinking(_make_thinking_message(), 0)

    assert fake_st.caption.call_count == 1, "Expected a single caption call for the placeholder."
    caption_text = fake_st.caption.call_args.args[0]
    assert "Thinking" in caption_text, f"Caption text should mention 'Thinking', got: {caption_text!r}"
    fake_st.expander.assert_not_called()
    fake_st.markdown.assert_not_called()


def test_render_thinking_on_renders_expander(monkeypatch):
    """When the toggle is on, render the full expander with the message
    content — preserves pre-toggle behavior exactly."""
    import utils.chat_bot_helper as helper

    fake_st = _fake_st({"show_thinking_process": True, "show_elapsed_time": True})
    monkeypatch.setattr(helper, "st", fake_st)

    msg = _make_thinking_message(content="my full chain-of-thought")
    helper._render_thinking(msg, 0)

    fake_st.expander.assert_called_once_with("🤔 AI Thinking Process", expanded=False)
    fake_st.markdown.assert_called_once_with("my full chain-of-thought")
    # `show_elapsed_time` is on AND `elapsed_time` is not None → caption with the timing.
    timing_captions = [c for c in fake_st.caption.call_args_list if "Thinking time" in c.args[0]]
    assert len(timing_captions) == 1, "Expected exactly one 'Thinking time' caption when both flags are on."


def test_render_thinking_defaults_to_off_when_pref_missing(monkeypatch):
    """If `show_thinking_process` isn't set in session state (e.g. a brand-new
    session before settings load), the default must be off per Epic #222
    acceptance criteria."""
    import utils.chat_bot_helper as helper

    fake_st = _fake_st({})  # no key for show_thinking_process
    monkeypatch.setattr(helper, "st", fake_st)

    helper._render_thinking(_make_thinking_message(), 0)

    assert fake_st.caption.call_count == 1
    fake_st.expander.assert_not_called()
