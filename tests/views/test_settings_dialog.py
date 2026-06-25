"""Unit tests for the Settings dialog's display form (Epic #222 / Feature #224).

Verifies the ``show_thinking_process`` checkbox is wired into
``_render_display_form`` and that submitting the form copies the value into
``st.session_state`` before persisting via ``save_user_settings``. The actual
persistence layer is covered by ``tests/orm/test_user_show_thinking_process.py``.
"""

from unittest.mock import MagicMock


def _build_fake_st(submit: bool, checkbox_overrides: dict | None = None) -> MagicMock:
    """Return a ``MagicMock`` shaped to stand in for the ``streamlit`` module
    inside ``_render_display_form``.

    ``checkbox_overrides`` maps checkbox labels to the value that
    ``st.checkbox`` should return for that label — this is what the form
    bindings (``form_show_*``) capture inside the function. Any label not in
    the dict falls back to the ``value=`` kwarg the caller passed in.
    """
    fake_st = MagicMock()
    fake_st.session_state.get = MagicMock(side_effect=lambda key, default=None: default)

    overrides = checkbox_overrides or {}

    def checkbox_side_effect(label, *args, value=None, help=None, **_kwargs):
        return overrides.get(label, value)

    fake_st.checkbox.side_effect = checkbox_side_effect

    # ``st.form`` is used as a context manager — make the MagicMock return a
    # context manager that yields ``None``.
    fake_st.form.return_value.__enter__ = MagicMock(return_value=None)
    fake_st.form.return_value.__exit__ = MagicMock(return_value=None)

    fake_st.form_submit_button.return_value = submit
    return fake_st


def test_render_display_form_includes_show_thinking_process_checkbox(monkeypatch):
    """The Display section must surface the new checkbox so users can toggle
    the thinking-process display from the Settings dialog."""
    import views.chat_bot as page

    fake_st = _build_fake_st(submit=False)
    monkeypatch.setattr(page, "st", fake_st)

    page._render_display_form()

    labels = [call.args[0] for call in fake_st.checkbox.call_args_list]
    assert "Show AI thinking process" in labels, (
        f"Settings dialog is missing the show_thinking_process checkbox; observed checkboxes={labels}"
    )

    # The default value must come from session_state with a False fallback —
    # matches the Epic acceptance criteria that the toggle defaults to hidden.
    keys_queried = [c.args[0] for c in fake_st.session_state.get.call_args_list if c.args]
    assert "show_thinking_process" in keys_queried, (
        "Checkbox default must be read from st.session_state.get('show_thinking_process', False) "
        "so toggled-and-saved state survives a rerun."
    )
    show_thinking_get_calls = [
        c for c in fake_st.session_state.get.call_args_list if c.args and c.args[0] == "show_thinking_process"
    ]
    assert any((c.args[1] is False) for c in show_thinking_get_calls if len(c.args) >= 2), (
        "Default value passed to st.session_state.get must be False per Epic #222 acceptance criteria."
    )


def test_render_display_form_submit_persists_show_thinking_process(monkeypatch):
    """Clicking Save must copy the form value to ``st.session_state`` AND
    call ``save_user_settings`` so the DB column is written."""
    import views.chat_bot as page

    fake_st = _build_fake_st(
        submit=True,
        checkbox_overrides={"Show AI thinking process": True},
    )
    monkeypatch.setattr(page, "st", fake_st)

    save_mock = MagicMock()
    monkeypatch.setattr(page, "save_user_settings", save_mock)

    page._render_display_form()

    assert fake_st.session_state.show_thinking_process is True, (
        "Submit branch must copy form_show_thinking_process into st.session_state.show_thinking_process."
    )
    save_mock.assert_called_once()
