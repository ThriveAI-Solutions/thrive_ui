"""View-layer tests for the By-Patient audit sub-tab (Epic #190, Phase 3).

Exercises the tab body in ``views.admin_audit_by_patient`` against a small
streamlit stub. Asserts:
  * autocomplete option formatting,
  * paste-textarea parsing,
  * source_ids wiring through to the Phase 1 data layer,
  * Phase 2 helpers being called with ``key_prefix="by_patient"`` so the
    new tab's widget state doesn't collide with the Queries tab.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from orm.models import RoleTypeEnum


def _make_options() -> list[dict]:
    return [
        {
            "source_id": "pat-1",
            "display_name": "Alice Anders",
            "last_touched_at": datetime(2026, 6, 1, 12, 0, 0),
            "access_count": 3,
        },
        {
            "source_id": "pat-2",
            "display_name": None,
            "last_touched_at": datetime(2026, 6, 1, 11, 0, 0),
            "access_count": 1,
        },
    ]


def _make_stub(
    *,
    multiselect_returns=None,
    text_area_returns=None,
    radio_value="Grouped",
    selectbox_returns=None,
):
    captured_multiselect: list[dict] = []
    captured_data_layer_calls: list[dict] = []
    multiselect_returns = multiselect_returns or {}
    text_area_returns = text_area_returns or {}
    selectbox_returns = selectbox_returns or {}

    class _Stub:
        def __init__(self):
            self.session_state = {"user_role": RoleTypeEnum.ADMIN.value}
            self.secrets = {"agent_logging": {"mode": "full"}}
            self.column_config = MagicMock()

        def multiselect(self, label, options=None, key=None, format_func=None, **kw):
            opts_list = list(options or [])
            captured_multiselect.append(
                {
                    "label": label,
                    "options": opts_list,
                    "key": key,
                    "format_func": format_func,
                    "kwargs": kw,
                }
            )
            val = multiselect_returns.get(key, [])
            self.session_state.setdefault(key, val)
            self.session_state[key] = val
            return val

        def text_area(self, label, key=None, **kw):
            val = text_area_returns.get(key, "")
            self.session_state.setdefault(key, val)
            return val

        def text_input(self, *_a, key=None, **_kw):
            self.session_state.setdefault(key, "")
            return ""

        def selectbox(self, *_a, options=None, index=0, key=None, **_kw):
            opts = list(options or [])
            val = selectbox_returns.get(key, opts[index] if opts else None)
            self.session_state.setdefault(key, val)
            return val

        def number_input(self, *_a, key=None, **_kw):
            self.session_state.setdefault(key, 1)
            return 1

        def radio(self, *_a, options=None, index=0, key=None, **_kw):
            self.session_state.setdefault(key, radio_value)
            return radio_value

        def columns(self, spec):
            n = spec if isinstance(spec, int) else len(spec)
            return [MagicMock() for _ in range(n)]

        def data_editor(self, df, **_kw):
            return df

        def dataframe(self, *_a, **_kw):
            return MagicMock()

        def expander(self, *_a, **_kw):
            return MagicMock()

        def info(self, *_a, **_kw):
            pass

        def button(self, *_a, key=None, **_kw):
            return False

        def caption(self, *_a, **_kw):
            pass

        def markdown(self, *_a, **_kw):
            pass

        def write(self, *_a, **_kw):
            pass

        def divider(self):
            pass

        def warning(self, *_a, **_kw):
            pass

        def error(self, *_a, **_kw):
            pass

        def download_button(self, *_a, **_kw):
            pass

        def code(self, *_a, **_kw):
            pass

        def rerun(self):
            pass

        def cache_data(self, *_a, **_kw):
            def _decorator(fn):
                return fn

            return _decorator

        def dialog(self, _title):
            def _decorator(fn):
                return fn

            return _decorator

    return _Stub(), {
        "multiselect": captured_multiselect,
        "data_layer_calls": captured_data_layer_calls,
    }


# ---------------------------------------------------------------------------
# (1) Empty selection → info card, no data-layer call
# ---------------------------------------------------------------------------


def test_no_patients_selected_renders_info_card_and_skips_data_layer():
    from views import admin_audit_by_patient

    stub, _ = _make_stub()
    page_calls: list[dict] = []

    def _fake_page(filters, page=1, page_size=50):
        page_calls.append(filters)
        return {"items": [], "total": 0}

    with (
        patch.object(admin_audit_by_patient, "st", stub),
        patch.object(admin_audit_by_patient, "_cached_patient_options", return_value=_make_options()),
        patch.object(admin_audit_by_patient._q, "get_per_query_audit_page", side_effect=_fake_page),
    ):
        admin_audit_by_patient._render_by_patient_tab(30)

    # No selection → no data-layer call.
    assert page_calls == []


# ---------------------------------------------------------------------------
# (2) format_func renders "<display> (<source_id>)" / "<source_id>"
# ---------------------------------------------------------------------------


def test_format_option_with_and_without_display_name():
    from views.admin_audit_by_patient import _format_option

    assert _format_option({"source_id": "pat-1", "display_name": "Alice"}) == "Alice (pat-1)"
    assert _format_option({"source_id": "pat-2", "display_name": None}) == "pat-2"


# ---------------------------------------------------------------------------
# (3) Selected patient(s) plumbed to the data layer via source_ids
# ---------------------------------------------------------------------------


def test_selected_patients_reach_data_layer_via_source_ids():
    from views import admin_audit_by_patient

    options = _make_options()
    stub, captures = _make_stub(
        multiselect_returns={"by_patient_picker": [options[0]]},
    )

    page_calls: list[dict] = []

    def _fake_page(filters, page=1, page_size=50):
        page_calls.append(filters)
        return {"items": [], "total": 0}

    with (
        patch.object(admin_audit_by_patient, "st", stub),
        patch.object(admin_audit_by_patient, "_cached_patient_options", return_value=options),
        patch.object(admin_audit_by_patient._q, "get_per_query_audit_page", side_effect=_fake_page),
    ):
        admin_audit_by_patient._render_by_patient_tab(30)

    assert page_calls, "data layer must be called once a patient is selected"
    assert page_calls[-1]["source_ids"] == ["pat-1"]
    # Pipeline is pinned to agentic for the By-Patient tab.
    assert page_calls[-1]["pipelines"] == ["agentic"]


# ---------------------------------------------------------------------------
# (4) Paste textarea contents merge into source_ids (deduped, trimmed)
# ---------------------------------------------------------------------------


def test_paste_textarea_appends_source_ids_deduped_and_trimmed():
    from views import admin_audit_by_patient

    options = _make_options()
    stub, _ = _make_stub(
        multiselect_returns={"by_patient_picker": [options[0]]},
        text_area_returns={"by_patient_paste": "  pat-99  \npat-1\n\n  pat-100\n"},
    )

    page_calls: list[dict] = []

    def _fake_page(filters, page=1, page_size=50):
        page_calls.append(filters)
        return {"items": [], "total": 0}

    with (
        patch.object(admin_audit_by_patient, "st", stub),
        patch.object(admin_audit_by_patient, "_cached_patient_options", return_value=options),
        patch.object(admin_audit_by_patient._q, "get_per_query_audit_page", side_effect=_fake_page),
    ):
        admin_audit_by_patient._render_by_patient_tab(30)

    assert page_calls[-1]["source_ids"] == ["pat-1", "pat-99", "pat-100"]


# ---------------------------------------------------------------------------
# (5) Widget keys use the by_patient prefix
# ---------------------------------------------------------------------------


def test_tab_uses_by_patient_key_prefix_for_widgets():
    from views import admin_audit_by_patient

    options = _make_options()
    stub, captures = _make_stub(
        multiselect_returns={"by_patient_picker": [options[0]]},
    )

    with (
        patch.object(admin_audit_by_patient, "st", stub),
        patch.object(admin_audit_by_patient, "_cached_patient_options", return_value=options),
        patch.object(
            admin_audit_by_patient._q,
            "get_per_query_audit_page",
            return_value={"items": [], "total": 0},
        ),
    ):
        admin_audit_by_patient._render_by_patient_tab(30)

    picker_chip = next(
        (m for m in captures["multiselect"] if m["key"] == "by_patient_picker"),
        None,
    )
    assert picker_chip is not None
    # No widget key from this tab should collide with the Queries tab's
    # ``queries_*`` keys.
    assert all(not str(k).startswith("queries_") for k in stub.session_state.keys())


# ---------------------------------------------------------------------------
# (6) Parser tests
# ---------------------------------------------------------------------------


def test_parse_paste_handles_blank_and_dedup():
    from views.admin_audit_by_patient import _parse_paste

    assert _parse_paste(None) == []
    assert _parse_paste("") == []
    assert _parse_paste("\n\n   \n") == []
    assert _parse_paste("a\nb\na\n  c  \nb") == ["a", "b", "c"]
