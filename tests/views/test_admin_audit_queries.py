"""View-layer tests for the Queries audit sub-tab (Epic #190, Phase 2).

The tab body in ``views.admin_audit_queries`` is exercised against a small
streamlit stub so we can assert column shape, mode banners, and role-gated
SQL visibility without spinning up the full Streamlit runtime.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from orm.models import RoleTypeEnum


# ---------------------------------------------------------------------------
# Per-query item factory
# ---------------------------------------------------------------------------


def _make_legacy_item(*, user_message_id: int = 100, **overrides) -> dict:
    base = {
        "asked_at": datetime(2026, 6, 1, 12, 0, 0),
        "user_id": 7,
        "username": "alice",
        "organization": "Acme",
        "question": "How many patients today?",
        "user_message_id": user_message_id,
        "scope": "Legacy/Unknown",
        "pipeline": "legacy",
        "run_id": None,
        "logging_mode": None,
        "tool_call_id": None,
        "tool_name": None,
        "call_index": None,
        "sql_statements": ["SELECT count(*) FROM patient"],
        "non_sql_summary": None,
        "elapsed_ms": 1250,
        "success": True,
        "error": None,
        "patients_touched": [],
    }
    base.update(overrides)
    return base


def _make_agentic_item(*, user_message_id: int = 200, tool_call_id: int = 1, **overrides) -> dict:
    base = {
        "asked_at": datetime(2026, 6, 1, 12, 5, 0),
        "user_id": 7,
        "username": "alice",
        "organization": "Acme",
        "question": "Patient X labs",
        "user_message_id": user_message_id,
        "scope": "Patient",
        "pipeline": "agentic",
        "run_id": "run-abc",
        "logging_mode": "full",
        "tool_call_id": tool_call_id,
        "tool_name": "run_sql",
        "call_index": 0,
        "sql_statements": ["SELECT * FROM lab WHERE patient_id = 'pat-1'"],
        "non_sql_summary": None,
        "elapsed_ms": 87,
        "success": True,
        "error": None,
        "patients_touched": [{"source_id": "pat-1", "display_name": "Alice A"}],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# (1)–(6) Row-component shape tests
# ---------------------------------------------------------------------------


def test_row_dict_legacy_item():
    from views.admin_audit_queries import _per_query_row_to_table_dict

    row = _per_query_row_to_table_dict(_make_legacy_item())
    assert row["Pipeline"] == "legacy"
    assert row["Tool"] == "(legacy SQL)"
    assert row["SQL"].startswith("SELECT count(*) FROM patient")
    assert row["Patient(s)"] == ""
    assert row["Status"] == "✅ Success"
    assert row["Elapsed (ms)"] == 1250


def test_row_dict_agentic_multi_sql_shows_statement_count():
    from views.admin_audit_queries import _per_query_row_to_table_dict

    item = _make_agentic_item(
        sql_statements=["SELECT a FROM t1", "SELECT b FROM t2", "SELECT c FROM t3"],
        tool_name="run_sql",
    )
    row = _per_query_row_to_table_dict(item)
    assert row["Tool"] == "run_sql"
    assert row["SQL"].startswith("3 statements:")
    assert "SELECT a FROM t1" in row["SQL"]


def test_row_dict_agentic_non_sql_tool_call_uses_sentinel():
    from views.admin_audit_queries import _per_query_row_to_table_dict

    item = _make_agentic_item(
        tool_name="search_knowledge_base",
        sql_statements=[],
        non_sql_summary="3 KB hits",
        call_index=2,
    )
    row = _per_query_row_to_table_dict(item)
    assert row["Tool"] == "search_knowledge_base"
    assert row["SQL"] == "(no SQL — tool result)"


def test_row_dict_disabled_logging_mode_uses_disabled_sentinel():
    from views.admin_audit_queries import _per_query_row_to_table_dict

    item = _make_agentic_item(
        logging_mode="disabled",
        sql_statements=[],
        non_sql_summary="(logging disabled)",
    )
    row = _per_query_row_to_table_dict(item)
    assert row["SQL"] == "(logging disabled)"


def test_row_dict_scrubbed_mode_prefixes_sql_preview():
    from views.admin_audit_queries import _per_query_row_to_table_dict

    item = _make_agentic_item(
        logging_mode="scrubbed",
        sql_statements=["SELECT * FROM t WHERE id = '<hash:abc>'"],
    )
    row = _per_query_row_to_table_dict(item)
    assert row["SQL"].startswith("(scrubbed) ")


def test_row_dict_patient_touched_joined_by_comma():
    from views.admin_audit_queries import _per_query_row_to_table_dict

    item = _make_agentic_item(
        patients_touched=[
            {"source_id": "pat-1", "display_name": "Alice"},
            {"source_id": "pat-2", "display_name": "Bob"},
        ]
    )
    row = _per_query_row_to_table_dict(item)
    assert "pat-1" in row["Patient(s)"]
    assert "pat-2" in row["Patient(s)"]
    assert "," in row["Patient(s)"]


# ---------------------------------------------------------------------------
# (7)–(9) Status derivation
# ---------------------------------------------------------------------------


def test_derive_status_success():
    from views.admin_audit_queries import _derive_status

    assert _derive_status(_make_agentic_item(success=True)) == "✅ Success"


def test_derive_status_error_takes_priority_over_payload():
    from views.admin_audit_queries import _derive_status

    item = _make_agentic_item(success=False, error="boom", sql_statements=["SELECT 1"])
    assert _derive_status(item) == "❌ Error"


def test_derive_status_empty_when_no_payload_no_error():
    from views.admin_audit_queries import _derive_status

    item = _make_agentic_item(
        success=None,
        sql_statements=[],
        non_sql_summary=None,
        error=None,
    )
    assert _derive_status(item) == "⚪ Empty"


# ---------------------------------------------------------------------------
# (10) Grouping by user_message_id
# ---------------------------------------------------------------------------


def test_group_items_by_question_clusters_by_message_id_in_order():
    from views.admin_audit_queries import _group_items_by_question

    items = [
        _make_agentic_item(user_message_id=200, tool_call_id=1, elapsed_ms=10, call_index=0),
        _make_agentic_item(user_message_id=200, tool_call_id=2, elapsed_ms=15, call_index=1),
        _make_legacy_item(user_message_id=100),  # 1250 ms
        _make_agentic_item(user_message_id=200, tool_call_id=3, elapsed_ms=5, call_index=2),
        _make_legacy_item(user_message_id=101, asked_at=datetime(2026, 6, 1, 11, 0, 0)),
    ]

    groups = _group_items_by_question(items)
    # Three distinct questions, order preserved.
    assert [hdr["user_message_id"] for hdr, _ in groups] == [200, 100, 101]
    # First group has 3 units; total_elapsed_ms sums.
    hdr200, units200 = groups[0]
    assert hdr200["query_count"] == 3
    assert hdr200["total_elapsed_ms"] == 30
    assert len(units200) == 3


def test_group_inherits_most_alarming_logging_mode():
    from views.admin_audit_queries import _group_items_by_question

    items = [
        _make_agentic_item(user_message_id=300, tool_call_id=1, logging_mode="full"),
        _make_agentic_item(user_message_id=300, tool_call_id=2, logging_mode="scrubbed"),
        _make_agentic_item(user_message_id=300, tool_call_id=3, logging_mode="full"),
    ]

    groups = _group_items_by_question(items)
    assert groups[0][0]["logging_mode"] == "scrubbed"


# ---------------------------------------------------------------------------
# (11)–(15) Detail dialog body rendering (role gating + mode banners)
# ---------------------------------------------------------------------------


class _RecordingStub:
    """Minimal Streamlit stub recording the calls dialog bodies make."""

    def __init__(self, *, role: int = RoleTypeEnum.ADMIN.value, button_returns: dict | None = None):
        self.session_state = {"user_role": role}
        self.code_calls: list[tuple[str, dict]] = []
        self.warning_calls: list[str] = []
        self.markdown_calls: list[str] = []
        self.write_calls: list[str] = []
        self.caption_calls: list[str] = []
        self.button_calls: list[dict] = []
        self.switch_page_calls: list[str] = []
        self.button_returns = dict(button_returns or {})
        self.components = MagicMock()
        self.components.v1 = MagicMock()
        self.components.v1.html = MagicMock()

    def markdown(self, text, *_a, **_kw):
        self.markdown_calls.append(text)

    def write(self, text, *_a, **_kw):
        self.write_calls.append(str(text))

    def warning(self, text, *_a, **_kw):
        self.warning_calls.append(str(text))

    def code(self, text, **kw):
        self.code_calls.append((str(text), kw))

    def caption(self, text, *_a, **_kw):
        self.caption_calls.append(str(text))

    def columns(self, n):
        return [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]

    def divider(self):
        pass

    def button(self, label, key=None, **_kw):
        self.button_calls.append({"label": label, "key": key})
        return self.button_returns.get(key, False)

    def switch_page(self, target):
        self.switch_page_calls.append(target)


def test_dialog_body_renders_sql_via_st_code_when_role_allowed():
    from views import admin_audit_queries

    stub = _RecordingStub(role=RoleTypeEnum.ADMIN.value)
    with patch.object(admin_audit_queries, "st", stub):
        admin_audit_queries._render_query_detail_dialog_body(_make_agentic_item())

    sql_texts = [c[0] for c in stub.code_calls if c[1].get("language") == "sql"]
    assert any("SELECT * FROM lab" in t for t in sql_texts)


def test_dialog_body_hides_sql_when_role_cannot_see_query_details():
    from views import admin_audit_queries

    stub = _RecordingStub(role=RoleTypeEnum.PATIENT.value)
    with patch.object(admin_audit_queries, "st", stub):
        admin_audit_queries._render_query_detail_dialog_body(_make_agentic_item())

    sql_calls = [c for c in stub.code_calls if c[1].get("language") == "sql"]
    assert sql_calls == []
    assert any("restricted" in w for w in stub.write_calls)


def test_dialog_body_shows_scrubbed_banner():
    from views import admin_audit_queries

    stub = _RecordingStub(role=RoleTypeEnum.ADMIN.value)
    with patch.object(admin_audit_queries, "st", stub):
        admin_audit_queries._render_query_detail_dialog_body(_make_agentic_item(logging_mode="scrubbed"))

    assert any("Scrubbed mode" in w for w in stub.warning_calls)


def test_dialog_body_shows_disabled_banner_and_skips_sql():
    from views import admin_audit_queries

    stub = _RecordingStub(role=RoleTypeEnum.ADMIN.value)
    with patch.object(admin_audit_queries, "st", stub):
        admin_audit_queries._render_query_detail_dialog_body(
            _make_agentic_item(logging_mode="disabled", sql_statements=[])
        )

    assert any("Logging disabled" in w for w in stub.warning_calls)
    sql_calls = [c for c in stub.code_calls if c[1].get("language") == "sql"]
    assert sql_calls == []


def test_dialog_body_numbers_multiple_sql_statements():
    from views import admin_audit_queries

    stub = _RecordingStub(role=RoleTypeEnum.ADMIN.value)
    item = _make_agentic_item(
        sql_statements=["SELECT 1", "SELECT 2", "SELECT 3"],
    )
    with patch.object(admin_audit_queries, "st", stub):
        admin_audit_queries._render_query_detail_dialog_body(item)

    sql_texts = [c[0] for c in stub.code_calls if c[1].get("language") == "sql"]
    assert sql_texts == ["SELECT 1", "SELECT 2", "SELECT 3"]
    # And each gets a numbered header.
    numbered_headers = [m for m in stub.markdown_calls if m.startswith("**SQL ") and " of " in m]
    assert len(numbered_headers) == 3


# ---------------------------------------------------------------------------
# (16) Question-level dialog renders one unit per query
# ---------------------------------------------------------------------------


def test_question_dialog_body_stacks_units_for_question():
    from views import admin_audit_queries

    units = [
        _make_agentic_item(user_message_id=300, tool_call_id=1, call_index=0, sql_statements=["SELECT A"]),
        _make_agentic_item(user_message_id=300, tool_call_id=2, call_index=1, sql_statements=["SELECT B"]),
        _make_agentic_item(user_message_id=300, tool_call_id=3, call_index=2, sql_statements=["SELECT C"]),
    ]
    groups = admin_audit_queries._group_items_by_question(units)
    header, _ = groups[0]

    stub = _RecordingStub(role=RoleTypeEnum.ADMIN.value)
    with patch.object(admin_audit_queries, "st", stub):
        admin_audit_queries._render_question_detail_dialog_body(header, units)

    # One "### Query N of M" markdown per unit.
    query_headers = [m for m in stub.markdown_calls if m.startswith("### Query ")]
    assert len(query_headers) == 3
    sql_texts = [c[0] for c in stub.code_calls if c[1].get("language") == "sql"]
    assert sql_texts == ["SELECT A", "SELECT B", "SELECT C"]


# ---------------------------------------------------------------------------
# (17) Full-tab smoke: confirms Phase 1 helper wiring + filter assembly
# ---------------------------------------------------------------------------


def _make_full_tab_stub(*, multiselect_returns=None, button_returns=None, radio_value="Grouped"):
    captured_multiselect: list[dict] = []
    captured_page_calls: list[dict] = []
    multiselect_returns = multiselect_returns or {}
    button_returns = button_returns or {}

    class _Stub:
        def __init__(self):
            self.session_state = {"user_role": RoleTypeEnum.ADMIN.value}
            self.secrets = {"agent_logging": {"mode": "full"}}
            self.column_config = MagicMock()

        def multiselect(self, label, options=None, key=None, **kw):
            captured_multiselect.append({"label": label, "options": list(options or []), "key": key, "kwargs": kw})
            val = multiselect_returns.get(key, [])
            self.session_state.setdefault(key, val)
            self.session_state[key] = val
            return val

        def text_input(self, *_a, key=None, **_kw):
            self.session_state.setdefault(key, "")
            return ""

        def selectbox(self, *_a, options=None, index=0, key=None, **_kw):
            opts = list(options or [])
            val = opts[index] if opts else None
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
            return button_returns.get(key, False)

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

        def dialog(self, _title):
            def _decorator(fn):
                return fn

            return _decorator

    return _Stub(), {"multiselect": captured_multiselect, "page_calls": captured_page_calls}


def test_tab_renders_filter_row_with_pipeline_chip_and_calls_data_layer():
    from views import admin_audit_queries

    stub, captures = _make_full_tab_stub(
        multiselect_returns={"queries_pipeline_filter": ["agentic"]},
    )

    captured_filters: list[dict] = []

    def _fake_page(filters, page=1, page_size=50):
        captured_filters.append(filters)
        return {"items": [], "total": 0}

    with (
        patch.object(admin_audit_queries, "st", stub),
        patch.object(
            admin_audit_queries,
            "_cached_audit_filter_options",
            return_value={"usernames": [], "orgs": []},
        ),
        patch.object(admin_audit_queries, "get_per_query_audit_page", side_effect=_fake_page),
    ):
        admin_audit_queries._render_queries_tab(30)

    # Pipeline chip rendered with the expected options.
    pipeline_chip = next(
        (m for m in captures["multiselect"] if m["key"] == "queries_pipeline_filter"),
        None,
    )
    assert pipeline_chip is not None
    assert pipeline_chip["options"] == ["legacy", "agentic"]
    # The selected pipeline reached the data layer.
    assert captured_filters, "data layer not called"
    assert captured_filters[-1]["pipelines"] == ["agentic"]


def test_csv_export_writes_per_query_audit_columns():
    from views import admin_audit_queries

    stub, _ = _make_full_tab_stub(
        radio_value="Flat",
        button_returns={"queries_export_btn": True},
    )

    captured_export_df: list[pd.DataFrame] = []

    def _fake_download(_label, data=None, **_kw):
        try:
            from io import StringIO

            captured_export_df.append(pd.read_csv(StringIO(data.decode("utf-8"))))
        except Exception:
            pass

    stub.download_button = _fake_download  # type: ignore[assignment]

    item = _make_agentic_item(
        sql_statements=["SELECT * FROM t1", "SELECT * FROM t2"],
        patients_touched=[{"source_id": "pat-1", "display_name": "Alice"}],
    )

    with (
        patch.object(admin_audit_queries, "st", stub),
        patch.object(
            admin_audit_queries,
            "_cached_audit_filter_options",
            return_value={"usernames": [], "orgs": []},
        ),
        patch.object(admin_audit_queries, "get_per_query_audit_page", return_value={"items": [item], "total": 1}),
        patch.object(admin_audit_queries, "get_per_query_audit_export", return_value=[item]),
    ):
        admin_audit_queries._render_queries_tab(30)

    assert captured_export_df, "CSV download_button was not called"
    df = captured_export_df[0]
    assert set(df.columns) >= {
        "Asked At",
        "User",
        "Pipeline",
        "Scope",
        "Tool",
        "SQL",
        "Patient(s)",
        "Logging mode",
        "Status",
        "Elapsed (ms)",
    }
    # The two SQL statements are joined with the row-level separator.
    assert " ;; " in str(df.iloc[0]["SQL"])


# ---------------------------------------------------------------------------
# Phase 4 — "View user in Manage Users →" deep-link button on the per-query
# detail dialog. Inherits from the retired Questions tab's equivalent button.
# ---------------------------------------------------------------------------


def test_dialog_body_renders_view_user_in_manage_users_button():
    from views import admin_audit_queries

    item = _make_agentic_item(user_message_id=200, tool_call_id=42, user_id=7)
    stub = _RecordingStub(role=RoleTypeEnum.ADMIN.value)
    with patch.object(admin_audit_queries, "st", stub):
        admin_audit_queries._render_query_detail_dialog_body(item)

    button_keys = [b["key"] for b in stub.button_calls]
    assert any(k and k.startswith("queries_dialog_goto_users_200_42") for k in button_keys)
    button_labels = [b["label"] for b in stub.button_calls]
    assert "View user in Manage Users →" in button_labels


def test_dialog_button_press_sets_pref_and_switches_page():
    from views import admin_audit_queries

    item = _make_agentic_item(user_message_id=200, tool_call_id=42, user_id=7)
    btn_key = f"queries_dialog_goto_users_{item['user_message_id']}_{item['tool_call_id']}"
    stub = _RecordingStub(role=RoleTypeEnum.ADMIN.value, button_returns={btn_key: True})

    with patch.object(admin_audit_queries, "st", stub):
        admin_audit_queries._render_query_detail_dialog_body(item)

    assert stub.session_state.get("manage_users_pref_user_id") == 7
    assert stub.switch_page_calls == ["views/admin.py"]
