import importlib


def test_module_imports_and_exposes_entrypoint():
    # The agent-analytics view was consolidated into views.admin_agentic by the
    # 3-route IA refactor (commit 15e5c14); its entry point is render(), not main().
    mod = importlib.import_module("views.admin_agentic")
    assert hasattr(mod, "render")
    assert hasattr(mod, "_guard_admin")


def test_render_run_timeline_orders_events():
    from views.admin_agentic import _timeline_rows

    detail = {
        "events": [
            {
                "seq": 2,
                "event_type": "tool_call_completed",
                "tool_name": "run_sql",
                "payload_json": '{"result": {"row_count": 3}}',
                "turn_index": 1,
            },
            {
                "seq": 1,
                "event_type": "run_started",
                "payload_json": '{"question": "q"}',
                "turn_index": None,
            },
        ],
    }
    rows = _timeline_rows(detail)
    assert [r["seq"] for r in rows] == [1, 2]
