import importlib


def test_module_imports_and_exposes_main():
    mod = importlib.import_module("views.agent_analytics")
    assert hasattr(mod, "main")
    assert hasattr(mod, "_guard_admin")


def test_render_run_timeline_orders_events():
    from views.agent_analytics import _timeline_rows

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
