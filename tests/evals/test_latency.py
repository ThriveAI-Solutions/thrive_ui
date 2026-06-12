from evals.latency import attribute_latency


def test_three_way_split():
    tool_calls = [
        {"elapsed_ms": 2300, "sql_executed": [{"sql": "SELECT 1", "db_elapsed_ms": 2100}]},
        {
            "elapsed_ms": 500,
            "sql_executed": [{"sql": "SELECT 2", "db_elapsed_ms": 300}, {"sql": "SELECT 3", "db_elapsed_ms": 100}],
        },
    ]
    out = attribute_latency(10000, tool_calls)
    assert out == {
        "total_ms": 10000,
        "redshift_ms": 2500,
        "tool_overhead_ms": 300,  # (2300+500) - 2500
        "llm_ms": 7200,  # 10000 - 2800
    }


def test_handles_missing_timing_fields():
    tool_calls = [
        {"elapsed_ms": None, "sql_executed": [{"sql": "SELECT 1", "db_elapsed_ms": None}]},
        {"sql_executed": []},
        {},
    ]
    out = attribute_latency(1000, tool_calls)
    assert out["redshift_ms"] == 0
    assert out["llm_ms"] == 1000


def test_never_negative():
    # clock skew: tools report more time than the turn total
    out = attribute_latency(100, [{"elapsed_ms": 500, "sql_executed": [{"db_elapsed_ms": 900}]}])
    assert out["llm_ms"] == 0
    assert out["tool_overhead_ms"] == 0
