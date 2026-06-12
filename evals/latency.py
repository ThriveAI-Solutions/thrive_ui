"""Split a turn's wall clock into LLM / our-code / warehouse buckets.

redshift_ms is the sum of db_elapsed_ms over executed statements — time
spent waiting on the warehouse, including network round-trip and fetch.
"""

from __future__ import annotations


def attribute_latency(total_elapsed_ms: int, tool_calls: list[dict]) -> dict:
    redshift_ms = 0
    tool_total_ms = 0
    for tc in tool_calls:
        tool_total_ms += int(tc.get("elapsed_ms") or 0)
        for stmt in tc.get("sql_executed") or []:
            redshift_ms += int(stmt.get("db_elapsed_ms") or 0)
    return {
        "total_ms": total_elapsed_ms,
        "redshift_ms": redshift_ms,
        "tool_overhead_ms": max(0, tool_total_ms - redshift_ms),
        "llm_ms": max(0, total_elapsed_ms - tool_total_ms),
    }
