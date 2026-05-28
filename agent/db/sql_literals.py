"""Inline named bind params into a SQL string for display/handoff only.

The breakdown tool returns generated_sql so the LLM can paste it into
run_sql, which accepts a bare SQL string (no bind params). Values here are
simple scalars produced by our own builders (str, int, ISO date strings),
never free user text, and run_sql's AST guard re-validates before execution.
"""

from __future__ import annotations
import re
from typing import Any


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    # strings / ISO date strings: single-quote, double embedded quotes
    return "'" + str(value).replace("'", "''") + "'"


def inline_sql_literals(sql: str, params: dict[str, Any]) -> str:
    """Replace :name placeholders in `sql` with quoted literals from `params`.

    Single-pass: the original SQL is scanned exactly once, so an inserted
    literal can never be re-matched (a param value that happens to contain a
    ":identifier" substring is left intact). Placeholders with no matching
    param are left untouched.
    """

    def _sub(match: "re.Match") -> str:
        name = match.group(1)
        if name in params:
            return _literal(params[name])
        return match.group(0)

    return re.sub(r":(\w+)", _sub, sql)
