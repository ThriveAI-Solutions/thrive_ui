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


def inline_sql_literals(sql: str, params: dict) -> str:
    """Replace :name placeholders in `sql` with quoted literals from `params`.

    Keys are applied longest-first and matched with a trailing word boundary
    so :dx_1 never clobbers :dx_10. Placeholders with no matching param are
    left untouched.
    """
    out = sql
    for name in sorted(params, key=len, reverse=True):
        out = re.sub(rf":{re.escape(name)}\b", _literal(params[name]), out)
    return out
