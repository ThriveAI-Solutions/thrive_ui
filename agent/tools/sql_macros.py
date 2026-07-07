"""{{codes:<set_id>}} macro for run_sql.

Ported from chiron's core/agent/tools/sql_macros.py (spec 2026-07-03).

Expanded BEFORE the AST guard (_ast_guard in agent/tools/run_sql.py), so the
safety layer only ever sees plain SQL. The IN-list is built exclusively from
vocab-DB values (never from model-supplied strings), so the macro cannot
smuggle SQL past validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from agent.codes.match_forms import code_match_forms
from agent.codes.service import expand_sets

CODES_MACRO_RE = re.compile(r"\{\{codes:([^}]+)\}\}")


@dataclass(frozen=True)
class MacroExpansion:
    sql: str
    expansions: dict[str, int]


def _quoted_in_list(values: list[str]) -> str:
    quoted = ",".join("'" + v.replace("'", "''") + "'" for v in values)
    return f"({quoted})"


def expand_code_macros(sql: str, session: Session) -> MacroExpansion:
    set_ids = [m.strip() for m in CODES_MACRO_RE.findall(sql)]
    if not set_ids:
        return MacroExpansion(sql=sql, expansions={})

    rendered: dict[str, str] = {}
    counts: dict[str, int] = {}
    for set_id in dict.fromkeys(set_ids):  # order-preserving dedup
        forms = code_match_forms(expand_sets(session, [set_id]))
        rendered[set_id] = _quoted_in_list(forms)
        counts[set_id] = len(forms)

    expanded = CODES_MACRO_RE.sub(lambda m: rendered[m.group(1).strip()], sql)
    return MacroExpansion(sql=expanded, expansions=counts)
