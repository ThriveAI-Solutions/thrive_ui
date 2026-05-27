"""Tests for agent.db.sql_literals.inline_sql_literals."""

from __future__ import annotations

from agent.db.sql_literals import inline_sql_literals


def test_inlines_string_with_quotes_escaped():
    out = inline_sql_literals("WHERE name = :n", {"n": "O'Brien"})
    assert out == "WHERE name = 'O''Brien'"


def test_inlines_int_unquoted():
    out = inline_sql_literals("WHERE age >= :age_min", {"age_min": 65})
    assert out == "WHERE age >= 65"


def test_longer_keys_not_clobbered_by_prefixes():
    # :dx_1 must not match inside :dx_10
    sql = "WHERE a = :dx_1 AND b = :dx_10"
    out = inline_sql_literals(sql, {"dx_1": "A", "dx_10": "B"})
    assert out == "WHERE a = 'A' AND b = 'B'"


def test_leaves_unknown_placeholders_untouched():
    out = inline_sql_literals("WHERE x = :known AND y = :other", {"known": 1})
    assert out == "WHERE x = 1 AND y = :other"
