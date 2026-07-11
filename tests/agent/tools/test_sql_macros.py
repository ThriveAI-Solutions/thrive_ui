"""Pure-mock tests for agent.tools.sql_macros.expand_code_macros.

Ported from chiron's tests/core/agent/tools/test_sql_macros.py (spec
2026-07-03); adapted imports only.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.tools.sql_macros import expand_code_macros
from agent.codes.service import UnknownCodeSetError


def _session():
    return MagicMock(name="session")


def test_no_token_passthrough_and_no_session_touch():
    session = _session()
    out = expand_code_macros("SELECT 1 FROM t WHERE code = 'E11.9'", session)
    assert out.sql == "SELECT 1 FROM t WHERE code = 'E11.9'"
    assert out.expansions == {}
    session.assert_not_called()


@patch("agent.tools.sql_macros.expand_sets", return_value=["E11.9"])
def test_single_token_expands_to_quoted_in_list(mock_expand):
    out = expand_code_macros("SELECT COUNT(*) FROM dx WHERE code IN {{codes:dx:diabetes-mellitus}}", _session())
    # code_match_forms(["E11.9"]) == ["E11.9", "E119"]
    assert "code IN ('E11.9','E119')" in out.sql
    assert "{{codes:" not in out.sql
    assert out.expansions == {"dx:diabetes-mellitus": 2}
    mock_expand.assert_called_once()


@patch("agent.tools.sql_macros.expand_sets", return_value=["O'BRIEN"])
def test_quotes_escaped_by_doubling(mock_expand):
    out = expand_code_macros("… IN {{codes:weird:set}}", _session())
    assert "'O''BRIEN'" in out.sql


@patch("agent.tools.sql_macros.expand_sets", return_value=["E11.9"])
def test_duplicate_token_expands_both_but_fetches_once(mock_expand):
    sql = "SELECT * FROM a WHERE x IN {{codes:s:1}} OR y IN {{codes:s:1}}"
    out = expand_code_macros(sql, _session())
    assert out.sql.count("('E11.9','E119')") == 2
    assert mock_expand.call_count == 1


@patch("agent.tools.sql_macros.expand_sets", side_effect=UnknownCodeSetError("nope", []))
def test_unknown_set_propagates(mock_expand):
    with pytest.raises(UnknownCodeSetError):
        expand_code_macros("… IN {{codes:nope}}", _session())
