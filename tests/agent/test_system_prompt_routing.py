from agent.system_prompt import SYSTEM_PROMPT
from agent.tools.run_sql import _RUN_SQL_BASE_DESCRIPTION


def test_prompt_mentions_breakdown_routing():
    assert "breakdown" in SYSTEM_PROMPT.lower()


def test_run_sql_description_defers_single_dimension_breakdowns():
    assert "single-dimension" in _RUN_SQL_BASE_DESCRIPTION.lower()
