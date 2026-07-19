"""run_sql consent gate (#244).

Under enforcement, freeform SQL against patient-bearing views is refused so it
can't bypass the consent-gated curated tools. Non-patient SQL and the
enforcement-off path are unaffected.
"""

from unittest.mock import MagicMock

import pytest
from pydantic_ai.exceptions import ModelRetry

from agent.consent.sql_gate import references_patient_view
from agent.tools.run_sql import RunSqlInput, run_sql


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT COUNT(*) FROM federated_demographic_v", True),
        ("SELECT * FROM internal_patient_profile_v", True),
        ("WITH x AS (SELECT 1) SELECT * FROM federated_results_v", True),
        ("SELECT * FROM metric_federated_data_v", True),
        ("SELECT 1", False),
        ("SELECT code, description FROM vocab_icd10", False),
    ],
)
def test_references_patient_view(sql, expected):
    assert references_patient_view(sql) is expected


def _ctx(*, enforce_consent):
    ctx = MagicMock()
    deps = MagicMock()
    deps.enforce_consent = enforce_consent
    deps.user_role = MagicMock(value=1)
    deps.sqlite_session = None
    deps.analytics_db = None  # not reached when the gate refuses first
    ctx.deps = deps
    return ctx


def test_enforcement_refuses_patient_sql():
    ctx = _ctx(enforce_consent=True)
    with pytest.raises(ModelRetry, match="Consent enforcement blocks"):
        run_sql(ctx, RunSqlInput(sql="SELECT source_id FROM federated_demographic_v"))


def test_enforcement_off_does_not_gate():
    # Enforcement off => the consent gate is skipped; execution proceeds to the
    # "analytics database not configured" retry (analytics_db is None here),
    # proving the consent gate did not fire.
    ctx = _ctx(enforce_consent=False)
    with pytest.raises(ModelRetry, match="Analytics database is not configured"):
        run_sql(ctx, RunSqlInput(sql="SELECT source_id FROM federated_demographic_v"))
