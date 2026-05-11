"""Tests for agent.tools.search_patients_by_criteria.

Phase 4 design §3.1, §3.5. Verifies:
- Tool operates WITHOUT a selected_patient (does not raise ModelRetry).
- total_count + sample shape.
- data_availability flips correctly.
- reliability_note set when code-based filter is present.
- sample_size=0 → counts only.
- ctx.deps.last_dataframe populated via cohort_result_to_df.
- Tool builds correctly against the synthetic fixture for the
  acceptance scenarios in spec §7.3.
"""

from __future__ import annotations
from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from agent.deps import AgentDeps, SelectedPatient
from agent.db.analytics_adapter import AnalyticsDbAdapter


def _deps(synthetic_db, selected: SelectedPatient | None = None) -> AgentDeps:
    return AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=selected,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite"),
        rag=None,
        sqlite_session=None,
        audit_logger=MagicMock(),
    )


def test_tool_operates_without_selected_patient(synthetic_db):
    """Unlike every other clinical-data tool, this one must NOT raise
    ModelRetry when selected_patient is None."""
    from agent.tools.search_patients_by_criteria import (
        search_patients_by_criteria,
        CohortCriteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected=None)
    result = search_patients_by_criteria(ctx, CohortCriteria())
    assert result.data_availability == "data_present"
    assert result.total_count >= 5


def test_diabetic_kaleida_over_65_acceptance(synthetic_db):
    """The §7.3 acceptance question: diabetic patients over 65 at Kaleida."""
    from agent.tools.search_patients_by_criteria import (
        search_patients_by_criteria,
        CohortCriteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db)
    result = search_patients_by_criteria(
        ctx,
        CohortCriteria(diagnosis_codes=["E11.9"], age_min=65, facility="Kaleida"),
    )
    assert result.total_count == 2
    assert len(result.sample) == 2
    src_ids = sorted(m.source_id for m in result.sample)
    assert src_ids == ["src-mary-1956", "src-susan-1955"]


def test_no_matches_returns_no_records_found(synthetic_db):
    from agent.tools.search_patients_by_criteria import (
        search_patients_by_criteria,
        CohortCriteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db)
    result = search_patients_by_criteria(
        ctx,
        CohortCriteria(diagnosis_codes=["X99.99"]),  # nonexistent
    )
    assert result.data_availability == "no_records_found"
    assert result.total_count == 0
    assert result.sample == []


def test_reliability_note_present_when_diagnosis_codes_set(synthetic_db):
    from agent.tools.search_patients_by_criteria import (
        search_patients_by_criteria,
        CohortCriteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db)
    result = search_patients_by_criteria(ctx, CohortCriteria(diagnosis_codes=["E11.9"]))
    assert result.reliability_note is not None
    assert "ICD-10" in result.reliability_note


def test_reliability_note_present_when_medication_codes_set(synthetic_db):
    from agent.tools.search_patients_by_criteria import (
        search_patients_by_criteria,
        CohortCriteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db)
    result = search_patients_by_criteria(ctx, CohortCriteria(medication_rxnorm_codes=["6809"]))
    assert result.reliability_note is not None


def test_reliability_note_absent_when_demographics_only(synthetic_db):
    """No code-based filter → no reliability badge."""
    from agent.tools.search_patients_by_criteria import (
        search_patients_by_criteria,
        CohortCriteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db)
    result = search_patients_by_criteria(ctx, CohortCriteria(age_min=65, facility="Kaleida"))
    assert result.reliability_note is None


def test_sample_size_zero_returns_count_only(synthetic_db):
    from agent.tools.search_patients_by_criteria import (
        search_patients_by_criteria,
        CohortCriteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db)
    result = search_patients_by_criteria(ctx, CohortCriteria(diagnosis_codes=["E11.9"], sample_size=0))
    assert result.total_count >= 3
    assert result.sample == []


def test_last_dataframe_populated_on_success(synthetic_db):
    from agent.tools.search_patients_by_criteria import (
        search_patients_by_criteria,
        CohortCriteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db)
    result = search_patients_by_criteria(ctx, CohortCriteria(medication_rxnorm_codes=["6809"]))
    assert isinstance(ctx.deps.last_dataframe, pd.DataFrame)
    assert len(ctx.deps.last_dataframe) == len(result.sample)
