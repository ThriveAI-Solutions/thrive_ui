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
from pydantic import ValidationError

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
    # age_min=0 is a permissive criterion; the point of the test is to
    # confirm the tool does not require selected_patient, not to test
    # the no-criteria path (which is now refused at the model layer).
    result = search_patients_by_criteria(ctx, CohortCriteria(age_min=0))
    assert result.data_availability == "data_present"
    assert result.total_count >= 5


def test_cohort_criteria_rejects_extra_fields():
    """Weak models sometimes invent a free-text `query` field; reject it
    so pydantic-ai surfaces a retry."""
    from agent.tools.search_patients_by_criteria import CohortCriteria

    with pytest.raises(ValidationError):
        CohortCriteria(query="diabetic patients over 65")


def test_cohort_criteria_rejects_empty_criteria():
    """All-None criteria would scan the entire patient table — refuse."""
    from agent.tools.search_patients_by_criteria import CohortCriteria

    with pytest.raises(ValidationError):
        CohortCriteria()


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


def test_truncated_true_when_population_exceeds_sample(synthetic_db):
    """sample_size smaller than the matching cohort → truncated=True."""
    from agent.tools.search_patients_by_criteria import (
        search_patients_by_criteria,
        CohortCriteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db)
    result = search_patients_by_criteria(
        ctx,
        CohortCriteria(diagnosis_codes=["E11.9"], sample_size=1),
    )
    assert result.total_count >= 3
    assert len(result.sample) == 1
    assert result.truncated is True


def test_truncated_false_when_sample_covers_population(synthetic_db):
    """sample_size ≥ population → truncated=False."""
    from agent.tools.search_patients_by_criteria import (
        search_patients_by_criteria,
        CohortCriteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db)
    result = search_patients_by_criteria(
        ctx,
        CohortCriteria(diagnosis_codes=["E11.9"], age_min=65, facility="Kaleida", sample_size=20),
    )
    assert result.total_count == 2
    assert result.truncated is False


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


def test_cohort_criteria_accepts_zip_code_alone():
    from agent.tools.search_patients_by_criteria import CohortCriteria

    c = CohortCriteria(zip_code="14223")
    assert c.zip_code == "14223"


def test_cohort_criteria_accepts_city_alone():
    from agent.tools.search_patients_by_criteria import CohortCriteria

    c = CohortCriteria(city="Buffalo")
    assert c.city == "Buffalo"


def test_cohort_criteria_accepts_state_alone():
    from agent.tools.search_patients_by_criteria import CohortCriteria

    c = CohortCriteria(state="NY")
    assert c.state == "NY"


def test_cohort_criteria_still_rejects_truly_empty():
    from agent.tools.search_patients_by_criteria import CohortCriteria

    with pytest.raises(ValidationError):
        CohortCriteria()


def test_geo_only_attaches_geo_reliability(synthetic_db):
    from agent.tools.search_patients_by_criteria import (
        CohortCriteria,
        search_patients_by_criteria,
        _RELIABILITY_GEO,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected=None)
    out = search_patients_by_criteria(ctx, CohortCriteria(zip_code="14223"))
    assert out.reliability_note == _RELIABILITY_GEO


def test_geo_no_records_still_attaches_geo_reliability(synthetic_db):
    """A geo query that finds nothing should STILL surface the geo caveat —
    'no one in zip 99999' could be a real negative or a data-quality miss."""
    from agent.tools.search_patients_by_criteria import (
        CohortCriteria,
        search_patients_by_criteria,
        _RELIABILITY_GEO,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected=None)
    out = search_patients_by_criteria(ctx, CohortCriteria(zip_code="99999"))
    assert out.data_availability == "no_records_found"
    assert out.reliability_note == _RELIABILITY_GEO


def test_dx_plus_geo_concatenates_both_notes(synthetic_db):
    from agent.tools.search_patients_by_criteria import (
        CohortCriteria,
        search_patients_by_criteria,
        _RELIABILITY_DX,
        _RELIABILITY_GEO,
    )
    from sqlalchemy import text as _sql_text

    # Insert a diagnosis row matching src-john-1962 so dx filter has data.
    with synthetic_db.begin() as conn:
        conn.execute(
            _sql_text(
                "INSERT INTO metric_federated_data_v "
                "(patient_id, code, code_type, start_date, is_claims_data) "
                "VALUES (1, 'I10', 'ICD-10', '2025-01-01', 0)"
            )
        )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected=None)
    out = search_patients_by_criteria(
        ctx,
        CohortCriteria(diagnosis_codes=["I10"], zip_code="14223"),
    )
    assert _RELIABILITY_DX in (out.reliability_note or "")
    assert _RELIABILITY_GEO in (out.reliability_note or "")
