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


def _deps(
    synthetic_db,
    selected: SelectedPatient | None = None,
    *,
    enforce_consent: bool = False,
    small_cell_threshold: int = 11,
) -> AgentDeps:
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
        run_logger=MagicMock(),
        enforce_consent=enforce_consent,
        small_cell_threshold=small_cell_threshold,
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


def test_cohort_criteria_accepts_diagnosis_date_range_alone():
    """'How many patients had a diagnosis in December 2024' — a date window
    with no codes is now a valid standalone criterion."""
    from agent.tools.search_patients_by_criteria import CohortCriteria, DateRange

    c = CohortCriteria(diagnosis_date_range=DateRange(start=date(2024, 12, 1), end=date(2024, 12, 31)))
    assert c.diagnosis_date_range.start == date(2024, 12, 1)


def test_cohort_criteria_rejects_empty_diagnosis_date_range():
    """An all-None DateRange carries no filter and must NOT satisfy the
    at-least-one-criterion guard — otherwise it scans the whole table."""
    from agent.tools.search_patients_by_criteria import CohortCriteria, DateRange

    with pytest.raises(ValidationError):
        CohortCriteria(diagnosis_date_range=DateRange())


def test_diagnosis_date_range_alone_count(synthetic_db):
    """End-to-end: a bare diagnosis window counts distinct patients with any
    ICD-10/SNOMED diagnosis in that window (Daniel + Susan in Nov-Dec 2025)."""
    from agent.tools.search_patients_by_criteria import (
        CohortCriteria,
        DateRange,
        search_patients_by_criteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected=None)
    out = search_patients_by_criteria(
        ctx,
        CohortCriteria(
            diagnosis_date_range=DateRange(start=date(2025, 11, 1), end=date(2025, 12, 31)),
            sample_size=0,
        ),
    )
    assert out.data_availability == "data_present"
    assert out.total_count == 2


def test_diagnosis_date_range_alone_attaches_dx_reliability(synthetic_db):
    """A diagnosis-window query hits the same low-coverage problems data, so
    the ICD-10/SNOMED coverage caveat MUST be surfaced."""
    from agent.tools.search_patients_by_criteria import (
        CohortCriteria,
        DateRange,
        search_patients_by_criteria,
        _RELIABILITY_DX,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected=None)
    out = search_patients_by_criteria(
        ctx,
        CohortCriteria(diagnosis_date_range=DateRange(start=date(2025, 11, 1), end=date(2025, 12, 31))),
    )
    assert out.reliability_note == _RELIABILITY_DX


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


def test_cohort_wraps_db_errors_as_model_retry(synthetic_db):
    """Live regression on 2026-05-13: 'how many people in NY have diabetes'
    triggered the new 30s curated-query timeout, the SQLAlchemyError
    propagated unwrapped through pydantic-ai, and the entire agent
    stream crashed. The cohort tool must convert DB errors to ModelRetry
    so the LLM gets a chance to narrow the criteria instead of dying."""
    from agent.tools.search_patients_by_criteria import (
        CohortCriteria,
        search_patients_by_criteria,
    )
    from pydantic_ai.exceptions import ModelRetry
    from sqlalchemy.exc import OperationalError

    deps = _deps(synthetic_db, selected=None)

    def boom(sql, params=None):
        raise OperationalError("statement", {}, Exception("simulated query timeout"))

    deps.analytics_db.fetch_all = boom

    ctx = MagicMock()
    ctx.deps = deps

    with pytest.raises(ModelRetry) as excinfo:
        search_patients_by_criteria(ctx, CohortCriteria(state="NY"))
    msg = str(excinfo.value)
    assert "Cohort query failed" in msg
    assert "simulated query timeout" in msg or "OperationalError" in msg


def test_cohort_count_only_path_returns_total_count(synthetic_db):
    """sample_size=0 must take the count-only fast path and return
    total_count without sample rows. Empty sample, data_present when
    count > 0."""
    from agent.tools.search_patients_by_criteria import (
        CohortCriteria,
        search_patients_by_criteria,
    )

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected=None)
    out = search_patients_by_criteria(ctx, CohortCriteria(state="NY", sample_size=0))
    assert out.data_availability == "data_present"
    assert out.total_count >= 2  # Both Johns + extras seeded for other tests
    assert out.sample == []


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


def test_criteria_accepts_breakdown_list():
    from agent.db.queries.cohort_breakdown import BreakdownDimension
    from agent.tools.search_patients_by_criteria import CohortCriteria

    c = CohortCriteria(gender="F", breakdown=[BreakdownDimension.GENDER])
    assert c.breakdown == [BreakdownDimension.GENDER]


def test_criteria_breakdown_defaults_empty():
    from agent.tools.search_patients_by_criteria import CohortCriteria

    c = CohortCriteria(gender="F")
    assert c.breakdown == []


def test_cohort_result_carries_breakdown_fields():
    from agent.tools.search_patients_by_criteria import BreakdownBucket, CohortResult

    r = CohortResult(
        total_count=3,
        sample=[],
        data_availability="data_present",
        buckets=[BreakdownBucket(bucket_label="F", patient_count=3)],
        non_additive=False,
        generated_sql="SELECT 1",
        breakdown_status="single_dimension",
    )
    assert r.buckets[0].patient_count == 3
    assert r.non_additive is False
    assert r.breakdown_status == "single_dimension"


def _ctx(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected=None)
    return ctx


def test_single_gender_breakdown_returns_buckets(synthetic_db):
    from agent.tools.search_patients_by_criteria import CohortCriteria, search_patients_by_criteria
    from agent.db.queries.cohort_breakdown import BreakdownDimension

    crit = CohortCriteria(age_min=0, breakdown=[BreakdownDimension.GENDER])
    result = search_patients_by_criteria(_ctx(synthetic_db), crit)
    assert result.breakdown_status == "single_dimension"
    assert result.buckets, "expected gender buckets"
    assert result.sample == [], "sample suppressed in breakdown mode"
    assert result.non_additive is False
    assert result.generated_sql and ":age_min" not in result.generated_sql


def test_diagnosis_month_breakdown_is_non_additive(synthetic_db):
    from datetime import date
    from agent.tools.search_patients_by_criteria import CohortCriteria, search_patients_by_criteria
    from agent.db.queries.cohort_breakdown import BreakdownDimension

    crit = CohortCriteria(
        diagnosis_date_range={"start": date(2025, 1, 1), "end": date(2025, 12, 31)},
        breakdown=[BreakdownDimension.DIAGNOSIS_MONTH],
    )
    result = search_patients_by_criteria(_ctx(synthetic_db), crit)
    assert result.non_additive is True
    assert result.reliability_note and "do not sum" in result.reliability_note.lower()


def test_two_dimensions_escalates_without_executing(synthetic_db):
    from datetime import date
    from agent.tools.search_patients_by_criteria import CohortCriteria, search_patients_by_criteria
    from agent.db.queries.cohort_breakdown import BreakdownDimension

    crit = CohortCriteria(
        diagnosis_date_range={"start": date(2025, 1, 1), "end": date(2025, 12, 31)},
        breakdown=[BreakdownDimension.DIAGNOSIS_MONTH, BreakdownDimension.GENDER],
    )
    result = search_patients_by_criteria(_ctx(synthetic_db), crit)
    assert result.breakdown_status == "unsupported_multi_dimension"
    assert result.buckets == []
    assert result.generated_sql, "must hand back a template for the first dimension"
    assert result.notes_to_agent and "run_sql" in result.notes_to_agent


def test_time_breakdown_missing_anchor_signals(synthetic_db):
    from agent.tools.search_patients_by_criteria import CohortCriteria, search_patients_by_criteria
    from agent.db.queries.cohort_breakdown import BreakdownDimension

    crit = CohortCriteria(gender="F", breakdown=[BreakdownDimension.DIAGNOSIS_MONTH])
    result = search_patients_by_criteria(_ctx(synthetic_db), crit)
    assert result.breakdown_status == "missing_diagnosis_anchor"
    assert result.buckets == []
    assert result.notes_to_agent and "diagnosis" in result.notes_to_agent.lower()


# condition_sets tests -----------------------------------------------------
# The vocab_session fixture (tests/conftest.py) seeds dx:diabetes-mellitus
# with members E11.9 and E11.65. Security-sensitive: expansion must union
# INTO diagnosis_codes server-side, from DB values only, and must not run
# code_match_forms here (that stays in the query layer, agent/db/queries/cohort.py).


def _deps_with_vocab(synthetic_db, vocab_session):
    deps = _deps(synthetic_db, selected=None)
    deps.sqlite_session = vocab_session
    return deps


def test_condition_sets_alone_satisfies_at_least_one_criterion():
    from agent.tools.search_patients_by_criteria import CohortCriteria

    # Must not raise — condition_sets is a standalone criterion.
    c = CohortCriteria(condition_sets=["dx:diabetes-mellitus"])
    assert c.condition_sets == ["dx:diabetes-mellitus"]


def test_condition_sets_expands_and_matches_diagnosis_codes_result(synthetic_db, vocab_session):
    """condition_sets=['dx:diabetes-mellitus'] (members E11.9, E11.65) must
    find at least as many patients as diagnosis_codes=['E11.9'] alone —
    the acceptance fixture from test_diabetic_kaleida_over_65_acceptance."""
    from agent.tools.search_patients_by_criteria import CohortCriteria, search_patients_by_criteria

    ctx = MagicMock()
    ctx.deps = _deps_with_vocab(synthetic_db, vocab_session)
    result = search_patients_by_criteria(
        ctx,
        CohortCriteria(condition_sets=["dx:diabetes-mellitus"], age_min=65, facility="Kaleida"),
    )
    assert result.total_count >= 2
    src_ids = {m.source_id for m in result.sample}
    assert {"src-mary-1956", "src-susan-1955"} <= src_ids


def test_condition_sets_unions_into_diagnosis_codes_without_dupes(synthetic_db, vocab_session, monkeypatch):
    """diagnosis_codes=['E11.9'] + condition_sets expanding to ['E11.9', 'E11.65']
    must union to exactly ['E11.9', 'E11.65'] — no duplicate 'E11.9', order preserved."""
    import agent.tools.search_patients_by_criteria as mod

    captured: dict = {}

    def fake_cohort_sql(criteria, schema_prefix="", dialect="sqlite"):
        captured["diagnosis_codes"] = list(criteria.diagnosis_codes or [])
        return "SELECT 0 AS total_count", {}

    monkeypatch.setattr(mod, "cohort_sql", fake_cohort_sql)

    ctx = MagicMock()
    ctx.deps = _deps_with_vocab(synthetic_db, vocab_session)
    mod.search_patients_by_criteria(
        ctx,
        mod.CohortCriteria(diagnosis_codes=["E11.9"], condition_sets=["dx:diabetes-mellitus"], sample_size=0),
    )
    assert captured["diagnosis_codes"] == ["E11.9", "E11.65"]


def test_condition_sets_unknown_set_raises_actionable_model_retry(synthetic_db, vocab_session):
    from agent.tools.search_patients_by_criteria import CohortCriteria, search_patients_by_criteria
    from pydantic_ai.exceptions import ModelRetry

    ctx = MagicMock()
    ctx.deps = _deps_with_vocab(synthetic_db, vocab_session)
    with pytest.raises(ModelRetry) as excinfo:
        search_patients_by_criteria(ctx, CohortCriteria(condition_sets=["dx:not-a-real-set"]))
    msg = str(excinfo.value)
    assert "not-a-real-set" in msg
    assert "search_codes first" in msg


# --- Consent aggregate-only mode (#244): under enforce_consent the cohort tool
# suppresses small count cells AND withholds the identified patient sample
# (identified rows aren't covered by the de-identified-cohort exemption). ---


def test_enforcement_off_returns_raw_count_and_sample(synthetic_db):
    from agent.tools.search_patients_by_criteria import CohortCriteria, search_patients_by_criteria

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, enforce_consent=False)
    # Small cohort (total_count == 2). With enforcement OFF, behavior is unchanged:
    # the real count and identified sample are returned.
    result = search_patients_by_criteria(
        ctx, CohortCriteria(diagnosis_codes=["E11.9"], age_min=65, facility="Kaleida")
    )
    assert result.total_count == 2
    assert len(result.sample) == 2


def test_enforcement_on_suppresses_small_total_and_withholds_sample(synthetic_db):
    from agent.tools.search_patients_by_criteria import CohortCriteria, search_patients_by_criteria

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, enforce_consent=True, small_cell_threshold=11)
    result = search_patients_by_criteria(
        ctx, CohortCriteria(diagnosis_codes=["E11.9"], age_min=65, facility="Kaleida")
    )
    # Count of 2 is below the floor -> suppressed label; identified sample withheld.
    assert result.total_count == "fewer than 11"
    assert result.sample == []


def test_enforcement_on_withholds_sample_even_when_count_not_suppressed(synthetic_db):
    from agent.tools.search_patients_by_criteria import CohortCriteria, search_patients_by_criteria

    ctx = MagicMock()
    # threshold=2 means a count of 2 is NOT suppressed (2 >= 2), proving the
    # sample is withheld independently of small-cell suppression.
    ctx.deps = _deps(synthetic_db, enforce_consent=True, small_cell_threshold=2)
    result = search_patients_by_criteria(
        ctx, CohortCriteria(diagnosis_codes=["E11.9"], age_min=65, facility="Kaleida")
    )
    assert result.total_count == 2
    assert result.sample == []


def test_enforcement_on_suppresses_small_breakdown_buckets(synthetic_db):
    from agent.tools.search_patients_by_criteria import CohortCriteria, search_patients_by_criteria
    from agent.db.queries.cohort_breakdown import BreakdownDimension
    from agent.consent.suppression import suppressed_label

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, enforce_consent=True, small_cell_threshold=1000)
    crit = CohortCriteria(age_min=0, breakdown=[BreakdownDimension.GENDER])
    result = search_patients_by_criteria(ctx, crit)
    # With an absurdly high floor every non-zero bucket is a "small cell".
    label = suppressed_label(1000)
    assert result.buckets, "expected gender buckets"
    for b in result.buckets:
        assert b.patient_count == label, f"bucket {b.bucket_label} not suppressed"
