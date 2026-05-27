"""Tests for agent.db.queries.cohort_breakdown."""

from __future__ import annotations

import pytest

from agent.db.queries.cohort_breakdown import (
    BreakdownDimension,
    breakdown_bucket,
)


def test_gender_bucket_is_additive_no_anchor():
    spec = breakdown_bucket(BreakdownDimension.GENDER, dialect="sqlite")
    assert spec.non_additive is False
    assert spec.requires_diagnosis_anchor is False
    assert "p.gender" in spec.group_expr
    assert "Unknown" in spec.group_expr  # null/U folded to a bucket


def test_age_band_bucket_is_additive_no_anchor():
    spec = breakdown_bucket(BreakdownDimension.AGE_BAND, dialect="sqlite")
    assert spec.non_additive is False
    assert spec.requires_diagnosis_anchor is False
    assert "p.age" in spec.group_expr
    assert "65" in spec.group_expr


def test_diagnosis_month_is_non_additive_needs_anchor_sqlite():
    spec = breakdown_bucket(BreakdownDimension.DIAGNOSIS_MONTH, dialect="sqlite")
    assert spec.non_additive is True
    assert spec.requires_diagnosis_anchor is True
    assert "strftime" in spec.group_expr
    assert "dx.start_date" in spec.group_expr


def test_diagnosis_month_uses_date_trunc_on_postgres():
    spec = breakdown_bucket(BreakdownDimension.DIAGNOSIS_MONTH, dialect="postgres")
    assert "DATE_TRUNC" in spec.group_expr or "TO_CHAR" in spec.group_expr
    assert "dx.start_date" in spec.group_expr


def test_unknown_dialect_rejected():
    with pytest.raises(ValueError):
        breakdown_bucket(BreakdownDimension.DIAGNOSIS_YEAR, dialect="oracle")
