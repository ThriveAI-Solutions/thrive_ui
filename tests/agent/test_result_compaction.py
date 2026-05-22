"""Tests for agent.result_compaction."""

from __future__ import annotations
from typing import ClassVar, List, Optional

from pydantic import BaseModel

from agent.result_compaction import CompactingListResult, compact_rows


# --- compact_rows ---------------------------------------------------


def test_compact_rows_drops_columns_null_across_all_rows():
    rows = [
        {"a": 1, "b": None, "c": "x"},
        {"a": 2, "b": None, "c": "y"},
    ]
    cleaned, dead = compact_rows(rows)
    assert dead == ["b"]
    assert cleaned == [{"a": 1, "c": "x"}, {"a": 2, "c": "y"}]


def test_compact_rows_keeps_column_with_at_least_one_value():
    rows = [
        {"a": 1, "b": None},
        {"a": 2, "b": "present"},
    ]
    cleaned, dead = compact_rows(rows)
    assert dead == []
    assert cleaned == rows


def test_compact_rows_empty_input():
    assert compact_rows([]) == ([], [])


def test_compact_rows_preserves_row_count_even_when_dropping_columns():
    rows = [{"a": None, "b": 1}, {"a": None, "b": 2}, {"a": None, "b": 3}]
    cleaned, dead = compact_rows(rows)
    assert dead == ["a"]
    assert len(cleaned) == 3


def test_compact_rows_handles_jagged_keysets():
    """Rows with different keysets: a key missing from a row is treated as None."""
    rows = [
        {"a": 1, "b": None},
        {"a": 2},  # 'b' absent (== None for compaction purposes)
    ]
    cleaned, dead = compact_rows(rows)
    assert dead == ["b"]
    assert cleaned == [{"a": 1}, {"a": 2}]


# --- CompactingListResult mixin -------------------------------------


class _Row(BaseModel):
    name: str
    value: Optional[int] = None
    notes: Optional[str] = None


class _Container(CompactingListResult):
    _list_field: ClassVar[str] = "rows"

    rows: List[_Row]
    total: int


def test_mixin_compacts_dead_columns_on_model_dump():
    container = _Container(
        total=2,
        rows=[
            _Row(name="alpha", value=1),  # notes is None
            _Row(name="beta", value=2),  # notes is None
        ],
    )
    dumped = container.model_dump(mode="json")
    assert dumped["omitted_empty_fields"] == ["notes"]
    for row in dumped["rows"]:
        assert "notes" not in row
        assert "value" in row


def test_mixin_no_compaction_when_every_column_has_at_least_one_value():
    container = _Container(
        total=2,
        rows=[
            _Row(name="alpha", value=1, notes=None),
            _Row(name="beta", value=None, notes="here"),
        ],
    )
    dumped = container.model_dump(mode="json")
    assert "omitted_empty_fields" not in dumped
    assert all("value" in r and "notes" in r for r in dumped["rows"])


def test_mixin_python_attribute_access_is_unchanged():
    """Compaction is JSON-only — typed attribute access still sees None fields."""
    container = _Container(
        total=1,
        rows=[_Row(name="alpha", value=None, notes=None)],
    )
    assert container.rows[0].value is None
    assert container.rows[0].notes is None


def test_mixin_handles_empty_list():
    container = _Container(total=0, rows=[])
    dumped = container.model_dump(mode="json")
    assert dumped["rows"] == []
    assert "omitted_empty_fields" not in dumped


# --- Integration with real result types -----------------------------


def test_clinical_result_compacts_dead_columns():
    from agent.tools.get_patient_clinical_data import ClinicalResult, MedicationItem

    result = ClinicalResult(
        domain="medications",
        items=[
            MedicationItem(source_id="s1", med_name="metformin", med_strength="500", med_strength_unit="mg"),
            MedicationItem(source_id="s2", med_name="lisinopril", med_strength="10", med_strength_unit="mg"),
        ],
        data_availability="data_present",
    )
    dumped = result.model_dump(mode="json")
    # Both rows have ndc_code=None, rxnorm_code=None, etc — those should be pruned.
    assert "omitted_empty_fields" in dumped
    dead = set(dumped["omitted_empty_fields"])
    assert "ndc_code" in dead
    assert "rxnorm_code" in dead
    # Fields that ARE populated must survive.
    for row in dumped["items"]:
        assert "med_name" in row
        assert "med_strength" in row


def test_cohort_result_compacts_sample():
    from agent.tools.search_patients_by_criteria import CohortResult, PatientMatch

    result = CohortResult(
        total_count=2,
        sample=[
            PatientMatch(source_id="s1", display_name="A", age=70, gender="F"),
            PatientMatch(source_id="s2", display_name="B", age=71, gender="F"),
        ],
        data_availability="data_present",
    )
    dumped = result.model_dump(mode="json")
    # last_date_of_visit, practice_name are None for both → pruned.
    assert "omitted_empty_fields" in dumped
    dead = set(dumped["omitted_empty_fields"])
    assert "last_date_of_visit" in dead
    assert "practice_name" in dead
    # age + gender + display_name survive.
    for row in dumped["sample"]:
        assert "age" in row
        assert "display_name" in row
