"""search_patients_by_criteria — cohort / population tool.

Phase 4 design §3.1. Operates WITHOUT a selected patient (unlike every
other clinical-data tool); the docstring + system prompt route population
questions here instead of through find_patient + get_patient_clinical_data.

The tool body is added in the next task; this module defines the shapes
the dataframe adapter (Task 3) and the test fixtures (Task 4) reference.
"""

from __future__ import annotations
from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DateRange(BaseModel):
    start: Optional[date] = None
    end: Optional[date] = None


class CohortCriteria(BaseModel):
    diagnosis_codes: Optional[List[str]] = None
    diagnosis_date_range: Optional[DateRange] = None
    medication_rxnorm_codes: Optional[List[str]] = None
    condition_text: Optional[str] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    gender: Optional[Literal["M", "F", "U"]] = None
    facility: Optional[str] = None
    last_visit_after: Optional[date] = None
    last_visit_before: Optional[date] = None
    sample_size: int = Field(default=20, ge=0, le=100)


class PatientMatch(BaseModel):
    source_id: str
    display_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    last_date_of_visit: Optional[date] = None
    practice_name: Optional[str] = None


class CohortResult(BaseModel):
    total_count: int
    sample: List[PatientMatch]
    data_availability: Literal["data_present", "no_records_found", "error"]
    reliability_note: Optional[str] = None
    notes_to_agent: Optional[str] = None
