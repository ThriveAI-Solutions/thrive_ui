"""Strictness audit for every tool-input pydantic model.

Each input model must reject unknown fields (extra="forbid"). Weak local
models sometimes invent argument names like `query` or `description`;
silent acceptance would let an underconstrained call execute, as
previously happened with find_patient and search_patients_by_criteria.

Result/Item models are NOT in scope — the LLM does not construct them.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from agent.tools.find_patient import PatientSearchQuery
from agent.tools.search_patients_by_criteria import (
    CohortCriteria,
    DateRange as CohortDateRange,
)
from agent.tools.list_patient_documents import (
    DocumentIndexQuery,
    DateRange as DocumentsDateRange,
)
from agent.tools.get_patient_clinical_data import (
    DateRange as ClinicalDateRange,
    DemographicsQuery,
    EncountersQuery,
    LabsQuery,
    DiagnosesQuery,
    MedicationsQuery,
    ImmunizationsQuery,
    ProceduresQuery,
    ImagingQuery,
)
from agent.tools.summarize_results import SummarizeResultsInput
from agent.tools.make_chart import MakeChartInput
from agent.tools.search_codes import CodeSearchInput
from agent.tools.run_sql import RunSqlInput


# (model_cls, kwargs that satisfy required fields + criterion-required validators)
STRICT_INPUT_MODELS: list[tuple[type[BaseModel], dict]] = [
    (PatientSearchQuery, {"last_name": "Smith"}),
    (CohortCriteria, {"age_min": 0}),
    (CohortDateRange, {}),
    (DocumentIndexQuery, {}),
    (DocumentsDateRange, {}),
    (ClinicalDateRange, {}),
    (DemographicsQuery, {}),
    (EncountersQuery, {}),
    (LabsQuery, {}),
    (DiagnosesQuery, {}),
    (MedicationsQuery, {}),
    (ImmunizationsQuery, {}),
    (ProceduresQuery, {}),
    (ImagingQuery, {}),
    (SummarizeResultsInput, {"question": "describe the labs"}),
    (MakeChartInput, {"question": "glucose over time"}),
    (CodeSearchInput, {"vocabulary": "icd10", "query": "diabetes"}),
    (RunSqlInput, {"sql": "SELECT 1"}),
]


@pytest.mark.parametrize("model_cls,valid_kwargs", STRICT_INPUT_MODELS)
def test_tool_input_model_rejects_extra_fields(model_cls, valid_kwargs):
    """Adding a stray unknown field must raise ValidationError so
    pydantic-ai feeds the error back to the LLM as a retry."""
    with pytest.raises(ValidationError):
        model_cls(**valid_kwargs, totally_made_up_field="nope")


@pytest.mark.parametrize("model_cls,valid_kwargs", STRICT_INPUT_MODELS)
def test_tool_input_model_accepts_valid_baseline(model_cls, valid_kwargs):
    """Sanity: the valid baseline used in the strictness test still
    constructs cleanly — guards against the strictness test passing
    for the wrong reason (e.g., a required field that the model is
    quietly missing)."""
    instance = model_cls(**valid_kwargs)
    assert isinstance(instance, model_cls)
