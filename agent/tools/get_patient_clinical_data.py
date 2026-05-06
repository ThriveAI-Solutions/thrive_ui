"""get_patient_clinical_data — the workhorse clinical retrieval tool.

Phase 1 implements demographics + encounters. Phase 2 adds:
labs, diagnoses, procedures, immunizations, imaging, medications.

Per spec §7.2: discriminated union by domain. Reads source_id from
ctx.deps.selected_patient. ModelRetry if no selection.
"""

from __future__ import annotations
from datetime import date
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry, RunContext

from agent.deps import AgentDeps
from agent.db.queries.clinical import demographics_sql, encounters_sql
from agent.db.queries.labs import labs_sql
from agent.db.queries.diagnoses import diagnoses_sql
from agent.code_normalizer import normalize_token, variants_for


# --- Query shapes (Phase 1 subset) ----------------------------------


class DateRange(BaseModel):
    start: Optional[date] = None
    end: Optional[date] = None


class DemographicsQuery(BaseModel):
    domain: Literal["demographics"] = "demographics"


class EncountersQuery(BaseModel):
    domain: Literal["encounters"] = "encounters"
    date_range: Optional[DateRange] = None
    facility_type: Optional[Literal["inpatient", "outpatient", "ed", "ltc", "any"]] = "any"


class LabsQuery(BaseModel):
    domain: Literal["labs"] = "labs"
    date_range: Optional[DateRange] = None
    loinc_codes: Optional[List[str]] = None
    test_name_text: Optional[str] = None
    result_filter: Optional[Literal["positive", "negative", "abnormal", "any"]] = "any"


class DiagnosesQuery(BaseModel):
    domain: Literal["diagnoses"] = "diagnoses"
    icd10_codes: Optional[List[str]] = None
    condition_text: Optional[str] = None
    most_recent_only: bool = False


PatientClinicalQuery = Annotated[
    Union[DemographicsQuery, EncountersQuery, LabsQuery, DiagnosesQuery],
    Field(discriminator="domain"),
]


# --- Result shapes --------------------------------------------------


DataAvailability = Literal[
    "data_present",
    "no_records_found",
    "domain_not_available",
    "partial_data",
    "permission_denied",
    "error",
]


class DemographicsItem(BaseModel):
    item_type: Literal["demographics"] = "demographics"
    source_id: str
    first_name: Optional[str]
    last_name: Optional[str]
    date_of_birth: Optional[date]
    gender: Optional[str]


class EncounterItem(BaseModel):
    item_type: Literal["encounter"] = "encounter"
    source_id: str
    encounter_id: Optional[str]
    type: Optional[str]
    status: Optional[str]
    event_datetime: Optional[str]
    location: Optional[str]
    rendering_provider: Optional[str]
    facility_name: Optional[str]
    place_of_service: Optional[str]


class LabItem(BaseModel):
    item_type: Literal["lab"] = "lab"
    source_id: str
    code: Optional[str]
    code_type: Optional[str]
    name: Optional[str]
    result: Optional[str]
    clean_result: Optional[str]
    unit: Optional[str]
    event_datetime: Optional[str]
    service_provider: Optional[str]


class DiagnosisItem(BaseModel):
    item_type: Literal["diagnosis"] = "diagnosis"
    source_id: str
    code: Optional[str]
    code_type: Optional[str]
    diagnosis: Optional[str]
    diagnosis_datetime: Optional[str]
    chronic_ind: Optional[str]
    service_provider_npi: Optional[str]


ClinicalItem = Annotated[
    Union[DemographicsItem, EncounterItem, LabItem, DiagnosisItem],
    Field(discriminator="item_type"),
]


class ClinicalResult(BaseModel):
    domain: str
    items: List[ClinicalItem]
    data_availability: DataAvailability
    notes_to_agent: Optional[str] = None
    reliability_note: Optional[str] = None


# --- Tool implementation --------------------------------------------


def get_patient_clinical_data(
    ctx: RunContext[AgentDeps],
    query: PatientClinicalQuery,
) -> ClinicalResult:
    if ctx.deps.selected_patient is None:
        raise ModelRetry(
            "No patient is currently selected. Call find_patient first or ask the user to select a patient."
        )
    source_id = ctx.deps.selected_patient.source_id
    adapter = ctx.deps.analytics_db

    if isinstance(query, DemographicsQuery):
        sql, params = demographics_sql(source_id=source_id)
        rows = adapter.fetch_all(sql, params)
        if not rows:
            return ClinicalResult(
                domain="demographics",
                items=[],
                data_availability="no_records_found",
            )
        items = [
            DemographicsItem(
                source_id=r["source_id"],
                first_name=r.get("first_name"),
                last_name=r.get("last_name"),
                date_of_birth=(
                    r["date_of_birth"]
                    if isinstance(r.get("date_of_birth"), date)
                    else (date.fromisoformat(r["date_of_birth"]) if r.get("date_of_birth") else None)
                ),
                gender=r.get("gender"),
            )
            for r in rows
        ]
        return ClinicalResult(
            domain="demographics",
            items=items,
            data_availability="data_present",
        )

    if isinstance(query, EncountersQuery):
        dr = query.date_range
        sql, params = encounters_sql(
            source_id=source_id,
            start_date=dr.start.isoformat() if dr and dr.start else None,
            end_date=dr.end.isoformat() if dr and dr.end else None,
            facility_type=query.facility_type,
        )
        rows = adapter.fetch_all(sql, params)
        if not rows:
            return ClinicalResult(
                domain="encounters",
                items=[],
                data_availability="no_records_found",
            )
        items = [
            EncounterItem(
                source_id=r["source_id"],
                encounter_id=r.get("encounter_id"),
                type=r.get("type"),
                status=r.get("status"),
                event_datetime=str(r.get("event_datetime")) if r.get("event_datetime") else None,
                location=r.get("location"),
                rendering_provider=r.get("rendering_provider"),
                facility_name=r.get("facility_name"),
                place_of_service=r.get("place_of_service"),
            )
            for r in rows
        ]
        return ClinicalResult(
            domain="encounters",
            items=items,
            data_availability="data_present",
        )

    if isinstance(query, LabsQuery):
        dr = query.date_range
        sql, params = labs_sql(
            source_id=source_id,
            loinc_codes=query.loinc_codes,
            test_name_text=query.test_name_text,
            start_date=dr.start.isoformat() if dr and dr.start else None,
            end_date=dr.end.isoformat() if dr and dr.end else None,
            result_filter=query.result_filter,
        )
        rows = adapter.fetch_all(sql, params)
        if not rows:
            return ClinicalResult(
                domain="labs",
                items=[],
                data_availability="no_records_found",
            )
        items = [
            LabItem(
                source_id=r["source_id"],
                code=r.get("code"),
                code_type=r.get("code_type"),
                name=r.get("name"),
                result=r.get("result"),
                clean_result=r.get("clean_result"),
                unit=r.get("unit"),
                event_datetime=str(r["event_datetime"]) if r.get("event_datetime") else None,
                service_provider=r.get("service_provider"),
            )
            for r in rows
        ]
        non_loinc = [i for i in items if normalize_token(i.code_type or "") != "loinc"]
        reliability_note = None
        if non_loinc:
            reliability_note = (
                f"LOINC coverage in source data is ~50%; "
                f"{len(non_loinc)} of {len(items)} returned rows use non-LOINC vocabularies."
            )
        return ClinicalResult(
            domain="labs",
            items=items,
            data_availability="data_present",
            reliability_note=reliability_note,
        )

    if isinstance(query, DiagnosesQuery):
        sql, params = diagnoses_sql(
            source_id=source_id,
            icd10_codes=query.icd10_codes,
            condition_text=query.condition_text,
            most_recent_only=query.most_recent_only,
        )
        rows = adapter.fetch_all(sql, params)
        if not rows:
            return ClinicalResult(
                domain="diagnoses",
                items=[],
                data_availability="no_records_found",
            )
        items = [
            DiagnosisItem(
                source_id=r["source_id"],
                code=r.get("code"),
                code_type=r.get("code_type"),
                diagnosis=r.get("diagnosis"),
                diagnosis_datetime=str(r["diagnosis_datetime"]) if r.get("diagnosis_datetime") else None,
                chronic_ind=r.get("chronic_ind"),
                service_provider_npi=r.get("service_provider_npi"),
            )
            for r in rows
        ]
        non_icd10 = [i for i in items if normalize_token(i.code_type or "") != "icd10"]
        reliability_note = None
        if non_icd10:
            reliability_note = (
                f"ICD-10 coverage in source data is ~57%; "
                f"{len(non_icd10)} of {len(items)} returned rows use SNOMED/ICD-9/other vocabularies."
            )
        return ClinicalResult(
            domain="diagnoses",
            items=items,
            data_availability="data_present",
            reliability_note=reliability_note,
        )

    raise NotImplementedError(f"Unhandled domain: {query.domain}")
