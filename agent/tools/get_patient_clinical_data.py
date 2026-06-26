"""get_patient_clinical_data — the workhorse clinical retrieval tool.

All domains are implemented: demographics, encounters, labs, diagnoses,
medications, immunizations, procedures, surgeries, imaging, admissions.

Per spec §7.2: discriminated union by domain. Reads source_id from
ctx.deps.selected_patient. ModelRetry if no selection.
"""

from __future__ import annotations
from datetime import date
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import ModelRetry, RunContext

from agent.deps import AgentDeps
from agent.result_compaction import CompactingListResult
from agent.db.queries.clinical import demographics_sql, encounters_sql
from agent.db.queries.labs import labs_sql
from agent.db.queries.diagnoses import diagnoses_sql
from agent.db.queries.medications import medications_sql
from agent.db.queries.immunizations import immunizations_sql
from agent.db.queries.procedures import procedures_sql
from agent.db.queries.surgeries import surgeries_sql
from agent.db.queries.imaging import imaging_sql
from agent.db.queries.adt import admissions_sql
from agent.db.queries.allergies import allergies_sql
from agent.code_normalizer import normalize_token
from agent.codes.allergies import find_drug_allergy_conflicts
from agent.dataframe_adapters import clinical_result_to_df


# --- Query shapes (Phase 1 subset) ----------------------------------


_STRICT = ConfigDict(extra="forbid")


class DateRange(BaseModel):
    model_config = _STRICT

    start: Optional[date] = None
    end: Optional[date] = None


class DemographicsQuery(BaseModel):
    model_config = _STRICT

    domain: Literal["demographics"] = "demographics"


class EncountersQuery(BaseModel):
    model_config = _STRICT

    domain: Literal["encounters"] = "encounters"
    date_range: Optional[DateRange] = None
    facility_type: Optional[Literal["inpatient", "outpatient", "ed", "ltc", "any"]] = "any"


class LabsQuery(BaseModel):
    model_config = _STRICT

    domain: Literal["labs"] = "labs"
    date_range: Optional[DateRange] = None
    loinc_codes: Optional[List[str]] = None
    test_name_text: Optional[str] = None
    result_filter: Optional[Literal["positive", "negative", "abnormal", "any"]] = "any"
    most_recent_only: bool = False


class DiagnosesQuery(BaseModel):
    model_config = _STRICT

    domain: Literal["diagnoses"] = "diagnoses"
    icd10_codes: Optional[List[str]] = None
    condition_text: Optional[str] = None
    most_recent_only: bool = False


class MedicationsQuery(BaseModel):
    model_config = _STRICT

    domain: Literal["medications"] = "medications"
    date_range: Optional[DateRange] = None


class ImmunizationsQuery(BaseModel):
    model_config = _STRICT

    domain: Literal["immunizations"] = "immunizations"
    cvx_codes: Optional[List[str]] = None
    vaccine_text: Optional[str] = None
    date_range: Optional[DateRange] = None


class ProceduresQuery(BaseModel):
    model_config = _STRICT

    domain: Literal["procedures"] = "procedures"
    date_range: Optional[DateRange] = None
    cpt_codes: Optional[List[str]] = None
    procedure_text: Optional[str] = None


class SurgeriesQuery(BaseModel):
    domain: Literal["surgeries"] = "surgeries"
    date_range: Optional[DateRange] = None
    cpt_codes: Optional[List[str]] = None
    procedure_text: Optional[str] = None


class ImagingQuery(BaseModel):
    model_config = _STRICT

    domain: Literal["imaging"] = "imaging"
    date_range: Optional[DateRange] = None
    modality: Optional[Literal["xray", "ct", "mri", "us", "pet", "any"]] = "any"
    body_region: Optional[str] = None


class AdmissionsQuery(BaseModel):
    model_config = _STRICT

    domain: Literal["admissions"] = "admissions"
    date_range: Optional[DateRange] = None
    facility_type: Optional[Literal["inpatient", "ltc", "snf", "ed", "outpatient", "any"]] = "any"


class AllergiesQuery(BaseModel):
    model_config = _STRICT

    domain: Literal["allergies"] = "allergies"
    snomed_codes: Optional[List[str]] = None
    allergen_text: Optional[str] = None
    category: Optional[Literal["drug", "food", "environmental", "contact", "anaphylaxis"]] = None
    include_inactive: bool = False
    date_range: Optional[DateRange] = None


PatientClinicalQuery = Annotated[
    Union[
        DemographicsQuery,
        EncountersQuery,
        LabsQuery,
        DiagnosesQuery,
        MedicationsQuery,
        ImmunizationsQuery,
        ProceduresQuery,
        SurgeriesQuery,
        ImagingQuery,
        AdmissionsQuery,
        AllergiesQuery,
    ],
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
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None


class EncounterItem(BaseModel):
    item_type: Literal["encounter"] = "encounter"
    source_id: str
    encounter_id: Optional[str] = None
    type: Optional[str] = None
    status: Optional[str] = None
    event_datetime: Optional[str] = None
    location: Optional[str] = None
    rendering_provider: Optional[str] = None
    facility_name: Optional[str] = None
    place_of_service: Optional[str] = None


class LabItem(BaseModel):
    item_type: Literal["lab"] = "lab"
    source_id: str
    code: Optional[str] = None
    code_type: Optional[str] = None
    name: Optional[str] = None
    result: Optional[str] = None
    clean_result: Optional[str] = None
    unit: Optional[str] = None
    event_datetime: Optional[str] = None
    service_provider: Optional[str] = None


class DiagnosisItem(BaseModel):
    item_type: Literal["diagnosis"] = "diagnosis"
    source_id: str
    code: Optional[str] = None
    code_type: Optional[str] = None
    diagnosis: Optional[str] = None
    diagnosis_datetime: Optional[str] = None
    chronic_ind: Optional[str] = None
    service_provider_npi: Optional[str] = None


class MedicationItem(BaseModel):
    item_type: Literal["medication"] = "medication"
    source_id: str
    ndc_code: Optional[str] = None
    rxnorm_code: Optional[str] = None
    med_name: Optional[str] = None
    date_prescribed: Optional[str] = None
    prescribing_provider_npi: Optional[str] = None
    med_strength: Optional[str] = None
    med_strength_unit: Optional[str] = None
    med_form: Optional[str] = None
    med_sig: Optional[str] = None
    drug_supply_days: Optional[int] = None
    number_of_refills: Optional[int] = None
    status: Optional[str] = None
    date_stopped: Optional[str] = None


class ImmunizationItem(BaseModel):
    item_type: Literal["immunization"] = "immunization"
    source_id: str
    cvx: Optional[str] = None
    ndc: Optional[str] = None
    vaccine: Optional[str] = None
    manufacturer: Optional[str] = None
    lot: Optional[str] = None
    event_datetime: Optional[str] = None
    location_name: Optional[str] = None
    mvx_code: Optional[str] = None


class ProcedureItem(BaseModel):
    item_type: Literal["procedure"] = "procedure"
    source: Literal["orders", "problems", "claims"]
    # source_id is NULL for the claims branch: federated_claims_icd_procedure_detail_v
    # has no source_id column per redshift_tables.json (Task 4 reconciliation).
    source_id: Optional[str] = None
    code: Optional[str] = None
    code_type: Optional[str] = None
    description: Optional[str] = None
    event_date: Optional[str] = None
    place_of_service: Optional[str] = None
    provider_npi: Optional[str] = None
    facility_name: Optional[str] = None


class SurgeryItem(BaseModel):
    item_type: Literal["surgery"] = "surgery"
    source: Literal["orders", "problems", "claims"]
    source_id: str
    code: Optional[str]
    code_type: Optional[str]
    description: Optional[str]
    event_date: Optional[str]
    place_of_service: Optional[str]
    provider_npi: Optional[str]
    performing_provider: Optional[str]
    provider_ambiguous: bool = False
    facility_name: Optional[str]


class ImagingItem(BaseModel):
    item_type: Literal["imaging"] = "imaging"
    source: Literal["orders", "documents"]
    source_id: str
    code: Optional[str] = None
    code_type: Optional[str] = None
    description: Optional[str] = None
    mnemonic: Optional[str] = None
    event_date: Optional[str] = None
    place_of_service: Optional[str] = None
    location_name: Optional[str] = None


class AdmissionStay(BaseModel):
    item_type: Literal["admission_stay"] = "admission_stay"
    source_id: str
    visit_number: Optional[str] = None
    admit_date: Optional[str] = None
    discharge_date: Optional[str] = None
    setting: Optional[str] = None
    is_inpatient_admission: bool = False
    event_location: Optional[str] = None
    location_type: Optional[str] = None
    admit_from: Optional[str] = None
    discharge_disposition: Optional[str] = None
    discharge_location: Optional[str] = None


class AllergyItem(BaseModel):
    item_type: Literal["allergy"] = "allergy"
    source_id: str
    allergy: Optional[str] = None
    code: Optional[str] = None
    code_type: Optional[str] = None
    # Category is normalized from the warehouse `type` column (e.g.,
    # 'Drug allergy' → 'drug'); raw value preserved in `type_raw` for callers
    # that need the original wording.
    category: Optional[Literal["drug", "food", "environmental", "contact", "anaphylaxis", "other"]] = None
    type_raw: Optional[str] = None
    severity: Optional[str] = None
    status: Optional[str] = None
    onset_date: Optional[str] = None
    event_datetime: Optional[str] = None
    reaction: Optional[str] = None
    comments: Optional[str] = None


ClinicalItem = Annotated[
    Union[
        DemographicsItem,
        EncounterItem,
        LabItem,
        DiagnosisItem,
        MedicationItem,
        ImmunizationItem,
        ProcedureItem,
        SurgeryItem,
        ImagingItem,
        AdmissionStay,
        AllergyItem,
    ],
    Field(discriminator="item_type"),
]


class ClinicalResult(CompactingListResult):
    _list_field = "items"

    domain: str
    items: List[ClinicalItem]
    data_availability: DataAvailability
    notes_to_agent: Optional[str] = None
    reliability_note: Optional[str] = None
    # First-class flag for 'NO KNOWN ALLERGIES'. Only meaningful on
    # allergies-domain results; left False everywhere else.
    negative_assertion: bool = False


# --- Per-domain helpers ---------------------------------------------


def _build_demographics_result(adapter: Any, source_id: str, schema_prefix: str) -> ClinicalResult:
    sql, params = demographics_sql(source_id=source_id, schema_prefix=schema_prefix)
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


def _build_encounters_result(
    adapter: Any, source_id: str, schema_prefix: str, query: EncountersQuery
) -> ClinicalResult:
    dr = query.date_range
    sql, params = encounters_sql(
        source_id=source_id,
        start_date=dr.start.isoformat() if dr and dr.start else None,
        end_date=dr.end.isoformat() if dr and dr.end else None,
        facility_type=query.facility_type,
        schema_prefix=schema_prefix,
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


def _build_labs_result(adapter: Any, source_id: str, schema_prefix: str, query: LabsQuery) -> ClinicalResult:
    dr = query.date_range
    sql, params = labs_sql(
        source_id=source_id,
        loinc_codes=query.loinc_codes,
        test_name_text=query.test_name_text,
        start_date=dr.start.isoformat() if dr and dr.start else None,
        end_date=dr.end.isoformat() if dr and dr.end else None,
        result_filter=query.result_filter,
        most_recent_only=query.most_recent_only,
        schema_prefix=schema_prefix,
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


def _build_diagnoses_result(adapter: Any, source_id: str, schema_prefix: str, query: DiagnosesQuery) -> ClinicalResult:
    sql, params = diagnoses_sql(
        source_id=source_id,
        icd10_codes=query.icd10_codes,
        condition_text=query.condition_text,
        most_recent_only=query.most_recent_only,
        schema_prefix=schema_prefix,
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


def _build_medications_result(
    adapter: Any, source_id: str, schema_prefix: str, query: MedicationsQuery
) -> ClinicalResult:
    dr = query.date_range
    sql, params = medications_sql(
        source_id=source_id,
        start_date=dr.start.isoformat() if dr and dr.start else None,
        end_date=dr.end.isoformat() if dr and dr.end else None,
        schema_prefix=schema_prefix,
    )
    rows = adapter.fetch_all(sql, params)
    if not rows:
        return ClinicalResult(
            domain="medications",
            items=[],
            data_availability="no_records_found",
        )
    items = [
        MedicationItem(
            source_id=r["source_id"],
            ndc_code=r.get("ndc_code"),
            rxnorm_code=r.get("rxnorm_code"),
            med_name=r.get("med_name"),
            date_prescribed=str(r["date_prescribed"]) if r.get("date_prescribed") else None,
            prescribing_provider_npi=r.get("prescribing_provider_npi"),
            med_strength=r.get("med_strength"),
            med_strength_unit=r.get("med_strength_unit"),
            med_form=r.get("med_form"),
            med_sig=r.get("med_sig"),
            drug_supply_days=r.get("drug_supply_days"),
            number_of_refills=r.get("number_of_refills"),
            status=r.get("status"),
            date_stopped=str(r["date_stopped"]) if r.get("date_stopped") else None,
        )
        for r in rows
    ]
    return ClinicalResult(
        domain="medications",
        items=items,
        data_availability="data_present",
    )


def _build_immunizations_result(
    adapter: Any, source_id: str, schema_prefix: str, query: ImmunizationsQuery
) -> ClinicalResult:
    dr = query.date_range
    sql, params = immunizations_sql(
        source_id=source_id,
        cvx_codes=query.cvx_codes,
        vaccine_text=query.vaccine_text,
        start_date=dr.start.isoformat() if dr and dr.start else None,
        end_date=dr.end.isoformat() if dr and dr.end else None,
        schema_prefix=schema_prefix,
    )
    rows = adapter.fetch_all(sql, params)
    if not rows:
        return ClinicalResult(
            domain="immunizations",
            items=[],
            data_availability="no_records_found",
        )
    items = [
        ImmunizationItem(
            source_id=r["source_id"],
            cvx=r.get("cvx"),
            ndc=r.get("ndc"),
            vaccine=r.get("vaccine"),
            manufacturer=r.get("manufacturer"),
            lot=r.get("lot"),
            event_datetime=str(r["event_datetime"]) if r.get("event_datetime") else None,
            location_name=r.get("location_name"),
            mvx_code=r.get("mvx_code"),
        )
        for r in rows
    ]
    return ClinicalResult(
        domain="immunizations",
        items=items,
        data_availability="data_present",
    )


def _build_procedures_result(
    adapter: Any, source_id: str, schema_prefix: str, query: ProceduresQuery
) -> ClinicalResult:
    dr = query.date_range
    sql, params = procedures_sql(
        source_id=source_id,
        cpt_codes=query.cpt_codes,
        procedure_text=query.procedure_text,
        start_date=dr.start.isoformat() if dr and dr.start else None,
        end_date=dr.end.isoformat() if dr and dr.end else None,
        schema_prefix=schema_prefix,
    )
    rows = adapter.fetch_all(sql, params)
    if not rows:
        return ClinicalResult(
            domain="procedures",
            items=[],
            data_availability="no_records_found",
            reliability_note=(
                "Procedure coverage is split across federated_orders_v (CPT subset), "
                "federated_problems_v (ICD-10-PCS), and the claims feed (monthly refresh; "
                "may lag bi-weekly clinical data by up to 30 days)."
            ),
        )
    items = [
        ProcedureItem(
            source=r["source"],
            source_id=r["source_id"],
            code=r.get("code"),
            code_type=r.get("code_type"),
            description=r.get("description"),
            event_date=str(r["event_date"]) if r.get("event_date") else None,
            place_of_service=r.get("place_of_service"),
            provider_npi=r.get("provider_npi"),
            facility_name=r.get("facility_name"),
        )
        for r in rows
    ]
    return ClinicalResult(
        domain="procedures",
        items=items,
        data_availability="data_present",
        reliability_note=(
            "Procedure coverage in this result is from federated_orders_v (CPT "
            "subset) and federated_problems_v (ICD-10-PCS only). The claims-feed "
            "branch (federated_claims_icd_procedure_detail_v) is suppressed in "
            "Phase 3 because that view has no patient identifier; results may "
            "therefore under-report compared to the full claims history."
        ),
    )


def _build_surgeries_result(adapter: Any, source_id: str, schema_prefix: str, query: SurgeriesQuery) -> ClinicalResult:
    dr = query.date_range
    db_dialect = getattr(adapter, "dialect", "sqlite")
    sql, params = surgeries_sql(
        source_id=source_id,
        cpt_codes=query.cpt_codes,
        procedure_text=query.procedure_text,
        start_date=dr.start.isoformat() if dr and dr.start else None,
        end_date=dr.end.isoformat() if dr and dr.end else None,
        schema_prefix=schema_prefix,
        dialect=db_dialect,
    )
    rows = adapter.fetch_all(sql, params)
    reliability = (
        "Surgery identification uses CPT surgery range (10004-69990) for orders "
        "and invasive ICD-10-PCS root operations for problems. Claims procedures "
        "are broadly included. Performing provider is resolved from encounter date "
        "matching and may be unavailable if no encounter aligns. When multiple "
        "providers match the same date, all are listed (comma-separated). Claims "
        "data refreshes monthly and may lag clinical data by up to 30 days."
    )
    if not rows:
        return ClinicalResult(
            domain="surgeries",
            items=[],
            data_availability="no_records_found",
            reliability_note=reliability,
        )
    items = []
    for r in rows:
        raw_provider = r.get("performing_provider")
        ambiguous = isinstance(raw_provider, str) and "," in raw_provider
        items.append(
            SurgeryItem(
                source=r["source"],
                source_id=r["source_id"],
                code=r.get("code"),
                code_type=r.get("code_type"),
                description=r.get("description"),
                event_date=str(r["event_date"]) if r.get("event_date") else None,
                place_of_service=r.get("place_of_service"),
                provider_npi=r.get("provider_npi"),
                performing_provider=raw_provider,
                provider_ambiguous=ambiguous,
                facility_name=r.get("facility_name"),
            )
        )
    return ClinicalResult(
        domain="surgeries",
        items=items,
        data_availability="data_present",
        reliability_note=reliability,
    )


def _build_imaging_result(adapter: Any, source_id: str, schema_prefix: str, query: ImagingQuery) -> ClinicalResult:
    dr = query.date_range
    sql, params = imaging_sql(
        source_id=source_id,
        modality=query.modality,
        body_region=query.body_region,
        start_date=dr.start.isoformat() if dr and dr.start else None,
        end_date=dr.end.isoformat() if dr and dr.end else None,
        schema_prefix=schema_prefix,
    )
    rows = adapter.fetch_all(sql, params)
    impression_note = (
        "Imaging report impressions are NOT stored in this warehouse — only "
        "order/document metadata. Tell the user to retrieve the full report "
        "via HEALTHeLINK or the source EHR if they need the impression text."
    )
    if not rows:
        return ClinicalResult(
            domain="imaging",
            items=[],
            data_availability="no_records_found",
            notes_to_agent=impression_note,
        )
    items = [
        ImagingItem(
            source=r["source"],
            source_id=r["source_id"],
            code=r.get("code"),
            code_type=r.get("code_type"),
            description=r.get("description"),
            mnemonic=r.get("mnemonic"),
            event_date=str(r["event_date"]) if r.get("event_date") else None,
            place_of_service=r.get("place_of_service"),
            location_name=r.get("location_name"),
        )
        for r in rows
    ]
    return ClinicalResult(
        domain="imaging",
        items=items,
        data_availability="data_present",
        notes_to_agent=impression_note,
        reliability_note=(
            "Modality and body region are inferred from order/document text; "
            "accuracy depends on source-system order naming conventions."
        ),
    )


def _build_admissions_result(
    adapter: Any, source_id: str, schema_prefix: str, query: AdmissionsQuery
) -> ClinicalResult:
    dr = query.date_range
    sql, params = admissions_sql(
        source_id=source_id,
        dialect=adapter.dialect,
        facility_type=query.facility_type,
        start_date=dr.start.isoformat() if dr and dr.start else None,
        end_date=dr.end.isoformat() if dr and dr.end else None,
        schema_prefix=schema_prefix,
    )
    rows = adapter.fetch_all(sql, params)
    if not rows:
        return ClinicalResult(
            domain="admissions",
            items=[],
            data_availability="no_records_found",
        )
    items = [
        AdmissionStay(
            source_id=r["source_id"],
            visit_number=r.get("visit_number"),
            admit_date=str(r["admit_date"]) if r.get("admit_date") else None,
            discharge_date=str(r["discharge_date"]) if r.get("discharge_date") else None,
            setting=r.get("setting"),
            is_inpatient_admission=bool(r.get("is_inpatient_admission")),
            event_location=r.get("event_location"),
            location_type=r.get("location_type"),
            admit_from=r.get("admit_from"),
            discharge_disposition=r.get("discharge_disposition"),
            discharge_location=r.get("discharge_location"),
        )
        for r in rows
    ]
    return ClinicalResult(
        domain="admissions",
        items=items,
        data_availability="data_present",
        reliability_note=(
            "Admissions are rolled up to one row per visit when visit_number is "
            "present; rows with missing/blank visit_number are kept separate by "
            "event so unrelated ADT events do not collapse into one stay. "
            "is_inpatient_admission marks visits with inpatient-class evidence "
            "(clean_setting INPATIENT or an A06 outpatient->inpatient conversion), "
            "excluding pre-admit/pending statuses and cancelled admits. A bare "
            "ADMIT/A01 event is not treated as inpatient on its own."
        ),
    )


_TYPE_TO_CATEGORY = {
    "drug allergy": "drug",
    "food allergy": "food",
    "environmental allergy": "environmental",
    "contact allergy": "contact",
    "adverse reaction": "anaphylaxis",
}


_ALLERGIES_RELIABILITY = (
    "Source: federated_allergies_v (dedicated allergies view). Captures "
    "structured allergens, severity, and category. Uncoded allergies that "
    "live only in clinical notes are NOT included."
)


def _normalize_category(type_raw: Optional[str]) -> Optional[str]:
    if not type_raw:
        return None
    return _TYPE_TO_CATEGORY.get(type_raw.strip().lower(), "other")


def _build_allergies_result(adapter: Any, source_id: str, schema_prefix: str, query: AllergiesQuery) -> ClinicalResult:
    dr = query.date_range
    sql, params = allergies_sql(
        source_id=source_id,
        snomed_codes=query.snomed_codes,
        allergen_text=query.allergen_text,
        category=query.category,
        include_inactive=query.include_inactive,
        start_date=dr.start.isoformat() if dr and dr.start else None,
        end_date=dr.end.isoformat() if dr and dr.end else None,
        schema_prefix=schema_prefix,
    )
    rows = adapter.fetch_all(sql, params)

    # No rows at all = no allergy assertion of any kind for this patient.
    if not rows:
        return ClinicalResult(
            domain="allergies",
            items=[],
            data_availability="no_records_found",
            reliability_note=_ALLERGIES_RELIABILITY,
        )

    # Separate NKA rows from real allergy rows. NKA is a first-class
    # negative assertion, not an item.
    real_rows = [r for r in rows if (r.get("allergy") or "").upper() != "NO KNOWN ALLERGIES"]
    has_nka = len(real_rows) < len(rows)

    if not real_rows and has_nka:
        # Pure NKA — patient asserts no known allergies.
        return ClinicalResult(
            domain="allergies",
            items=[],
            data_availability="data_present",
            negative_assertion=True,
            reliability_note=_ALLERGIES_RELIABILITY,
        )

    items = [
        AllergyItem(
            source_id=r["source_id"],
            allergy=r.get("allergy"),
            code=r.get("code"),
            code_type=r.get("code_type"),
            category=_normalize_category(r.get("type")),
            type_raw=r.get("type"),
            severity=r.get("severity"),
            status=r.get("status"),
            onset_date=str(r["onset_date"]) if r.get("onset_date") else None,
            event_datetime=str(r["event_datetime"]) if r.get("event_datetime") else None,
            reaction=r.get("reaction"),
            comments=r.get("comments"),
        )
        for r in real_rows
    ]

    # Drug-allergy soft conflict signal. Only meaningful when there's at
    # least one drug allergy AND an active medication list. The check is a
    # second SQL fetch — adds <1 round-trip; well under tool-call budget.
    notes_to_agent = _maybe_drug_allergy_signal(adapter, source_id, schema_prefix, real_rows)

    return ClinicalResult(
        domain="allergies",
        items=items,
        data_availability="data_present",
        notes_to_agent=notes_to_agent,
        reliability_note=_ALLERGIES_RELIABILITY,
    )


def _maybe_drug_allergy_signal(
    adapter: Any, source_id: str, schema_prefix: str, allergy_rows: list[dict]
) -> Optional[str]:
    """Returns an advisory string when a recorded allergen overlaps an active
    med. Returns None when there's no overlap (or the meds fetch fails)."""
    if not allergy_rows:
        return None
    has_drug_concern = any(
        (r.get("type") or "").lower() == "drug allergy" or _normalize_category(r.get("type")) == "drug"
        for r in allergy_rows
    )
    if not has_drug_concern:
        return None
    try:
        med_sql, med_params = medications_sql(source_id=source_id, schema_prefix=schema_prefix)
        meds = adapter.fetch_all(med_sql, med_params)
    except Exception:
        return None
    conflicts = find_drug_allergy_conflicts(allergies=allergy_rows, medications=meds)
    if not conflicts:
        return None
    pairs = "; ".join(
        f"{c.allergen_label} allergy ↔ active {c.conflicting_med_name} (RxNorm {c.conflicting_rxnorm})"
        for c in conflicts
    )
    return (
        f"Drug-allergy advisory: {pairs}. This is a soft signal for "
        "clinician review only — not a clinical decision-support verdict."
    )


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
    schema_prefix = getattr(adapter, "schema_prefix", "")

    if isinstance(query, DemographicsQuery):
        result = _build_demographics_result(adapter, source_id, schema_prefix)
    elif isinstance(query, EncountersQuery):
        result = _build_encounters_result(adapter, source_id, schema_prefix, query)
    elif isinstance(query, LabsQuery):
        result = _build_labs_result(adapter, source_id, schema_prefix, query)
    elif isinstance(query, DiagnosesQuery):
        result = _build_diagnoses_result(adapter, source_id, schema_prefix, query)
    elif isinstance(query, MedicationsQuery):
        result = _build_medications_result(adapter, source_id, schema_prefix, query)
    elif isinstance(query, ImmunizationsQuery):
        result = _build_immunizations_result(adapter, source_id, schema_prefix, query)
    elif isinstance(query, ProceduresQuery):
        result = _build_procedures_result(adapter, source_id, schema_prefix, query)
    elif isinstance(query, SurgeriesQuery):
        result = _build_surgeries_result(adapter, source_id, schema_prefix, query)
    elif isinstance(query, ImagingQuery):
        result = _build_imaging_result(adapter, source_id, schema_prefix, query)
    elif isinstance(query, AdmissionsQuery):
        result = _build_admissions_result(adapter, source_id, schema_prefix, query)
    elif isinstance(query, AllergiesQuery):
        result = _build_allergies_result(adapter, source_id, schema_prefix, query)
    else:
        raise ModelRetry(f"Unknown clinical query variant: {type(query).__name__}")

    ctx.deps.last_dataframe = clinical_result_to_df(result)
    return result
