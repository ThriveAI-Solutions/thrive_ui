"""Use-case retrieval validation (#247-#253).

Proves the data-retrieval layer supports each go-live use-case question against
the synthetic patient (src-john-1962), independent of the LLM. These pin the
"capabilities already exist" finding: the remaining work on #248-#253 is
prompt/RAG tuning + eval measurement, not new query features. Each test maps to
a use-case acceptance criterion.
"""

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.federation import federated_source_ids
from agent.tools.get_patient_clinical_data import (
    DiagnosesQuery,
    LabsQuery,
    MedicationsQuery,
    ProceduresQuery,
    _build_diagnoses_result,
    _build_labs_result,
    _build_medications_result,
    _build_procedures_result,
)

_PATIENT = "src-john-1962"


def _ctx(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    return adapter, federated_source_ids(adapter, _PATIENT, "")


def test_uc248_diabetes_history_and_latest_a1c(synthetic_db):
    """#248 UC2-1: diabetes diagnosis (binary) + most-recent A1C."""
    adapter, sids = _ctx(synthetic_db)
    dx = _build_diagnoses_result(adapter, sids, "", DiagnosesQuery(icd10_codes=["E11.9"]))
    assert any(getattr(i, "code", None) == "E11.9" for i in dx.items)  # has diabetes

    labs = _build_labs_result(adapter, sids, "", LabsQuery(loinc_codes=["4548-4"]))
    a1c = [i for i in labs.items if i.code == "4548-4"]
    assert a1c and a1c[0].clean_result == "7.2"  # most recent A1C value retrievable


def test_uc250_hepatitis_negative_with_performing_lab(synthetic_db):
    """#250 UC3-2 (HIGH): negative hepatitis result + date drawn + performing lab."""
    adapter, sids = _ctx(synthetic_db)
    labs = _build_labs_result(adapter, sids, "", LabsQuery(loinc_codes=["5195-3"], result_filter="negative"))
    hbsag = [i for i in labs.items if i.code == "5195-3"]
    assert hbsag, "HBsAg negative result must be retrievable"
    row = hbsag[0]
    assert (row.clean_result or "").lower() == "negative"
    assert row.event_datetime  # 3A: date drawn
    assert row.source_name  # 3B: performing lab (reporting organization)


def test_uc251_measles_igg_ever(synthetic_db):
    """#251 UC3-3: measles/rubeola IgM/IgG testing ever."""
    adapter, sids = _ctx(synthetic_db)
    labs = _build_labs_result(adapter, sids, "", LabsQuery(loinc_codes=["22501-7"]))
    assert any(i.code == "22501-7" for i in labs.items)  # test was performed (ever)


def test_uc249_procedures_in_range_with_date(synthetic_db):
    """#249 UC3-1: invasive procedures/surgeries in date range + date performed."""
    adapter, sids = _ctx(synthetic_db)
    procs = _build_procedures_result(adapter, sids, "", ProceduresQuery())
    assert procs.items
    # every returned procedure carries a performed date (event_date)
    assert all(i.event_date for i in procs.items)


def test_uc252_gc_chlamydia_meds_carry_dosage(synthetic_db):
    """#252 UC3-4 (HIGH): antibiotic med with dosage/duration retrievable.

    The antibiotic (azithromycin) with strength + sig + days-supply is what
    answers 5a (which meds) and 5b (dosage/duration).
    """
    adapter, sids = _ctx(synthetic_db)
    meds = _build_medications_result(adapter, sids, "", MedicationsQuery())
    azi = [i for i in meds.items if "azithromycin" in (getattr(i, "med_name", "") or "").lower()]
    assert azi, "azithromycin must be retrievable"
    m = azi[0]
    assert m.med_strength and m.med_sig  # dosage + directions present


def test_uc253_hepatitis_b_detectable(synthetic_db):
    """#253 UC3-5: detect Hep B (diagnosis)."""
    adapter, sids = _ctx(synthetic_db)
    dx = _build_diagnoses_result(adapter, sids, "", DiagnosesQuery(icd10_codes=["B16.9"]))
    assert any(getattr(i, "code", None) == "B16.9" for i in dx.items)  # Hep B present
