from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.tools.get_patient_clinical_data import (
    AdmissionsQuery,
    AdmissionStay,
    _build_admissions_result,
)


def _adapter(engine):
    return AnalyticsDbAdapter(engine=engine, dialect="sqlite")


def test_admissions_result_maps_inpatient_stay(synthetic_db):
    res = _build_admissions_result(
        _adapter(synthetic_db), "src-john-1962", "", AdmissionsQuery(facility_type="inpatient")
    )
    assert res.data_availability == "data_present"
    assert len(res.items) == 1
    stay = res.items[0]
    assert isinstance(stay, AdmissionStay)
    assert stay.item_type == "admission_stay"
    assert stay.is_inpatient_admission is True
    assert stay.visit_number == "V100"
    assert stay.source_id == "src-john-1962"


def test_admissions_any_includes_non_inpatient_flag_false(synthetic_db):
    res = _build_admissions_result(_adapter(synthetic_db), "src-john-1962", "", AdmissionsQuery(facility_type="any"))
    by_visit = {s.visit_number: s for s in res.items}
    assert by_visit["V101"].is_inpatient_admission is False  # ED-only
    assert by_visit["V100"].is_inpatient_admission is True


def test_admissions_no_records(synthetic_db):
    res = _build_admissions_result(_adapter(synthetic_db), "src-nobody", "", AdmissionsQuery())
    assert res.data_availability == "no_records_found"
    assert res.items == []


def test_admissions_query_rejects_removed_field():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AdmissionsQuery(include_discharge_details=False)
