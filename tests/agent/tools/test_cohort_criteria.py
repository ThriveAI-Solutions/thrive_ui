import pytest
from pydantic import ValidationError

from agent.tools.search_patients_by_criteria import CohortCriteria
from agent.db.queries.cohort import cohort_sql


def test_inpatient_admission_is_standalone_criterion():
    c = CohortCriteria(inpatient_admission=True)
    assert c.inpatient_admission is True


def test_empty_criteria_still_rejected():
    with pytest.raises(ValidationError):
        CohortCriteria()


def test_cohort_sql_guard_accepts_inpatient_admission():
    # duck-typed namespace; cohort_sql must not raise for inpatient-only criteria
    class _C:
        inpatient_admission = True
        inpatient_admission_date_range = None
        sample_size = 0
        # everything else absent -> getattr(..., None)

    cohort_sql(_C(), schema_prefix="")  # must not raise ValueError
