"""sqlglot 3-mode consent classifier for freeform run_sql (#244).

classify_sql sorts a statement into:
  - non_patient : references no patient-bearing view -> run untouched
  - aggregate   : only COUNT / SUM(CASE 0/1) measures over non-identifying
                  group keys, every UNION arm included -> run then suppress
  - patient_row : anything else touching patient data -> refuse
  - unparseable : sqlglot could not parse it -> refuse (fail-closed)

The classifier is a disclosure control, so its bias is conservative: when in
doubt it must NOT return "aggregate".
"""

import pytest

from agent.consent.sql_gate import classify_sql, references_patient_view


# references_patient_view is unchanged (still the family regex).
@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT COUNT(*) FROM federated_demographic_v", True),
        ("SELECT 1", False),
        ("SELECT code FROM vocab_icd10", False),
    ],
)
def test_references_patient_view(sql, expected):
    assert references_patient_view(sql) is expected


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT code, description FROM vocab_icd10",
        "SELECT COUNT(*) FROM vocab_icd10",
    ],
)
def test_non_patient(sql):
    assert classify_sql(sql) == "non_patient"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM federated_demographic_v",
        "SELECT COUNT(DISTINCT source_id) FROM federated_demographic_v",
        "SELECT gender, COUNT(*) FROM federated_demographic_v GROUP BY gender",
        "SELECT SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) AS f FROM federated_demographic_v",
        # aggregate joined to a non-patient lookup — only projection is a count
        "SELECT COUNT(*) FROM federated_demographic_v d JOIN vocab_icd10 v ON d.x = v.code",
        # every UNION arm is an aggregate
        (
            "SELECT COUNT(*) FROM federated_demographic_v "
            "UNION ALL SELECT COUNT(*) FROM internal_patient_profile_v"
        ),
    ],
)
def test_aggregate(sql):
    assert classify_sql(sql) == "aggregate"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM federated_demographic_v",
        "SELECT source_id FROM federated_demographic_v",
        # bare non-identifier column, no aggregate -> row-level enumeration
        "SELECT age FROM federated_demographic_v",
        # identifier as a GROUP BY key
        "SELECT source_id, COUNT(*) FROM federated_demographic_v GROUP BY source_id",
        "SELECT last_name, COUNT(*) FROM federated_demographic_v GROUP BY last_name",
        # UNION smuggling: clean first arm, row-returning second arm
        (
            "SELECT COUNT(*) FROM federated_demographic_v "
            "UNION ALL SELECT source_id FROM federated_demographic_v"
        ),
        # SUM(CASE) with a non-0/1 literal is an existence oracle for one person
        "SELECT SUM(CASE WHEN last_name = 'Smith' THEN 1000 ELSE 0 END) FROM federated_demographic_v",
        # MIN/MAX return an actual value, not a count
        "SELECT MAX(last_name) FROM federated_demographic_v",
        "SELECT AVG(age) FROM federated_demographic_v",
        # SUM of a raw column is an oracle for a single-row scope
        "SELECT SUM(age) FROM federated_demographic_v",
        # WHERE ... OR TRUE cannot rescue a row-level projection
        "SELECT source_id FROM federated_demographic_v WHERE source_id = 'x' OR TRUE",
        # count measure present but an identifier column also projected
        "SELECT source_id, COUNT(*) FROM federated_demographic_v GROUP BY source_id, gender",
    ],
)
def test_patient_row(sql):
    assert classify_sql(sql) == "patient_row"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(* FROM ((( federated_demographic_v",
        "NOT SQL AT ALL ;;;",
    ],
)
def test_unparseable(sql):
    assert classify_sql(sql) == "unparseable"
