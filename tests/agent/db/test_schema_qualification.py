"""Every SQL template in agent/db/queries/ must accept a schema_prefix
kwarg and qualify all warehouse view references with it. Production
Redshift requires `dw.federated_*` because its search_path doesn't
include `dw`; SQLite default of "" leaves the SQL unchanged.

These are pure string-shape tests — they don't execute the SQL, they
just confirm the prefix lands on every view name.
"""

from __future__ import annotations

import pytest

from agent.db.queries.patient import find_patient_sql, related_source_ids_sql
from agent.db.queries.clinical import demographics_sql, encounters_sql
from agent.db.queries.diagnoses import diagnoses_sql
from agent.db.queries.medications import medications_sql
from agent.db.queries.labs import labs_sql
from agent.db.queries.immunizations import immunizations_sql
from agent.db.queries.imaging import imaging_sql
from agent.db.queries.documents import documents_sql
from agent.db.queries.procedures import procedures_sql
from agent.db.queries.allergies import allergies_sql


# (label, callable, expected_views_referenced)
_CASES = [
    (
        "find_patient",
        lambda **kw: find_patient_sql(last_name="x", **kw),
        {"internal_patient_profile_v", "internal_source_reference_v"},
    ),
    (
        "related_source_ids",
        lambda **kw: related_source_ids_sql(**kw),
        {"internal_source_reference_v"},
    ),
    (
        "demographics",
        lambda **kw: demographics_sql(source_id="x", **kw),
        {"federated_demographic_v"},
    ),
    (
        "encounters",
        lambda **kw: encounters_sql(source_id="x", **kw),
        {"federated_encounters_v"},
    ),
    (
        "diagnoses",
        lambda **kw: diagnoses_sql(source_id="x", **kw),
        {"federated_problems_v"},
    ),
    (
        "medications",
        lambda **kw: medications_sql(source_id="x", **kw),
        {"federated_meds_v"},
    ),
    (
        "labs",
        lambda **kw: labs_sql(source_id="x", **kw),
        {"federated_results_v"},
    ),
    (
        "immunizations",
        lambda **kw: immunizations_sql(source_id="x", **kw),
        {"federated_vaccination_v"},
    ),
    (
        "imaging",
        lambda **kw: imaging_sql(source_id="x", **kw),
        {"federated_orders_v", "federated_documents_v"},
    ),
    (
        "documents",
        lambda **kw: documents_sql(source_id="x", **kw),
        {"federated_documents_v"},
    ),
    (
        "procedures",
        lambda **kw: procedures_sql(source_id="x", **kw),
        {
            "federated_orders_v",
            "federated_problems_v",
            "federated_claims_icd_procedure_detail_v",
        },
    ),
    (
        "allergies",
        lambda **kw: allergies_sql(source_id="x", **kw),
        {"federated_allergies_v"},
    ),
]


@pytest.mark.parametrize("label,factory,views", _CASES)
def test_default_prefix_leaves_views_unqualified(label, factory, views):
    """SQLite has no schema concept; default behavior must not break it."""
    sql, _ = factory()
    for view in views:
        assert f"dw.{view}" not in sql, f"[{label}] unexpected dw. prefix in default-prefix SQL"


@pytest.mark.parametrize("label,factory,views", _CASES)
def test_dw_prefix_qualifies_every_view(label, factory, views):
    """When schema_prefix='dw.', every view reference must be schema-qualified."""
    sql, _ = factory(schema_prefix="dw.")
    for view in views:
        assert f"dw.{view}" in sql, f"[{label}] view {view} not qualified in:\n{sql}"
        # The bare unqualified form must not appear in FROM/JOIN positions.
        for clause in ("FROM ", "JOIN "):
            assert f"{clause}{view}" not in sql, f"[{label}] unqualified {clause}{view} still present after dw. prefix"
