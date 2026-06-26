"""End-to-end: agent talks to the loaded sample DB.

Requires:
- Docker Postgres running (docker compose up -d)
- Sample dump loaded (uv run python scripts/load_sample_db.py)
- [postgres] in .streamlit/secrets.toml pointing at it
- An LLM configured in [ai_keys] (the agent-end tests, if added later)

Run: uv run pytest -m sample_db

Without a reachable Postgres these tests SKIP rather than fail.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa


def _engine_or_skip() -> sa.Engine:
    """Build a Postgres engine and probe it; skip the test if unreachable."""
    import tomllib

    try:
        pg = tomllib.loads(open(".streamlit/secrets.toml").read())["postgres"]
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"Postgres config not found: {e}")

    url = f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}"
    eng = sa.create_engine(url)
    try:
        with eng.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
            # Verify the sample DB tables are present — internal_source_reference_v
            # is created exclusively by load_sample_db.py, so its absence means the
            # sample dump hasn't been loaded yet.
            table_exists = conn.execute(
                sa.text(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'dw'
                      AND table_name = 'internal_source_reference_v'
                    """
                )
            ).scalar()
            if not table_exists:
                pytest.skip("Sample DB tables not found — run 'uv run python scripts/load_sample_db.py' first")
    except pytest.skip.Exception:
        raise
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Postgres sample DB not reachable: {e}")
    return eng


@pytest.mark.sample_db
def test_can_count_diabetics():
    """SQL-level sanity: the loaded DB returns a non-trivial diabetic cohort."""
    eng = _engine_or_skip()
    with eng.connect() as conn:
        n = conn.execute(
            sa.text(
                """
                SELECT COUNT(DISTINCT isr.patient_id)
                FROM dw.internal_source_reference_v isr
                JOIN dw.federated_problems_v fp ON fp.source_id = isr.source_id
                WHERE isr.empi_rank != 99
                  AND (fp.code IN ('E11.9', 'E11') OR fp.code = '44054006')
                """
            )
        ).scalar()
    assert n is not None and n >= 1, f"Expected diabetic cohort >=1, got {n}"


@pytest.mark.sample_db
def test_can_find_metformin_users():
    eng = _engine_or_skip()
    with eng.connect() as conn:
        n = conn.execute(
            sa.text(
                """
                SELECT COUNT(DISTINCT isr.patient_id)
                FROM dw.internal_source_reference_v isr
                JOIN dw.federated_meds_v fm ON fm.source_id = isr.source_id
                WHERE isr.empi_rank != 99
                  AND fm.rxnorm_code = '6809'
                """
            )
        ).scalar()
    assert n is not None and n >= 0


@pytest.mark.sample_db
def test_admissions_table_present_with_inpatient_rows():
    """Per epic #173 AC: federated_adt_v is part of the V1 whitelist and the
    sample DB must carry rows for inpatient + ED encounters so the
    admissions domain on get_patient_clinical_data can be exercised."""
    eng = _engine_or_skip()
    with eng.connect() as conn:
        # Table must exist.
        n_rows = conn.execute(sa.text("SELECT COUNT(*) FROM dw.federated_adt_v")).scalar()
        assert n_rows is not None and n_rows >= 1, (
            f"federated_adt_v should be populated from inpatient/ED encounters; got {n_rows}"
        )
        # At least one inpatient row — otherwise the admissions tool's
        # facility_type='inpatient' query has nothing to verify against.
        n_inpatient = conn.execute(
            sa.text("SELECT COUNT(*) FROM dw.federated_adt_v WHERE clean_setting = 'INPATIENT'")
        ).scalar()
        assert n_inpatient is not None and n_inpatient >= 1


@pytest.mark.sample_db
def test_admissions_tool_returns_data_for_admitted_patient():
    """Per epic #173 AC: 'Was patient X admitted? When?' must work end-to-end
    against the sample DB for a patient that has admission records."""
    from datetime import datetime
    from unittest.mock import MagicMock
    from agent.deps import AgentDeps, SelectedPatient
    from agent.db.analytics_adapter import AnalyticsDbAdapter
    from agent.tools.get_patient_clinical_data import (
        AdmissionsQuery,
        get_patient_clinical_data,
    )

    eng = _engine_or_skip()
    # Pick the first patient that actually has an INPATIENT ADT row.
    with eng.connect() as conn:
        row = conn.execute(
            sa.text(
                """
                SELECT isr.source_id
                FROM dw.federated_adt_v adt
                JOIN dw.internal_source_reference_v isr
                  ON CAST(isr.patient_id AS VARCHAR) = adt.patient_id
                WHERE adt.clean_setting = 'INPATIENT' AND isr.empi_rank = 1
                LIMIT 1
                """
            )
        ).fetchone()
        if not row:
            pytest.skip("No admitted patient available in loaded sample DB")
        admitted_source_id = row[0]

    adapter = AnalyticsDbAdapter(engine=eng, dialect="postgres", schema="dw")
    deps = AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=SelectedPatient(
            source_id=admitted_source_id,
            display_name="(sample patient)",
            dob=None,
            selected_at=datetime.now(),
            selection_origin="user_click",
        ),
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=adapter,
        rag=None,
        sqlite_session=None,
        run_logger=MagicMock(),
    )
    ctx = MagicMock()
    ctx.deps = deps

    result = get_patient_clinical_data(ctx, AdmissionsQuery(facility_type="inpatient"))
    assert result.domain == "admissions"
    assert result.data_availability == "data_present"
    assert len(result.items) >= 1
    item = result.items[0]
    assert item.admit_date is not None
    assert item.discharge_date is not None
    assert item.event_location is not None
    assert (item.setting or "").upper() == "INPATIENT"


@pytest.mark.sample_db
def test_admissions_tool_distinguishes_no_records_from_no_admissions():
    """Per epic #173 AC: when no admission records exist, the tool must
    explicitly signal that — not silently return [] that looks like 'no
    admissions occurred'. This is the meeting-transcript failure mode."""
    from datetime import datetime
    from unittest.mock import MagicMock
    from agent.deps import AgentDeps, SelectedPatient
    from agent.db.analytics_adapter import AnalyticsDbAdapter
    from agent.tools.get_patient_clinical_data import (
        AdmissionsQuery,
        get_patient_clinical_data,
    )

    eng = _engine_or_skip()
    with eng.connect() as conn:
        # Pick a patient with NO inpatient/ED rows in federated_adt_v.
        row = conn.execute(
            sa.text(
                """
                SELECT isr.source_id
                FROM dw.internal_source_reference_v isr
                WHERE isr.empi_rank = 1
                  AND NOT EXISTS (
                      SELECT 1 FROM dw.federated_adt_v adt
                      WHERE adt.patient_id = CAST(isr.patient_id AS VARCHAR)
                  )
                LIMIT 1
                """
            )
        ).fetchone()
        if not row:
            pytest.skip("Every patient in sample has admissions — cannot test the no-records branch")
        non_admitted_source_id = row[0]

    adapter = AnalyticsDbAdapter(engine=eng, dialect="postgres", schema="dw")
    deps = AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=SelectedPatient(
            source_id=non_admitted_source_id,
            display_name="(sample patient)",
            dob=None,
            selected_at=datetime.now(),
            selection_origin="user_click",
        ),
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=adapter,
        rag=None,
        sqlite_session=None,
        run_logger=MagicMock(),
    )
    ctx = MagicMock()
    ctx.deps = deps

    result = get_patient_clinical_data(ctx, AdmissionsQuery())
    assert result.domain == "admissions"
    # Critical: explicit "no records" signal, not silent empty data_present.
    assert result.data_availability == "no_records_found"
    assert result.items == []


@pytest.mark.sample_db
def test_polyglot_code_types_present_in_problems():
    """Spec §6.3 #1 — multiple spelling variants of the same code system."""
    eng = _engine_or_skip()
    with eng.connect() as conn:
        types = {
            r[0] for r in conn.execute(sa.text("SELECT DISTINCT code_type FROM dw.federated_problems_v")).fetchall()
        }
    icd_variants = {t for t in types if t and "ICD" in t.upper()}
    assert len(icd_variants) >= 2, f"Expected >=2 ICD variants, got {icd_variants}"
    assert "" in types, "Expected at least one empty code_type"
