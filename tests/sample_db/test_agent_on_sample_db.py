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
