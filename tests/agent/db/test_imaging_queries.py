from sqlalchemy import text

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.imaging import imaging_sql


def test_imaging_unions_orders_and_documents(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = imaging_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    sources = {r["source"] for r in rows}
    assert "orders" in sources
    assert "documents" in sources


def test_imaging_filtered_by_modality_xray(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = imaging_sql(source_id="src-john-1962", modality="xray")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) >= 1
    assert all("ray" in (r["description"] or "").lower() or "ray" in (r["mnemonic"] or "").lower() for r in rows)


def test_imaging_body_region_chest_uses_keywords(synthetic_db):
    """body_region='chest' should match Chest X-ray order via keyword expansion."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = imaging_sql(source_id="src-john-1962", body_region="chest")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) >= 1
    # The Chest X-ray order should match
    descriptions = [r["description"].lower() for r in rows if r["description"]]
    assert any("chest" in d for d in descriptions)


def test_imaging_body_region_knee_sql_is_valid(synthetic_db):
    """body_region='knee' generates valid SQL; no knee imaging row in test data so result may be empty."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = imaging_sql(source_id="src-john-1962", body_region="knee")
    rows = adapter.fetch_all(sql, params)
    # knee keyword expansion produces valid SQL; result depends on fixture data
    assert isinstance(rows, list)


def test_imaging_body_region_fallback_for_unknown_region(synthetic_db):
    """Unknown body region should fall back to raw LIKE matching."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = imaging_sql(source_id="src-john-1962", body_region="elbow")
    rows = adapter.fetch_all(sql, params)
    # No imaging rows should match 'elbow' in our test data
    assert isinstance(rows, list)


def _insert_imaging_precision_rows(synthetic_db):
    with synthetic_db.begin() as conn:
        order_rows = [
            ("POCT GLUCOSE METER", "POCT-1"),
            ("Product Notification", "PRODUCT-1"),
            ("CT HEAD WO CONTRAST", "CT-1"),
            ("CHEST CT", "CT-2"),
            ("PET SCAN WHOLE BODY", "PET-1"),
            ("CAT SCAN PETITE", "PETITE-1"),
        ]
        conn.execute(
            text(
                "INSERT INTO federated_orders_v VALUES "
                "('src-john-1962', :code, 'CPT', :name, '2026-04-01 09:00', "
                "'2026-04-01 08:00', '22', 'completed')"
            ),
            [{"name": name, "code": code} for name, code in order_rows],
        )
        document_rows = [
            ("Transition of Care Visit", "AS::ECTOCIVCH"),
            ("Doctor note", "DOCTOR_NOTES"),
            ("CT Report", "CT HEAD"),
        ]
        conn.execute(
            text(
                "INSERT INTO federated_documents_v VALUES "
                "('src-john-1962', '2026-04-01 09:30', :name, :mnemonic, "
                "'final', 'enc-precision', '22', 'Buffalo Medical Group')"
            ),
            [{"name": name, "mnemonic": mnemonic} for name, mnemonic in document_rows],
        )


def _imaging_descriptions(synthetic_db, *, modality=None):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = imaging_sql(source_id="src-john-1962", modality=modality)
    return {row["description"] for row in adapter.fetch_all(sql, params)}


def test_imaging_base_predicates_require_ct_xr_pet_boundaries(synthetic_db):
    _insert_imaging_precision_rows(synthetic_db)

    descriptions = _imaging_descriptions(synthetic_db)

    assert {"CT HEAD WO CONTRAST", "CHEST CT", "PET SCAN WHOLE BODY", "CT Report"} <= descriptions
    assert {
        "POCT GLUCOSE METER",
        "Product Notification",
        "CAT SCAN PETITE",
        "Transition of Care Visit",
        "Doctor note",
    }.isdisjoint(descriptions)
    assert "Radiology Report" in descriptions  # XRREPORT remains a genuine document match.


def test_imaging_modality_predicates_preserve_true_ct_and_pet_matches(synthetic_db):
    _insert_imaging_precision_rows(synthetic_db)

    ct_descriptions = _imaging_descriptions(synthetic_db, modality="ct")
    pet_descriptions = _imaging_descriptions(synthetic_db, modality="pet")

    assert {"CT HEAD WO CONTRAST", "CHEST CT", "CT Report"} <= ct_descriptions
    assert {"POCT GLUCOSE METER", "Transition of Care Visit", "Doctor note"}.isdisjoint(ct_descriptions)
    assert "PET SCAN WHOLE BODY" in pet_descriptions
    assert "CAT SCAN PETITE" not in pet_descriptions
