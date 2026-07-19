from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.medications import medications_sql


def test_medications_returns_full_list_with_status(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = medications_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 2
    by_name = {r["med_name"]: r for r in rows}
    assert by_name["Metformin"]["status"] == "active"
    assert by_name["Metformin"]["date_stopped"] is None
    assert by_name["Azithromycin"]["status"] == "completed"
    assert by_name["Azithromycin"]["status_date"] == "2026-04-07 00:00"


def test_medications_sql_has_no_rxnorm_filter():
    sql, params = medications_sql(source_id="src-john-1962")
    assert "rxnorm_code IN" not in sql
    assert not any(k.startswith("rx_") for k in params)


def test_medications_filtered_by_date_range(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = medications_sql(
        source_id="src-john-1962",
        start_date="2026-04-01",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["med_name"] == "Azithromycin"


def test_medications_projection_selects_only_expected_warehouse_fields():
    sql, _ = medications_sql(source_id="s")
    projection = sql.split("SELECT", 1)[1].split("FROM", 1)[0]
    assert [column.strip() for column in projection.split(",")] == [
        "source_id",
        "ndc_code",
        "rxnorm_code",
        "med_name",
        "date_prescribed",
        "prescribing_provider_npi",
        "med_strength",
        "med_strength_unit",
        "med_form",
        "med_sig",
        "drug_supply_days",
        "number_of_refills",
        "status",
        "status_date",
        "date_stopped",
    ]


def test_medications_combined_date_filters_and_order_keep_sql_shape():
    sql, params = medications_sql(
        source_id="s",
        start_date="2026-01-01",
        end_date="2026-06-30",
        schema_prefix="dw.",
    )
    normalized_sql = " ".join(sql.split())

    assert "FROM dw.federated_meds_v" in normalized_sql
    assert (
        "WHERE source_id = :source_id AND date_prescribed >= :start_date "
        "AND date_prescribed <= :end_date" in normalized_sql
    )
    assert normalized_sql.endswith("ORDER BY date_prescribed DESC")
    assert params == {
        "source_id": "s",
        "start_date": "2026-01-01",
        "end_date": "2026-06-30",
    }
