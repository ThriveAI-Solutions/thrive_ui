"""Per spec §8.7 and §11.4 — HIPAA scrubbing assertions.

Run before each PR merge. If a future change accidentally leaks PHI to
audit logs, these tests fail loudly.
"""

import json
import pytest
from agent.audit import AuditLogger, scrub_arguments_json, summarize_result


def test_scrub_run_sql_hashes_predicate_literals():
    raw = {"sql": "SELECT * FROM patients WHERE name = 'John Smith'"}
    scrubbed = json.loads(scrub_arguments_json("run_sql", raw))
    assert "John Smith" not in scrubbed["sql"]
    assert "<HASH:" in scrubbed["sql"]


def test_scrub_other_tools_pass_through():
    raw = {"first_name": "John", "last_name": "Smith"}
    scrubbed = json.loads(scrub_arguments_json("find_patient", raw))
    assert scrubbed == raw


def test_summarize_result_no_dataframe_contents():
    import pandas as pd

    df = pd.DataFrame({"name": ["John"], "ssn": ["123-45-6789"]})
    summary = summarize_result(
        tool_name="run_sql",
        result_obj={"dataframe": df, "row_count": 1, "data_availability": "data_present"},
    )
    assert "John" not in summary
    assert "123-45-6789" not in summary
    assert "row_count=1" in summary or "1 rows" in summary
    assert "data_present" in summary


def test_summarize_result_no_source_id_in_clinical_result():
    summary = summarize_result(
        tool_name="get_patient_clinical_data",
        result_obj={
            "domain": "encounters",
            "items": [{"source_id": "src-secret-123", "encounter_id": "e1"}],
            "data_availability": "data_present",
        },
    )
    assert "src-secret-123" not in summary
    assert "encounters" in summary
    assert "data_present" in summary


def test_audit_logger_writes_row(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from orm.models import Base, ToolCall

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    logger = AuditLogger(session=session, session_id="s1", user_id=1, user_role=1)
    logger.log(
        tool_name="find_patient",
        selected_patient_source_id=None,
        arguments={"first_name": "John"},
        result_obj={"matches": [{"source_id": "src-1"}], "total_unique": 1},
        elapsed_ms=42,
        success=True,
        error=None,
    )
    # No outer commit — AuditLogger.log() commits per call so it does not
    # hold the SQLite write lock open while the agent yields more events.

    row = session.query(ToolCall).one()
    assert row.tool_name == "find_patient"
    assert "src-1" not in row.result_summary
    assert row.success is True


def test_audit_logger_releases_write_lock_immediately(tmp_path):
    """A second connection to the same SQLite file must be able to
    INSERT immediately after AuditLogger.log() returns. If the audit
    session held an open transaction, the second writer would block on
    SQLite's BUSY/locked timeout and fail.
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from orm.models import Base, ToolCall

    db_file = tmp_path / "audit_lock.sqlite3"
    url = f"sqlite:///{db_file}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    audit_session = Session()

    logger = AuditLogger(session=audit_session, session_id="s1", user_id=1, user_role=1)
    logger.log(
        tool_name="find_patient",
        selected_patient_source_id=None,
        arguments={"q": "x"},
        result_obj={"matches": [], "total_unique": 0},
        elapsed_ms=1,
        success=True,
        error=None,
    )

    # Concurrent writer with a tiny timeout — should not block.
    other = create_engine(url, connect_args={"timeout": 0.5})
    with other.begin() as conn:
        conn.execute(text("CREATE TABLE other (id INTEGER)"))
        conn.execute(text("INSERT INTO other VALUES (1)"))


def test_summarize_cohort_result_contains_counts_not_names():
    """A CohortResult.sample contains patient display_names. summarize_result
    must persist counts ONLY — never names or source_ids — per spec §8.7.
    """

    result_obj = {
        "total_count": 147,
        "sample": [
            {
                "source_id": "src-mary-1956",
                "display_name": "Mary Jones",
                "age": 70,
                "gender": "F",
                "last_date_of_visit": "2026-04-15",
                "practice_name": "Kaleida",
            },
            {
                "source_id": "src-susan-1955",
                "display_name": "Susan Park",
                "age": 71,
                "gender": "F",
                "last_date_of_visit": "2026-03-20",
                "practice_name": "Kaleida",
            },
        ],
        "data_availability": "data_present",
        "reliability_note": "ICD-10 coverage in problems ~57%; SNOMED ~25%. ...",
    }

    summary = summarize_result("search_patients_by_criteria", result_obj)

    # Counts present
    assert "total_count=147" in summary
    assert "sample_size=2" in summary

    # PHI MUST NOT appear
    assert "Mary" not in summary
    assert "Jones" not in summary
    assert "Susan" not in summary
    assert "Park" not in summary
    assert "src-mary-1956" not in summary
    assert "src-susan-1955" not in summary
