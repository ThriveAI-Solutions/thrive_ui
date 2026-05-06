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
    session.commit()

    row = session.query(ToolCall).one()
    assert row.tool_name == "find_patient"
    assert "src-1" not in row.result_summary
    assert row.success is True
