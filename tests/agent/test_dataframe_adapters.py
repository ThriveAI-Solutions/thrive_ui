"""Tests for agent.dataframe_adapters.

Each adapter flattens a typed tool result into a pandas DataFrame. The
shape rules:
- Each row corresponds to one ClinicalItem / document / sql row.
- Item-type discriminator (e.g., 'lab', 'diagnosis') is preserved as a
  column so multi-domain results stay self-describing.
- None values pass through as NaN; pandas handles that natively.
- Empty result → empty DataFrame with no rows but stable column set
  (well-defined columns for the empty case keep downstream chart code
  from blowing up).
"""

from __future__ import annotations
from datetime import date
import pandas as pd
import pytest

from agent.dataframe_adapters import (
    clinical_result_to_df,
    document_index_result_to_df,
)


def _labs_result():
    from agent.tools.get_patient_clinical_data import ClinicalResult, LabItem

    return ClinicalResult(
        domain="labs",
        items=[
            LabItem(
                source_id="src-1",
                code="2345-7",
                code_type="LOINC",
                name="Glucose",
                result="120",
                clean_result="120",
                unit="mg/dL",
                event_date="2026-04-01",
            ),
            LabItem(
                source_id="src-1",
                code=None,
                code_type=None,
                name="Hgb",
                result="14.2",
                clean_result="14.2",
                unit="g/dL",
                event_date="2026-04-02",
            ),
        ],
        data_availability="data_present",
    )


def test_clinical_result_to_df_labs():
    df = clinical_result_to_df(_labs_result())
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    # item_type column is preserved so heterogeneous unions stay readable
    assert "item_type" in df.columns
    assert df["name"].tolist() == ["Glucose", "Hgb"]
    # None becomes NaN
    assert pd.isna(df["code"].iloc[1])


def test_clinical_result_to_df_empty():
    from agent.tools.get_patient_clinical_data import ClinicalResult

    result = ClinicalResult(domain="labs", items=[], data_availability="no_records_found")
    df = clinical_result_to_df(result)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_clinical_result_to_df_demographics_single_row():
    from agent.tools.get_patient_clinical_data import ClinicalResult, DemographicsItem

    result = ClinicalResult(
        domain="demographics",
        items=[
            DemographicsItem(
                source_id="src-1",
                first_name="John",
                last_name="Smith",
                date_of_birth=date(1962, 5, 3),
                gender="M",
            )
        ],
        data_availability="data_present",
    )
    df = clinical_result_to_df(result)
    assert len(df) == 1
    assert df["first_name"].iloc[0] == "John"


def test_document_index_result_to_df():
    from agent.tools.list_patient_documents import DocumentEntry, DocumentIndexResult

    result = DocumentIndexResult(
        documents=[
            DocumentEntry(
                source_id="src-1",
                event_datetime="2026-03-01",
                name="H&P",
                mnemonic="HP",
                status=None,
                encounter_id=None,
                place_of_service=None,
                location_name=None,
            )
        ],
        data_availability="data_present",
    )
    df = document_index_result_to_df(result)
    assert len(df) == 1
    assert df["name"].iloc[0] == "H&P"


def test_document_index_result_to_df_empty():
    from agent.tools.list_patient_documents import DocumentIndexResult

    result = DocumentIndexResult(documents=[], data_availability="no_records_found")
    df = document_index_result_to_df(result)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@pytest.mark.xfail(reason="RunSqlResult lands in Task 14")
def test_run_sql_result_to_df():
    from agent.tools.run_sql import RunSqlResult
    from agent.dataframe_adapters import run_sql_result_to_df

    result = RunSqlResult(
        sql="SELECT a, b FROM t",
        columns=["a", "b"],
        rows=[[1, "x"], [2, "y"]],
        row_count=2,
        truncated=False,
    )
    df = run_sql_result_to_df(result)
    assert df.columns.tolist() == ["a", "b"]
    assert df["a"].tolist() == [1, 2]
    assert df["b"].tolist() == ["x", "y"]


@pytest.mark.xfail(reason="RunSqlResult lands in Task 14")
def test_run_sql_result_to_df_empty():
    from agent.tools.run_sql import RunSqlResult
    from agent.dataframe_adapters import run_sql_result_to_df

    result = RunSqlResult(
        sql="SELECT 1",
        columns=["a"],
        rows=[],
        row_count=0,
        truncated=False,
    )
    df = run_sql_result_to_df(result)
    assert len(df) == 0
    assert df.columns.tolist() == ["a"]
