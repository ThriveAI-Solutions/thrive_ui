"""Adapters from typed tool results to pandas DataFrames.

Phase 3 design §3.2 — tools that produce row sets stash a DataFrame on
ctx.deps.last_dataframe (and the runtime mirrors that to
st.session_state["df"] for slash-command compatibility). The Pydantic
result still flows back to the LLM unchanged; this module only handles
the side-channel DataFrame.

Adapters live here, not on the Pydantic models, so pandas stays out of
the model definitions and the models stay JSON-clean for the audit log.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from agent.tools.get_patient_clinical_data import ClinicalResult
    from agent.tools.list_patient_documents import DocumentIndexResult
    from agent.tools.run_sql import RunSqlResult


def clinical_result_to_df(result: "ClinicalResult") -> pd.DataFrame:
    """Flatten a ClinicalResult into one row per item.

    Preserves the item_type discriminator so heterogeneous unions stay
    readable. Empty results return an empty DataFrame (no rows, no
    enforced columns — downstream code should handle empty).
    """
    if not result.items:
        return pd.DataFrame()
    return pd.DataFrame([item.model_dump(mode="json") for item in result.items])


def document_index_result_to_df(result: "DocumentIndexResult") -> pd.DataFrame:
    if not result.documents:
        return pd.DataFrame()
    return pd.DataFrame([entry.model_dump(mode="json") for entry in result.documents])


def run_sql_result_to_df(result: "RunSqlResult") -> pd.DataFrame:
    """Build a DataFrame from the (columns, rows) pair on a RunSqlResult.

    Empty result returns a zero-row DataFrame with stable column set so
    downstream charts don't fail on missing keys.
    """
    return pd.DataFrame(result.rows, columns=result.columns)
