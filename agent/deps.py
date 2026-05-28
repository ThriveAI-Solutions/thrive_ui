"""Dependency container injected into Pydantic AI tools via RunContext.

Per spec §8 — the safety perimeter for the agent. Tools read state here;
they do not see raw database connections, user IDs in prompts, or
session-state directly.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from orm.models import RoleTypeEnum


SelectionOrigin = Literal["user_click", "agent_disambiguation"]


@dataclass
class SelectedPatient:
    source_id: str
    display_name: str
    dob: Optional[date]
    selected_at: datetime
    selection_origin: SelectionOrigin

    def __post_init__(self) -> None:
        if self.selection_origin not in ("user_click", "agent_disambiguation"):
            raise ValueError(f"Invalid selection_origin: {self.selection_origin!r}")


@dataclass
class QueryMeta:
    tool_name: str
    row_count: int
    elapsed_ms: int
    truncated: bool


@dataclass
class AgentDeps:
    user_id: int
    user_role: "RoleTypeEnum"
    session_id: str
    selected_patient: Optional[SelectedPatient]
    last_dataframe: Optional[pd.DataFrame]
    last_sql: Optional[str]
    last_query_meta: Optional[QueryMeta]
    analytics_db: object
    rag: object
    sqlite_session: "Session"
    run_logger: object  # AgentRunLogger | None
    group_id: Optional[str] = None
    user_message_id: Optional[int] = None
    parent_run_id: Optional[str] = None
    resume_reason: Optional[str] = None
