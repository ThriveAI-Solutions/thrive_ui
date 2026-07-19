"""Dependency container injected into Pydantic AI tools via RunContext.

Per spec §8 — the safety perimeter for the agent. Tools read state here;
they do not see raw database connections, user IDs in prompts, or
session-state directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import FrozenSet, Literal, Optional, TYPE_CHECKING
import pandas as pd

from agent.consent.suppression import SMALL_CELL_THRESHOLD

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
    date_of_death: Optional[date] = None
    # Person-grain EMPI identity (#318). Resolved once during find_patient and
    # carried here so tools/consent gate key on patient_id instead of
    # re-resolving the source_id each call. May be None for legacy sessions.
    internal_patient_id: Optional[int] = None

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
    # Consent enforcement (#244). Default off: enabling it denies every
    # patient the warehouse has no explicit TRUE consent for, so it must not
    # flip on until consent data readiness + HeL policy (threshold, role
    # bypass) are confirmed. Set from [security].enforce_consent.
    enforce_consent: bool = False
    # Roles exempt from consent enforcement (Erie-County-clinical principle,
    # #244). Empty until HeL ratifies the mapping; parsed from
    # [security].consent_bypass_roles by agent.consent.gate.parse_bypass_roles.
    consent_bypass_roles: "FrozenSet[RoleTypeEnum]" = field(default_factory=frozenset)
    # Small-cell suppression floor for the aggregate/cohort exemption (#244/#312).
    # Informal default 11 (not yet ratified); from [security].small_cell_threshold.
    small_cell_threshold: int = SMALL_CELL_THRESHOLD
