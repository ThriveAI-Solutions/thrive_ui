from enum import Enum


class MessageType(Enum):
    SQL = "sql"
    PYTHON = "python"
    PLOTLY_CHART = "plotly_chart"
    ST_LINE_CHART = "st_line_chart"
    ST_BAR_CHART = "st_bar_chart"
    ST_AREA_CHART = "st_area_chart"
    ST_SCATTER_CHART = "st_scatter_chart"
    DATAFRAME = "dataframe"
    SUMMARY = "summary"
    FOLLOWUP = "followup"
    ERROR = "error"
    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    PATIENT_CHOOSER = "patient_chooser"


class RoleType(Enum):
    ASSISTANT = "assistant"
    USER = "user"


class ThemeType(Enum):
    THRIVEAI = "ThriveAI"
    WELLTELLAI = "WellTellAI"
    HEALTHELINK = "HEALTHeLINK"


def user_selectable_themes() -> list[str]:
    """Themes shown in user-facing dropdowns. WellTellAI is intentionally
    excluded (#108) but remains in ThemeType so persisted DB rows still
    parse — render-time fallback coerces them to the default theme."""
    hidden = {ThemeType.WELLTELLAI.value}
    return [t.value for t in ThemeType if t.value not in hidden]
