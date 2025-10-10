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


class RoleType(Enum):
    ASSISTANT = "assistant"
    USER = "user"
