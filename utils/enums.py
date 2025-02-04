from enum import Enum

class MessageType(Enum):
    SQL = "sql"
    PYTHON = "python"
    PLOTLY_CHART = "plotly_chart"
    DATAFRAME = "dataframe"
    SUMMARY = "summary"
    FOLLOWUP = "followup"
    ERROR = "error"
    TEXT = "text"

class RoleType(Enum):
    ASSISTANT = "assistant"
    USER = "user"