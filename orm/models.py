import json
from decimal import Decimal

import pandas as pd
import streamlit as st
from sqlalchemy import TIMESTAMP, Boolean, Column, ForeignKey, Integer, Numeric, String, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from utils.enums import MessageType, RoleType

# Load database settings from st.secrets
db_settings = st.secrets["postgres"]

# Construct the database URL
DATABASE_URL = f"postgresql://{db_settings['user']}:{db_settings['password']}@{db_settings['host']}:{db_settings['port']}/{db_settings['database']}"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for the models to inherit from
Base = declarative_base()


def content_to_json(type, content):
    if type == MessageType.DATAFRAME.value and not isinstance(content, str):
        panda_frame = pd.DataFrame(content)
        return panda_frame.to_json()
    if type == MessageType.PLOTLY_CHART.value and not isinstance(content, str):
        json_content = content.to_json()
        return json_content
    return content


class UserRole(Base):
    __tablename__ = "thrive_user_role"
    id = Column(Integer, primary_key=True)
    role_name = Column(String(50), nullable=False, unique=True)
    description = Column(String)


class User(Base):
    __tablename__ = "thrive_user"
    id = Column(Integer, primary_key=True)
    user_role_id = Column(Integer, ForeignKey("thrive_user_role.id"))
    username = Column(String(50), nullable=False, unique=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    show_sql = Column(Boolean)
    show_table = Column(Boolean)
    show_plotly_code = Column(Boolean)
    show_chart = Column(Boolean)
    show_question_history = Column(Boolean)
    show_summary = Column(Boolean)
    voice_input = Column(Boolean)
    speak_summary = Column(Boolean)
    show_suggested = Column(Boolean)
    show_followup = Column(Boolean)
    show_elapsed_time = Column(Boolean)
    llm_fallback = Column(Boolean)
    min_message_id = Column(Integer)

    role = relationship("UserRole")

    def to_dict(self):
        return {
            "id": self.id,
            "user_role_id": self.user_role_id,
            "username": self.username,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "show_sql": self.show_sql,
            "show_table": self.show_table,
            "show_plotly_code": self.show_plotly_code,
            "show_chart": self.show_chart,
            "show_question_history": self.show_question_history,
            "show_summary": self.show_summary,
            "voice_input": self.voice_input,
            "speak_summary": self.speak_summary,
            "show_suggested": self.show_suggested,
            "show_followup": self.show_followup,
            "show_elapsed_time": self.show_elapsed_time,
            "llm_fallback": self.llm_fallback,
            "min_message_id": self.min_message_id,
        }


class Message(Base):
    __tablename__ = "thrive_message"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("thrive_user.id"))
    role = Column(String(50), nullable=False)
    content = Column(String, nullable=False)
    type = Column(String(50), nullable=False)
    feedback = Column(String(50))
    query = Column(String)
    question = Column(String(1000))
    dataframe = Column(String)
    elapsed_time = Column(Numeric(10, 6))
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def __init__(
        self,
        role: RoleType,
        content: str,
        type: MessageType,
        query: str = None,
        question: str = None,
        dataframe: pd.DataFrame | str = None,
        elapsed_time: Decimal = None,
        user_id: int = None,
    ):
        # Try to get user_id from session_state if not provided directly
        if user_id is None:
            try:
                # Get user_id from session_state cookies if available
                if hasattr(st.session_state, "cookies") and st.session_state.cookies is not None:
                    user_id_str = st.session_state.cookies.get("user_id")
                    if user_id_str:
                        user_id = json.loads(user_id_str)

                # If still None, default to 1 for testing purposes
                if user_id is None:
                    user_id = 1
            except Exception:
                # Default to user_id 1 for testing purposes if there's any error
                user_id = 1

        self.user_id = user_id
        self.role = role.value
        self.content = content_to_json(type.value, content)
        self.type = type.value
        self.feedback = None
        self.query = query
        self.question = question
        self.elapsed_time = elapsed_time

        # Serialize the DataFrame to JSON if provided
        if dataframe is not None:
            # Handle both DataFrame objects and serialized JSON strings
            if isinstance(dataframe, pd.DataFrame):
                self.dataframe = dataframe.to_json(orient="records")  # Convert DataFrame to JSON
            elif isinstance(dataframe, str):
                # Assume it's already a JSON string
                self.dataframe = dataframe
            else:
                # Try to convert other types to string
                self.dataframe = str(dataframe)
        else:
            self.dataframe = None

    def to_dict(self):
        return {"role": self.role, "content": self.content, "question": self.question}

    def save(self):
        session = SessionLocal()

        self.content = content_to_json(self.type, self.content)

        # Add the new message to the session and commit
        if self.id is None:
            session.add(self)
            session.commit()
            # Refresh the db_message object to get the auto-generated fields
            session.refresh(self)
        else:
            session.merge(self)
            session.commit()

        # Close the session
        session.close()

        return self


# Create tables
Base.metadata.create_all(bind=engine)
