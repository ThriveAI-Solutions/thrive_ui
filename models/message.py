import streamlit as st
import uuid
from models.database import DB_Message, SessionLocal
from utils.enums import (MessageType, RoleType)
import json
import pandas as pd

class Message:
    def __init__(self, role:RoleType, content:str, type:MessageType, query:str=None, question:str=None):
        self.key = self.generate_guid()
        self.role = role
        self.content = content
        self.type = type
        self.feedback = None
        self.query = query
        self.question = question

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "question": self.question
        }
    
    def generate_guid(self):
        return str(uuid.uuid4())
    
    def save_to_db(self):
        # if self.type == MessageType.SUMMARY.value:
            session = SessionLocal()

            user_id = st.session_state.cookies.get("user_id")
            user_id = json.loads(user_id)

            content = self.content

            if(self.type == MessageType.DATAFRAME.value):
                df = pd.DataFrame(content)
                content = df.to_json()

            # Create a new DB_Message object
            db_message = DB_Message(
                user_id=user_id,
                role=self.role,
                content=content,
                type=self.type,
                feedback=self.feedback,
                query=self.query,
                question=self.question
            )

            # Add the new message to the session and commit
            session.add(db_message)
            session.commit()

            # Close the session
            session.close()
