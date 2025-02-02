import uuid
from helperClasses.Enums import (MessageType, RoleType)

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
            "type": self.type.value,
            "feedback": self.feedback,
            "query": self.query,
            "question": self.question
        }
    
    def generate_guid(self):
        return str(uuid.uuid4())
    
def display_info(self):
    return f"Role: {self.role}, Content: {self.content}, Type: {self.type.value}, Feedback: {self.feedback}, Query: {self.query}, Question: {self.question}"
