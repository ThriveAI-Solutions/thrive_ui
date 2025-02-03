import streamlit as st
import openai
from models.message import Message

def chat_gpt(message:Message):
    openai.api_key = st.secrets["ai_keys"]["openai_api"]
    client = openai

    # Generate a response using the OpenAI API.
    stream  = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[message.to_dict()],
        # messages=[
        #     {"role": m["role"], "content": m["content"]}
        #     for m in st.session_state.messages
        # ],
        stream=True,
    )

    return stream
