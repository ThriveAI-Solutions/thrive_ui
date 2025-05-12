import logging

import openai
import streamlit as st

from orm.models import Message

logger = logging.getLogger(__name__)


def chat_gpt(message: Message):
    try:
        if "openai_api" in st.secrets.ai_keys and "openai_model" in st.secrets.ai_keys:
            openai.api_key = st.secrets["ai_keys"]["openai_api"]
            client = openai

            # Generate a response using the OpenAI API.
            stream = client.chat.completions.create(
                model=st.secrets["ai_keys"]["openai_model"],
                messages=[message.to_dict()],
                # messages=[
                #     {"role": m["role"], "content": m["content"]}
                #     for m in st.session_state.messages
                # ],
                stream=True,
            )

            return stream
        else:
            return "No API key found"
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"An error occurred: {e}")
