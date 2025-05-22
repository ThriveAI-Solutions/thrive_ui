import streamlit as st
import requests
import json
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
            response = client.chat.completions.create(
                model=st.secrets["ai_keys"]["openai_model"],
                messages=[message.to_dict()],
                stream=False,
            )

            return response
        else:
            return "No API key found"
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.exception(f"An error occurred: {e}")

def ask_message(message:Message):
    try:
        if "ollama_host" in st.secrets.ai_keys and "ollama_model" in st.secrets.ai_keys:
            return ask_dict([message.to_dict()])
        else:
            chat_gpt(message)
    except Exception as e:
        st.error(f"An error occurred converting message to dictionary: {e}")
        logger.exception(e)

def ask(message:str):
    try:
        message_dict = {"content": message}
        
        return ask_dict(message_dict)
    except Exception as e:
        st.error(f"An error occurred converting string to dictionary: {e}")
        logger.exception(e)

def ask_dict(message:dict):
    try:
        if "ollama_host" in st.secrets.ai_keys and "ollama_model" in st.secrets.ai_keys:
            # Convert the message dictionary to a string
            message_str = json.dumps(message)

            # Set up the request payload
            data = {
                "model": st.secrets["ai_keys"]["ollama_model"],
                "prompt": message_str,
                "stream": False  # Set to True for streaming responses
            }

            # Send the request
            response = requests.post(f"{st.secrets["ai_keys"]["ollama_host"]}/api/generate", json=data)

            if(response.status_code != 200):
                logger.error(f"Error: Received status code {response.status_code}")
                return f"Error: Received status code {response.status_code}"
            
            response_json = response.json()
            
            # Print the response
            logger.info(response_json["response"])

            return response_json["response"]
        else:
            return "No Ollama API key found"
    except Exception as e:
        st.error(f"An error occurred asking ollama: {e}")
        logger.exception(e)
