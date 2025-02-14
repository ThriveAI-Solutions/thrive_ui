import streamlit as st
import time
import json
from io import StringIO
from utils.vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    is_sql_valid_cached,
    generate_summary_cached,
    write_to_file
)
from utils.communicate import (speak, listen)
from utils.llm_calls import (chat_gpt)
from utils.enums import (MessageType, RoleType)
from orm.functions import (save_user_settings, get_recent_messages)
from orm.models import Message
import pandas as pd

# Initialize session state variables
if "messages" not in st.session_state or st.session_state.messages == []:
    st.session_state.messages = get_recent_messages()
if st.session_state.messages is None:
    st.session_state.messages = []

def set_question(question:str, render = True):
    if question is None:
        # Clear questions history when resetting
        st.session_state.my_question = None
        st.session_state.messages = None
    else:
        # Set question
        st.session_state.my_question = question
        addMessage(Message(RoleType.USER, question, MessageType.TEXT), render)

def get_unique_messages():
    # Assuming st.session_state.messages is a list of dictionaries
    messages = st.session_state.messages

    # Filter messages to ensure uniqueness based on the "content" field
    seen_content = set()
    unique_messages = []
    for message in messages:
        if message.role == RoleType.USER.value:
            content = message.content
            if content not in seen_content:
                seen_content.add(content)
                unique_messages.append(message)

    return unique_messages

def set_feedback(index:int, value: str):
    message = st.session_state.messages[index]
    message.feedback = value
    message.save()
    new_entry = {
        "question": st.session_state.messages[index].question,
        "query": st.session_state.messages[index].query,
    }
    write_to_file(new_entry)

def renderMessage(message:Message, index:int):
   with st.chat_message(message.role):
        match message.type:
            case MessageType.SQL.value:
                st.code(message.content, language="sql", line_numbers=True)
            case MessageType.PYTHON.value:
                st.code(message.content, language="python", line_numbers=True)
            case MessageType.PLOTLY_CHART.value:
                chart = json.loads(message.content)
                st.plotly_chart(chart, key=f"message_{index}")
            case MessageType.ERROR.value:
                st.error(message.content)
            case MessageType.DATAFRAME.value:
                df = pd.read_json(StringIO(message.content))
                st.dataframe(df, key=f"message_{index}")
                # st.markdown(message.content)
            case MessageType.SUMMARY.value:
                st.code(message.content, language=None)
                # Add feedback buttons below the summary
                cols = st.columns([0.1, 0.1, 0.8])
                with cols[0]:
                    st.button(
                        "👍",
                        key=f"thumbs_up_{index}",
                        type="primary" if message.feedback == "up" else "secondary",
                        on_click=set_feedback,
                        args=(index, "up")
                    )
                with cols[1]:
                    st.button(
                        "👎",
                        key=f"thumbs_down_{index}",
                        type="primary" if message.feedback == "down" else "secondary",
                        on_click=set_feedback,
                        args=(index, "down")
                    )
            case MessageType.FOLLOWUP.value:
                 if len(message.content) > 0:
                    st.text(
                        "Here are some possible follow-up questions"
                    )
                    # Print the first 5 follow-up questions
                    for question in message.content[:5]:
                        st.button(question, on_click=set_question, args=(question,), key=message.generate_guid(), use_container_width=True)
            case _:
                st.markdown(message.content)

def addMessage(message:Message, render=True):
    message = message.save()
    st.session_state.messages.append(message)
    if len(st.session_state.messages) > 0 and render:
        renderMessage(st.session_state.messages[-1], len(st.session_state.messages)-1)

def callLLM(my_question:str):
    stream = chat_gpt(Message(RoleType.ASSISTANT, my_question, MessageType.SQL))
    with st.chat_message(RoleType.ASSISTANT.value):
        response = st.write(f"{st.secrets["ai_keys"]["openai_model"]}:", stream)        
        print("response", response) #TODO: why isnt this storing the response
        message = Message(RoleType.ASSISTANT, f"{st.secrets["ai_keys"]["openai_model"]}: {response}", MessageType.TEXT)
        message = message.save()
        st.session_state.messages.append(message)

######### Sidebar settings #########
with st.sidebar.expander("Settings"):    
    st.checkbox("Show SQL", key="show_sql")
    st.checkbox("Show Table", key="show_table")
    # st.checkbox("Show Plotly Code", value=False, key="show_plotly_code")
    st.checkbox("Show Chart", key="show_chart")
    st.checkbox("Show Question History", key="show_question_history")
    st.checkbox("Voice Input", key="voice_input")
    st.checkbox("Speak Summary", key="speak_summary")
    st.checkbox("Show Suggested Questions", key="show_suggested")
    st.checkbox("Show Follow-up Questions", key="show_followup")
    if "openai_api" in st.secrets.ai_keys and "openai_model" in st.secrets.ai_keys:
        st.checkbox("LLM Fallback on Error", key="llm_fallback")
    st.button("Save", on_click=save_user_settings, use_container_width=True)

st.sidebar.button("Reset", on_click=lambda: set_question(None), use_container_width=True, type="primary")

if st.session_state.get("voice_input", True):
    with st.sidebar.popover("🎤 Speak Your Question", use_container_width=True):
        if st.button("Listen", use_container_width=True):
            text = listen()
            if text:
                st.success(f"Recognized text: {text}")
            else:
                st.error("No input detected.")
            if text:
                set_question(text, False)

# Show suggested questions
if st.session_state.get("show_suggested", True):
    with st.sidebar.popover("Click to show suggested questions", use_container_width=True):
        questions = generate_questions_cached()
        for i, question in enumerate(questions):
            time.sleep(0.05)
            button = st.button(
                question,
                on_click=set_question,
                args=(question, False),
                key=f"suggested_question_{i}",
                use_container_width=True,
            )

# Display questions history in sidebar
if st.session_state.get("show_question_history", True):
    with st.sidebar:
        st.title("Question History")
    filtered_messages = get_unique_messages()
    if len(filtered_messages) > 0:
        for past_question in filtered_messages:
            st.sidebar.button(past_question.content, on_click=set_question, args=(past_question.content,), use_container_width=True)
    else:
        st.sidebar.text("No questions asked yet")

# for debugging
# st.sidebar.write(st.session_state)
######### Sidebar settings #########

st.title("Thrive AI")

if st.session_state.messages == []:
    with st.chat_message(RoleType.ASSISTANT.value):
            st.markdown("Ask me a question about your data")

# Populate messages in the chat message component everytime the streamlit is run
index = 0
for message in st.session_state.messages:
    renderMessage(message, index)
    index = index+1

# Always show chat input
chat_input = st.chat_input("Ask me a question about your data")

######### Handle new chat input #########
if chat_input:
    set_question(chat_input)

# Get question from session state
my_question = st.session_state.get("my_question", None)

if my_question:
    sql = generate_sql_cached(question=my_question)
    st.session_state.my_question = None

    if sql:
        if is_sql_valid_cached(sql=sql):
            if st.session_state.get("show_sql", True):
                addMessage(Message(RoleType.ASSISTANT, sql, MessageType.SQL, sql, my_question))
        else:
            addMessage(Message(RoleType.ASSISTANT, sql, MessageType.ERROR, sql, my_question))
            # TODO: not sure if calling the LLM here is the correct spot or not, it seems to be necessary
            if st.session_state.get("llm_fallback", True):
                callLLM(my_question)
            st.stop()

        df = run_sql_cached(sql=sql)

        if df is not None:
            st.session_state["df"] = df

        if st.session_state.get("df") is not None:
            if st.session_state.get("show_table", True):
                df = st.session_state.get("df")
                addMessage(Message(RoleType.ASSISTANT, df, MessageType.DATAFRAME, sql, my_question))

            if should_generate_chart_cached(question=my_question, sql=sql, df=df):
                code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                if st.session_state.get("show_plotly_code", False):
                    addMessage(Message(RoleType.ASSISTANT, code, MessageType.PYTHON, sql, my_question))

                if code is not None and code != "":
                    if st.session_state.get("show_chart", True):
                        fig = generate_plot_cached(code=code, df=df)
                        if fig is not None:
                            addMessage(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, my_question))
                        else:
                            addMessage(Message(RoleType.ASSISTANT, "I couldn't generate a chart", MessageType.ERROR, sql, my_question))

            if st.session_state.get("show_summary", True) or st.session_state.get("speak_summary", True):
                summary = generate_summary_cached(question=my_question, df=df)
                if summary is not None:
                    if st.session_state.get("show_summary", True):
                        addMessage(Message(RoleType.ASSISTANT, summary, MessageType.SUMMARY, sql, my_question))
                                
                    if st.session_state.get("speak_summary", True):
                        speak(summary)
                else:
                    addMessage(Message(RoleType.ASSISTANT, "Could not generate a summary", MessageType.SUMMARY, sql, my_question))
                    if st.session_state.get("speak_summary", True):
                        speak("Could not generate a summary")

            if st.session_state.get("show_followup", True):
                followup_questions = generate_followup_cached(
                    question=my_question, sql=sql, df=df
                )
                st.session_state["df"] = None

                addMessage(Message(RoleType.ASSISTANT, followup_questions, MessageType.FOLLOWUP,  sql, my_question))
    else:
        addMessage(Message(RoleType.ASSISTANT, "I wasn't able to generate SQL for that question", MessageType.ERROR, sql, my_question))
        if st.session_state.get("llm_fallback", True):
            callLLM(my_question)
######### Handle new chat input #########