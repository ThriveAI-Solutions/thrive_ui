import streamlit as st
import time
from utils.vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    is_sql_valid_cached,
    generate_summary_cached
)
from utils.train_vanna import (train, write_to_file)
from utils.communicate import (speak, listen)
from utils.llm_calls import (chat_gpt)
from utils.enums import (MessageType, RoleType)
from models.message import Message
from models.user import (save_user_settings)

# Train Vanna on database schema
train()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

def set_question(question:str):
    if question is None:
        # Clear questions history when resetting
        st.session_state.my_question = None
        st.session_state.messages = []
    else:
        # Set question
        st.session_state.my_question = question
        addMessage(Message(RoleType.USER.value, question, "text"))

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
    #TODO: update feedback in the database
    st.session_state.messages[index].feedback = value
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
                st.plotly_chart(message.content, key=message.key)
            case MessageType.ERROR.value:
                st.error(message.content)
            case MessageType.DATAFRAME.value:
                st.dataframe(message.content, key=message.key)
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

def addMessage(message:Message):
    st.session_state.messages.append(message)
    if len(st.session_state.messages) > 0:
        renderMessage(st.session_state.messages[-1], len(st.session_state.messages)-1)

    message.save_to_db()

def callLLM(my_question:str):
    stream = chat_gpt(Message(RoleType.ASSISTANT.value, my_question, MessageType.SQL))
    with st.chat_message(RoleType.ASSISTANT.value):
        response = st.write_stream(stream)
        st.session_state.messages.append(Message(RoleType.ASSISTANT.value, response, "text"))

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
    st.checkbox("LLM Fallback on Error", key="llm_fallback")
    st.button("Save", on_click=save_user_settings, use_container_width=True)

st.sidebar.button("Reset", on_click=lambda: set_question(None), use_container_width=True, type="primary")

if st.session_state.get("voice_input", True):
    if st.sidebar.button("🎤 Speak Your Question", use_container_width=True):
        text = listen()
        if text:
            st.success(f"Recognized text: {text}")
        else:
            st.error("No input detected.")
        if text:
            set_question(text)

# Show suggested questions
if st.session_state.get("show_suggested", True):
    if st.sidebar.button("Click to show suggested questions", use_container_width=True):
        st.session_state.my_question = None
        questions = generate_questions_cached()
        for i, question in enumerate(questions):
            time.sleep(0.05)
            button = st.button(
                question,
                on_click=set_question,
                args=(question,),
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
                addMessage(Message(RoleType.ASSISTANT.value, sql, "sql", sql, my_question))
        else:            
            addMessage(Message(RoleType.ASSISTANT.value, sql, "error", sql, my_question))
            if st.session_state.get("llm_fallback", True):
                callLLM(my_question)
            st.stop()

        df = run_sql_cached(sql=sql)

        if df is not None:
            st.session_state["df"] = df

        if st.session_state.get("df") is not None:
            if st.session_state.get("show_table", True):
                df = st.session_state.get("df")
                addMessage(Message(RoleType.ASSISTANT.value, df, "dataframe", sql, my_question))

            if should_generate_chart_cached(question=my_question, sql=sql, df=df):
                code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                if st.session_state.get("show_plotly_code", False):
                    addMessage(Message(RoleType.ASSISTANT.value, code, "python", sql, my_question))

                if code is not None and code != "":
                    if st.session_state.get("show_chart", True):
                        fig = generate_plot_cached(code=code, df=df)
                        if fig is not None:
                            addMessage(Message(RoleType.ASSISTANT.value, fig, "plotly_chart", sql, my_question))
                        else:
                            addMessage(Message(RoleType.ASSISTANT.value, "I couldn't generate a chart", "error", sql, my_question))
                            if st.session_state.get("llm_fallback", True):
                                callLLM(my_question)

            if st.session_state.get("show_summary", True) or st.session_state.get("speak_summary", True):
                summary = generate_summary_cached(question=my_question, df=df)
                if summary is not None:
                    if st.session_state.get("show_summary", True):
                        addMessage(Message(RoleType.ASSISTANT.value, summary, "summary", sql, my_question))
                                
                    if st.session_state.get("speak_summary", True):
                        speak(summary)
                else:
                    addMessage(Message(RoleType.ASSISTANT.value, "Could not generate a summary", "summary", sql, my_question))
                    if st.session_state.get("speak_summary", True):
                        speak("Could not generate a summary")

            if st.session_state.get("show_followup", True):
                followup_questions = generate_followup_cached(
                    question=my_question, sql=sql, df=df
                )
                st.session_state["df"] = None

                addMessage(Message(RoleType.ASSISTANT.value, followup_questions, "followup",  sql, my_question))
    else:
        addMessage(Message(RoleType.ASSISTANT.value, "I wasn't able to generate SQL for that question", "error", sql, my_question))
        if st.session_state.get("llm_fallback", True):
            callLLM(my_question)
######### Handle new chat input #########