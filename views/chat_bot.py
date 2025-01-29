import time
import streamlit as st
from helperClasses.vanna_calls import (
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
from helperClasses.train_vanna import (train)
from helperClasses.communicate import (speak, listen, copy_to_clipboard)
print('chat_bot')
# if st.session_state["authentication_status"] is False or st.session_state["authentication_status"] is None:
#     st.switch_page("views/login.py")

# --- HIDE LOGIN NAVIGATION ---
st.markdown(
    """
    <style>
    [data-testid="stSidebarNavItems"] li:first-child {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- HIDE LOGIN NAVIGATION ---

# Train Vanna on database schema
train()

# Initialize session state variables
if "questions_history" not in st.session_state:
    st.session_state.questions_history = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

with st.sidebar.expander("Options"):
    st.checkbox("Conversational Controls", value=False, key="show_conversational_controls")
    st.checkbox("Suggested Questions", value=False, key="show_suggested_questions")
    st.checkbox("Question History", value=False, key="show_question_history")

st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL", value=False, key="show_sql")
st.sidebar.checkbox("Show Table", value=True, key="show_table")
show_plotly_code = False
# st.sidebar.checkbox("Show Plotly Code", value=False, key="show_plotly_code")
st.sidebar.checkbox("Show Chart", value=False, key="show_chart")
show_summary = True
# st.sidebar.checkbox("Show Summary", value=True, key="show_summary")
if st.session_state.get("show_conversational_controls", True):
    st.sidebar.checkbox("Speak Summary", value=False, key="speak_summary")
else:
    st.session_state["speak_summary"] = False
if st.session_state.get("show_suggested_questions", True):
    st.sidebar.checkbox("Show Follow-up Questions", value=False, key="show_followup")
else:
    st.session_state["show_followup"] = False
st.sidebar.button("Reset", on_click=lambda: set_question(None, True), use_container_width=True)

st.title("Thrive AI")
# st.sidebar.write(st.session_state)

def set_feedback(key: str, value: str):
    """Set feedback state for a specific summary"""
    current = st.session_state.get(key)
    # Toggle off if same value, otherwise set new value
    st.session_state[key] = None if current == value else value
    set_question(st.session_state.questions_history[-1], False)

def set_question(question, rerun=False):
    if question is None:
        # Clear questions history when resetting
        st.session_state.questions_history = []
        st.session_state.my_question = None
        st.session_state.is_processing = False
    else:
        # Set question and processing flag
        st.session_state.my_question = question
        st.session_state.is_processing = True
        if rerun is True:
            st.rerun()

# Display questions history in sidebar
if st.session_state.get("show_question_history", True):
    st.sidebar.title("Questions History")
    if len(st.session_state.questions_history) > 0:
        for past_question in st.session_state.questions_history:
            st.sidebar.button(past_question, on_click=set_question, args=(past_question,False), use_container_width=True)
    else:
        st.sidebar.text("No questions asked yet")

if st.session_state.get("show_conversational_controls", True) or st.session_state.get("show_suggested_questions", True):
    assistant_message_suggested = st.chat_message(
        "assistant"
    )

    if st.session_state.get("show_conversational_controls", True) and assistant_message_suggested.button("🎤 Speak Human"):
        text = listen()
        if text:
            st.success(f"Recognized text: {text}")
        else:
            st.error("No input detected.")
        if text:
            set_question(text, True)

    if st.session_state.get("show_suggested_questions", True) and assistant_message_suggested.button("Click to show suggested questions"):
        st.session_state.my_question = None
        questions = generate_questions_cached()
        for i, question in enumerate(questions):
            time.sleep(0.05)
            button = st.button(
                question,
                on_click=set_question,
                args=(question, False),
            )

# Always show chat input
chat_input = st.chat_input("Ask me a question about your data")

# Handle new chat input
if chat_input:
    set_question(chat_input, True)

# Get question from session state
my_question = st.session_state.get("my_question", None)

if my_question and st.session_state.is_processing:
    # Process the question and add to history
    # Add question to history if it's not already there
    if my_question not in st.session_state.questions_history:
        st.session_state.questions_history.append(my_question)
    user_message = st.chat_message("user")
    user_message.write(f"{my_question}")

    sql = generate_sql_cached(question=my_question)
    
    # Clear processing flag and question after processing
    st.session_state.is_processing = False
    st.session_state.my_question = None

    if sql:
        if is_sql_valid_cached(sql=sql):
            if st.session_state.get("show_sql", True):
                assistant_message_sql = st.chat_message(
                    "assistant"
                )
                assistant_message_sql.code(sql, language="sql", line_numbers=True)
        else:
            assistant_message = st.chat_message(
                "assistant"
            )
            assistant_message.write(sql)
            st.stop()

        df = run_sql_cached(sql=sql)

        if df is not None:
            st.session_state["df"] = df

        if st.session_state.get("df") is not None:
            if st.session_state.get("show_table", True):
                df = st.session_state.get("df")
                assistant_message_table = st.chat_message(
                    "assistant"
                )
                if len(df) > 10:
                    assistant_message_table.text("First 10 rows of data")
                    assistant_message_table.dataframe(df.head(10))
                else:
                    assistant_message_table.dataframe(df)

            if should_generate_chart_cached(question=my_question, sql=sql, df=df):

                code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                if st.session_state.get("show_plotly_code", False):
                    assistant_message_plotly_code = st.chat_message(
                        "assistant"
                    )
                    assistant_message_plotly_code.code(
                        code, language="python", line_numbers=True
                    )

                if code is not None and code != "":
                    if st.session_state.get("show_chart", True):
                        assistant_message_chart = st.chat_message(
                            "assistant"
                        )
                        fig = generate_plot_cached(code=code, df=df)
                        if fig is not None:
                            assistant_message_chart.plotly_chart(fig)
                        else:
                            assistant_message_chart.error("I couldn't generate a chart")

            if st.session_state.get("show_summary", True) or st.session_state.get("speak_summary", True):
                assistant_message_summary = st.chat_message("assistant")
                summary = generate_summary_cached(question=my_question, df=df)
                if summary is not None:
                    if st.session_state.get("show_summary", True):
                        # Display summary in a code block for easy copying
                        assistant_message_summary.code(summary, language=None)
                        
                        # Initialize feedback state for this summary if not exists
                        feedback_key = f"feedback_{len(st.session_state.questions_history)}"
                        if feedback_key not in st.session_state:
                            st.session_state[feedback_key] = None
                        
                        # Add feedback buttons below the summary
                        cols = assistant_message_summary.columns([0.1, 0.1, 0.1, 0.7])
                        with cols[0]: #TODO: why does this trigger a redraw?
                            st.button("📋 Copy", 
                            key=f"copy_btn",
                            type="secondary",
                            on_click=copy_to_clipboard, 
                            args=(summary,))
                        with cols[1]:
                            st.button(
                                "👍",
                                key=f"thumbs_up_{len(st.session_state.questions_history)}",
                                type="primary" if st.session_state[feedback_key] == "up" else "secondary",
                                on_click=set_feedback,
                                args=(feedback_key, "up")
                            )
                        with cols[2]:
                            st.button(
                                "👎",
                                key=f"thumbs_down_{len(st.session_state.questions_history)}",
                                type="primary" if st.session_state[feedback_key] == "down" else "secondary",
                                on_click=set_feedback,
                                args=(feedback_key, "down")
                            )
                                
                    if st.session_state.get("speak_summary", True):
                        speak(summary)
                else:
                    if st.session_state.get("speak_summary", True):
                        speak("No results found")

            if st.session_state.get("show_followup", True):
                assistant_message_followup = st.chat_message(
                    "assistant"
                )
                followup_questions = generate_followup_cached(
                    question=my_question, sql=sql, df=df
                )
                st.session_state["df"] = None

                if len(followup_questions) > 0:
                    assistant_message_followup.text(
                        "Here are some possible follow-up questions"
                    )
                    # Print the first 5 follow-up questions
                    for question in followup_questions[:5]:
                        assistant_message_followup.button(question, on_click=set_question, args=(question, True))

    else:
        assistant_message_error = st.chat_message(
            "assistant"
        )
        assistant_message_error.error("I wasn't able to generate SQL for that question")
