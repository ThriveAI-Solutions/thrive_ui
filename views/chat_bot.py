import streamlit as st
import time
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
from helperClasses.communicate import (speak, listen)
from helperClasses.auth import (check_authenticate)

check_authenticate()

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
        addMessage({"role": "user", "content": question, "type": "text"})

def get_unique_messages():
    # Assuming st.session_state.messages is a list of dictionaries
    messages = st.session_state.messages

    # Filter messages to ensure uniqueness based on the "content" field
    seen_content = set()
    unique_messages = []
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if content not in seen_content:
                seen_content.add(content)
                unique_messages.append(message)

    return unique_messages

def set_feedback(index:int, value: str):
    st.session_state.messages[index]["feedback"] = value
    # TODO: send this to the database at this point 

def renderMessage(message:object, index:int):
   with st.chat_message(message["role"]):
        match message["type"]:
            case "sql":
                st.code(message["content"], language="sql", line_numbers=True)
            case "python":
                st.code(message["content"], language="python", line_numbers=True)
            case "plotly_chart":
                st.plotly_chart(message["content"])
            case "error":
                st.error(message["content"])
            case "dataframe":
                st.dataframe(message["content"])
            case "summary":
                st.code(message["content"], language=None)
                # Add feedback buttons below the summary
                cols = st.columns([0.1, 0.1, 0.8])
                with cols[0]:
                    st.button(
                        "👍",
                        key=f"thumbs_up_{index}",
                        type="primary" if message["feedback"] == "up" else "secondary",
                        on_click=set_feedback,
                        args=(index, "up")
                    )
                with cols[1]:
                    st.button(
                        "👎",
                        key=f"thumbs_down_{index}",
                        type="primary" if message["feedback"] == "down" else "secondary",
                        on_click=set_feedback,
                        args=(index, "down")
                    )
            case "followup":
                 if len(message["content"]) > 0:
                    st.text(
                        "Here are some possible follow-up questions"
                    )
                    # Print the first 5 follow-up questions
                    for question in message["content"][:5]:
                        st.button(question, on_click=set_question, args=(question))
            case _:
                st.markdown(message["content"])

def addMessage(message:object):
    st.session_state.messages.append(message)
    if len(st.session_state.messages) > 0:
        renderMessage(st.session_state.messages[-1], len(st.session_state.messages)-1)

######### Sidebar settings #########
st.sidebar.button("Reset", on_click=lambda: set_question(None), use_container_width=True,type="primary")

with st.sidebar.expander("Output Settings"):
    st.checkbox("Show SQL", value=False, key="show_sql")
    st.checkbox("Show Table", value=True, key="show_table")
    # st.checkbox("Show Plotly Code", value=False, key="show_plotly_code")
    st.checkbox("Show Chart", value=False, key="show_chart")
    # st.checkbox("Show Summary", value=True, key="show_summary")
    st.checkbox("Speak Summary", value=False, key="speak_summary")
    st.checkbox("Show Follow-up Questions", value=False, key="show_followup")

if st.sidebar.button("🎤 Speak Your Question", use_container_width=True):
    text = listen()
    if text:
        st.success(f"Recognized text: {text}")
    else:
        st.error("No input detected.")
    if text:
        set_question(text)

# TODO: this seems to be broken for whatever reason
# if st.sidebar.button("Click to show suggested questions", use_container_width=True):
#     st.session_state.my_question = None
#     questions = generate_questions_cached()
#     for i, question in enumerate(questions):
#         time.sleep(0.05)
#         button = st.button(
#             question,
#             on_click=set_question,
#             args=(question),
#             use_container_width=True,
#         )

# Display questions history in sidebar
st.sidebar.title("Questions History")
filtered_messages = get_unique_messages()
if len(filtered_messages) > 0:
    for past_question in filtered_messages:
        st.sidebar.button(past_question["content"], on_click=set_question, args=(past_question["content"]), use_container_width=True)
else:
    st.sidebar.text("No questions asked yet")

# for debugging
# st.sidebar.write(st.session_state)
######### Sidebar settings #########

show_plotly_code = False
show_summary = True

st.title("Thrive AI")

if st.session_state.messages == []:
    with st.chat_message("assistant"):
            st.markdown("Ask me a question about your data")

# Populate messages in the chat message component everytime the streamlit is run
index = 0
for message in st.session_state.messages:
    renderMessage(message, index)
    index = index+1

# Always show chat input
chat_input = st.chat_input("Ask me a question about your data")

# Handle new chat input
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
                addMessage({"role": "assistant", "content": sql, "type": "sql", "feedback": None})
        else:
            # addMessage({"role": "assistant", "content": sql, "type": "sql"})
            addMessage({"role": "assistant", "content": sql, "type": "error", "feedback": None})
            st.stop()

        df = run_sql_cached(sql=sql)

        if df is not None:
            st.session_state["df"] = df

        if st.session_state.get("df") is not None:
            if st.session_state.get("show_table", True):
                df = st.session_state.get("df")
                addMessage({"role": "assistant", "content": df, "type": "dataframe", "feedback": None})

            if should_generate_chart_cached(question=my_question, sql=sql, df=df):
                code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                if st.session_state.get("show_plotly_code", False):
                    addMessage({"role": "assistant", "content": code, "type": "python", "feedback": None})

                if code is not None and code != "":
                    if st.session_state.get("show_chart", True):
                        fig = generate_plot_cached(code=code, df=df)
                        if fig is not None:
                            addMessage({"role": "assistant", "content": fig, "type": "plotly_chart", "feedback": None})
                        else:
                            addMessage({"role": "assistant", "content": "I couldn't generate a chart", "type": "error", "feedback": None})

            if st.session_state.get("show_summary", True) or st.session_state.get("speak_summary", True):
                summary = generate_summary_cached(question=my_question, df=df)
                if summary is not None:
                    if st.session_state.get("show_summary", True):
                        addMessage({"role": "assistant", "content": summary, "type": "summary", "feedback": None})
                                
                    if st.session_state.get("speak_summary", True):
                        speak(summary)
                else:
                    addMessage({"role": "assistant", "content": "Could not generate a summary", "type": "summary", "feedback": None})
                    if st.session_state.get("speak_summary", True):
                        speak("Could not generate a summary")

            if st.session_state.get("show_followup", True):
                followup_questions = generate_followup_cached(
                    question=my_question, sql=sql, df=df
                )
                st.session_state["df"] = None

                addMessage({"role": "assistant", "content": followup_questions, "type": "followup", "feedback": None})
    else:
        addMessage({"role": "assistant", "content": "I wasn't able to generate SQL for that question", "type": "error", "feedback": None})