import json
from decimal import Decimal
from unittest.mock import ANY, MagicMock, call, patch

import pandas as pd
import pytest

# Import the function to be tested
from views.chat_bot import Message, MessageType, RoleType, get_chart, render_message


# Mock VannaService globally for this test module
@pytest.fixture(autouse=True)
def mock_vanna_service():
    with patch("views.chat_bot.vn") as mock_vn:
        # Configure default return values for vn methods used in get_chart
        mock_vn.should_generate_chart.return_value = True
        mock_vn.generate_plotly_code.return_value = ("mock_plotly_code", 0.1)
        mock_vn.generate_plot.return_value = (MagicMock(), 0.2)  # Assuming generate_plot returns a figure object
        yield mock_vn


@pytest.fixture
def mock_streamlit_session_state():
    # Mock st.session_state
    with patch("views.chat_bot.st.session_state", MagicMock()) as mock_session_state:
        # Set default session state values if needed
        mock_session_state.get = MagicMock(side_effect=lambda key, default=None: default)

        # Mock cookies attribute and its get method
        mock_cookies = MagicMock()
        # Ensure user_id returns a string that json.loads can parse
        mock_cookies.get.side_effect = lambda key: '"1"' if key == "user_id" else None
        mock_session_state.cookies = mock_cookies
        yield mock_session_state


@pytest.fixture
def mock_add_message():
    with patch("views.chat_bot.add_message") as mock_add_msg:
        yield mock_add_msg


class TestGetChart:
    def test_get_chart_generates_plot_and_adds_message(
        self, mock_vanna_service, mock_streamlit_session_state, mock_add_message
    ):
        my_question = "Test question"
        sql = "SELECT * FROM test_table"
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

        # Call the function
        get_chart(my_question, sql, df)

        # Assert that VannaService methods were called
        mock_vanna_service.should_generate_chart.assert_called_once_with(question=my_question, sql=sql, df=df)
        mock_vanna_service.generate_plotly_code.assert_called_once_with(question=my_question, sql=sql, df=df)
        mock_vanna_service.generate_plot.assert_called_once_with(code="mock_plotly_code", df=df)

        # Assert that add_message was called to add the plot
        # We expect one call to add_message for the plot
        assert mock_add_message.call_count == 1

        # Check the arguments of the call to add_message
        args, kwargs = mock_add_message.call_args
        message_arg = args[0]

        assert isinstance(message_arg, Message)
        assert message_arg.role == RoleType.ASSISTANT.value
        assert message_arg.type == MessageType.PLOTLY_CHART.value
        assert message_arg.query == sql
        assert message_arg.question == my_question
        # The content of the message should be the figure returned by generate_plot, after to_json() is called
        assert message_arg.content == mock_vanna_service.generate_plot.return_value[0].to_json()

    def test_get_chart_shows_plotly_code_if_enabled(
        self, mock_vanna_service, mock_streamlit_session_state, mock_add_message
    ):
        my_question = "Test question for code"
        sql = "SELECT * FROM another_table"
        df = pd.DataFrame({"data": [3, 4]})

        # Enable showing plotly code in session state
        mock_streamlit_session_state.get.side_effect = (
            lambda key, default=None: True if key == "show_plotly_code" else default
        )

        get_chart(my_question, sql, df)

        # Assert that add_message was called twice (once for code, once for plot)
        assert mock_add_message.call_count == 2

        # Check the first call (for Python code)
        code_call_args, _ = mock_add_message.call_args_list[0]
        code_message_arg = code_call_args[0]
        assert isinstance(code_message_arg, Message)
        assert code_message_arg.type == MessageType.PYTHON.value
        assert code_message_arg.content == "mock_plotly_code"
        assert code_message_arg.query == sql
        assert code_message_arg.question == my_question

        # Check the second call (for the plot)
        plot_call_args, _ = mock_add_message.call_args_list[1]
        plot_message_arg = plot_call_args[0]
        assert isinstance(plot_message_arg, Message)
        assert plot_message_arg.type == MessageType.PLOTLY_CHART.value

    def test_get_chart_handles_no_chart_generation(
        self, mock_vanna_service, mock_streamlit_session_state, mock_add_message
    ):
        my_question = "Test no chart"
        sql = "SELECT * FROM empty_table"
        df = pd.DataFrame()  # Empty dataframe

        # Mock vn.should_generate_chart to return False
        mock_vanna_service.should_generate_chart.return_value = False

        get_chart(my_question, sql, df)

        # Assert that generate_plotly_code and generate_plot were NOT called
        mock_vanna_service.generate_plotly_code.assert_not_called()
        mock_vanna_service.generate_plot.assert_not_called()

        # Assert that add_message was called with an error message
        mock_add_message.assert_called_once()
        args, kwargs = mock_add_message.call_args
        message_arg = args[0]

        assert isinstance(message_arg, Message)
        assert message_arg.role == RoleType.ASSISTANT.value
        assert message_arg.type == MessageType.ERROR.value
        assert "unable to generate a chart" in message_arg.content.lower()
        assert message_arg.query == sql
        assert message_arg.question == my_question

    def test_get_chart_handles_plot_generation_failure(
        self, mock_vanna_service, mock_streamlit_session_state, mock_add_message
    ):
        my_question = "Test plot failure"
        sql = "SELECT * FROM fail_table"
        df = pd.DataFrame({"col1": [1]})

        # Mock generate_plot to return None (failure)
        mock_vanna_service.generate_plot.return_value = (None, 0.2)

        get_chart(my_question, sql, df)

        assert mock_add_message.call_count == 1
        args, kwargs = mock_add_message.call_args
        message_arg = args[0]

        assert isinstance(message_arg, Message)
        assert message_arg.role == RoleType.ASSISTANT.value
        assert message_arg.type == MessageType.ERROR.value
        assert message_arg.content == "I couldn't generate a chart"
        assert message_arg.query == sql
        assert message_arg.question == my_question


class TestRenderMessage:
    def common_asserts(self, mock_st, expected_role):
        mock_st.chat_message.assert_called_once_with(expected_role)
        # Check that the context manager was entered
        mock_st.chat_message.return_value.__enter__.assert_called_once()

    @patch("views.chat_bot.generate_guid")
    @patch("views.chat_bot.set_question")
    @patch("views.chat_bot.get_chart")
    @patch("views.chat_bot.add_message")
    @patch("views.chat_bot.get_followup_questions")
    @patch("views.chat_bot.speak")
    @patch("views.chat_bot.set_feedback")
    @patch("views.chat_bot.st")
    def test_render_sql_message_with_elapsed_time(
        self,
        mock_st,
        mock_set_feedback,
        mock_speak,
        mock_get_followup_questions,
        mock_add_message,
        mock_get_chart,
        mock_set_question,
        mock_generate_guid,
    ):
        mock_st.session_state.get.return_value = True  # show_elapsed_time = True
        msg = Message(role=RoleType.ASSISTANT, content="SELECT 1;", type=MessageType.SQL, elapsed_time=Decimal("0.123"))

        render_message(msg, 0)

        self.common_asserts(mock_st, RoleType.ASSISTANT.value)
        mock_st.write.assert_called_once_with("Elapsed Time: 0.123")
        mock_st.code.assert_called_once_with("SELECT 1;", language="sql", line_numbers=True)

    @patch("views.chat_bot.generate_guid")
    @patch("views.chat_bot.set_question")
    @patch("views.chat_bot.get_chart")
    @patch("views.chat_bot.add_message")
    @patch("views.chat_bot.get_followup_questions")
    @patch("views.chat_bot.speak")
    @patch("views.chat_bot.set_feedback")
    @patch("views.chat_bot.st")
    def test_render_sql_message_without_elapsed_time(
        self,
        mock_st,
        mock_set_feedback,
        mock_speak,
        mock_get_followup_questions,
        mock_add_message,
        mock_get_chart,
        mock_set_question,
        mock_generate_guid,
    ):
        mock_st.session_state.get.return_value = False  # show_elapsed_time = False
        msg = Message(role=RoleType.USER, content="SELECT 2;", type=MessageType.SQL, elapsed_time=Decimal("0.456"))

        render_message(msg, 1)

        self.common_asserts(mock_st, RoleType.USER.value)
        mock_st.write.assert_not_called()
        mock_st.code.assert_called_once_with("SELECT 2;", language="sql", line_numbers=True)

    @patch("views.chat_bot.st")
    def test_render_python_message(self, mock_st):
        msg = Message(role=RoleType.ASSISTANT, content="print('hello')", type=MessageType.PYTHON)

        render_message(msg, 2)

        self.common_asserts(mock_st, RoleType.ASSISTANT.value)
        mock_st.code.assert_called_once_with("print('hello')", language="python", line_numbers=True)

    @patch("views.chat_bot.st")
    def test_render_plotly_chart_message(self, mock_st):
        mock_st.session_state.get.return_value = True
        chart_data = {"data": [], "layout": {}}
        msg = Message(
            role=RoleType.ASSISTANT,
            content=json.dumps(chart_data),
            type=MessageType.PLOTLY_CHART,
            elapsed_time=Decimal("0.321"),
        )

        render_message(msg, 3)

        self.common_asserts(mock_st, RoleType.ASSISTANT.value)
        mock_st.write.assert_called_once_with("Elapsed Time: 0.321")
        mock_st.plotly_chart.assert_called_once_with(chart_data, key="message_3")

    @patch("views.chat_bot.st")
    def test_render_error_message(self, mock_st):
        msg = Message(role=RoleType.ASSISTANT, content="An error occurred", type=MessageType.ERROR)

        render_message(msg, 4)

        self.common_asserts(mock_st, RoleType.ASSISTANT.value)
        mock_st.error.assert_called_once_with("An error occurred")

    @patch("views.chat_bot.st")
    def test_render_dataframe_message(self, mock_st):
        df_dict = {"col1": [1, 2], "col2": ["a", "b"]}
        df_json = pd.DataFrame(df_dict).to_json()
        msg = Message(role=RoleType.ASSISTANT, content=df_json, type=MessageType.DATAFRAME)

        with patch("pandas.read_json", return_value=pd.DataFrame(df_dict)) as mock_read_json:
            render_message(msg, 5)

        self.common_asserts(mock_st, RoleType.ASSISTANT.value)
        mock_read_json.assert_called_once()
        # We check that st.dataframe is called with a pandas DataFrame.
        # The first argument of the first call to mock_st.dataframe
        called_df = mock_st.dataframe.call_args[0][0]
        pd.testing.assert_frame_equal(called_df, pd.DataFrame(df_dict))
        assert mock_st.dataframe.call_args[1]["key"] == "message_5"

    @patch("views.chat_bot.generate_guid", return_value="mock_guid")
    @patch("views.chat_bot.set_question")
    @patch("views.chat_bot.get_chart")
    @patch("views.chat_bot.add_message")
    @patch("views.chat_bot.get_followup_questions")
    @patch("views.chat_bot.speak")
    @patch("views.chat_bot.set_feedback")
    @patch("views.chat_bot.st")
    def test_render_summary_message(
        self,
        mock_st,
        mock_set_feedback,
        mock_speak,
        mock_get_followup_questions,
        mock_add_message,
        mock_get_chart,
        mock_set_question,
        mock_generate_guid,
    ):
        mock_st.session_state.get.return_value = True  # show_elapsed_time
        summary_content = "This is a summary."
        sql_query = "SELECT * FROM summary_table;"
        question_text = "What is the summary?"
        df_for_actions = pd.DataFrame({"column1": [1], "column2": [2]})  # Need at least 2 columns for chart buttons
        df_json_for_actions = df_for_actions.to_json(orient="records")

        msg = Message(
            role=RoleType.ASSISTANT,
            content=summary_content,
            type=MessageType.SUMMARY,
            query=sql_query,
            question=question_text,
            dataframe=df_json_for_actions,  # This is message.dataframe
            elapsed_time=Decimal("0.789"),
        )
        msg.feedback = "up"  # For testing button type

        # Mock popover and expander to be context managers
        mock_st.popover.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.popover.return_value.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=None)

        # Mock columns to return context managers
        mock_cols = [MagicMock(), MagicMock(), MagicMock()]
        for col_mock in mock_cols:
            col_mock.__enter__.return_value = None
            col_mock.__exit__.return_value = None
        mock_st.columns.return_value = mock_cols

        # Also need to mock the 5-column layout for chart buttons
        mock_chart_cols = [MagicMock() for _ in range(5)]
        for col_mock in mock_chart_cols:
            col_mock.__enter__.return_value = None
            col_mock.__exit__.return_value = None

        # Set up columns mock to return different layouts based on parameters
        def columns_side_effect(spec):
            if spec == [0.1, 0.1, 0.6]:
                return mock_cols  # For feedback buttons
            elif spec == (1, 1, 1, 1, 1):
                return mock_chart_cols  # For chart buttons
            else:
                return mock_cols  # Default fallback

        mock_st.columns.side_effect = columns_side_effect

        # Mock for pd.read_json inside the summary rendering for actions
        with patch("pandas.read_json") as mock_pd_read_json:
            mock_pd_read_json.return_value = df_for_actions  # This is my_df

            render_message(msg, 6)

        self.common_asserts(mock_st, RoleType.ASSISTANT.value)
        mock_st.write.assert_called_once_with("Elapsed Time: 0.789")
        mock_st.code.assert_any_call(summary_content, language=None, wrap_lines=True)  # First code call for summary

        # Check that columns was called for both feedback buttons and chart buttons
        mock_st.columns.assert_any_call([0.1, 0.1, 0.6])  # For feedback buttons
        mock_st.columns.assert_any_call((1, 1, 1, 1, 1))  # For chart buttons

        # Check feedback buttons
        mock_st.button.assert_any_call(
            "👍", key="thumbs_up_6", type="primary", on_click=mock_set_feedback, args=(6, "up")
        )
        mock_st.button.assert_any_call(
            "👎", key="thumbs_down_6", type="secondary", on_click=mock_set_feedback, args=(6, "down")
        )

        # Check popover and its contents
        mock_st.popover.assert_called_once_with("Actions", use_container_width=True)
        assert mock_pd_read_json.call_count == 1
        assert mock_pd_read_json.call_args[1]["orient"] == "records"

        # Check buttons within popover (order might vary, use assert_any_call or check call_args_list)
        speak_call = call("Speak Summary", key="speak_summary_6", on_click=ANY)  # ANY for lambda
        follow_up_call = call("Follow-up Questions", key="follow_up_questions_6", on_click=ANY)
        generate_table_call = call("Generate Table", key="table_6")  # No on_click for this one as it's conditional
        generate_graph_call = call("Generate Plotly", key="graph_6", on_click=ANY)

        # Brittle check for button calls, assumes order. A more robust check would inspect call_args_list.
        # This part needs careful checking of exact lambda behavior or restructuring for easier testing if lambdas are complex.
        # For now, checking they are called:
        assert any(c == speak_call for c in mock_st.button.call_args_list), (
            "Speak Summary button not called as expected"
        )
        assert any(c == follow_up_call for c in mock_st.button.call_args_list), (
            "Follow-up Questions button not called as expected"
        )
        assert any(c == generate_table_call for c in mock_st.button.call_args_list), (
            "Generate Table button not called as expected"
        )
        assert any(c == generate_graph_call for c in mock_st.button.call_args_list), (
            "Generate Plotly button not called as expected"
        )

        # Check expander and its content
        mock_st.expander.assert_called_once_with("Show SQL")
        mock_st.code.assert_any_call(
            sql_query, language="sql", line_numbers=True
        )  # Second code call for SQL in expander

        # Test the on_click for speak (example)
        speak_button_call = next(c for c in mock_st.button.call_args_list if c[1].get("key") == "speak_summary_6")
        speak_button_call[1]["on_click"]()  # Execute the lambda
        mock_speak.assert_called_once_with(summary_content)

        # Test on_click for "Generate Table" by simulating button click
        # This is tricky because the add_message is inside an if st.button().
        # We assume the button returns True when clicked for testing the effect.
        # A better way might be to extract the logic from the if block.
        # For now, this test doesn't directly test the add_message call from conditional st.button.

    @patch("views.chat_bot.generate_guid", return_value="mock_guid_fw")
    @patch("views.chat_bot.set_question")
    @patch("views.chat_bot.st")
    def test_render_followup_message(self, mock_st, mock_set_question, mock_generate_guid):
        followup_list = ["Question 1?", "Question 2?"]
        msg = Message(role=RoleType.ASSISTANT, content=str(followup_list), type=MessageType.FOLLOWUP)

        render_message(msg, 7)

        self.common_asserts(mock_st, RoleType.ASSISTANT.value)
        mock_st.text.assert_called_once_with("Here are some possible follow-up questions")

        expected_calls = [
            call(
                "Question 1?",
                on_click=mock_set_question,
                args=("Question 1?",),
                key="mock_guid_fw",
                use_container_width=True,
            ),
            call(
                "Question 2?",
                on_click=mock_set_question,
                args=("Question 2?",),
                key="mock_guid_fw",
                use_container_width=True,
            ),
        ]
        mock_st.button.assert_has_calls(
            expected_calls, any_order=False
        )  # Relies on generate_guid being called for each
        assert mock_generate_guid.call_count == len(followup_list)

    @patch("views.chat_bot.st")
    def test_render_default_message(self, mock_st):
        # Assuming MessageType.TEXT or any unknown type would fall here
        # The Message class init uses type.value, so message.type will be the string
        # Let's simulate an unknown type by using a value not in MessageType explicitly handled
        msg = Message(
            role=RoleType.USER, content="This is a default message.", type=MessageType.TEXT
        )  # type=MessageType.TEXT will hit default

        render_message(msg, 8)

        self.common_asserts(mock_st, RoleType.USER.value)
        mock_st.markdown.assert_called_once_with("This is a default message.")

    @patch("views.chat_bot.generate_guid")
    @patch("views.chat_bot.set_question")
    @patch("views.chat_bot.st")
    def test_render_followup_message_empty_content(self, mock_st, mock_set_question, mock_generate_guid):
        msg = Message(role=RoleType.ASSISTANT, content="[]", type=MessageType.FOLLOWUP)  # Empty list

        render_message(msg, 9)
        self.common_asserts(mock_st, RoleType.ASSISTANT.value)
        mock_st.text.assert_called_once_with("Here are some possible follow-up questions")
        mock_st.button.assert_not_called()  # No buttons for empty list

    @patch("views.chat_bot.generate_guid")
    @patch("views.chat_bot.set_question")
    @patch("views.chat_bot.st")
    def test_render_followup_message_malformed_content(self, mock_st, mock_set_question, mock_generate_guid):
        # The code now has error handling for ast.literal_eval
        # and doesn't raise the ValueError anymore
        malformed_content = "This is not a list"
        msg = Message(role=RoleType.ASSISTANT, content=malformed_content, type=MessageType.FOLLOWUP)

        # Call render_message which should handle the malformed content gracefully
        render_message(msg, 10)

        # Verify that the message was rendered correctly
        self.common_asserts(mock_st, RoleType.ASSISTANT.value)
        mock_st.text.assert_called_once_with("Here are some possible follow-up questions")
        # No buttons should be created for malformed content
        mock_st.button.assert_not_called()
