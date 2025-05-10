from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import streamlit as st

# Import the function to be tested
from views.chat_bot import Message, MessageType, RoleType, add_message, get_chart


# Mock VannaService globally for this test module
@pytest.fixture(autouse=True)
def mock_vanna_service():
    with patch('views.chat_bot.vn') as mock_vn:
        # Configure default return values for vn methods used in get_chart
        mock_vn.should_generate_chart.return_value = True
        mock_vn.generate_plotly_code.return_value = ("mock_plotly_code", 0.1)
        mock_vn.generate_plot.return_value = (MagicMock(), 0.2) # Assuming generate_plot returns a figure object
        yield mock_vn

@pytest.fixture
def mock_streamlit_session_state():
    # Mock st.session_state
    with patch('views.chat_bot.st.session_state', MagicMock()) as mock_session_state:
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
    with patch('views.chat_bot.add_message') as mock_add_msg:
        yield mock_add_msg


class TestGetChart:
    def test_get_chart_generates_plot_and_adds_message(self, mock_vanna_service, mock_streamlit_session_state, mock_add_message):
        my_question = "Test question"
        sql = "SELECT * FROM test_table"
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})

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


    def test_get_chart_shows_plotly_code_if_enabled(self, mock_vanna_service, mock_streamlit_session_state, mock_add_message):
        my_question = "Test question for code"
        sql = "SELECT * FROM another_table"
        df = pd.DataFrame({'data': [3, 4]})

        # Enable showing plotly code in session state
        mock_streamlit_session_state.get.side_effect = lambda key, default=None: True if key == "show_plotly_code" else default

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


    def test_get_chart_handles_no_chart_generation(self, mock_vanna_service, mock_streamlit_session_state, mock_add_message):
        my_question = "Test no chart"
        sql = "SELECT * FROM empty_table"
        df = pd.DataFrame() # Empty dataframe

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

    def test_get_chart_handles_plot_generation_failure(self, mock_vanna_service, mock_streamlit_session_state, mock_add_message):
        my_question = "Test plot failure"
        sql = "SELECT * FROM fail_table"
        df = pd.DataFrame({'col1': [1]})

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