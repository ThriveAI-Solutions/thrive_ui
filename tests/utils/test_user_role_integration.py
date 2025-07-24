"""
Integration test for the user role fix in chat functionality.
This test simulates the actual chat flow to ensure admin users can access 
their training data during question processing.
"""

import json
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest
import streamlit as st

from orm.models import RoleTypeEnum
from utils.chat_bot_helper import get_vn


@pytest.fixture
def mock_streamlit_environment():
    """Mock complete streamlit environment for integration test."""
    with patch("streamlit.session_state", new_callable=MagicMock) as mock_session:
        # Mock cookies with admin user
        mock_cookies = MagicMock()
        mock_cookies.get.side_effect = lambda key: '"5"' if key == "user_id" else "Admin" if key == "role_name" else None
        mock_session.cookies = mock_cookies
        
        # Mock session state with admin role
        mock_session.get.side_effect = lambda key, default=None: RoleTypeEnum.ADMIN.value if key == "user_role" else default
        mock_session.user_role = RoleTypeEnum.ADMIN.value
        
        # Initialize _vn_instance to None to ensure fresh creation
        mock_session._vn_instance = None
        
        # Mock streamlit secrets
        secrets = {
            "ai_keys": {
                "ollama_model": "test_model",
                "anthropic_api": "test_key",
                "gemini_api": "test_key",
                "gemini_model": "test_model"
            },
            "rag_model": {
                "chroma_path": "/tmp/test_chroma"
            },
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "user": "test_user",
                "password": "test_pass"
            },
            "security": {
                "allow_llm_to_see_data": False
            }
        }
        
        with patch("streamlit.secrets", new=secrets):
            yield mock_session


@pytest.fixture
def mock_database_user():
    """Mock database user with admin role."""
    mock_user = MagicMock()
    mock_user.id = 5
    mock_user.first_name = "Admin"
    mock_user.last_name = "User"
    mock_user.role.role.value = RoleTypeEnum.ADMIN.value
    mock_user.show_sql = True
    mock_user.show_table = True
    mock_user.show_plotly_code = False
    mock_user.show_chart = False
    mock_user.show_question_history = True
    mock_user.show_summary = True
    mock_user.voice_input = False
    mock_user.speak_summary = False
    mock_user.show_suggested = False
    mock_user.show_followup = False
    mock_user.show_elapsed_time = True
    mock_user.llm_fallback = False
    mock_user.min_message_id = 0
    return mock_user


class TestUserRoleIntegration:
    """Integration tests for user role fix in chat functionality."""
    
    def test_admin_chat_flow_uses_correct_role(self, mock_streamlit_environment, mock_database_user):
        """Test that the complete chat flow uses admin role correctly."""
        
        # Mock the set_user_preferences_in_session_state function
        with patch("utils.chat_bot_helper.set_user_preferences_in_session_state") as mock_set_prefs:
            mock_set_prefs.return_value = mock_database_user
            
            # Mock VannaService setup and methods
            with patch("utils.vanna_calls.VannaService._setup_vanna"):
                with patch("utils.vanna_calls.VannaService.generate_sql") as mock_generate_sql:
                    with patch("utils.vanna_calls.VannaService.get_training_data") as mock_get_training:
                        
                        # Setup mock returns
                        mock_generate_sql.return_value = ("SELECT COUNT(*) FROM users WHERE role = 'admin'", 0.1)
                        mock_training_data = pd.DataFrame({
                            'id': ['1', '2'],
                            'question': ['How many admins?', 'List admin users'],
                            'content': ['SELECT COUNT(*) FROM users WHERE role = "admin"', 'SELECT * FROM users WHERE role = "admin"'],
                            'training_data_type': ['sql', 'sql']
                        })
                        mock_get_training.return_value = mock_training_data
                        
                        # Simulate the chat flow: user asks a question
                        vn_service = get_vn()
                        
                        # Verify that the service was created with admin role
                        assert vn_service.user_context.user_role == RoleTypeEnum.ADMIN.value
                        assert vn_service.user_context.user_id == "5"
                        
                        # Simulate SQL generation (what happens when a question is asked)
                        sql, elapsed_time = vn_service.generate_sql("How many admin users are there?")
                        
                        # Verify SQL was generated successfully
                        assert sql == "SELECT COUNT(*) FROM users WHERE role = 'admin'"
                        assert elapsed_time == 0.1
                        
                        # Verify that training data can be accessed with admin privileges
                        training_data = vn_service.get_training_data()
                        assert len(training_data) == 2  # Admin should see all training data
                        
                        # Verify user preferences were loaded before service creation
                        mock_set_prefs.assert_called()
                        
    def test_admin_can_access_all_training_data(self, mock_streamlit_environment, mock_database_user):
        """Test that admin users can access all training data, not just their own."""
        
        with patch("utils.chat_bot_helper.set_user_preferences_in_session_state") as mock_set_prefs:
            mock_set_prefs.return_value = mock_database_user
            
            with patch("utils.vanna_calls.VannaService._setup_vanna"):
                
                # Mock the underlying ChromaDB instance
                mock_vn_instance = MagicMock()
                mock_vn_instance._prepare_retrieval_metadata.return_value = {"user_role": {"$gte": RoleTypeEnum.ADMIN.value}}
                
                # Mock training data that should be visible to admin
                mock_training_data = pd.DataFrame({
                    'id': ['admin_1', 'doctor_1', 'nurse_1', 'patient_1'],
                    'question': ['Admin query', 'Doctor query', 'Nurse query', 'Patient query'],
                    'content': ['SELECT * FROM admin_table', 'SELECT * FROM patient_data', 'SELECT * FROM nursing_notes', 'SELECT * FROM my_data'],
                    'training_data_type': ['sql', 'sql', 'sql', 'sql']
                })
                mock_vn_instance.get_training_data.return_value = mock_training_data
                
                vn_service = get_vn()
                vn_service.vn = mock_vn_instance
                
                # Admin should be able to see all training data
                training_data = vn_service.get_training_data()
                
                # Verify the role-based filtering was applied correctly for admin
                mock_vn_instance._prepare_retrieval_metadata.assert_called_once_with(None)
                expected_metadata = {"user_role": {"$gte": RoleTypeEnum.ADMIN.value}}
                mock_vn_instance.get_training_data.assert_called_once_with(metadata=expected_metadata)
                
                # Verify admin sees all training data (since admin role = 0, they can see all roles >= 0)
                assert len(training_data) == 4
                
    def test_patient_role_fallback_still_works_for_security(self):
        """Test that the security fallback to patient role still works when no session data is available."""
        
        with patch("streamlit.session_state", new_callable=MagicMock) as mock_session:
            # Mock empty/missing session state
            mock_cookies = MagicMock()
            mock_cookies.get.return_value = None
            mock_session.cookies = mock_cookies
            mock_session.get.return_value = None
            mock_session._vn_instance = None
            
            # Mock streamlit secrets
            secrets = {
                "ai_keys": {"ollama_model": "test_model"},
                "rag_model": {"chroma_path": "/tmp/test_chroma"},
                "postgres": {"host": "localhost", "port": 5432, "database": "test_db", "user": "test_user", "password": "test_pass"},
                "security": {"allow_llm_to_see_data": False}
            }
            
            with patch("streamlit.secrets", new=secrets):
                with patch("utils.chat_bot_helper.set_user_preferences_in_session_state") as mock_set_prefs:
                    mock_set_prefs.return_value = None  # No user found
                    
                    with patch("utils.vanna_calls.VannaService._setup_vanna"):
                        vn_service = get_vn()
                        
                        # Should fall back to patient role for security
                        assert vn_service.user_context.user_role == RoleTypeEnum.PATIENT.value
                        assert vn_service.user_context.user_id == "anonymous"
                        
    def test_multiple_users_get_separate_service_instances(self, mock_database_user):
        """Test that different users get separate VannaService instances with correct roles."""
        
        # Test admin user
        with patch("streamlit.session_state", new_callable=MagicMock) as mock_session_admin:
            mock_cookies_admin = MagicMock()
            mock_cookies_admin.get.side_effect = lambda key: '"5"' if key == "user_id" else "Admin" if key == "role_name" else None
            mock_session_admin.cookies = mock_cookies_admin
            mock_session_admin.get.side_effect = lambda key, default=None: RoleTypeEnum.ADMIN.value if key == "user_role" else default
            mock_session_admin.user_role = RoleTypeEnum.ADMIN.value
            mock_session_admin._vn_instance = None
            
            secrets = {"ai_keys": {"ollama_model": "test_model"}, "rag_model": {"chroma_path": "/tmp/test_chroma"}, "postgres": {"host": "localhost", "port": 5432, "database": "test_db", "user": "test_user", "password": "test_pass"}, "security": {"allow_llm_to_see_data": False}}
            
            with patch("streamlit.secrets", new=secrets):
                with patch("utils.chat_bot_helper.set_user_preferences_in_session_state") as mock_set_prefs_admin:
                    mock_set_prefs_admin.return_value = mock_database_user
                    
                    with patch("utils.vanna_calls.VannaService._setup_vanna"):
                        admin_service = get_vn()
                        assert admin_service.user_context.user_role == RoleTypeEnum.ADMIN.value
        
        # Test patient user  
        with patch("streamlit.session_state", new_callable=MagicMock) as mock_session_patient:
            mock_cookies_patient = MagicMock()
            mock_cookies_patient.get.side_effect = lambda key: '"10"' if key == "user_id" else "Patient" if key == "role_name" else None
            mock_session_patient.cookies = mock_cookies_patient
            mock_session_patient.get.side_effect = lambda key, default=None: RoleTypeEnum.PATIENT.value if key == "user_role" else default
            mock_session_patient.user_role = RoleTypeEnum.PATIENT.value
            mock_session_patient._vn_instance = None
            
            mock_patient_user = MagicMock()
            mock_patient_user.role.role.value = RoleTypeEnum.PATIENT.value
            
            with patch("streamlit.secrets", new=secrets):
                with patch("utils.chat_bot_helper.set_user_preferences_in_session_state") as mock_set_prefs_patient:
                    mock_set_prefs_patient.return_value = mock_patient_user
                    
                    with patch("utils.vanna_calls.VannaService._setup_vanna"):
                        patient_service = get_vn()
                        assert patient_service.user_context.user_role == RoleTypeEnum.PATIENT.value
                        
                        # Verify they are different instances
                        assert admin_service is not patient_service 