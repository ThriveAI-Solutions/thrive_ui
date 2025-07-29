"""
Test file to verify that the user_role fix is working correctly.
This test ensures that when an admin user asks a question, the VannaService
correctly uses their admin role instead of defaulting to PATIENT role.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

from orm.models import RoleTypeEnum
from utils.chat_bot_helper import get_vanna_service, get_vn
from utils.vanna_calls import UserContext, VannaService, extract_user_context_from_streamlit


@pytest.fixture
def mock_streamlit_session_admin():
    """Mock streamlit session state for admin user."""
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
        
        yield mock_session


@pytest.fixture
def mock_streamlit_secrets():
    """Mock streamlit secrets."""
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
        yield


@pytest.fixture
def mock_user_preferences():
    """Mock the set_user_preferences_in_session_state function."""
    with patch("utils.chat_bot_helper.set_user_preferences_in_session_state") as mock_prefs:
        mock_prefs.return_value = MagicMock()
        yield mock_prefs


class TestUserRoleFix:
    """Test cases for the user role fix."""
    
    def test_extract_user_context_gets_correct_admin_role(self, mock_streamlit_session_admin):
        """Test that extract_user_context_from_streamlit gets the correct admin role."""
        context = extract_user_context_from_streamlit()
        
        assert context.user_id == "5"
        assert context.user_role == RoleTypeEnum.ADMIN.value
        
    def test_get_vanna_service_uses_admin_role(self, mock_streamlit_session_admin, mock_streamlit_secrets, mock_user_preferences):
        """Test that get_vanna_service creates service with admin role."""
        with patch.object(VannaService, '_setup_vanna'):
            service = get_vanna_service()
            
            assert service.user_context.user_role == RoleTypeEnum.ADMIN.value
            assert service.user_context.user_id == "5"
            
    def test_get_vn_function_preserves_admin_role(self, mock_streamlit_session_admin, mock_streamlit_secrets, mock_user_preferences):
        """Test that get_vn() function preserves admin role across calls."""
        with patch.object(VannaService, '_setup_vanna'):
            # First call should create the service
            service1 = get_vn()
            assert service1.user_context.user_role == RoleTypeEnum.ADMIN.value
            
            # Second call should return the same cached instance
            service2 = get_vn()
            assert service2 is service1
            assert service2.user_context.user_role == RoleTypeEnum.ADMIN.value
            
    def test_user_role_not_defaulting_to_patient_during_question(self, mock_streamlit_session_admin, mock_streamlit_secrets, mock_user_preferences):
        """Test that user role doesn't default to PATIENT when asking a question."""
        with patch.object(VannaService, '_setup_vanna'):
            with patch.object(VannaService, 'generate_sql') as mock_generate_sql:
                mock_generate_sql.return_value = ("SELECT 1", 0.1)
                
                service = get_vn()
                sql, elapsed_time = service.generate_sql("What is the count of users?")
                
                # Verify the service was created with admin role, not patient
                assert service.user_context.user_role == RoleTypeEnum.ADMIN.value
                assert service.user_context.user_role != RoleTypeEnum.PATIENT.value
                
    def test_training_data_filtering_uses_correct_role(self, mock_streamlit_session_admin, mock_streamlit_secrets, mock_user_preferences):
        """Test that training data filtering uses the correct admin role."""
        with patch.object(VannaService, '_setup_vanna'):
            # Mock the underlying vn object to have the _prepare_retrieval_metadata method
            mock_vn_instance = MagicMock()
            mock_vn_instance._prepare_retrieval_metadata.return_value = {"user_role": {"$gte": RoleTypeEnum.ADMIN.value}}
            mock_vn_instance.get_training_data.return_value = MagicMock()
            
            service = get_vn()
            service.vn = mock_vn_instance
            
            # Call get_training_data
            service.get_training_data()
            
            # Verify that the role filtering uses admin role (0), not patient role (3)
            mock_vn_instance._prepare_retrieval_metadata.assert_called_once_with(None)
            expected_metadata = {"user_role": {"$gte": RoleTypeEnum.ADMIN.value}}
            mock_vn_instance.get_training_data.assert_called_once_with(metadata=expected_metadata)


class TestUserRoleEdgeCases:
    """Test edge cases for user role handling."""
    
    def test_session_state_missing_defaults_to_patient(self):
        """Test that when session state is missing, it still defaults to patient for security."""
        with patch("streamlit.session_state", new_callable=MagicMock) as mock_session:
            # Mock empty session state
            mock_cookies = MagicMock()
            mock_cookies.get.return_value = None
            mock_session.cookies = mock_cookies
            mock_session.get.return_value = None
            
            context = extract_user_context_from_streamlit()
            
            # Should default to patient for security
            assert context.user_role == RoleTypeEnum.PATIENT.value
            assert context.user_id == "anonymous"
            
    def test_user_preferences_called_before_service_creation(self, mock_streamlit_session_admin, mock_streamlit_secrets, mock_user_preferences):
        """Test that set_user_preferences_in_session_state is called before creating VannaService."""
        with patch.object(VannaService, '_setup_vanna'):
            get_vanna_service()
            
            # Verify that user preferences were loaded
            mock_user_preferences.assert_called_once() 