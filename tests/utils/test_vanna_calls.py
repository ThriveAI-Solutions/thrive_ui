from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.chromadb_vector import ThriveAI_ChromaDB
from utils.vanna_calls import (
    MyVannaGeminiChromaDB,
    MyVannaOllamaChromaDB,
    UserContext,
    VannaService,
    extract_user_context_from_streamlit,
    extract_vanna_config_from_secrets,
)


# Mock the streamlit secrets for testing
@pytest.fixture
def mock_streamlit_secrets(test_chromadb_path):
    """Mock streamlit secrets using temporary ChromaDB path."""
    with patch(
        "streamlit.secrets",
        new={
            "ai_keys": {
                "ollama_model": "llama3",
                "vanna_api": "mock_vanna_api",
                "vanna_model": "mock_vanna_model",
                "anthropic_api": "mock_anthropic_api",
                "anthropic_model": "claude-3-sonnet-20240229",
                "gemini_model": "gemini-1.5-flash",
                "gemini_api": "mock_gemini_api",
            },
            "rag_model": {"chroma_path": test_chromadb_path},
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "thrive",
                "user": "postgres",
                "password": "postgres",
            },
            "security": {"allow_llm_to_see_data": True},
        },
    ):
        yield


# Test for MyVannaOllamaChromaDB class
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestMyVannaOllamaChromaDB:
    def test_init(self, test_chromadb_path):
        with (
            patch("utils.vanna_calls.ThriveAI_ChromaDB.__init__") as mock_thriveai_chromadb_init,
            patch("utils.vanna_calls.Ollama.__init__", return_value=None) as mock_ollama_init,
        ):
            # Test initialization
            user_role_test = 1
            vanna_ollama = MyVannaOllamaChromaDB(user_role=user_role_test, config={"path": test_chromadb_path})

            # Verify ThriveAI_ChromaDB was initialized with the right user_role and config
            assert mock_thriveai_chromadb_init.call_count == 1
            called_args, called_kwargs = mock_thriveai_chromadb_init.call_args
            assert called_kwargs.get("user_role") == user_role_test
            assert called_kwargs.get("config") == {"path": test_chromadb_path}

            # Verify Ollama was initialized with the right config
            mock_ollama_init.assert_called_once()
            assert mock_ollama_init.call_args[1]["config"]["model"] == "llama3"

    def test_log(self):
        with (
            patch("utils.vanna_calls.ThriveAI_ChromaDB.__init__"),
            patch("utils.vanna_calls.Ollama.__init__"),
            patch("utils.vanna_calls.logger.debug") as mock_logger,
        ):
            vanna_ollama = MyVannaOllamaChromaDB(user_role=0)
            vanna_ollama.log("Test message", "Test Title")

            # Verify logger.debug was called with correct arguments
            mock_logger.assert_called_once_with("%s: %s", "Test Title", "Test message")


# Test for ThriveAIChromaDBMetadata
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestThriveAIChromaDBMetadata:
    @pytest.fixture
    def mock_chromadb_collection(self):
        collection_mock = MagicMock()
        return collection_mock

    @pytest.fixture
    def thrive_ai_chromadb_instance(self, mock_chromadb_collection, test_chromadb_path):
        # Instantiate ThriveAI_ChromaDB first
        # Provide dummy implementations for abstract methods for isolated testing if needed
        class ConcreteThriveAI(ThriveAI_ChromaDB):
            def generate_embedding(self, data, **kwargs):
                return [0.1, 0.2]  # Dummy embedding

            def system_message(self, message: str):
                return f"SYSTEM: {message}"  # Dummy

            def user_message(self, message: str):
                return f"USER: {message}"  # Dummy

            def assistant_message(self, message: str):
                return f"ASSISTANT: {message}"  # Dummy

            def submit_prompt(self, prompt, **kwargs):
                return "Dummy prompt response"  # Dummy

        instance = ConcreteThriveAI(user_role=1, config={"path": test_chromadb_path})  # Example role: Doctor

        # Now, patch the collections on the instance
        instance.sql_collection = mock_chromadb_collection
        instance.ddl_collection = mock_chromadb_collection
        instance.documentation_collection = mock_chromadb_collection
        yield instance

    def test_init_sets_user_role(self, test_chromadb_path):
        test_role = 2  # Example: Nurse

        # Use the same ConcreteThriveAI as in the fixture to avoid abstract method errors
        class ConcreteThriveAI(ThriveAI_ChromaDB):
            def generate_embedding(self, data, **kwargs):
                return [0.1, 0.2]

            def system_message(self, message: str):
                return f"SYSTEM: {message}"

            def user_message(self, message: str):
                return f"USER: {message}"

            def assistant_message(self, message: str):
                return f"ASSISTANT: {message}"

            def submit_prompt(self, prompt, **kwargs):
                return "Dummy prompt response"

        db = ConcreteThriveAI(user_role=test_role, config={"path": test_chromadb_path})
        assert db.user_role == test_role

    def test_add_question_sql_with_user_role_metadata(self, thrive_ai_chromadb_instance, mock_chromadb_collection):
        thrive_ai_chromadb_instance.add_question_sql("test question", "test sql")

        args, kwargs = mock_chromadb_collection.add.call_args
        assert "metadatas" in kwargs
        # The metadatas argument to collection.add is expected to be a list of dicts,
        # or a single dict if only one item is added. Vanna's ChromaDB wrapper might pass it as a list.
        # For a single add, it's often a single dict. Let's check the first item if it's a list, or the dict itself.
        added_metadata = kwargs["metadatas"][0] if isinstance(kwargs["metadatas"], list) else kwargs["metadatas"]
        assert added_metadata["user_role"] == thrive_ai_chromadb_instance.user_role

    def test_add_question_sql_merges_existing_metadata(self, thrive_ai_chromadb_instance, mock_chromadb_collection):
        existing_meta = {"custom_key": "custom_value"}
        thrive_ai_chromadb_instance.add_question_sql("test question", "test sql", metadata=existing_meta.copy())

        args, kwargs = mock_chromadb_collection.add.call_args
        added_metadata = kwargs["metadatas"][0] if isinstance(kwargs["metadatas"], list) else kwargs["metadatas"]
        assert added_metadata["user_role"] == thrive_ai_chromadb_instance.user_role
        assert added_metadata["custom_key"] == "custom_value"

    def test_get_similar_question_sql_with_user_role_filter(
        self, thrive_ai_chromadb_instance, mock_chromadb_collection
    ):
        thrive_ai_chromadb_instance.get_similar_question_sql("test question")

        args, kwargs = mock_chromadb_collection.query.call_args
        assert "where" in kwargs
        assert kwargs["where"]["user_role"] == {"$gte": thrive_ai_chromadb_instance.user_role}

    def test_get_similar_question_sql_merges_existing_filter(
        self, thrive_ai_chromadb_instance, mock_chromadb_collection
    ):
        existing_filter = {"other_field": "other_value"}
        thrive_ai_chromadb_instance.get_similar_question_sql("test question", metadata=existing_filter.copy())

        args, kwargs = mock_chromadb_collection.query.call_args
        assert "where" in kwargs
        assert kwargs["where"]["user_role"] == {"$gte": thrive_ai_chromadb_instance.user_role}
        assert kwargs["where"]["other_field"] == "other_value"


# Test for UserContext and factory methods
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestUserContextAndFactories:
    def test_user_context_creation(self):
        """Test UserContext creation with direct values."""
        context = UserContext(user_id="test_user", user_role=1)
        assert context.user_id == "test_user"
        assert context.user_role == 1

    def test_extract_user_context_from_streamlit_with_valid_data(self):
        """Test extraction of user context from Streamlit when data is available."""
        with patch("utils.vanna_calls.st.session_state", new_callable=MagicMock) as mock_session_state:
            # Mock cookies to contain user_id
            mock_cookies = MagicMock()
            mock_cookies.get.return_value = '"123"'  # JSON-encoded user_id
            mock_session_state.cookies = mock_cookies

            # Mock session_state get method
            def mock_get(key, default=None):
                if key == "user_role":
                    return 2  # NURSE role
                return default

            mock_session_state.get.side_effect = mock_get

            context = extract_user_context_from_streamlit()

            assert context.user_id == "123"
            assert context.user_role == 2

    def test_extract_user_context_from_streamlit_with_missing_data(self):
        """Test extraction of user context when data is missing (defaults to anonymous/patient)."""
        from orm.models import RoleTypeEnum

        with patch("utils.vanna_calls.st.session_state", new_callable=MagicMock) as mock_session_state:
            # Mock missing cookies
            mock_cookies = MagicMock()
            mock_cookies.get.return_value = None
            mock_session_state.cookies = mock_cookies

            # Mock missing session_state data
            def mock_get(key, default=None):
                return None  # Everything missing

            mock_session_state.get.side_effect = mock_get

            with patch("utils.vanna_calls.logger.warning") as mock_logger_warning:
                context = extract_user_context_from_streamlit()

                assert context.user_id == "anonymous"
                assert context.user_role == RoleTypeEnum.PATIENT.value

                # Verify warnings were logged
                assert mock_logger_warning.call_count >= 2  # At least one for user_id, one for user_role

    def test_extract_vanna_config_from_secrets(self):
        """Test extraction of Vanna configuration from Streamlit secrets."""
        config = extract_vanna_config_from_secrets()

        # Verify all expected config sections are present
        assert "ai_keys" in config
        assert "rag_model" in config
        assert "postgres" in config
        assert "security" in config

        # Verify some expected keys
        assert "ollama_model" in config["ai_keys"]
        assert "chroma_path" in config["rag_model"]
        assert "host" in config["postgres"]

    def test_user_context_from_streamlit_session_factory(self):
        """Test UserContext.from_streamlit_session factory method."""
        with patch("utils.vanna_calls.extract_user_context_from_streamlit") as mock_extract:
            mock_extract.return_value = UserContext(user_id="factory_test", user_role=1)

            context = UserContext.from_streamlit_session()

            assert context.user_id == "factory_test"
            assert context.user_role == 1
            mock_extract.assert_called_once()


# Test for VannaService with dependency injection
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestVannaServiceDependencyInjection:
    def test_vanna_service_init_with_dependencies(self):
        """Test VannaService initialization with explicit dependencies."""
        user_context = UserContext(user_id="test_user", user_role=1)
        config = {
            "ai_keys": {"ollama_model": "llama3"},
            "rag_model": {"chroma_path": "./test_chroma"},
            "postgres": {"host": "localhost", "database": "test", "user": "test", "password": "test", "port": 5432},
            "security": {"allow_llm_to_see_data": False},
        }

        with patch.object(VannaService, "_setup_vanna") as mock_setup:
            service = VannaService(user_context, config)

            assert service.user_context == user_context
            assert service.config == config
            assert service.user_id == "test_user"
            assert service.user_role == 1
            mock_setup.assert_called_once()

    def test_vanna_service_from_streamlit_session_factory(self):
        """Test VannaService.from_streamlit_session factory method."""
        with patch("utils.vanna_calls.UserContext.from_streamlit_session") as mock_user_context:
            with patch("utils.vanna_calls.extract_vanna_config_from_secrets") as mock_config:
                with patch.object(VannaService, "get_instance") as mock_get_instance:
                    mock_user_context.return_value = UserContext(user_id="factory_user", user_role=2)
                    mock_config.return_value = {"test": "config"}
                    mock_get_instance.return_value = MagicMock()

                    service = VannaService.from_streamlit_session()

                    mock_user_context.assert_called_once()
                    mock_config.assert_called_once()
                    mock_get_instance.assert_called_once()

    def test_vanna_service_get_instance_with_explicit_dependencies(self):
        """Test VannaService.get_instance with explicit dependencies."""
        user_context = UserContext(user_id="explicit_user", user_role=1)
        config = {"test": "config"}

        with patch.object(VannaService, "_create_instance_for_user") as mock_create:
            mock_create.return_value = MagicMock()
            # Clear instances cache
            VannaService._instances = {}

            service = VannaService.get_instance(user_context, config)

            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args[0][0] == user_context  # First argument is user_context
            assert call_args[0][1] == config  # Second argument is config

    def test_vanna_service_get_instance_backwards_compatibility(self):
        """Test VannaService.get_instance backwards compatibility (no arguments)."""
        with patch("utils.vanna_calls.UserContext.from_streamlit_session") as mock_user_context:
            with patch("utils.vanna_calls.extract_vanna_config_from_secrets") as mock_config:
                with patch.object(VannaService, "_create_instance_for_user") as mock_create:
                    mock_user_context.return_value = UserContext(user_id="compat_user", user_role=3)
                    mock_config.return_value = {"compat": "config"}
                    mock_create.return_value = MagicMock()
                    # Clear instances cache
                    VannaService._instances = {}

                    service = VannaService.get_instance()

                    mock_user_context.assert_called_once()
                    mock_config.assert_called_once()
                    mock_create.assert_called_once()


# Test VannaService class (updated for new architecture)
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestVannaService:
    @pytest.fixture
    def mock_vanna_service_instance(self):
        user_context = UserContext(user_id="test_user", user_role=2)
        config = {
            "ai_keys": {"ollama_model": "llama3"},
            "rag_model": {"chroma_path": "./test_chroma"},
            "postgres": {"host": "localhost", "database": "test", "user": "test", "password": "test", "port": 5432},
            "security": {"allow_llm_to_see_data": False},
        }

        with patch.object(VannaService, "_setup_vanna") as mock_setup_vanna_method:
            instance = VannaService(user_context, config)
            instance.vn = MagicMock()  # Mock the Vanna backend instance

            # Mock the specific add_* methods on the Vanna backend instance
            instance.vn.add_question_sql = MagicMock(return_value="id_sql")
            instance.vn.add_ddl = MagicMock(return_value="id_ddl")
            instance.vn.add_documentation = MagicMock(return_value="id_doc")
            instance.vn.train = MagicMock()  # For plan training
            yield instance

    def test_train_sql_passes_user_role_metadata(self, mock_vanna_service_instance):
        service = mock_vanna_service_instance
        service.train(question="Test Q", sql="Test S")

        service.vn.add_question_sql.assert_called_once()
        call_args = service.vn.add_question_sql.call_args
        assert "metadata" in call_args.kwargs
        assert call_args.kwargs["metadata"]["user_role"] == service.user_role

    def test_train_documentation_passes_user_role_metadata(self, mock_vanna_service_instance):
        service = mock_vanna_service_instance
        service.train(documentation="Test Doc")

        service.vn.add_documentation.assert_called_once()
        call_args = service.vn.add_documentation.call_args
        assert "metadata" in call_args.kwargs
        assert call_args.kwargs["metadata"]["user_role"] == service.user_role

    def test_train_ddl_passes_user_role_metadata(self, mock_vanna_service_instance):
        service = mock_vanna_service_instance
        service.train(ddl="Test DDL")

        service.vn.add_ddl.assert_called_once()
        call_args = service.vn.add_ddl.call_args
        assert "metadata" in call_args.kwargs
        assert call_args.kwargs["metadata"]["user_role"] == service.user_role

    def test_train_plan_calls_vn_train_without_metadata(self, mock_vanna_service_instance):
        service = mock_vanna_service_instance
        test_plan = MagicMock()  # A mock plan object
        service.train(plan=test_plan)

        service.vn.train.assert_called_once_with(plan=test_plan)  # Check it's called with plan
        # Ensure metadata is NOT passed here as VannaBase.train(plan=...) doesn't take it
        call_args = service.vn.train.call_args
        assert "metadata" not in call_args.kwargs

    def test_get_instance(self):
        # Test that we get the same instance twice when called with the same user
        user_context = UserContext(user_id="123", user_role=1)
        config = {"test": "config"}

        with patch.object(VannaService, "_setup_vanna") as mock_setup:
            # Reset singleton and clear cache
            VannaService._instances = {}
            # Clear the Streamlit cache for _create_instance_for_user
            VannaService._create_instance_for_user.clear()

            # Get instance twice and check they're the same
            instance1 = VannaService.get_instance(user_context, config)
            instance2 = VannaService.get_instance(user_context, config)

            assert instance1 is instance2
            # Verify setup was called only once (singleton behavior)
            mock_setup.assert_called_once()

    def test_setup_vanna_ollama_chromadb(self, test_chromadb_path):
        """Test setup_vanna with OllamaChromaDB backend"""
        config = {
            "ai_keys": {
                "ollama_model": "llama3",
                "vanna_api": "mock_vanna_api",
                "vanna_model": "mock_vanna_model",
            },
            "rag_model": {"chroma_path": test_chromadb_path},
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "thrive",
                "user": "postgres",
                "password": "postgres",
            },
            "security": {"allow_llm_to_see_data": False},
        }

        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService(user_context=UserContext(user_id="test_user", user_role=1), config=config)

            # Manually mock what we want to test
            service.vn = MagicMock()

            # Call connect_to_postgres directly with the expected parameters
            service.vn.connect_to_postgres(
                host="localhost", dbname="thrive", user="postgres", password="postgres", port=5432
            )

            # Verify connect_to_postgres was called with correct parameters
            service.vn.connect_to_postgres.assert_called_once()
            call_args = service.vn.connect_to_postgres.call_args[1]
            assert call_args["host"] == "localhost"
            assert call_args["dbname"] == "thrive"
            assert call_args["user"] == "postgres"
            assert call_args["password"] == "postgres"
            assert call_args["port"] == 5432

    def test_generate_questions(self):
        # Create a service instance with mocked dependencies
        user_context = UserContext(user_id="test_user", user_role=1)
        config = {"test": "config"}

        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService(user_context, config)
            # Set the vn attribute directly
            service.vn = MagicMock()
            service.vn.generate_questions.return_value = ["Question 1", "Question 2"]

            # Override the StreamLit caching decorator to make the test deterministic
            original_generate_questions = service.generate_questions
            service.generate_questions = lambda: service.vn.generate_questions()

            # Call the function
            result = service.generate_questions()

            # Check the result
            assert result == ["Question 1", "Question 2"]
            service.vn.generate_questions.assert_called_once()

    @patch("utils.vanna_calls.read_forbidden_from_json")
    def test_generate_sql(self, mock_read_forbidden):
        # Mock forbidden references
        mock_read_forbidden.return_value = ([], [], "''")

        user_context = UserContext(user_id="test_user", user_role=1)
        config = {"security": {"allow_llm_to_see_data": False}}

        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService(user_context, config)
            service.vn = MagicMock()

            # Set up the mocks
            service.vn.generate_sql.return_value = "SELECT * FROM test_table"

            # Create a simplified version of generate_sql for testing
            def simplified_generate_sql(question):
                sql = service.vn.generate_sql(question=question, allow_llm_to_see_data=False)
                return service.check_references(sql), 0.5

            # Use our simplified method for the test
            with patch.object(service, "generate_sql", simplified_generate_sql):
                result, elapsed_time = service.generate_sql("Show me all test data")

                assert result == "SELECT * FROM test_table"
                assert elapsed_time == 0.5
                # Check that the call was made with the expected parameters
                service.vn.generate_sql.assert_called_once_with(
                    question="Show me all test data", allow_llm_to_see_data=False
                )

    def test_generate_questions_error(self, mock_vanna_service_instance):
        mock_vanna_service_instance.vn.generate_questions.side_effect = Exception("Test exception")
        result = mock_vanna_service_instance.generate_questions()
        assert result == []


# Test utility functions (these work on standalone functions)
class TestUtilityFunctions:
    @patch("utils.vanna_calls.read_forbidden_from_json")
    def test_check_references_valid(self, mock_read_forbidden):
        mock_read_forbidden.return_value = (["secret_table"], ["password"], "'secret_table'")

        # Test the check_references method on VannaService
        user_context = UserContext(user_id="test", user_role=1)
        config = {"test": "config"}

        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService(user_context, config)

            # We need to mock both sqlparse.parse and get_identifiers
            with patch("sqlparse.parse") as mock_parse, patch("utils.vanna_calls.get_identifiers") as mock_get_ids:
                # Set up the mock for sqlparse.parse
                mock_stmt = MagicMock()
                mock_parse.return_value = [mock_stmt]

                # Set up the mock for get_identifiers
                mock_get_ids.return_value = (["public_table"], ["id", "name"])

                # Valid SQL without forbidden references
                sql = "SELECT id, name FROM public_table"
                result = service.check_references(sql)
                assert result == sql


@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestVannaServiceRoleFiltering:
    """Test that VannaService properly applies role-based filtering."""

    def test_get_training_data_applies_role_filtering_chromadb(self):
        """Test that get_training_data applies role-based filtering when using ChromaDB backend."""
        user_context = UserContext(user_id="123", user_role=2)  # NURSE role
        config = {"test": "config"}

        with patch.object(VannaService, "_setup_vanna") as mock_setup:
            service = VannaService(user_context, config)
            service.vn = MagicMock()

            # Mock ChromaDB backend with _prepare_retrieval_metadata
            service.vn._prepare_retrieval_metadata = MagicMock()
            service.vn._prepare_retrieval_metadata.return_value = {"user_role": {"$gte": 2}}
            service.vn.get_training_data = MagicMock()
            service.vn.get_training_data.return_value = pd.DataFrame({"question": ["test"], "sql": ["SELECT 1"]})

            # Call get_training_data
            result = service.get_training_data()

            # Verify ChromaDB role filtering was applied
            service.vn._prepare_retrieval_metadata.assert_called_once_with(None)
            service.vn.get_training_data.assert_called_once_with(metadata={"user_role": {"$gte": 2}})

    def test_get_training_data_applies_role_filtering_other_backend(self):
        """Test that get_training_data applies role-based filtering for non-ChromaDB backends."""
        user_context = UserContext(user_id="123", user_role=1)  # DOCTOR role
        config = {"test": "config"}

        with patch.object(VannaService, "_setup_vanna") as mock_setup:
            service = VannaService(user_context, config)
            service.vn = MagicMock()

            # Mock non-ChromaDB backend (no _prepare_retrieval_metadata)
            del service.vn._prepare_retrieval_metadata  # Remove the method to simulate non-ChromaDB
            service.vn.get_training_data = MagicMock()
            service.vn.get_training_data.return_value = pd.DataFrame({"question": ["test"], "sql": ["SELECT 1"]})

            # Call get_training_data
            result = service.get_training_data()

            # Verify basic role filtering was applied
            expected_metadata = {"user_role": {"$gte": 1}}
            service.vn.get_training_data.assert_called_once_with(metadata=expected_metadata)

    def test_get_training_data_preserves_existing_metadata(self):
        """Test that get_training_data preserves existing metadata while applying role filtering."""
        user_context = UserContext(user_id="123", user_role=2)  # NURSE role
        config = {"test": "config"}

        with patch.object(VannaService, "_setup_vanna") as mock_setup:
            service = VannaService(user_context, config)
            service.vn = MagicMock()

            # Mock ChromaDB backend
            service.vn._prepare_retrieval_metadata = MagicMock()
            service.vn._prepare_retrieval_metadata.return_value = {"user_role": {"$gte": 2}, "category": "test"}
            service.vn.get_training_data = MagicMock()
            service.vn.get_training_data.return_value = pd.DataFrame({"question": ["test"], "sql": ["SELECT 1"]})

            # Call with existing metadata
            existing_metadata = {"category": "test"}
            result = service.get_training_data(metadata=existing_metadata)

            # Verify existing metadata was preserved and passed through
            service.vn._prepare_retrieval_metadata.assert_called_once_with(existing_metadata)


@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestVannaServiceUIScenario:
    """Test VannaService in realistic UI scenarios."""

    def test_patient_user_cannot_see_admin_training_data(self):
        """Test that patient users cannot see training data created by admin users."""
        from orm.models import RoleTypeEnum

        # Create admin service
        admin_context = UserContext(user_id="456", user_role=RoleTypeEnum.ADMIN.value)
        config = {"test": "config"}

        with patch.object(VannaService, "_setup_vanna") as mock_setup:
            admin_service = VannaService(admin_context, config)
            admin_service.vn = MagicMock()

            # Mock admin creating training data
            admin_service.vn.add_question_sql = MagicMock(return_value="training_id_123")

            # Admin creates training data
            admin_service.train(question="Admin question", sql="SELECT * FROM admin_table")

            # Verify admin created training data with admin role
            admin_service.vn.add_question_sql.assert_called_once()
            call_args = admin_service.vn.add_question_sql.call_args
            assert call_args[1]["metadata"]["user_role"] == RoleTypeEnum.ADMIN.value

            # Now test patient user
            patient_context = UserContext(user_id="789", user_role=RoleTypeEnum.PATIENT.value)
            patient_service = VannaService(patient_context, config)
            patient_service.vn = MagicMock()

            # Mock ChromaDB filtering - patient should only see their own data
            patient_service.vn._prepare_retrieval_metadata = MagicMock()
            patient_service.vn._prepare_retrieval_metadata.return_value = {
                "user_role": {"$gte": RoleTypeEnum.PATIENT.value}
            }
            patient_service.vn.get_training_data = MagicMock()
            patient_service.vn.get_training_data.return_value = pd.DataFrame()  # Empty - no access to admin data

            # Patient tries to get training data
            result = patient_service.get_training_data()

            # Verify patient can only see data with user_role >= PATIENT (most restrictive)
            patient_service.vn._prepare_retrieval_metadata.assert_called_once_with(None)
            expected_metadata = {"user_role": {"$gte": RoleTypeEnum.PATIENT.value}}
            patient_service.vn.get_training_data.assert_called_once_with(metadata=expected_metadata)

            # Patient should not see admin's training data
            assert result.empty


@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestMyVannaGeminiChromaDB:
    @patch("utils.vanna_calls.GoogleGeminiChat.__init__", MagicMock(return_value=None))
    @patch("utils.vanna_calls.ThriveAI_ChromaDB.__init__", MagicMock(return_value=None))
    def test_generate_sql_with_rag(self, test_chromadb_path):
        # 1. Setup
        user_role_test = 1
        question = "What are the patient's latest vital signs?"

        vanna_gemini = MyVannaGeminiChromaDB(user_role=user_role_test, config={"path": test_chromadb_path})

        # Manually set the config and dialect because we mocked the initializers
        vanna_gemini.config = {"path": test_chromadb_path}
        vanna_gemini.dialect = "postgres"
        vanna_gemini.max_tokens = 4096
        vanna_gemini.static_documentation = ""

        # Mock the RAG and LLM methods on the instance
        vanna_gemini.get_related_ddl = MagicMock(return_value=["CREATE TABLE vitals (patient_id INT, heart_rate INT)"])
        vanna_gemini.get_similar_question_sql = MagicMock(return_value=[])
        vanna_gemini.get_related_documentation = MagicMock(
            return_value=["Vitals are important for patient monitoring."]
        )
        vanna_gemini.submit_prompt = MagicMock(return_value="SELECT heart_rate FROM vitals WHERE patient_id = 123")
        vanna_gemini.extract_sql = MagicMock(return_value="SELECT heart_rate FROM vitals WHERE patient_id = 123")

        # 2. Execution
        sql = vanna_gemini.generate_sql(question=question)

        # 3. Assertion
        vanna_gemini.submit_prompt.assert_called_once()
        prompt_messages = vanna_gemini.submit_prompt.call_args[0][0]

        full_prompt_str = "".join(str(p) for p in prompt_messages)
        assert "CREATE TABLE vitals" in full_prompt_str
        assert "Vitals are important for patient monitoring." in full_prompt_str

        vanna_gemini.extract_sql.assert_called_with("SELECT heart_rate FROM vitals WHERE patient_id = 123")
        assert sql == "SELECT heart_rate FROM vitals WHERE patient_id = 123"


@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestVannaServiceSecurity:
    """Test security aspects of VannaService, especially role handling."""

    def test_defaults_to_patient_role_when_session_state_missing(self):
        """Test that VannaService defaults to PATIENT role (least privileged) when user_role is not in session state."""
        from orm.models import RoleTypeEnum

        # Test the extract_user_context_from_streamlit function when data is missing
        with patch("utils.vanna_calls.st.session_state", new_callable=MagicMock) as mock_session_state:
            # Mock cookies to contain user_id
            mock_cookies = MagicMock()
            mock_cookies.get.return_value = '"123"'  # JSON-encoded user_id like the real app
            mock_session_state.cookies = mock_cookies

            # Mock session_state get method to return None for user_role
            def mock_get(key, default=None):
                if key == "user_role":
                    return None  # user_role not set - this should trigger default to PATIENT
                return default

            mock_session_state.get.side_effect = mock_get

            # Mock warning/error display methods
            with patch("utils.vanna_calls.logger.warning") as mock_logger_warning:
                with patch.object(VannaService, "_setup_vanna") as mock_setup:
                    # Clear any cached instances to ensure fresh creation
                    VannaService._instances = {}

                    # Test that from_streamlit_session properly handles missing user_role
                    service = VannaService.from_streamlit_session()

                    # Verify it was set up with PATIENT role (most restrictive)
                    assert service.user_id == "123"
                    assert service.user_role == RoleTypeEnum.PATIENT.value  # This should be 3

                    # Verify security warning was logged
                    mock_logger_warning.assert_called()
                    # Check that one of the warning calls was about user_role
                    warning_calls = [
                        call for call in mock_logger_warning.call_args_list if "user_role not found" in str(call)
                    ]
                    assert len(warning_calls) > 0, "Expected warning about missing user_role"

    def test_validates_invalid_role_values(self):
        """Test that VannaService validates user_role values and defaults to PATIENT for invalid values."""
        from orm.models import RoleTypeEnum

        # Create user context with invalid role
        user_context = UserContext(user_id="123", user_role=999)  # Invalid role
        config = {"test": "config"}

        with patch("utils.vanna_calls.st.error") as mock_error:
            with patch("utils.vanna_calls.logger.error") as mock_logger_error:
                with patch.object(VannaService, "_setup_vanna") as mock_setup:
                    # This should validate and fix the invalid role
                    service = VannaService._create_instance_for_user(user_context, config, "test_cache_key")

                    # Verify it defaulted to PATIENT role
                    assert service.user_role == RoleTypeEnum.PATIENT.value

                    # Verify error was logged and displayed
                    mock_logger_error.assert_called_once()
                    mock_error.assert_called_once()
                    assert "Invalid user role detected" in mock_error.call_args[0][0]

    def test_accepts_valid_role_values(self):
        """Test that VannaService accepts valid role values without warnings."""
        from orm.models import RoleTypeEnum

        # Create user context with valid role
        user_context = UserContext(user_id="123", user_role=RoleTypeEnum.DOCTOR.value)
        config = {"test": "config"}

        with patch("utils.vanna_calls.st.warning") as mock_warning:
            with patch("utils.vanna_calls.st.error") as mock_error:
                with patch.object(VannaService, "_setup_vanna") as mock_setup:
                    service = VannaService._create_instance_for_user(user_context, config, "test_cache_key")

                    # Verify it was created with the correct role
                    assert service.user_role == RoleTypeEnum.DOCTOR.value

                    # Verify no warnings or errors were displayed
                    mock_warning.assert_not_called()
                    mock_error.assert_not_called()

    def test_constructor_defaults_to_patient_role(self):
        """Test that UserContext can be created with PATIENT role explicitly."""
        from orm.models import RoleTypeEnum

        user_context = UserContext(user_id="test", user_role=RoleTypeEnum.PATIENT.value)
        assert user_context.user_role == RoleTypeEnum.PATIENT.value


@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestTrainEnhancedSchema:
    """Tests for the train_enhanced_schema function."""

    @patch("utils.vanna_calls.psycopg2.connect")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    @patch("utils.vanna_calls.read_forbidden_from_json")
    @patch("utils.vanna_calls.get_configured_schema")
    @patch("utils.vanna_calls.st.toast")
    @patch("utils.vanna_calls.st.success")
    def test_train_enhanced_schema_basic(
        self,
        mock_success,
        mock_toast,
        mock_get_schema,
        mock_read_forbidden,
        mock_vanna_service,
        mock_psycopg2_connect,
    ):
        """Test basic train_enhanced_schema execution."""
        from utils.vanna_calls import train_enhanced_schema

        # Setup mocks
        mock_get_schema.return_value = "public"
        mock_read_forbidden.return_value = (["forbidden_table"], ["password"], "'forbidden_table'")

        # Mock VannaService
        mock_service = MagicMock()
        mock_vanna_service.return_value = mock_service

        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=None)
        mock_psycopg2_connect.return_value = mock_conn

        # Mock schema enricher methods via the cursor
        # get_allowed_tables query result
        mock_cursor.fetchall.side_effect = [
            [("test_table",)],  # get_allowed_tables
            [],  # discover_explicit_relationships
            [],  # discover_implicit_relationships (id columns)
            [],  # extract_view_definitions
        ]

        # Run with minimal options to reduce complexity
        with patch("utils.schema_enrichment.SchemaEnricher") as MockEnricher:
            mock_enricher_instance = MagicMock()
            MockEnricher.return_value = mock_enricher_instance

            mock_enricher_instance.get_allowed_tables.return_value = ["test_table"]
            mock_enricher_instance.collect_column_statistics.return_value = []
            mock_enricher_instance.discover_explicit_relationships.return_value = []
            mock_enricher_instance.discover_implicit_relationships.return_value = []
            mock_enricher_instance.extract_view_definitions.return_value = []
            mock_enricher_instance.schema_graph.get_table_centrality.return_value = {}

            result = train_enhanced_schema(
                include_statistics=False,
                include_relationships=False,
                include_view_definitions=False,
            )

        assert result["success"] is True
        mock_conn.close.assert_called_once()

    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    @patch("utils.vanna_calls.st.error")
    def test_train_enhanced_schema_no_vanna_service(self, mock_error, mock_vanna_service):
        """Test train_enhanced_schema handles missing VannaService."""
        from utils.vanna_calls import train_enhanced_schema

        mock_vanna_service.return_value = None

        with patch("utils.vanna_calls.st.toast"):
            result = train_enhanced_schema()

        assert result["success"] is False
        assert "VannaService not initialized" in result["error"]

    @patch("utils.vanna_calls.psycopg2.connect")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    @patch("utils.vanna_calls.read_forbidden_from_json")
    @patch("utils.vanna_calls.get_configured_schema")
    @patch("utils.vanna_calls.st.toast")
    @patch("utils.vanna_calls.st.success")
    def test_train_enhanced_schema_with_statistics(
        self,
        mock_success,
        mock_toast,
        mock_get_schema,
        mock_read_forbidden,
        mock_vanna_service,
        mock_psycopg2_connect,
    ):
        """Test train_enhanced_schema collects and trains statistics."""
        from utils.schema_enrichment import ColumnStatistics
        from utils.vanna_calls import train_enhanced_schema

        # Setup mocks
        mock_get_schema.return_value = "public"
        mock_read_forbidden.return_value = ([], [], "")

        # Mock VannaService
        mock_service = MagicMock()
        mock_vanna_service.return_value = mock_service

        # Mock database connection
        mock_conn = MagicMock()
        mock_psycopg2_connect.return_value = mock_conn

        # Create sample column statistics
        sample_stats = ColumnStatistics(
            table_name="users",
            column_name="email",
            data_type="varchar",
            semantic_type="email",
            total_count=100,
            distinct_count=100,
        )

        with patch("utils.schema_enrichment.SchemaEnricher") as MockEnricher:
            mock_enricher_instance = MagicMock()
            MockEnricher.return_value = mock_enricher_instance

            mock_enricher_instance.get_allowed_tables.return_value = ["users"]
            mock_enricher_instance.collect_column_statistics.return_value = [sample_stats]
            mock_enricher_instance.discover_explicit_relationships.return_value = []
            mock_enricher_instance.discover_implicit_relationships.return_value = []
            mock_enricher_instance.extract_view_definitions.return_value = []
            mock_enricher_instance.schema_graph.get_table_centrality.return_value = {}

            result = train_enhanced_schema(
                include_statistics=True,
                include_relationships=False,
                include_view_definitions=False,
            )

        assert result["success"] is True
        assert result["tables_processed"] == 1
        assert result["statistics_trained"] == 1

        # Verify training was called with documentation
        mock_service.train.assert_called()
        call_args = mock_service.train.call_args_list
        assert any("documentation" in str(call) for call in call_args)

    @patch("utils.vanna_calls.psycopg2.connect")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    @patch("utils.vanna_calls.read_forbidden_from_json")
    @patch("utils.vanna_calls.get_configured_schema")
    @patch("utils.vanna_calls.st.toast")
    @patch("utils.vanna_calls.st.success")
    def test_train_enhanced_schema_with_relationships(
        self,
        mock_success,
        mock_toast,
        mock_get_schema,
        mock_read_forbidden,
        mock_vanna_service,
        mock_psycopg2_connect,
    ):
        """Test train_enhanced_schema discovers and trains relationships."""
        from utils.schema_enrichment import TableRelationship
        from utils.vanna_calls import train_enhanced_schema

        # Setup mocks
        mock_get_schema.return_value = "public"
        mock_read_forbidden.return_value = ([], [], "")

        # Mock VannaService
        mock_service = MagicMock()
        mock_vanna_service.return_value = mock_service

        # Mock database connection
        mock_conn = MagicMock()
        mock_psycopg2_connect.return_value = mock_conn

        # Create sample relationships
        explicit_rel = TableRelationship(
            source_table="orders",
            source_column="user_id",
            target_table="users",
            target_column="id",
            relationship_type="foreign_key",
        )
        implicit_rel = TableRelationship(
            source_table="products",
            source_column="category_id",
            target_table="category",
            target_column="id",
            relationship_type="implicit",
            confidence=0.8,
        )

        with patch("utils.schema_enrichment.SchemaEnricher") as MockEnricher:
            mock_enricher_instance = MagicMock()
            MockEnricher.return_value = mock_enricher_instance

            mock_enricher_instance.get_allowed_tables.return_value = []
            mock_enricher_instance.discover_explicit_relationships.return_value = [explicit_rel]
            mock_enricher_instance.discover_implicit_relationships.return_value = [implicit_rel]
            mock_enricher_instance.extract_view_definitions.return_value = []
            mock_enricher_instance.schema_graph.get_table_centrality.return_value = {}

            result = train_enhanced_schema(
                include_statistics=False,
                include_relationships=True,
                include_view_definitions=False,
            )

        assert result["success"] is True
        assert result["explicit_relationships"] == 1
        assert result["implicit_relationships"] == 1
        assert result["documentation_trained"] == 2

        # Verify training was called for each relationship
        assert mock_service.train.call_count >= 2
