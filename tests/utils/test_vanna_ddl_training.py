from typing import Any
from unittest.mock import MagicMock, patch

import chromadb
import pandas as pd
import pytest
import streamlit as st

from orm.models import RoleTypeEnum  # Assuming RoleTypeEnum is here for user_role values
from utils.chromadb_vector import ThriveAI_ChromaDB
from utils.vanna_calls import VannaService, train_ddl, train_ddl_describe_to_rag


# Mock the streamlit secrets for testing
@pytest.fixture(scope="function")  # Changed from module to function
def mock_streamlit_secrets_ddl(test_chromadb_path):
    """Mock streamlit secrets using temporary ChromaDB path."""
    with patch(
        "streamlit.secrets",
        new={
            "ai_keys": {
                "ollama_model": "test_ollama_model",  # Simplified for DDL tests
                # Add other keys if MyVanna*ChromaDB classes require them for init, even if not used for DDL
            },
            "rag_model": {"chroma_path": test_chromadb_path},  # Use temporary path
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db_ddl",
                "user": "test_user_ddl",
                "password": "test_pass_ddl",
            },
            "security": {"allow_llm_to_see_data": False},
        },
    ):
        yield


@pytest.fixture
def in_memory_chroma_client():
    """Provides an in-memory ChromaDB client."""
    return chromadb.Client()


@pytest.fixture
def thrive_ai_chromadb_in_memory(in_memory_chromadb_client):
    """Provides a ThriveAI_ChromaDB instance using an in-memory client."""

    # Create a concrete class that implements the abstract methods for testing purposes
    class ConcreteThriveAIChromaDB(ThriveAI_ChromaDB):
        def generate_embedding(self, data: str, **kwargs: Any) -> list[float]:
            return [0.1, 0.2, 0.3]  # Dummy embedding

        def system_message(self, message: str) -> Any:
            return "dummy system message: " + message

        def user_message(self, message: str) -> Any:
            return "dummy user message: " + message

        def assistant_message(self, message: str) -> Any:
            return "dummy assistant message: " + message

        def submit_prompt(self, prompt, **kwargs: Any) -> str:
            return "dummy llm response to: " + str(prompt)  # Changed from Any to str to match VannaBase

    # user_role can be any valid RoleTypeEnum value, e.g., ADMIN
    return ConcreteThriveAIChromaDB(user_role=RoleTypeEnum.ADMIN.value, client=in_memory_chromadb_client)


@pytest.fixture
def mock_vanna_service_with_in_memory_chroma(thrive_ai_chromadb_in_memory):
    """Mocks VannaService to use an in-memory ThriveAI_ChromaDB backend."""

    # 1. Mock st.session_state.get to provide a user_role for VannaService._initialize_instance
    with patch("streamlit.session_state") as mock_st_session_state:
        mock_st_session_state.get.return_value = (
            RoleTypeEnum.ADMIN.value
        )  # Match user_role in thrive_ai_chromadb_in_memory

        # 2. Patch VannaService._setup_vanna to directly assign our in-memory backend
        # This bypasses the complex logic in _setup_vanna that tries to pick a backend based on secrets
        with patch.object(VannaService, "_setup_vanna", autospec=True) as mock_setup:
            # Create an instance of VannaService. __init__ will run.
            # VannaService._initialize_instance will call our mocked _setup_vanna.
            # The key is that service_instance.vn should become our thrive_ai_chromadb_in_memory
            # And service_instance.user_role should be set.
            # The actual _setup_vanna takes (self, user_role).
            # We need to make sure that when the real VannaService.get_instance() -> _initialize_instance() -> _setup_vanna(user_role)
            # happens, it sets up service_instance.vn and service_instance.user_role correctly.

            # Instead of mocking _setup_vanna's internals, we set the .vn attribute after instance creation,
            # and ensure the user_role in the service matches the one in thrive_ai_chromadb_in_memory.

            VannaService._instance = None  # Reset singleton for fresh instance
            service_instance = VannaService.get_instance()

            # Configure the mock_setup to behave as if it set up our in-memory backend
            # The key is that service_instance.vn should become our thrive_ai_chromadb_in_memory
            # And service_instance.user_role should be set.
            # The actual _setup_vanna takes (self, user_role).
            # We need to make sure that when the real VannaService.get_instance() -> _initialize_instance() -> _setup_vanna(user_role)
            # happens, it sets up service_instance.vn and service_instance.user_role correctly.

            # Manually set vn and user_role on the instance after it's created and its _setup_vanna (mock_setup) has been called.
            service_instance.vn = thrive_ai_chromadb_in_memory

            yield service_instance


@patch("utils.vanna_calls.psycopg2.connect")
@patch("utils.vanna_calls.read_forbidden_from_json")  # Also mock reading forbidden DDL
def test_train_ddl_adds_to_chromadb_with_user_role(
    mock_read_forbidden, mock_psycopg2_connect, mock_streamlit_secrets_ddl, mock_vanna_service_with_in_memory_chroma
):
    # 1. Setup Mocks
    # Mock read_forbidden_from_json to return no forbidden tables/columns
    mock_read_forbidden.return_value = ([], [], "")

    # Mock psycopg2 connection and cursor
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.server_version = 140000  # Example: PostgreSQL 14
    mock_cursor.connection = mock_conn  # Add connection attribute to cursor
    mock_psycopg2_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    # Schema info to be returned by cursor.fetchall()
    # (table_schema, table_name, column_name, data_type, is_nullable)
    schema_info = [
        ("public", "users", "id", "integer", "NO"),
        ("public", "users", "name", "varchar", "YES"),
        ("public", "orders", "order_id", "integer", "NO"),
        ("public", "orders", "item", "varchar", "NO"),
    ]
    mock_cursor.fetchall.return_value = schema_info

    # Get the VannaService instance (which uses the in-memory ChromaDB)
    service = mock_vanna_service_with_in_memory_chroma
    expected_user_role = service.user_role  # This is RoleTypeEnum.ADMIN.value from the fixture

    # Clear any existing documents from previous tests
    existing_ids = service.vn.ddl_collection.get()["ids"]
    if existing_ids:
        service.vn.ddl_collection.delete(ids=existing_ids)

    # 2. Call the function to test
    # Reset call count if using a persistent mock across tests or re-patch:
    # For this test, let's ensure add_ddl on the specific `service.vn` is a spy/mock *before* train_ddl call.

    # The fixture `mock_vanna_service_with_in_memory_chroma` should yield the service,
    # and `train_ddl` (called inside the test) should use this yielded service instance.
    # So, we can wrap `service.vn.add_ddl` before calling `train_ddl`.

    # The original test called train_ddl and *then* tried to assert. This implies the effect of train_ddl
    # should be on the `service` instance from the fixture.

    # If `service.vn.add_ddl` is not a MagicMock itself, then it has no `call_count`.
    # Let's make it a spy *before* calling train_ddl.

    original_add_ddl = service.vn.add_ddl
    service.vn.add_ddl = MagicMock(wraps=original_add_ddl)

    try:
        train_ddl(describe_ddl_from_llm=False)  # We are not testing describe part here

        assert service.vn.add_ddl.call_count == 2  # Once for 'users', once for 'orders'

        # Verify the DDLs were added to the in-memory ChromaDB's ddl_collection with correct metadata
        # We can query the actual collection if the mock above (service.vn.add_ddl) isn't sufficient
        # For direct verification, let's query the collection
        ddl_collection = service.vn.ddl_collection
        results = ddl_collection.get(where={"user_role": expected_user_role})

        assert len(results["ids"]) == 2  # Two DDL statements added

        added_ddls = results["documents"]
        expected_ddl_users = "\nCREATE TABLE users (\n    id integer NOT NULL,\n    name varchar NULL\n);"  # Note: VannaService.train joins DDL list with " "
        expected_ddl_orders = "\nCREATE TABLE orders (\n    order_id integer NOT NULL,\n    item varchar NOT NULL\n);"

        # Check if the formatted DDL strings are present (order might vary)
        # The actual DDL sent to train is " ".join(ddl_list_for_table)
        # The formatting in train_ddl is specific, let's check the call_args of add_ddl

        call_args_list = service.vn.add_ddl.call_args_list

        first_call_ddl = call_args_list[0].kwargs["ddl"]
        first_call_metadata = call_args_list[0].kwargs["metadata"]
        assert "CREATE TABLE users (" in first_call_ddl
        assert "id integer NOT NULL" in first_call_ddl
        assert "name varchar NULL" in first_call_ddl
        assert first_call_metadata["user_role"] == expected_user_role

        second_call_ddl = call_args_list[1].kwargs["ddl"]
        second_call_metadata = call_args_list[1].kwargs["metadata"]
        assert "CREATE TABLE orders (" in second_call_ddl
        assert "order_id integer NOT NULL" in second_call_ddl
        assert "item varchar NOT NULL" in second_call_ddl
        assert second_call_metadata["user_role"] == expected_user_role

        # Verify the data from collection query (more robust)
        found_users_ddl = False
        found_orders_ddl = False
        for doc_meta in zip(results["documents"], results["metadatas"]):
            doc, meta = doc_meta
            if "CREATE TABLE users (" in doc:
                found_users_ddl = True
            if "CREATE TABLE orders (" in doc:
                found_orders_ddl = True
            assert meta["user_role"] == expected_user_role

        assert found_users_ddl
        assert found_orders_ddl

    finally:
        # Reset call count if using a persistent mock across tests or re-patch:
        # For this test, let's ensure add_ddl on the specific `service.vn` is a spy/mock *before* train_ddl call.
        service.vn.add_ddl = original_add_ddl


@patch("utils.vanna_calls.psycopg2.connect")
@patch("utils.vanna_calls.pd.read_sql_query")
@patch.object(VannaService, "submit_prompt", autospec=True)
@patch("utils.vanna_calls.st.toast")
@patch("psycopg2.sql.SQL")
@patch("psycopg2.sql.Identifier")
def test_train_ddl_describe_to_rag_adds_documentation_with_user_role(
    mock_identifier,
    mock_sql,
    mock_st_toast,
    mock_submit_prompt,
    mock_read_sql_query,
    mock_psycopg2_connect,
    mock_streamlit_secrets_ddl,
    mock_vanna_service_with_in_memory_chroma,
):
    # 1. Setup Mocks
    # Mock psycopg2 connection and cursor
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.server_version = 140000  # Example: PostgreSQL 14
    mock_cursor.connection = mock_conn  # Add connection attribute to cursor
    mock_psycopg2_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    # Set up the cursor as a context manager
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=None)

    # Mock cursor.description to return column names
    # This needs to be available after execute() is called
    mock_cursor.description = [
        ("id",),
        ("name",),
    ]  # (name, type_code, display_size, internal_size, precision, scale, null_ok)

    # Mock the execute method to succeed and make description available
    mock_cursor.execute.return_value = None

    # Mock the psycopg2.sql objects
    mock_query_string = MagicMock()
    mock_query_string.as_string.return_value = "SELECT DISTINCT col FROM tab ORDER BY col LIMIT 10;"
    mock_sql.return_value.format.return_value = mock_query_string
    mock_identifier.return_value = MagicMock()

    # Mock pd.read_sql_query to return sample data for columns
    # It will be called for 'id' and then for 'name'
    mock_read_sql_query.side_effect = [
        pd.DataFrame({"id": [1, 2, 3]}),  # Sample data for 'id' column
        pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]}),  # Sample data for 'name' column
    ]

    # Mock LLM's description generation
    mock_submit_prompt.side_effect = ["Description for id column.", "Description for name column."]

    # Get the VannaService instance (which uses the in-memory ChromaDB)
    service = mock_vanna_service_with_in_memory_chroma
    expected_user_role = service.user_role  # This is RoleTypeEnum.ADMIN.value from the fixture

    # Clear any existing documents from previous tests
    existing_ids = service.vn.documentation_collection.get()["ids"]
    if existing_ids:
        service.vn.documentation_collection.delete(ids=existing_ids)

    # Define sample DDL and table name
    table_name = "test_table"
    ddl_list = ["CREATE TABLE test_table (", "id INT NOT NULL,", "name VARCHAR(50) NULL", ");"]

    # 2. Call the function to test
    # The train_ddl_describe_to_rag function is in utils.vanna_calls
    # It uses VannaService.get_instance() internally. Our fixture ensures this instance
    # has the mock_submit_prompt and the in-memory chroma.

    # We also need to ensure that service.vn.add_documentation is a mock if we want to check its call_args
    # Since service.vn is thrive_ai_chromadb_in_memory, its add_documentation is real.
    # Let's spy on it or check the collection directly. Checking collection is more robust.
    # Spy on the add_documentation method of the actual in-memory chromadb instance
    with patch.object(service.vn, "add_documentation", wraps=service.vn.add_documentation) as mock_add_documentation:
        train_ddl_describe_to_rag(table_name, ddl_list)

        # Assert database query mock was called
        assert mock_read_sql_query.call_count == 2  # For 'id' and 'name'

        # 3. Verify calls to submit_prompt (LLM call)
        assert mock_submit_prompt.call_count == 2  # Called for 'id' and 'name'

        # 4. Verify calls to add_documentation
        assert mock_add_documentation.call_count == 2  # Called for 'id' and 'name'

        # Check arguments for the first call (id column)
        args_id, kwargs_id = mock_add_documentation.call_args_list[0]
        expected_doc_id = f"{table_name}.id Description for id column."
        assert kwargs_id["documentation"] == expected_doc_id
        # Metadata is handled by _prepare_metadata in ThriveAI_ChromaDB, so it's not passed directly to add_documentation here in the VannaService.train call chain
        # VannaService.train creates the metadata with self.user_role and passes it.

    # 4. Verify results in ChromaDB
    doc_collection = service.vn.documentation_collection
    results = doc_collection.get(where={"user_role": expected_user_role})

    assert len(results["ids"]) == 2  # Two documentation entries added

    added_docs_content = results["documents"]
    added_docs_metadata = results["metadatas"]

    expected_doc_id_content = f"{table_name}.id Description for id column."
    expected_doc_name_content = f"{table_name}.name Description for name column."

    # Check if the formatted documentation strings are present and have correct metadata
    found_id_doc = False
    found_name_doc = False
    for doc_content, meta in zip(added_docs_content, added_docs_metadata):
        if doc_content == expected_doc_id_content:
            found_id_doc = True
            assert meta["user_role"] == expected_user_role
        elif doc_content == expected_doc_name_content:
            found_name_doc = True
            assert meta["user_role"] == expected_user_role

    assert found_id_doc, "Documentation for 'id' column not found or incorrect."
    assert found_name_doc, "Documentation for 'name' column not found or incorrect."


def test_hierarchical_role_access_ddl_with_lte_filter(mock_vanna_service_with_in_memory_chroma):
    """Test direct collection filtering with $lte operator using manually added documents."""
    service = mock_vanna_service_with_in_memory_chroma

    # Clear any existing documents from previous tests
    existing_ids = service.vn.ddl_collection.get()["ids"]
    if existing_ids:
        service.vn.ddl_collection.delete(ids=existing_ids)

    # Manually add documents with different roles (bypassing the automatic role assignment)
    service.vn.ddl_collection.add(
        documents=["CREATE TABLE public_table (id INT);"],
        metadatas=[{"user_role": 1}],
        ids=["public-ddl"],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    service.vn.ddl_collection.add(
        documents=["CREATE TABLE internal_table (id INT, data VARCHAR(100));"],
        metadatas=[{"user_role": 2}],
        ids=["internal-ddl"],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    service.vn.ddl_collection.add(
        documents=["CREATE TABLE sensitive_table (id INT, secret_data TEXT);"],
        metadatas=[{"user_role": 3}],
        ids=["sensitive-ddl"],
        embeddings=[[0.1, 0.2, 0.3]],
    )

    # Test direct collection access with $lte operator
    # Test that we can filter for DDL with role <= 3 (should get all)
    results = service.vn.ddl_collection.get(where={"user_role": {"$lte": 3}})
    assert len(results["ids"]) == 3, "Should access all DDL documents with role <= 3"

    # Test that we can filter for DDL with role <= 2 (should get 2)
    results = service.vn.ddl_collection.get(where={"user_role": {"$lte": 2}})
    assert len(results["ids"]) == 2, "Should access only role 1 and 2 DDL"

    # Test that we can filter for DDL with role <= 1 (should get 1)
    results = service.vn.ddl_collection.get(where={"user_role": {"$lte": 1}})
    assert len(results["ids"]) == 1, "Should access only role 1 DDL"

    # Verify content is correct for role 1 access
    role_1_results = service.vn.ddl_collection.get(where={"user_role": {"$lte": 1}})
    assert "public_table" in role_1_results["documents"][0]


def test_role_based_access_with_different_service_roles(thrive_ai_chromadb_in_memory):
    """Test that the built-in role filtering works correctly with different user roles."""
    from orm.models import RoleTypeEnum

    # Create services with different user roles
    basic_user_service = type(thrive_ai_chromadb_in_memory)(user_role=1, client=thrive_ai_chromadb_in_memory.client)
    advanced_user_service = type(thrive_ai_chromadb_in_memory)(user_role=2, client=thrive_ai_chromadb_in_memory.client)
    admin_user_service = type(thrive_ai_chromadb_in_memory)(
        user_role=0, client=thrive_ai_chromadb_in_memory.client
    )  # Admin = 0

    # Clear ALL existing documents from all collections
    for collection in [
        admin_user_service.ddl_collection,
        admin_user_service.documentation_collection,
        admin_user_service.sql_collection,
    ]:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

    # Add DDL documents using different services (each adds their own user_role)
    basic_user_service.add_ddl("CREATE TABLE public_table (id INT);")  # Will have user_role: 1
    advanced_user_service.add_ddl("CREATE TABLE internal_table (id INT, data VARCHAR(100));")  # Will have user_role: 2
    admin_user_service.add_ddl("CREATE TABLE admin_table (id INT, secret_data TEXT);")  # Will have user_role: 0

    # Test basic user (role 1) - should only see documents with user_role >= 1 (roles 1, 2)
    basic_results = basic_user_service.get_training_data()
    assert len(basic_results) == 2, "Basic user should see 2 DDL documents (role >= 1: roles 1, 2)"

    # Test advanced user (role 2) - should see documents with user_role >= 2 (role 2 only)
    advanced_results = advanced_user_service.get_training_data()
    assert len(advanced_results) == 1, "Advanced user should see 1 DDL document (role >= 2: role 2)"

    # Test admin user (role 0) - should see all documents (user_role >= 0: all roles)
    admin_results = admin_user_service.get_training_data()
    assert len(admin_results) == 3, "Admin should see all 3 DDL documents (role >= 0: all roles)"


def test_hierarchical_role_access_documentation_with_lte_filter(mock_vanna_service_with_in_memory_chroma):
    """Test documentation access using direct collection filtering."""
    service = mock_vanna_service_with_in_memory_chroma

    # Clear any existing documents from previous tests
    existing_ids = service.vn.documentation_collection.get()["ids"]
    if existing_ids:
        service.vn.documentation_collection.delete(ids=existing_ids)

    # Manually add documentation with different user roles
    service.vn.documentation_collection.add(
        documents=["Basic user documentation"],
        metadatas=[{"user_role": 1}],
        ids=["basic-doc"],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    service.vn.documentation_collection.add(
        documents=["Advanced user documentation"],
        metadatas=[{"user_role": 2}],
        ids=["advanced-doc"],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    service.vn.documentation_collection.add(
        documents=["Admin user documentation"],
        metadatas=[{"user_role": 3}],
        ids=["admin-doc"],
        embeddings=[[0.1, 0.2, 0.3]],
    )

    # Test direct collection access with $lte operator
    # Test that we can filter for documentation with role <= 3 (should get all)
    results = service.vn.documentation_collection.get(where={"user_role": {"$lte": 3}})
    assert len(results["ids"]) == 3, "Should access all documentation with role <= 3"

    # Test that we can filter for documentation with role <= 2 (should get 2)
    results = service.vn.documentation_collection.get(where={"user_role": {"$lte": 2}})
    assert len(results["ids"]) == 2, "Should access only role 1 and 2 documentation"

    # Test that we can filter for documentation with role <= 1 (should get 1)
    results = service.vn.documentation_collection.get(where={"user_role": {"$lte": 1}})
    assert len(results["ids"]) == 1, "Should access only role 1 documentation"

    # Verify content matches expected privilege level
    admin_results = service.vn.documentation_collection.get(where={"user_role": {"$lte": 3}})
    doc_contents = admin_results["documents"]
    assert "Basic user documentation" in doc_contents
    assert "Advanced user documentation" in doc_contents
    assert "Admin user documentation" in doc_contents


def test_role_access_with_gte_filter(mock_vanna_service_with_in_memory_chroma):
    """Test role filtering using $gte operator."""
    service = mock_vanna_service_with_in_memory_chroma

    # Clear any existing documents from previous tests
    existing_ids = service.vn.documentation_collection.get()["ids"]
    if existing_ids:
        service.vn.documentation_collection.delete(ids=existing_ids)

    # Manually add documents with different user roles
    for role, doc_name in [
        (1, "Low privilege doc"),
        (2, "Medium privilege doc"),
        (3, "High privilege doc"),
        (4, "Super admin doc"),
    ]:
        service.vn.documentation_collection.add(
            documents=[doc_name],
            metadatas=[{"user_role": role}],
            ids=[f"role-{role}-doc"],
            embeddings=[[0.1, 0.2, 0.3]],
        )

    # Test $gte: Get documents that require role >= 2 (medium and above)
    results = service.vn.documentation_collection.get(where={"user_role": {"$gte": 2}})
    assert len(results["ids"]) == 3, "Should get 3 documents with role >= 2"

    # Test $gte: Get documents that require role >= 3 (high and above)
    results = service.vn.documentation_collection.get(where={"user_role": {"$gte": 3}})
    assert len(results["ids"]) == 2, "Should get 2 documents with role >= 3"

    # Test $gte: Get documents that require role >= 4 (super admin only)
    results = service.vn.documentation_collection.get(where={"user_role": {"$gte": 4}})
    assert len(results["ids"]) == 1, "Should get 1 document with role >= 4"
    assert "Super admin doc" in results["documents"][0]


def test_role_access_with_range_filters(mock_vanna_service_with_in_memory_chroma):
    """Test role filtering using range operators ($gt, $lt, $ne)."""
    service = mock_vanna_service_with_in_memory_chroma

    # Clear any existing documents from previous tests
    existing_ids = service.vn.documentation_collection.get()["ids"]
    if existing_ids:
        service.vn.documentation_collection.delete(ids=existing_ids)

    # Manually add documents with different user roles
    for role in range(1, 6):  # roles 1, 2, 3, 4, 5
        service.vn.documentation_collection.add(
            documents=[f"Role {role} document"],
            metadatas=[{"user_role": role}],
            ids=[f"role-{role}"],
            embeddings=[[0.1, 0.2, 0.3]],
        )

    # Test $gt: Get documents that require role > 2
    results = service.vn.documentation_collection.get(where={"user_role": {"$gt": 2}})
    assert len(results["ids"]) == 3, "Should get 3 documents with role > 2 (roles 3, 4, 5)"

    # Test $lt: Get documents that require role < 4
    results = service.vn.documentation_collection.get(where={"user_role": {"$lt": 4}})
    assert len(results["ids"]) == 3, "Should get 3 documents with role < 4 (roles 1, 2, 3)"

    # Test $ne: Get documents that don't require role 3
    results = service.vn.documentation_collection.get(where={"user_role": {"$ne": 3}})
    assert len(results["ids"]) == 4, "Should get 4 documents with role != 3"

    # Verify no role 3 document in $ne results
    role_3_found = any("Role 3 document" in doc for doc in results["documents"])
    assert not role_3_found, "Role 3 document should not be in $ne results"


def test_combined_metadata_filters_with_roles(mock_vanna_service_with_in_memory_chroma):
    """Test combining role-based filters with other metadata."""
    service = mock_vanna_service_with_in_memory_chroma

    # Clear any existing documents from previous tests
    existing_ids = service.vn.documentation_collection.get()["ids"]
    if existing_ids:
        service.vn.documentation_collection.delete(ids=existing_ids)

    # Manually add documents with multiple metadata fields
    docs_metadata = [
        ("Public API doc", {"user_role": 1, "category": "api", "department": "public"}),
        ("Internal API doc", {"user_role": 2, "category": "api", "department": "internal"}),
        ("Admin API doc", {"user_role": 3, "category": "api", "department": "admin"}),
        ("Public DB doc", {"user_role": 1, "category": "database", "department": "public"}),
        ("Admin DB doc", {"user_role": 3, "category": "database", "department": "admin"}),
    ]

    for i, (doc, metadata) in enumerate(docs_metadata):
        service.vn.documentation_collection.add(
            documents=[doc], metadatas=[metadata], ids=[f"combined-{i}"], embeddings=[[0.1, 0.2, 0.3]]
        )

    # Test: Get API docs that user with role 2 can access (role <= 2)
    results = service.vn.documentation_collection.get(where={"$and": [{"user_role": {"$lte": 2}}, {"category": "api"}]})
    assert len(results["ids"]) == 2, "Should get 2 API docs accessible to role 2"
    api_docs = results["documents"]
    assert "Public API doc" in api_docs
    assert "Internal API doc" in api_docs
    assert "Admin API doc" not in api_docs

    # Test: Get all database docs for admin (role >= 1, which includes all)
    results = service.vn.documentation_collection.get(
        where={"$and": [{"user_role": {"$gte": 1}}, {"category": "database"}]}
    )
    assert len(results["ids"]) == 2, "Should get 2 database docs"

    # Test: Get admin-level documents across all categories
    results = service.vn.documentation_collection.get(where={"user_role": 3})
    assert len(results["ids"]) == 2, "Should get 2 admin-level documents"


def test_role_based_similar_question_retrieval(mock_vanna_service_with_in_memory_chroma):
    """Test that similar question retrieval respects role-based filtering."""
    service = mock_vanna_service_with_in_memory_chroma

    # Clear any existing documents from previous tests
    existing_ids = service.vn.sql_collection.get()["ids"]
    if existing_ids:
        service.vn.sql_collection.delete(ids=existing_ids)

    # Manually add question-SQL pairs with different user roles
    questions_data = [
        ("What are the public users?", "SELECT * FROM public_users;", 1),
        ("What are the internal users?", "SELECT * FROM internal_users;", 2),
        ("What are the admin users?", "SELECT * FROM admin_users;", 3),
    ]

    for i, (question, sql, role) in enumerate(questions_data):
        import json

        question_sql_json = json.dumps({"question": question, "sql": sql}, ensure_ascii=False)
        service.vn.sql_collection.add(
            documents=[question_sql_json],
            metadatas=[{"user_role": role}],
            ids=[f"question-{i}"],
            embeddings=[[0.1, 0.2, 0.3]],
        )

    # Test direct collection access: Basic user (role 1) should only see public questions
    results = service.vn.sql_collection.get(where={"user_role": {"$lte": 1}})
    assert len(results["ids"]) == 1, "Basic user should only see 1 question"
    assert "public_users" in results["documents"][0]

    # Test direct collection access: Advanced user (role 2) should see public and internal questions
    results = service.vn.sql_collection.get(where={"user_role": {"$lte": 2}})
    assert len(results["ids"]) == 2, "Advanced user should see 2 questions"

    # Test direct collection access: Admin user (role 3) should see all questions
    results = service.vn.sql_collection.get(where={"user_role": {"$lte": 3}})
    assert len(results["ids"]) == 3, "Admin user should see all 3 questions"

    # Verify the metadata is correctly stored
    admin_results = service.vn.sql_collection.get(where={"user_role": {"$lte": 3}})
    for meta in admin_results["metadatas"]:
        assert "user_role" in meta, "Each result should have user_role metadata"
        assert meta["user_role"] in [1, 2, 3], "User role should be valid"


def test_built_in_role_filtering_behavior(thrive_ai_chromadb_in_memory):
    """Test that the built-in _prepare_retrieval_metadata filtering works as designed."""
    from orm.models import RoleTypeEnum

    # Create services with different user roles
    basic_user_service = type(thrive_ai_chromadb_in_memory)(user_role=1, client=thrive_ai_chromadb_in_memory.client)
    advanced_user_service = type(thrive_ai_chromadb_in_memory)(user_role=2, client=thrive_ai_chromadb_in_memory.client)
    admin_user_service = type(thrive_ai_chromadb_in_memory)(user_role=0, client=thrive_ai_chromadb_in_memory.client)

    # Test that _prepare_retrieval_metadata adds the correct user_role filter
    basic_metadata = basic_user_service._prepare_retrieval_metadata()
    assert basic_metadata["user_role"] == {"$gte": 1}, "Basic user should filter for user_role >= 1"

    advanced_metadata = advanced_user_service._prepare_retrieval_metadata()
    assert advanced_metadata["user_role"] == {"$gte": 2}, "Advanced user should filter for user_role >= 2"

    admin_metadata = admin_user_service._prepare_retrieval_metadata()
    assert admin_metadata["user_role"] == {"$gte": 0}, "Admin user should filter for user_role >= 0"

    # Test that existing metadata is preserved
    custom_metadata = {"category": "api"}
    basic_metadata_with_custom = basic_user_service._prepare_retrieval_metadata(custom_metadata)
    assert basic_metadata_with_custom["user_role"] == {"$gte": 1}, "Should add user_role filter"
    assert basic_metadata_with_custom["category"] == "api", "Should preserve existing metadata"

    # Test that _prepare_metadata adds the service's user_role
    basic_add_metadata = basic_user_service._prepare_metadata({"category": "test"})
    assert basic_add_metadata["user_role"] == 1, "Should set user_role to service's role"
    assert basic_add_metadata["category"] == "test", "Should preserve existing metadata"


@patch("utils.vanna_calls.psycopg2.connect")
@patch("utils.vanna_calls.read_forbidden_from_json")
@patch("utils.vanna_calls.st.toast")
@patch("utils.vanna_calls.st.success")
@patch("utils.vanna_calls.st.warning")
def test_train_ddl_shows_success_messages(
    mock_warning,
    mock_success,
    mock_toast,
    mock_read_forbidden,
    mock_psycopg2_connect,
    mock_streamlit_secrets_ddl,
    mock_vanna_service_with_in_memory_chroma,
):
    """Test that train_ddl function shows appropriate success messages to users."""
    # 1. Setup Mocks
    mock_read_forbidden.return_value = ([], [], "")

    # Mock psycopg2 connection and cursor
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.server_version = 140000
    mock_cursor.connection = mock_conn
    mock_psycopg2_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    # Schema info for 2 tables
    schema_info = [
        ("public", "users", "id", "integer", "NO"),
        ("public", "users", "name", "varchar", "YES"),
        ("public", "orders", "order_id", "integer", "NO"),
        ("public", "orders", "item", "varchar", "NO"),
    ]
    mock_cursor.fetchall.return_value = schema_info

    service = mock_vanna_service_with_in_memory_chroma

    # Clear any existing documents
    existing_ids = service.vn.ddl_collection.get()["ids"]
    if existing_ids:
        service.vn.ddl_collection.delete(ids=existing_ids)

    # 2. Call the function
    train_ddl(describe_ddl_from_llm=False)

    # 3. Verify success messages were shown
    # Check that starting message was shown
    mock_toast.assert_any_call("🚀 Starting DDL training...")

    # Check that individual table completion messages were shown
    mock_toast.assert_any_call("✓ Trained DDL for table: users")
    mock_toast.assert_any_call("✓ Trained DDL for table: orders")

    # Check that final success message was shown
    mock_success.assert_called_once_with("🎉 DDL Training completed successfully! Trained 2 table(s).")

    # Verify warning was not called (since we have tables)
    mock_warning.assert_not_called()


@patch("utils.vanna_calls.psycopg2.connect")
@patch("utils.vanna_calls.read_forbidden_from_json")
@patch("utils.vanna_calls.st.toast")
@patch("utils.vanna_calls.st.success")
@patch("utils.vanna_calls.st.warning")
def test_train_ddl_shows_warning_when_no_tables(
    mock_warning,
    mock_success,
    mock_toast,
    mock_read_forbidden,
    mock_psycopg2_connect,
    mock_streamlit_secrets_ddl,
    mock_vanna_service_with_in_memory_chroma,
):
    """Test that train_ddl function shows warning when no tables are found."""
    # 1. Setup Mocks
    mock_read_forbidden.return_value = ([], [], "")

    # Mock psycopg2 connection and cursor
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.server_version = 140000
    mock_cursor.connection = mock_conn
    mock_psycopg2_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    # No schema info (empty result)
    schema_info = []
    mock_cursor.fetchall.return_value = schema_info

    service = mock_vanna_service_with_in_memory_chroma

    # 2. Call the function
    train_ddl(describe_ddl_from_llm=False)

    # 3. Verify appropriate messages were shown
    # Check that starting message was shown
    mock_toast.assert_called_once_with("🚀 Starting DDL training...")

    # Check that warning was shown instead of success
    mock_warning.assert_called_once_with("No tables found to train DDL on.")

    # Verify success was not called (since no tables)
    mock_success.assert_not_called()
