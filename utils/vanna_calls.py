import json
import logging
import re
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd
import psycopg2
import sqlparse
import streamlit as st
from pandas import DataFrame
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import DML, Keyword
from vanna.anthropic import Anthropic_Chat
from vanna.ollama import Ollama
from vanna.remote import VannaDefault
from vanna.vannadb import VannaDB_VectorStore

from utils.chromadb_vector import ThriveAI_ChromaDB

logger = logging.getLogger(__name__)


@dataclass
class UserContext:
    """User context containing authentication and authorization information."""

    user_id: str
    user_role: int

    @classmethod
    def from_streamlit_session(cls) -> "UserContext":
        """Factory method to create UserContext from Streamlit session/cookies."""
        return extract_user_context_from_streamlit()


def extract_user_context_from_streamlit() -> UserContext:
    """Extract user context from Streamlit session and cookies."""
    # Get user_id from cookies (where it's actually stored)
    user_id = None
    try:
        user_id_str = st.session_state.cookies.get("user_id")
        if user_id_str:
            user_id = json.loads(user_id_str)
            # Convert to string for consistent use
            user_id = str(user_id)
    except Exception as e:
        logger.warning(f"Error getting user_id from cookies: {e}")
        user_id = None

    if user_id is None:
        user_id = "anonymous"
        logger.warning("user_id not found in session state cookies - using anonymous user")

    # Get user_role from session state
    user_role = st.session_state.get("user_role", None)
    if user_role is None:
        from orm.models import RoleTypeEnum

        user_role = RoleTypeEnum.PATIENT.value
        logger.warning(f"user_role not found in session state for user {user_id} - defaulting to PATIENT role")

    return UserContext(user_id=user_id, user_role=user_role)


def extract_vanna_config_from_secrets() -> dict:
    """Extract Vanna configuration from Streamlit secrets."""
    return {
        "ai_keys": dict(st.secrets["ai_keys"]),
        "rag_model": dict(st.secrets["rag_model"]),
        "postgres": dict(st.secrets["postgres"]),
        "security": dict(st.secrets.get("security", {})),
    }


def get_configured_schema() -> str:
    """Get the configured database schema name from secrets."""
    schema_name = st.secrets.get("postgres", {}).get("schema_name", "public")
    if not schema_name or not isinstance(schema_name, str):
        logger.warning(f"Invalid schema_name '{schema_name}', defaulting to 'public'")
        schema_name = "public"
    return schema_name


def get_configured_object_type() -> str:
    """Get the configured database object type (tables or views) from secrets."""
    object_type = st.secrets.get("postgres", {}).get("object_type", "tables").lower()
    if object_type not in ["tables", "views"]:
        logger.warning(f"Invalid object_type '{object_type}', defaulting to 'tables'")
        object_type = "tables"
    return object_type


class MyVannaAnthropic(VannaDB_VectorStore, Anthropic_Chat):
    def __init__(self, config=None):
        try:
            logger.info("Using Anthropic and VannaDB")
            VannaDB_VectorStore.__init__(
                self,
                vanna_model=st.secrets["ai_keys"]["vanna_model"],
                vanna_api_key=st.secrets["ai_keys"]["vanna_api"],
                config=config,
            )
            Anthropic_Chat.__init__(
                self,
                config={
                    "api_key": st.secrets["ai_keys"]["anthropic_api"],
                    "model": st.secrets["ai_keys"]["anthropic_model"],
                },
            )
        except Exception as e:
            logger.exception("Error Configuring MyVannaAnthropic: %s", e)
            raise


class MyVannaAnthropicChromaDB(ThriveAI_ChromaDB, Anthropic_Chat):
    def __init__(self, user_role: int, config=None):
        try:
            logger.info("Using Anthropic and chromaDB")
            ThriveAI_ChromaDB.__init__(self, user_role=user_role, config=config)
            Anthropic_Chat.__init__(
                self,
                config={
                    "api_key": st.secrets["ai_keys"]["anthropic_api"],
                    "model": st.secrets["ai_keys"]["anthropic_model"],
                },
            )
        except Exception as e:
            logger.exception("Error Configuring MyVannaAnthropicChromaDB: %s", e)
            raise

    def log(self, message: str, title: str = "Info"):
        logger.debug("%s: %s", title, message)


class MyVannaOllama(VannaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        try:
            logger.info("Using Ollama and VannaDB")
            VannaDB_VectorStore.__init__(
                self,
                vanna_model=st.secrets["ai_keys"]["vanna_model"],
                vanna_api_key=st.secrets["ai_keys"]["vanna_api"],
                config=config,
            )
            Ollama.__init__(self, config={"model": st.secrets["ai_keys"]["ollama_model"]})
        except Exception as e:
            logger.exception("Error Configuring MyVannaOllama: %s", e)
            raise

    # override the log function to stop vannas annoying print statements
    def log(self, message: str, title: str = "Info"):
        logger.debug("%s: %s", title, message)


class MyVannaOllamaChromaDB(ThriveAI_ChromaDB, Ollama):
    def __init__(self, user_role: int, config=None):
        try:
            logger.info("Using Ollama and ChromaDB")
            ThriveAI_ChromaDB.__init__(self, user_role=user_role, config=config)
            Ollama.__init__(self, config={"model": st.secrets["ai_keys"]["ollama_model"]})
        except Exception as e:
            logger.exception("Error Configuring MyVannaOllamaChromaDB: %s", e)
            raise

    # override the log function to stop vannas annoying print statements
    def log(self, message: str, title: str = "Info"):
        logger.debug("%s: %s", title, message)


def read_forbidden_from_json():
    try:
        # Path to the forbidden_references.json file
        forbidden_file_path = Path(__file__).parent / "config/forbidden_references.json"

        # Load the forbidden references
        with forbidden_file_path.open("r") as file:
            forbidden_data = json.load(file)

        forbidden_tables = forbidden_data.get("tables", [])
        forbidden_columns = forbidden_data.get("columns", [])
    except Exception as e:
        logger.exception("Error reading forbidden_references.json: %s", e)
        return [], [], []
    else:
        return forbidden_tables, forbidden_columns, ", ".join(f"'{table}'" for table in forbidden_tables)


class VannaService:
    _instances = {}  # Store instances by user_id instead of just user_role

    def __init__(self, user_context: UserContext, config: dict):
        """Initialize VannaService with explicit dependencies."""
        self.user_context = user_context
        self.config = config
        self.vn = None
        self._setup_vanna()

    @classmethod
    def get_instance(cls, user_context: UserContext = None, config: dict = None):
        """Get or create VannaService instance for the given user context."""
        # If no user_context provided, extract from Streamlit (backwards compatibility)
        if user_context is None:
            user_context = UserContext.from_streamlit_session()

        # If no config provided, extract from secrets (backwards compatibility)
        if config is None:
            config = extract_vanna_config_from_secrets()

        # Create cache key that includes both user_id and user_role
        cache_key = f"vanna_service_{user_context.user_id}_{user_context.user_role}"

        # Check if we already have an instance for this specific user
        if user_context.user_id not in cls._instances:
            cls._instances[user_context.user_id] = cls._create_instance_for_user(user_context, config, cache_key)

        return cls._instances[user_context.user_id]

    @classmethod
    def from_streamlit_session(cls):
        """Factory method to create VannaService from Streamlit session (convenience method)."""
        user_context = UserContext.from_streamlit_session()
        config = extract_vanna_config_from_secrets()
        return cls.get_instance(user_context, config)

    @classmethod
    @st.cache_resource(ttl=3600)
    def _create_instance_for_user(cls, user_context: UserContext, config: dict, cache_key: str):
        """Create VannaService instance for specific user. Cached by user_id and user_role."""
        # Validate that user_role is a valid role value
        try:
            from orm.models import RoleTypeEnum

            if user_context.user_role not in [role.value for role in RoleTypeEnum]:
                logger.error(f"Invalid user_role value: {user_context.user_role}. Defaulting to PATIENT role.")
                st.error(f"Invalid user role detected: {user_context.user_role}. Using restricted access.")
                user_context.user_role = RoleTypeEnum.PATIENT.value
        except Exception as e:
            logger.error(f"Error validating user_role: {e}. Defaulting to PATIENT role.")
            st.error("Error validating user permissions. Using restricted access.")
            user_context.user_role = RoleTypeEnum.PATIENT.value

        return cls(user_context, config)

    def _setup_vanna(self):
        """Setup Vanna with appropriate configuration."""
        try:
            chroma_config = (
                {"path": self.config["rag_model"]["chroma_path"]} if "chroma_path" in self.config["rag_model"] else None
            )

            if "ollama_model" in self.config["ai_keys"]:
                if chroma_config:
                    self.vn = MyVannaOllamaChromaDB(user_role=self.user_context.user_role, config=chroma_config)
                elif "vanna_api" in self.config["ai_keys"] and "vanna_model" in self.config["ai_keys"]:
                    self.vn = MyVannaOllama()
                else:
                    raise ValueError("Missing ollama Configuration Values")
            elif "anthropic_api" in self.config["ai_keys"] and "anthropic_model" in self.config["ai_keys"]:
                if chroma_config:
                    self.vn = MyVannaAnthropicChromaDB(user_role=self.user_context.user_role, config=chroma_config)
                elif "vanna_api" in self.config["ai_keys"] and "vanna_model" in self.config["ai_keys"]:
                    self.vn = MyVannaAnthropic()
                else:
                    raise ValueError("Missing anthropic Configuration Values")
            else:
                logger.info("Using Default")
                self.vn = VannaDefault(
                    api_key=self.config["ai_keys"]["vanna_api"],
                    model=self.config["ai_keys"]["vanna_model"],
                )

            self.vn.connect_to_postgres(
                host=self.config["postgres"]["host"],
                dbname=self.config["postgres"]["database"],
                user=self.config["postgres"]["user"],
                password=self.config["postgres"]["password"],
                port=self.config["postgres"]["port"],
            )
        except Exception as e:
            st.error(f"Error setting up Vanna: {e}")
            logger.exception("%s", e)
            raise

    @property
    def user_id(self) -> str:
        """Get the user ID."""
        return self.user_context.user_id

    @property
    def user_role(self) -> int:
        """Get the user role."""
        return self.user_context.user_role

    @st.cache_data(show_spinner="Generating sample questions ...")
    def generate_questions(_self):
        """Generate sample questions using Vanna."""
        try:
            questions = _self.vn.generate_questions()
        except Exception as e:
            st.error(f"Error generating questions: {e}")
            logger.exception("%s", e)
            return []
        else:
            return questions

    @st.cache_data(show_spinner="Generating SQL query ...")
    def generate_sql(_self, question: str):
        """Generate SQL from a natural language question."""
        try:
            start_time = time.perf_counter()
            allow_see_data = _self.config["security"].get("allow_llm_to_see_data", False)

            if allow_see_data:
                logger.info("Allowing LLM to see data")
                sql_response = _self.vn.generate_sql(question=question, allow_llm_to_see_data=True)
            else:
                logger.info("NOT allowing LLM to see data")
                sql_response = _self.vn.generate_sql(question=question)

            response = _self.check_references(sql_response)

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.info("Response is %s", response)
            logger.info("Elapsed time is %s", elapsed_time)
        except Exception as e:
            st.error(f"Error generating SQL: {e}")
            logger.exception("%s", e)
            return None, 0
        else:
            return response, elapsed_time

    @st.cache_data(show_spinner="Checking for valid SQL ...")
    def is_sql_valid(_self, sql: str) -> bool:
        """Check if SQL is valid."""
        try:
            is_valid = _self.vn.is_sql_valid(sql=sql)
        except Exception as e:
            st.error(f"Error checking SQL validity: {e}")
            logger.exception("%s", e)
            return False
        else:
            return is_valid

    @st.cache_data(show_spinner="Running SQL query ...")
    def run_sql(_self, sql: str) -> DataFrame | None:
        """Run SQL query and return results as DataFrame."""
        try:
            df = _self.vn.run_sql(sql=sql)
        except Exception as e:
            st.error(f"Error running SQL: {e}")
            logger.exception("%s", e)
            return None
        else:
            return df

    @st.cache_data(show_spinner="Checking if we should generate a chart ...")
    def should_generate_chart(_self, question, sql, df):
        """Check if a chart should be generated for the result."""
        try:
            generate_chart = _self.vn.should_generate_chart(df=df)
        except Exception as e:
            st.error(f"Error checking if we should generate a chart: {e}")
            logger.exception("%s", e)
            return False
        else:
            return generate_chart

    @st.cache_data(show_spinner="Generating Plotly code ...")
    def generate_plotly_code(_self, question, sql, df):
        """Generate Plotly code for visualization."""
        try:
            start_time = time.perf_counter()
            code = _self.vn.generate_plotly_code(question=question, sql=sql, df=df)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.info("Plotly code generation elapsed time is %s", elapsed_time)
        except Exception as e:
            st.error(f"Error generating Plotly code: {e}")
            logger.exception("%s", e)
            return None, 0
        else:
            return code, elapsed_time

    @st.cache_data(show_spinner="Running Plotly code ...")
    def generate_plot(_self, code, df):
        """Generate Plotly figure from code and data."""
        try:
            start_time = time.perf_counter()
            plotly_fig = _self.vn.get_plotly_figure(plotly_code=code, df=df)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.info("Plotly figure generation elapsed time is %s", elapsed_time)
        except Exception as e:
            st.error(f"Error generating plot: {e}")
            logger.exception("%s", e)
            return None, 0
        else:
            return plotly_fig, elapsed_time

    @st.cache_data(show_spinner="Generating followup questions ...")
    def generate_followup_questions(_self, question, sql, df):
        """Generate follow-up questions based on results."""
        try:
            followup_questions = _self.vn.generate_followup_questions(question=question, sql=sql, df=df)
        except Exception as e:
            st.error(f"Error generating followup questions: {e}")
            logger.exception("%s", e)
            return []
        else:
            return followup_questions

    @st.cache_data(show_spinner="Generating summary ...")
    def generate_summary(_self, question: str, df: DataFrame) -> tuple[str | None, float]:
        """Generate a summary of the results."""
        try:
            start_time = time.perf_counter()
            response = _self.vn.generate_summary(question=question, df=df)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.info("Summary generation elapsed time is %s", elapsed_time)
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            logger.exception("%s", e)
            return None, 0
        else:
            return response, elapsed_time

    def submit_prompt(self, system_message, user_message):
        """Submit generic prompt to Vanna."""
        try:
            return self.vn.submit_prompt(
                prompt=[
                    self.vn.system_message(system_message),
                    self.vn.user_message(user_message),
                ]
            )
        except Exception as e:
            st.error(f"Error prompting Vanna: {e}")
            logger.exception("%s", e)
            return e

    def remove_from_training(self, entry_id):
        """Remove a training entry by ID."""
        try:
            self.vn.remove_training_data(entry_id)
            return True
        except Exception as e:
            st.error(f"Error removing training data: {e}")
            logger.exception("%s", e)
            return False

    def get_training_data(self, metadata: dict[str, Any] | None = None):
        """Get training data with role-based filtering applied."""
        try:
            # If using ChromaDB backend, it will automatically apply role-based filtering
            # For other backends, we need to ensure appropriate filtering
            if hasattr(self.vn, "_prepare_retrieval_metadata"):
                # ChromaDB backend - let it handle the role-based filtering
                effective_metadata = self.vn._prepare_retrieval_metadata(metadata)
                return self.vn.get_training_data(metadata=effective_metadata)
            else:
                # Other backends - apply basic role-based filtering
                effective_metadata = metadata.copy() if metadata is not None else {}
                effective_metadata["user_role"] = {"$gte": self.user_role}
                return self.vn.get_training_data(metadata=effective_metadata)
        except Exception as e:
            st.error(f"Error getting training data: {e}")
            logger.exception("%s", e)
            return DataFrame()

    def train(
        self, question=None, sql=None, documentation=None, ddl=None, plan=None, metadata: dict[str, Any] | None = None
    ):
        """Train Vanna with various types of data, ensuring user_role is in metadata."""
        try:
            effective_metadata = metadata.copy() if metadata is not None else {}
            effective_metadata["user_role"] = self.user_role

            if question and sql:
                self.vn.add_question_sql(question=question, sql=sql, metadata=effective_metadata)
            elif documentation:
                self.vn.add_documentation(documentation=documentation, metadata=effective_metadata)
            elif ddl:
                self.vn.add_ddl(ddl=ddl, metadata=effective_metadata)
            elif plan:
                self.vn.train(plan=plan)
            return True
        except Exception as e:
            st.error(f"Error training Vanna: {e}")
            logger.exception("%s", e)
            return False

    def get_training_plan_generic(self, df_information_schema):
        """Get a generic training plan."""
        try:
            plan = self.vn.get_training_plan_generic(df_information_schema)
        except Exception as e:
            st.error(f"Error getting training plan: {e}")
            logger.exception("%s", e)
            return None
        else:
            return plan

    def check_references(self, sql):
        """Check SQL for forbidden references."""
        try:
            forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()

            # TODO: should I make this role based? or user based?
            parsed = sqlparse.parse(sql)[0]
            tables, columns = get_identifiers(parsed)

            # Check for forbidden references
            referenced_tables = set(forbidden_tables).intersection(set(tables))
            referenced_columns = set(forbidden_columns).intersection(set(columns))
            if referenced_tables:
                raise ValueError(f"Referenced forbidden tables: {referenced_tables}")
            if referenced_columns:
                raise ValueError(f"Referenced forbidden columns: {referenced_columns}")
        except Exception as e:
            st.error(f"Error checking references: {e}")
            logger.exception("%s", e)
        else:
            return sql

    @classmethod
    def _initialize_instance(cls):
        """Initialize the VannaService instance with appropriate configuration."""
        # This method is now just a wrapper around get_instance for backwards compatibility
        return cls.get_instance()


# Backward compatibility functions that use the VannaService singleton
def setup_vanna():
    """
    Legacy function for backward compatibility.
    Returns the Vanna instance from the VannaService singleton.
    """
    return VannaService.from_streamlit_session().vn


@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    return VannaService.from_streamlit_session().generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    return VannaService.from_streamlit_session().generate_sql(question=question)


@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    return VannaService.from_streamlit_session().is_sql_valid(sql)


@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str) -> DataFrame:
    return VannaService.from_streamlit_session().run_sql(sql)


@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    return VannaService.from_streamlit_session().should_generate_chart(question, sql, df)


@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    return VannaService.from_streamlit_session().generate_plotly_code(question, sql, df)


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    return VannaService.from_streamlit_session().generate_plot(code, df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    return VannaService.from_streamlit_session().generate_followup_questions(question, sql, df)


@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question: str, df: DataFrame) -> tuple[str | None, float]:
    return VannaService.from_streamlit_session().generate_summary(question, df)


def remove_from_file_training(new_entry: dict):
    try:
        vanna_service = VannaService.from_streamlit_session()
        training_data = vanna_service.get_training_data()

        for index, row in training_data.iterrows():
            if row["question"] and row["question"] == new_entry["question"]:
                vanna_service.remove_from_training(row["id"])

        # Path to the training_data.json file
        training_file_path = Path(__file__).parent / "config/training_data.json"

        # Load the existing data
        with training_file_path.open("r") as file:
            training_data = json.load(file)

        # Check for duplicates based on the question text
        existing_questions = {entry["question"] for entry in training_data["sample_queries"]}
        if new_entry["question"] in existing_questions:
            # Remove the new entry from the sample_queries list if it's not a duplicate
            training_data["sample_queries"] = [
                entry for entry in training_data["sample_queries"] if entry["question"] != new_entry["question"]
            ]

            # Write the updated data back to the file
            with training_file_path.open("w") as file:
                json.dump(training_data, file, indent=4)

            logging.info("Entry removed from training_data.json")
        else:
            logging.warning("Entry not found, nothing removed from training_data.json")
    except Exception as e:
        st.error(f"Error removing entry from training_data.json: {e}")
        logging.exception("Error removing entry from training_data.json: %s", e)


def write_to_file_and_training(new_entry: dict):
    try:
        vanna_service = VannaService.from_streamlit_session()
        vanna_service.train(question=new_entry["question"], sql=new_entry["query"])

        # Path to the training_data.json file
        training_file_path = Path(__file__).parent / "config/training_data.json"

        # Load the existing data
        with training_file_path.open("r") as file:
            training_data = json.load(file)

        # Check for duplicates based on the question text
        existing_questions = {entry["question"] for entry in training_data["sample_queries"]}
        if new_entry["question"] not in existing_questions:
            # Append the new entry to the sample_queries list if it's not a duplicate
            training_data["sample_queries"].append(new_entry)

            # Write the updated data back to the file
            with training_file_path.open("w") as file:
                json.dump(training_data, file, indent=4)

            logging.info("New entry added to training_data.json")
        else:
            logging.info("Duplicate entry found. No new entry added.")
    except Exception as e:
        st.error(f"Error writing to training_data.json: {e}")
        logger.exception("%s", e)


def training_plan():
    """
    RAG-optimized training plan that captures comprehensive database schema information
    including relationships, constraints, sample data, and semantic context for superior SQL generation.
    
    Returns:
        bool: True if training completed successfully, False otherwise
    """
    try:
        st.toast("🚀 Starting RAG-optimized training plan...")
        logger.info("Starting RAG-optimized training plan generation")
        
        vanna_service = VannaService.from_streamlit_session()
        if not vanna_service:
            st.error("Failed to initialize VannaService")
            return False
        
        # Get forbidden tables with proper error handling
        try:
            forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()
            logger.info(f"Loaded {len(forbidden_tables)} forbidden tables")
        except Exception as e:
            logger.warning(f"Error reading forbidden tables: {e}")
            forbidden_tables_str = ""
        
        # Get object type and schema configuration
        object_type = get_configured_object_type()
        schema_name = get_configured_schema()
        
        object_name = "table" if object_type == "tables" else "view"
        logger.info(f"Training plan will process {object_type} in schema '{schema_name}'")
        
        # Build enhanced schema query with relationships and semantic information
        # Note: Views don't have constraints like foreign keys or primary keys
        if object_type == "tables":
            constraint_ctes = """
                foreign_keys AS (
                    SELECT
                        tc.table_name,
                        kcu.column_name,
                        ccu.table_name AS referenced_table,
                        ccu.column_name AS referenced_column,
                        tc.constraint_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_schema = '{schema_name}'
                ),
                primary_keys AS (
                    SELECT
                        tc.table_name,
                        kcu.column_name,
                        tc.constraint_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                    AND tc.table_schema = '{schema_name}'
                ),"""
            constraint_joins = """
                LEFT JOIN foreign_keys fk ON fk.table_name = c.table_name AND fk.column_name = c.column_name
                LEFT JOIN primary_keys pk ON pk.table_name = c.table_name AND pk.column_name = c.column_name"""
            constraint_columns = """
                    -- Constraints and relationships  
                    pk.constraint_name as primary_key,
                    fk.referenced_table,
                    fk.referenced_column,
                    fk.constraint_name as foreign_key_name,"""
            table_filter = f"AND c.table_name IN (SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema_name}' AND table_type = 'BASE TABLE')"
        else:  # views
            constraint_ctes = ""
            constraint_joins = ""
            constraint_columns = """
                    -- Views don't have constraints
                    NULL as primary_key,
                    NULL as referenced_table,
                    NULL as referenced_column,  
                    NULL as foreign_key_name,"""
            table_filter = f"AND c.table_name IN (SELECT table_name FROM information_schema.views WHERE table_schema = '{schema_name}')"

        if forbidden_tables_str:
            enhanced_query = f"""
                WITH {constraint_ctes}
                object_filter AS (
                    SELECT table_name FROM information_schema.{object_type} 
                    WHERE table_schema = '{schema_name}' 
                    AND table_name NOT IN ({forbidden_tables_str})
                )
                SELECT DISTINCT
                    c.table_catalog as database_name,
                    c.table_schema,
                    c.table_name,
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    c.column_default,
                    c.ordinal_position,
                    c.character_maximum_length,
                    c.numeric_precision,
                    c.numeric_scale,
                    CASE WHEN c.data_type = 'USER-DEFINED' THEN c.udt_name ELSE NULL END as enum_type,
                    {constraint_columns}
                    -- Semantic indicators based on column names
                    CASE 
                        WHEN c.column_name ILIKE '%id' OR c.column_name ILIKE '%_id' THEN 'identifier'
                        WHEN c.column_name ILIKE '%name%' OR c.column_name ILIKE '%title%' THEN 'name'
                        WHEN c.column_name ILIKE '%email%' THEN 'email'
                        WHEN c.column_name ILIKE '%phone%' THEN 'phone'
                        WHEN c.column_name ILIKE '%date%' OR c.column_name ILIKE '%time%' OR c.column_name ILIKE '%created%' OR c.column_name ILIKE '%updated%' THEN 'temporal'
                        WHEN c.column_name ILIKE '%price%' OR c.column_name ILIKE '%cost%' OR c.column_name ILIKE '%amount%' THEN 'monetary'
                        WHEN c.column_name ILIKE '%count%' OR c.column_name ILIKE '%quantity%' OR c.column_name ILIKE '%number%' THEN 'quantity'
                        WHEN c.column_name ILIKE '%status%' OR c.column_name ILIKE '%state%' THEN 'status'
                        WHEN c.column_name ILIKE '%address%' OR c.column_name ILIKE '%city%' OR c.column_name ILIKE '%zip%' THEN 'location'
                        ELSE 'general'
                    END as semantic_type
                    
                FROM information_schema.columns c
                JOIN object_filter o ON o.table_name = c.table_name
                {constraint_joins}
                WHERE c.table_schema = '{schema_name}'
                ORDER BY c.table_schema, c.table_name, c.ordinal_position
            """
        else:
            enhanced_query = f"""
                WITH {constraint_ctes}
                object_filter AS (
                    SELECT table_name FROM information_schema.{object_type} 
                    WHERE table_schema = '{schema_name}'
                )
                SELECT DISTINCT
                    c.table_catalog as database_name,
                    c.table_schema,
                    c.table_name,
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    c.column_default,
                    c.ordinal_position,
                    c.character_maximum_length,
                    c.numeric_precision,
                    c.numeric_scale,
                    CASE WHEN c.data_type = 'USER-DEFINED' THEN c.udt_name ELSE NULL END as enum_type,
                    {constraint_columns}
                    -- Semantic indicators based on column names
                    CASE 
                        WHEN c.column_name ILIKE '%id' OR c.column_name ILIKE '%_id' THEN 'identifier'
                        WHEN c.column_name ILIKE '%name%' OR c.column_name ILIKE '%title%' THEN 'name'
                        WHEN c.column_name ILIKE '%email%' THEN 'email'
                        WHEN c.column_name ILIKE '%phone%' THEN 'phone'
                        WHEN c.column_name ILIKE '%date%' OR c.column_name ILIKE '%time%' OR c.column_name ILIKE '%created%' OR c.column_name ILIKE '%updated%' THEN 'temporal'
                        WHEN c.column_name ILIKE '%price%' OR c.column_name ILIKE '%cost%' OR c.column_name ILIKE '%amount%' THEN 'monetary'
                        WHEN c.column_name ILIKE '%count%' OR c.column_name ILIKE '%quantity%' OR c.column_name ILIKE '%number%' THEN 'quantity'
                        WHEN c.column_name ILIKE '%status%' OR c.column_name ILIKE '%state%' THEN 'status'
                        WHEN c.column_name ILIKE '%address%' OR c.column_name ILIKE '%city%' OR c.column_name ILIKE '%zip%' THEN 'location'
                        ELSE 'general'
                    END as semantic_type
                    
                FROM information_schema.columns c
                JOIN object_filter o ON o.table_name = c.table_name
                {constraint_joins}
                WHERE c.table_schema = '{schema_name}'
                ORDER BY c.table_schema, c.table_name, c.ordinal_position
            """
        
        # Execute enhanced schema query
        st.toast("📊 Retrieving enhanced schema information...")
        df_information_schema = vanna_service.run_sql(enhanced_query)
        
        if df_information_schema is None or df_information_schema.empty:
            st.warning("No schema information retrieved")
            return False
        
        object_count = df_information_schema['table_name'].nunique()
        column_count = len(df_information_schema)
        logger.info(f"Retrieved enhanced schema for {object_count} {object_type} with {column_count} columns")
        
        # Train standard schema plan
        st.toast("🎓 Training enhanced schema plan...")
        plan = vanna_service.get_training_plan_generic(df_information_schema)
        if plan:
            vanna_service.train(plan=plan)
            logger.info("Standard schema training plan executed")
        
        # Train relationship documentation (only for tables, not views)
        relationships_trained = 0
        if object_type == "tables":
            st.toast("🔗 Training table relationships...")
            for table in df_information_schema['table_name'].unique():
                table_data = df_information_schema[df_information_schema['table_name'] == table]
                
                # Foreign key relationships
                fk_relationships = table_data[table_data['referenced_table'].notna()]
                for _, fk_row in fk_relationships.iterrows():
                    relationship_doc = f"""
                    Table {fk_row['table_name']} column {fk_row['column_name']} references {fk_row['referenced_table']}.{fk_row['referenced_column']}.
                    This creates a relationship where {fk_row['table_name']} belongs to {fk_row['referenced_table']}.
                    Use JOIN {fk_row['referenced_table']} ON {fk_row['table_name']}.{fk_row['column_name']} = {fk_row['referenced_table']}.{fk_row['referenced_column']} to connect these tables.
                    """
                    vanna_service.train(documentation=relationship_doc)
                    relationships_trained += 1
        
        # Train semantic column information
        st.toast("🏷️ Training semantic context...")
        semantics_trained = 0
        for _, row in df_information_schema.iterrows():
            if row.get('semantic_type') and row['semantic_type'] != 'general':
                semantic_doc = f"""
                Column {row['table_name']}.{row['column_name']} ({row['data_type']}) is a {row['semantic_type']} field.
                """
                vanna_service.train(documentation=semantic_doc)
                semantics_trained += 1
        
        # Train query patterns
        st.toast("🎯 Training query patterns...")
        patterns_trained = 0
        for table in df_information_schema['table_name'].unique():
            table_data = df_information_schema[df_information_schema['table_name'] == table]
            
            # Generate common query pattern documentation
            id_columns = table_data[table_data['semantic_type'] == 'identifier']['column_name'].tolist()
            name_columns = table_data[table_data['semantic_type'] == 'name']['column_name'].tolist()
            date_columns = table_data[table_data['semantic_type'] == 'temporal']['column_name'].tolist()
            
            query_patterns = []
            if id_columns:
                query_patterns.append(f"To find {object_name} {table} by ID: SELECT * FROM {table} WHERE {id_columns[0]} = value")
            if name_columns:
                query_patterns.append(f"To search {object_name} {table} by name: SELECT * FROM {table} WHERE {name_columns[0]} ILIKE '%value%'")
            if date_columns:
                query_patterns.append(f"To get recent {object_name} {table}: SELECT * FROM {table} ORDER BY {date_columns[0]} DESC")
            
            if query_patterns:
                pattern_doc = f"Common query patterns for {object_name} {table}:\n" + "\n".join(query_patterns)
                vanna_service.train(documentation=pattern_doc)
                patterns_trained += 1
        
        # Show final success message
        relationship_msg = f"🔗 Trained {relationships_trained} relationships\n        " if object_type == "tables" else ""
        st.success(f"""
        🎉 RAG-optimized training plan completed successfully! 
        📊 Processed {object_count} {object_type} with {column_count} columns
        {relationship_msg}🏷️ Enhanced {semantics_trained} columns with semantic context
        🎯 Added {patterns_trained} query pattern sets
        """)
        logger.info(f"Enhanced training plan completed: {object_count} {object_type}, {relationships_trained} relationships, {semantics_trained} semantics, {patterns_trained} patterns")
        return True
        
    except Exception as e:
        error_msg = f"Error in RAG-optimized training plan: {e}"
        st.error(error_msg)
        logger.exception("Error in training_plan: %s", e)
        return False


def train_ddl(describe_ddl: bool = False):
    """
    RAG-optimized DDL training function that captures comprehensive schema information
    including constraints, indexes, triggers, and enhanced metadata for superior SQL generation.
    
    Args:
        describe_ddl (bool): Whether to generate AI descriptions for DDL structures
        
    Returns:
        bool: True if training completed successfully, False otherwise
    """
    conn = None
    try:
        st.toast("🚀 Starting RAG-optimized DDL training...")
        logger.info("Starting comprehensive DDL training")
        
        vanna_service = VannaService.from_streamlit_session()
        if not vanna_service:
            st.error("Failed to initialize VannaService")
            return False
            
        # Get forbidden tables with proper error handling
        try:
            forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()
            logger.info(f"Loaded {len(forbidden_tables)} forbidden tables")
        except Exception as e:
            logger.warning(f"Error reading forbidden tables: {e}")
            forbidden_tables_str = ""
        
        # Get object type and schema configuration
        object_type = get_configured_object_type()
        schema_name = get_configured_schema()
        
        object_name = "table" if object_type == "tables" else "view"
        logger.info(f"DDL training will process {object_type} in schema '{schema_name}'")
        
        # Establish database connection with context manager
        from contextlib import closing
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            port=st.secrets["postgres"]["port"],
            database=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
        )
        
        with closing(conn), conn.cursor() as cursor:
            # Build schema query based on object type and forbidden objects
            # Note: Views don't have constraints like foreign keys or primary keys
            if object_type == "tables":
                constraint_ctes = """
                    table_constraints AS (
                        SELECT 
                            tc.table_name,
                            tc.constraint_name,
                            tc.constraint_type,
                            kcu.column_name,
                            rc.match_option,
                            rc.update_rule,
                            rc.delete_rule,
                            ccu.table_name AS referenced_table_name,
                            ccu.column_name AS referenced_column_name
                        FROM information_schema.table_constraints tc
                        LEFT JOIN information_schema.key_column_usage kcu 
                            ON tc.constraint_name = kcu.constraint_name
                        LEFT JOIN information_schema.referential_constraints rc 
                            ON tc.constraint_name = rc.constraint_name
                        LEFT JOIN information_schema.constraint_column_usage ccu 
                            ON rc.unique_constraint_name = ccu.constraint_name
                        WHERE tc.table_schema = '{schema_name}'
                    ),"""
                constraint_joins = """
                    LEFT JOIN table_constraints tc ON tc.table_name = c.table_name AND tc.column_name = c.column_name"""
                constraint_columns = """
                        -- Constraint information
                        tc.constraint_type,
                        tc.constraint_name,
                        tc.referenced_table_name,
                        tc.referenced_column_name,
                        tc.update_rule,
                        tc.delete_rule,"""
            else:  # views
                constraint_ctes = ""
                constraint_joins = ""
                constraint_columns = """
                        -- Views don't have constraints
                        NULL as constraint_type,
                        NULL as constraint_name,
                        NULL as referenced_table_name,
                        NULL as referenced_column_name,
                        NULL as update_rule,
                        NULL as delete_rule,"""

            if forbidden_tables_str:
                schema_query = f"""
                    WITH {constraint_ctes}
                    object_filter AS (
                        SELECT table_name FROM information_schema.{object_type} 
                        WHERE table_schema = '{schema_name}' 
                        AND table_name NOT IN ({forbidden_tables_str})
                    )
                    SELECT DISTINCT
                        c.table_schema,
                        c.table_name,
                        c.column_name,
                        c.data_type,
                        c.is_nullable,
                        c.column_default,
                        c.ordinal_position,
                        c.character_maximum_length,
                        c.numeric_precision,
                        c.numeric_scale,
                        c.datetime_precision,
                        c.udt_name,
                        CASE WHEN c.data_type = 'USER-DEFINED' THEN c.udt_name ELSE NULL END as enum_type,
                        {constraint_columns}
                        -- Enhanced semantic classification
                        CASE 
                            WHEN c.column_name ILIKE '%id' OR c.column_name ILIKE '%_id' OR c.column_name = 'id' THEN 'primary_key'
                            WHEN c.column_name ILIKE '%uuid%' OR c.column_name ILIKE '%guid%' THEN 'uuid'
                            WHEN c.column_name ILIKE '%name%' OR c.column_name ILIKE '%title%' THEN 'name'
                            WHEN c.column_name ILIKE '%email%' THEN 'email'
                            WHEN c.column_name ILIKE '%phone%' OR c.column_name ILIKE '%mobile%' THEN 'phone'
                            WHEN c.column_name ILIKE '%address%' OR c.column_name ILIKE '%street%' OR c.column_name ILIKE '%city%' OR c.column_name ILIKE '%zip%' OR c.column_name ILIKE '%postal%' THEN 'address'
                            WHEN c.column_name ILIKE '%date%' OR c.column_name ILIKE '%time%' OR c.column_name ILIKE '%created%' OR c.column_name ILIKE '%updated%' OR c.column_name ILIKE '%modified%' THEN 'temporal'
                            WHEN c.column_name ILIKE '%price%' OR c.column_name ILIKE '%cost%' OR c.column_name ILIKE '%amount%' OR c.column_name ILIKE '%fee%' OR c.column_name ILIKE '%total%' THEN 'monetary'
                            WHEN c.column_name ILIKE '%count%' OR c.column_name ILIKE '%quantity%' OR c.column_name ILIKE '%number%' OR c.column_name ILIKE '%size%' THEN 'quantity'
                            WHEN c.column_name ILIKE '%status%' OR c.column_name ILIKE '%state%' OR c.column_name ILIKE '%type%' THEN 'categorical'
                            WHEN c.column_name ILIKE '%description%' OR c.column_name ILIKE '%comment%' OR c.column_name ILIKE '%note%' THEN 'text'
                            WHEN c.column_name ILIKE '%url%' OR c.column_name ILIKE '%link%' OR c.column_name ILIKE '%website%' THEN 'url'
                            WHEN c.column_name ILIKE '%image%' OR c.column_name ILIKE '%photo%' OR c.column_name ILIKE '%picture%' THEN 'media'
                            WHEN c.column_name ILIKE '%json%' OR c.column_name ILIKE '%data%' OR c.column_name ILIKE '%config%' THEN 'structured_data'
                            ELSE 'general'
                        END as semantic_type
                        
                    FROM information_schema.columns c
                    JOIN object_filter o ON o.table_name = c.table_name
                    {constraint_joins}
                    WHERE c.table_schema = '{schema_name}'
                    ORDER BY c.table_schema, c.table_name, c.ordinal_position
                """
            else:
                schema_query = f"""
                    WITH {constraint_ctes}
                    object_filter AS (
                        SELECT table_name FROM information_schema.{object_type} 
                        WHERE table_schema = '{schema_name}'
                    )
                    SELECT DISTINCT
                        c.table_schema,
                        c.table_name,
                        c.column_name,
                        c.data_type,
                        c.is_nullable,
                        c.column_default,
                        c.ordinal_position,
                        c.character_maximum_length,
                        c.numeric_precision,
                        c.numeric_scale,
                        c.datetime_precision,
                        c.udt_name,
                        CASE WHEN c.data_type = 'USER-DEFINED' THEN c.udt_name ELSE NULL END as enum_type,
                        {constraint_columns}
                        -- Enhanced semantic classification
                        CASE 
                            WHEN c.column_name ILIKE '%id' OR c.column_name ILIKE '%_id' OR c.column_name = 'id' THEN 'primary_key'
                            WHEN c.column_name ILIKE '%uuid%' OR c.column_name ILIKE '%guid%' THEN 'uuid'
                            WHEN c.column_name ILIKE '%name%' OR c.column_name ILIKE '%title%' THEN 'name'
                            WHEN c.column_name ILIKE '%email%' THEN 'email'
                            WHEN c.column_name ILIKE '%phone%' OR c.column_name ILIKE '%mobile%' THEN 'phone'
                            WHEN c.column_name ILIKE '%address%' OR c.column_name ILIKE '%street%' OR c.column_name ILIKE '%city%' OR c.column_name ILIKE '%zip%' OR c.column_name ILIKE '%postal%' THEN 'address'
                            WHEN c.column_name ILIKE '%date%' OR c.column_name ILIKE '%time%' OR c.column_name ILIKE '%created%' OR c.column_name ILIKE '%updated%' OR c.column_name ILIKE '%modified%' THEN 'temporal'
                            WHEN c.column_name ILIKE '%price%' OR c.column_name ILIKE '%cost%' OR c.column_name ILIKE '%amount%' OR c.column_name ILIKE '%fee%' OR c.column_name ILIKE '%total%' THEN 'monetary'
                            WHEN c.column_name ILIKE '%count%' OR c.column_name ILIKE '%quantity%' OR c.column_name ILIKE '%number%' OR c.column_name ILIKE '%size%' THEN 'quantity'
                            WHEN c.column_name ILIKE '%status%' OR c.column_name ILIKE '%state%' OR c.column_name ILIKE '%type%' THEN 'categorical'
                            WHEN c.column_name ILIKE '%description%' OR c.column_name ILIKE '%comment%' OR c.column_name ILIKE '%note%' THEN 'text'
                            WHEN c.column_name ILIKE '%url%' OR c.column_name ILIKE '%link%' OR c.column_name ILIKE '%website%' THEN 'url'
                            WHEN c.column_name ILIKE '%image%' OR c.column_name ILIKE '%photo%' OR c.column_name ILIKE '%picture%' THEN 'media'
                            WHEN c.column_name ILIKE '%json%' OR c.column_name ILIKE '%data%' OR c.column_name ILIKE '%config%' THEN 'structured_data'
                            ELSE 'general'
                        END as semantic_type
                        
                    FROM information_schema.columns c
                    JOIN object_filter o ON o.table_name = c.table_name
                    {constraint_joins}
                    WHERE c.table_schema = '{schema_name}'
                    ORDER BY c.table_schema, c.table_name, c.ordinal_position
                """
            
            # Execute comprehensive schema query
            st.toast("📊 Retrieving comprehensive schema information...")
            cursor.execute(schema_query)
            schema_info = cursor.fetchall()
            
            if not schema_info:
                st.warning("No schema information found")
                return False
            
            # Organize data by table for enhanced DDL generation
            tables_data = {}
            for row in schema_info:
                table_name = row[1]
                if table_name not in tables_data:
                    tables_data[table_name] = {
                        'columns': [],
                        'constraints': [],
                        'indexes': [],
                        'semantic_info': []
                    }
                
                # Column information
                column_info = {
                    'name': row[2],
                    'type': row[3],
                    'nullable': row[4],
                    'default': row[5],
                    'position': row[6],
                    'max_length': row[7],
                    'precision': row[8],
                    'scale': row[9],
                    'datetime_precision': row[10],
                    'udt_name': row[11],
                    'enum_type': row[12],
                    'semantic_type': row[19] if len(row) > 19 else 'general'
                }
                tables_data[table_name]['columns'].append(column_info)
                
                # Constraint information
                if row[13]:  # constraint_type exists
                    constraint_info = {
                        'type': row[13],
                        'name': row[14],
                        'column': row[2],
                        'referenced_table': row[15],
                        'referenced_column': row[16],
                        'update_rule': row[17],
                        'delete_rule': row[18]
                    }
                    if constraint_info not in tables_data[table_name]['constraints']:
                        tables_data[table_name]['constraints'].append(constraint_info)
                
                # Note: Index information removed to simplify query
            
            # Generate and train enhanced DDL for each table
            st.toast("🎓 Training enhanced DDL structures...")
            tables_trained = 0
            constraints_trained = 0
            semantic_docs_trained = 0
            
            for table_name, table_data in tables_data.items():
                # Generate enhanced DDL
                ddl_lines = [f"\nCREATE TABLE {table_name} ("]
                
                # Add columns with enhanced information
                column_definitions = []
                for col in sorted(table_data['columns'], key=lambda x: x['position']):
                    nullable = "NULL" if col['nullable'] == "YES" else "NOT NULL"
                    default_clause = f" DEFAULT {col['default']}" if col['default'] else ""
                    
                    # Enhanced type information
                    data_type = col['type']
                    if col['max_length']:
                        data_type += f"({col['max_length']})"
                    elif col['precision'] and col['scale']:
                        data_type += f"({col['precision']},{col['scale']})"
                    elif col['datetime_precision']:
                        data_type += f"({col['datetime_precision']})"
                    
                    column_def = f"    {col['name']} {data_type}{default_clause} {nullable}"
                    column_definitions.append(column_def)
                
                ddl_lines.extend([',\n'.join(column_definitions)])
                ddl_lines.append(");")
                
                # Train basic DDL
                enhanced_ddl = '\n'.join(ddl_lines)
                success = vanna_service.train(ddl=enhanced_ddl)
                if success:
                    tables_trained += 1
                    st.toast(f"✓ Trained enhanced DDL for table: {table_name}")
                
                # Train constraint documentation
                for constraint in table_data['constraints']:
                    if constraint['type'] == 'FOREIGN KEY':
                        constraint_doc = f"""
                        Table {table_name} has a foreign key constraint on {constraint['column']} 
                        referencing {constraint['referenced_table']}.{constraint['referenced_column']}.
                        Update rule: {constraint['update_rule']}, Delete rule: {constraint['delete_rule']}.
                        This enforces referential integrity between {table_name} and {constraint['referenced_table']}.
                        """
                        vanna_service.train(documentation=constraint_doc)
                        constraints_trained += 1
                    elif constraint['type'] == 'PRIMARY KEY':
                        constraint_doc = f"""
                        Table {table_name} has primary key constraint on {constraint['column']}.
                        This uniquely identifies each row in {table_name}.
                        """
                        vanna_service.train(documentation=constraint_doc)
                        constraints_trained += 1
                    elif constraint['type'] == 'UNIQUE':
                        constraint_doc = f"""
                        Table {table_name} has unique constraint on {constraint['column']}.
                        This ensures no duplicate values in {constraint['column']}.
                        """
                        vanna_service.train(documentation=constraint_doc)
                        constraints_trained += 1
                
                # Note: Index training removed to simplify implementation
                
                # Train semantic type documentation
                semantic_groups = {}
                for col in table_data['columns']:
                    if col['semantic_type'] != 'general':
                        if col['semantic_type'] not in semantic_groups:
                            semantic_groups[col['semantic_type']] = []
                        semantic_groups[col['semantic_type']].append(col['name'])
                
                for semantic_type, columns in semantic_groups.items():
                    semantic_doc = f"""
                    Table {table_name} has {semantic_type} columns: {', '.join(columns)}.
                    These columns contain {semantic_type} data and should be handled accordingly in queries.
                    """
                    vanna_service.train(documentation=semantic_doc)
                    semantic_docs_trained += 1
                
                # Optional: Generate AI descriptions
                if describe_ddl:
                    try:
                        train_ddl_describe_to_rag(table_name, [enhanced_ddl])
                    except Exception as e:
                        logger.warning(f"Failed to generate AI description for {table_name}: {e}")
        
        # Show comprehensive success message
        relationship_msg = f"🔗 Documented {constraints_trained} constraint relationships\n        " if object_type == "tables" else ""
        st.success(f"""
        🎉 RAG-optimized DDL training completed successfully!
        📊 Enhanced {tables_trained} {object_name} DDL structures
        {relationship_msg}🏷️ Classified {semantic_docs_trained} semantic column groups
        """)
        logger.info(f"Enhanced DDL training completed: {tables_trained} {object_type}, {constraints_trained} constraints, {semantic_docs_trained} semantic docs")
        return True
        
    except Exception as e:
        error_msg = f"Error in RAG-optimized DDL training: {e}"
        st.error(error_msg)
        logger.exception("Error in train_ddl: %s", e)
        return False
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")


def train_ddl_describe_to_rag(table: str, ddl: list):
    """
    Generate statistical profiles for table columns and train RAG model with the insights.
    
    This function analyzes column data using pandas statistical methods to create
    comprehensive profiles that help the RAG model understand data characteristics
    for better SQL query generation.
    
    Args:
        table (str): Name of the table to analyze
        ddl (list): DDL definition (unused in current implementation)
    """
    conn = None
    try:
        # Establish database connection
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            port=st.secrets["postgres"]["port"],
            database=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
        )

        # Get column names from table structure
        column_names = _get_table_columns(conn, table)
        if not column_names:
            return

        # Fetch sample data for statistical analysis
        sample_data = _get_sample_data(conn, table)
        if sample_data is None or sample_data.empty:
            return

        # Initialize Vanna service for training
        vanna_service = VannaService.from_streamlit_session()
        st.toast(f"Analyzing {table} with statistical profiling...")

        # Analyze each column and train RAG model
        for column in column_names:
            if column not in sample_data.columns:
                logger.warning(f"Column {column} not found in sample data for table {table}")
                continue

            try:
                st.toast(f"Profiling column: {table}.{column}")
                profile = _generate_column_profile(table, column, sample_data[column])
                
                # Train RAG model with statistical profile
                vanna_service.train(documentation=f"{table}.{column} statistical profile: {profile}")
                logger.info(f"Trained profile for {table}.{column}")
                
            except Exception as col_err:
                logger.warning(f"Failed to analyze column {column} in table {table}: {col_err}")
                continue

    except psycopg2.Error as db_err:
        st.error(f"Database error during statistical analysis for {table}: {db_err}")
        logger.exception(f"Database error for table {table}: %s", db_err)
    except Exception as e:
        st.error(f"Error during statistical profiling for {table}: {e}")
        logger.exception(f"Statistical profiling error for table {table}: %s", e)
    finally:
        if conn:
            conn.close()


def _get_table_columns(conn, table: str) -> list:
    """Get column names from table using safe SQL execution."""
    try:
        with conn.cursor() as cur:
            # Use psycopg2.sql for safe identifier quoting
            from psycopg2 import sql as psycopg2_sql
            safe_table_ident = psycopg2_sql.Identifier(table)
            query = psycopg2_sql.SQL("SELECT * FROM {} LIMIT 1;").format(safe_table_ident)
            cur.execute(query)
            
            if cur.description is None:
                logger.warning(f"Table {table} appears to be empty or inaccessible")
                return []
            
            return [desc[0] for desc in cur.description]
            
    except ImportError:
        # Fallback for systems without psycopg2.sql
        with conn.cursor() as cur:
            cur.execute(f'SELECT * FROM "{table}" LIMIT 1;')
            if cur.description is None:
                return []
            return [desc[0] for desc in cur.description]
    except Exception as e:
        logger.warning(f"Could not retrieve columns for table {table}: {e}")
        return []


def _get_sample_data(conn, table: str) -> pd.DataFrame:
    """Fetch sample data for statistical analysis."""
    try:
        # Use psycopg2.sql for safe table name handling
        from psycopg2 import sql as psycopg2_sql
        safe_table_ident = psycopg2_sql.Identifier(table)
        query = psycopg2_sql.SQL("SELECT * FROM {} LIMIT 1000;").format(safe_table_ident)
        return pd.read_sql_query(query.as_string(conn), conn)
        
    except ImportError:
        # Fallback for systems without psycopg2.sql
        query = f'SELECT * FROM "{table}" LIMIT 1000;'
        return pd.read_sql_query(query, conn)
    except Exception as e:
        logger.warning(f"Could not fetch sample data for table {table}: {e}")
        return None


def _generate_column_profile(table: str, column: str, column_data: pd.Series) -> str:
    """
    Generate comprehensive statistical profile for a single column.
    
    Returns a detailed description including data type, statistics, patterns,
    and insights useful for SQL query generation.
    """
    # Basic statistics
    total_rows = len(column_data)
    non_null_count = column_data.count()
    null_count = column_data.isnull().sum()
    
    # Guard against division by zero
    null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0
    unique_count = column_data.nunique()
    unique_percentage = (unique_count / non_null_count * 100) if non_null_count > 0 else 0
    
    # Build profile description
    profile_parts = [
        f"Column '{column}' in table '{table}'",
        f"Data type: {column_data.dtype}",
        f"Total rows: {total_rows}, Non-null: {non_null_count}",
    ]
    
    # Add null information if present
    if null_count > 0:
        profile_parts.append(f"Null values: {null_count} ({null_percentage:.1f}%)")
    
    profile_parts.append(f"Unique values: {unique_count} ({unique_percentage:.1f}%)")
    
    # Column role classification
    if unique_percentage > 95:
        profile_parts.append("Likely primary key or unique identifier")
    elif unique_percentage < 10:
        profile_parts.append("Low cardinality - likely categorical")
    
    # Type-specific analysis
    if pd.api.types.is_numeric_dtype(column_data):
        profile_parts.extend(_analyze_numeric_column(column_data, non_null_count))
    elif pd.api.types.is_string_dtype(column_data):
        profile_parts.extend(_analyze_string_column(column_data))
    elif pd.api.types.is_datetime64_any_dtype(column_data):
        profile_parts.extend(_analyze_datetime_column(column_data))
    elif pd.api.types.is_bool_dtype(column_data):
        profile_parts.extend(_analyze_boolean_column(column_data, non_null_count))
    
    # Most frequent values (for all types)
    if non_null_count > 0:
        top_values = column_data.value_counts().head(3)
        if len(top_values) > 0:
            top_values_str = ", ".join([f"'{val}' ({count}x)" for val, count in top_values.items()])
            profile_parts.append(f"Most frequent: {top_values_str}")
    
    # Create final profile description
    description = ". ".join(profile_parts) + "."
    
    # Ensure description fits within reasonable length limits
    return description[:800] + "..." if len(description) > 800 else description


def _analyze_numeric_column(column_data: pd.Series, non_null_count: int) -> list:
    """Analyze numeric columns for statistical insights."""
    desc_stats = column_data.describe()
    analysis = [
        f"Numeric range: {desc_stats['min']:.2f} to {desc_stats['max']:.2f}",
        f"Mean: {desc_stats['mean']:.2f}, Median: {desc_stats['50%']:.2f}",
        f"Standard deviation: {desc_stats['std']:.2f}",
    ]
    
    # Outlier detection using IQR method
    q1, q3 = desc_stats['25%'], desc_stats['75%']
    iqr = q3 - q1
    outliers = column_data[(column_data < q1 - 1.5 * iqr) | (column_data > q3 + 1.5 * iqr)]
    
    if len(outliers) > 0:
        outlier_percentage = (len(outliers) / non_null_count * 100)
        analysis.append(f"Potential outliers: {len(outliers)} ({outlier_percentage:.1f}%)")
    
    return analysis


def _analyze_string_column(column_data: pd.Series) -> list:
    """Analyze string columns for length and pattern insights."""
    non_null_strings = column_data.dropna()
    if len(non_null_strings) == 0:
        return ["No non-null string values"]
    
    # Length statistics
    lengths = non_null_strings.str.len()
    analysis = [f"String length: avg {lengths.mean():.1f}, range {lengths.min()}-{lengths.max()}"]
    
    # Pattern detection
    if non_null_strings.str.contains('@', na=False).any():
        email_count = non_null_strings.str.contains('@', na=False).sum()
        analysis.append(f"Contains {email_count} email-like values")
    
    if non_null_strings.str.match(r'^\d+$', na=False).any():
        numeric_str_count = non_null_strings.str.match(r'^\d+$', na=False).sum()
        analysis.append(f"Contains {numeric_str_count} numeric strings")
    
    if non_null_strings.str.contains(r'\d{4}-\d{2}-\d{2}', na=False).any():
        date_like_count = non_null_strings.str.contains(r'\d{4}-\d{2}-\d{2}', na=False).sum()
        analysis.append(f"Contains {date_like_count} date-like values")
    
    return analysis


def _analyze_datetime_column(column_data: pd.Series) -> list:
    """Analyze datetime columns for temporal insights."""
    non_null_dates = column_data.dropna()
    if len(non_null_dates) == 0:
        return ["No non-null datetime values"]
    
    min_date, max_date = non_null_dates.min(), non_null_dates.max()
    date_range = max_date - min_date
    
    return [
        f"Date range: {min_date.date()} to {max_date.date()}",
        f"Span: {date_range.days} days",
    ]


def _analyze_boolean_column(column_data: pd.Series, non_null_count: int) -> list:
    """Analyze boolean columns for distribution insights."""
    if non_null_count == 0:
        return ["No non-null boolean values"]
    
    true_count = column_data.sum()
    false_count = non_null_count - true_count
    true_percentage = (true_count / non_null_count * 100)
    
    return [f"Boolean distribution: {true_count} true ({true_percentage:.1f}%), {false_count} false"]


# Train Vanna on database question/query pairs from file
def train_file():
    try:
        vanna_service = VannaService.from_streamlit_session()

        # Load training queries from JSON
        training_file = Path(__file__).parent / "config" / "training_data.json"
        with open(training_file, "r") as f:
            training_data = json.load(f)

        # Extract the sample queries
        sample_queries = training_data.get("sample_queries", [])

        # Iterate over the sample queries and send the question and sql to vn.train()
        for query in sample_queries:
            question = query.get("question")
            query = query.get("query")
            if question and query:
                vanna_service.train(question=question, sql=query)

        # Extract the sample documents
        sample_documents = training_data.get("sample_documents", [])

        # Iterate over the sample documents and send the documentation to vn.train()
        for doc in sample_documents:
            documentation = doc.get("documentation")
            if documentation:
                vanna_service.train(documentation=documentation)
    except Exception as e:
        st.error(f"Error training from file: {e}")
        logger.exception("%s", e)


def is_table_token(token):
    try:
        # Check if the token is an Identifier and not a Keyword or DML
        if isinstance(token, Identifier):
            for sub_token in token.tokens:
                if sub_token.ttype in (Keyword, DML):
                    return False
            return True
        return False
    except Exception as e:
        st.error(f"Error training from file: {e}")
        logger.exception("%s", e)


def get_identifiers(parsed):
    try:
        tables = []
        columns = []
        is_table_context = False
        for token in parsed.tokens:
            if token.ttype in Keyword and token.value.upper() in ("FROM", "JOIN", "INTO", "UPDATE"):
                is_table_context = True
            elif token.ttype in Keyword and token.value.upper() in ("SELECT", "WHERE", "GROUP BY", "ORDER BY"):
                is_table_context = False

            if isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    if is_table_context and is_table_token(identifier):
                        tables.append(identifier.get_real_name())
                    elif not is_table_context and isinstance(identifier, Identifier):
                        columns.append(identifier.get_real_name())
            elif isinstance(token, Identifier):
                if is_table_context and is_table_token(token):
                    tables.append(token.get_real_name())
                elif not is_table_context:
                    columns.append(token.get_real_name())
    except Exception as e:
        st.error(f"Error getting identifiers: {e}")
        logger.exception("%s", e)
    else:
        return tables, columns
