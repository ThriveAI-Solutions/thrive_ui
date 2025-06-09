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
        
    def get_closest_table_from_ddl(self, table_name: str) -> str:
        """Get the closest table from the DDL."""
        try:
            closest_table = self.vn.get_closest_table_from_ddl(table_name=table_name)
            return closest_table
        except Exception as e:
            st.error(f"Error getting closest table from DDL: {e}")
            logger.exception("%s", e)
            return None

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
    vanna_service = VannaService.from_streamlit_session()
    forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()

    # The information schema query may need some tweaking depending on your database. This is a good starting point.
    df_information_schema = vanna_service.run_sql(f"""
            SELECT 
                *
            FROM 
                information_schema.columns
            WHERE 
                table_schema = 'public'
            AND 
                table_name NOT IN ({forbidden_tables_str})
            ORDER BY 
                table_schema, table_name, ordinal_position;
        """)

    # This will break up the information schema into bite-sized chunks that can be referenced by the LLM
    plan = vanna_service.get_training_plan_generic(df_information_schema)
    vanna_service.train(plan=plan)


# Train Vanna on database schema
def train_ddl(describe_ddl_from_llm: bool = False):
    try:
        st.toast("🚀 Starting DDL training...")
        forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()

        # PostgreSQL Connection
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            port=st.secrets["postgres"]["port"],
            database=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
        )

        # Get database schema
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT 
                table_schema,
                table_name,
                column_name,
                data_type,
                is_nullable
            FROM 
                information_schema.columns
            WHERE 
                table_schema = 'public'
            AND 
                table_name NOT IN ({forbidden_tables_str})
            ORDER BY 
                table_schema, table_name, ordinal_position;
        """)
        schema_info = cursor.fetchall()

        # Format schema for training
        ddl = []
        current_table = None
        tables_trained = 0

        vanna_service = VannaService.from_streamlit_session()

        for row in schema_info:
            if current_table != row[1]:
                if current_table is not None:
                    ddl.append(");")
                    vanna_service.train(ddl=" ".join(ddl))
                    st.toast(f"✓ Trained DDL for table: {current_table}")
                    if describe_ddl_from_llm:
                        train_ddl_describe_to_rag(current_table, ddl)
                    ddl = []  # reset ddl for next table
                    tables_trained += 1
                current_table = row[1]
                ddl.append(f"\nCREATE TABLE {row[1]} (")
            else:
                ddl.append(",")

            nullable = "NULL" if row[4] == "YES" else "NOT NULL"
            ddl.append(f"\n    {row[2]} {row[3]} {nullable}")

        if ddl:  # Close the last table
            ddl.append(");")
            vanna_service.train(ddl=" ".join(ddl))
            st.toast(f"✓ Trained DDL for table: {current_table}")
            if describe_ddl_from_llm:
                train_ddl_describe_to_rag(current_table, ddl)
            ddl = []  # reset ddl for next table
            tables_trained += 1

        cursor.close()
        conn.close()

        # Show final success message
        if tables_trained > 0:
            st.success(f"🎉 DDL Training completed successfully! Trained {tables_trained} table(s).")
        else:
            st.warning("No tables found to train DDL on.")

    except Exception as e:
        st.error(f"Error training DDL: {e}")
        logger.exception("%s", e)


def train_ddl_describe_to_rag(table: str, ddl: list):
    conn = None  # Initialize conn for the finally block
    try:
        # Establish a new connection for this function
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            port=st.secrets["postgres"]["port"],
            database=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
        )

        # Use a cursor to get column names to avoid issues if table is empty or doesn't exist
        with conn.cursor() as cur:
            # Use sqlparse.sql.Identifier for safe table name formatting in query
            # To correctly use Identifier for a plain string table name, wrap it in a TokenList
            # containing a single Token of type Name.
            # However, a simpler way for just quoting is to use psycopg2's quoting or f-string with careful quoting.
            # Given sqlparse is already a dependency, let's try to use it correctly or fall back.
            # For now, let's assume the direct use of Identifier on a string was problematic, and rely on psycopg2's parameterization or SQL object composition.
            # Psycopg2 itself doesn't directly substitute table/column names in `execute`.
            # `sqlparse.sql.Identifier` is for manipulating parsed SQL structures.
            # A common way to safely quote identifiers is using `psycopg2.sql.Identifier`
            try:
                from psycopg2 import sql as psycopg2_sql  # Import it conditionally or at top if always used

                safe_table_ident = psycopg2_sql.Identifier(table)
                query_string = psycopg2_sql.SQL("SELECT * FROM {} LIMIT 1;").format(safe_table_ident)
                cur.execute(query_string)
            except ImportError:
                # Fallback if psycopg2.sql is not available or for some other reason, though it should be with psycopg2
                # This might still be unsafe if table name contains quotes, but Identifier was also causing issues.
                # The original code was: cur.execute(f"SELECT * FROM {sqlparse.sql.Identifier(table)} LIMIT 1;")
                # The issue was `sqlparse.sql.Identifier(table)` when table is a string.
                # For a direct f-string, one would need to ensure `table` is validated and sanitized.
                # Let's revert to a simpler quoting for the purpose of the fix, assuming `table` is a simple name.
                # A better fix would be to use psycopg2.sql.Identifier consistently.
                cur.execute(f'SELECT * FROM "{table}" LIMIT 1;')  # Simple quoting, assumes table is valid

            if cur.description is None:
                logger.warning(f"Skipping DDL description for table {table} as it might be empty or not found.")
                return
            colnames = [desc[0] for desc in cur.description]

        vanna_service = VannaService.from_streamlit_session()

        for column in colnames:
            # Properly quote column name for the query
            # Similar issue here as with table name for sqlparse.sql.Identifier
            # Using psycopg2.sql.Identifier is preferred.
            try:
                from psycopg2 import sql as psycopg2_sql

                safe_table_ident = psycopg2_sql.Identifier(table)
                safe_column_ident = psycopg2_sql.Identifier(column)
                query_string = psycopg2_sql.SQL("SELECT DISTINCT {col} FROM {tab} ORDER BY {col} LIMIT 10;").format(
                    col=safe_column_ident, tab=safe_table_ident
                )
                data = pd.read_sql_query(query_string.as_string(conn), conn)  # as_string(conn) for psycopg2 >2.8
            except ImportError:
                # Fallback for pd.read_sql_query, which can also take a SQL string directly
                query = f'SELECT DISTINCT "{column}" FROM "{table}" ORDER BY "{column}" LIMIT 10;'
                data = pd.read_sql_query(query, conn)
            except psycopg2.Error as col_query_err:
                logger.warning(f"Could not query column {column} from table {table}. Skipping. Error: {col_query_err}")
                continue  # Skip this column if it can't be queried (e.g., complex type not directly usable in DISTINCT/ORDER BY)

            st.toast(f"Describing column: {table}.{column}")
            column_data = data[column].tolist()

            system_message = "You are a PostgreSQL expert tasked with describing a specific column from a table. Your goal is to provide a detailed analysis of the column based on the provided information. Follow these steps:"
            ddl_string = " ".join(ddl)  # Convert list of ddl parts to a string
            prompt = textwrap.dedent(f"""
                1. First, review the DDL (Data Definition Language) for the entire table:
                <ddl>
                {ddl_string}
                </ddl>
            
                2. Now, focus on the specific column you need to describe:
                Column name: {column}
            
                3. Examine the sample data for this column:
                <sample_data>
                {column_data}
                </sample_data>
            
                4. Analyze the column based on the DDL and sample data. Consider the following aspects:
                - Data type
                - Constraints (e.g., NOT NULL, UNIQUE, PRIMARY KEY)
                - Default values
                - Any patterns or characteristics observed in the sample data
                - Potential use or purpose of the column in the context of the table
            
                5. Provide your analysis in plain text. Your response should include:
                - Observations about the sample data
                - Any insights or inferences you can make about the column's role or importance in the table
                - Potential considerations for querying or working with this column
                - A brief description of the column's properties as defined in the DDL
            
                Remember to keep your response concise yet informative, focusing on the most relevant details for a PostgreSQL expert. Do not include any XML tags in your response.
            """)

             # The response should be a maximum of 1000 characters in length.

            description = vanna_service.submit_prompt(system_message=system_message, user_message=prompt)
            description = re.sub(r"[•●▪️–—\-•·►★✓✔✗✘➔➤➢➣➤➥➦➧➨➩➪➫➬➭➮➯➱➲➳➴➵➶➷➸➹➺➻➼➽➾]", "", description)
            logger.info(f"Column: {table}.{column}, Description: {description}")
            vanna_service.train(documentation=f"{table}.{column} {description}")

    except psycopg2.Error as db_err:
        st.error(f"Database error during DDL description for {table}: {db_err}")
        logger.exception(f"Database error during DDL description for {table}: %s", db_err)
    except Exception as e:
        st.error(f"Error training Table DDL to RAG for {table}: {e}")
        logger.exception(f"Error training Table DDL to RAG for {table}: %s", e)
    finally:
        if conn:
            conn.close()  # Ensure connection is closed


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
