import json
import logging
import time
from pathlib import Path

import psycopg2
import sqlparse
import streamlit as st
from pandas import DataFrame
from psycopg2.extras import RealDictCursor
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import DML, Keyword
from vanna.anthropic import Anthropic_Chat
from vanna.chromadb import ChromaDB_VectorStore
from vanna.ollama import Ollama
from vanna.remote import VannaDefault
from vanna.vannadb import VannaDB_VectorStore

logger = logging.getLogger(__name__)


class RetryCapableMixin:
    """
    A mixin that adds SQL retry capability to any Vanna class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_question = None
        self.last_error = None
        self.original_prompt = None

    def retry_sql_generation(self, error_message, max_retries=1):
        """
        Retry SQL generation with feedback about the error.

        Args:
            error_message (str): The error message from SQL execution
            max_retries (int): Maximum retry attempts

        Returns:
            str: A new SQL query that hopefully fixes the error
        """
        logger.debug("RETRY SYSTEM ACTIVATED: Beginning retry_sql_generation")

        # Store the error message
        self.last_error = error_message
        logger.debug(f"RETRY SYSTEM: Stored error message: {error_message[:100]}...")

        if not hasattr(self, "last_question") or not self.last_question:
            logger.warning("Cannot retry SQL generation: no previous question stored")
            return None

        retries = 0
        last_sql = None

        logger.debug(f"RETRY SYSTEM: Question exists: '{self.last_question}'")
        logger.debug(
            f"RETRY SYSTEM: Original prompt exists: {hasattr(self, 'original_prompt') and self.original_prompt is not None}"
        )

        while retries < max_retries:
            # Increment retries first
            retries += 1
            logger.debug(f"RETRY SYSTEM: Attempt {retries}/{max_retries}")

            # Create retry prompt with error information
            retry_message = f"Looks like there was an error when trying to execute SQL: {error_message}. Please review the original question and construct a SQL query that will answer the users question. Here is the original prompt, which includes the users question, for reference: {self.last_question}"

            # If we have the original prompt stored, use it for better context
            if hasattr(self, "original_prompt") and self.original_prompt:
                logger.debug("RETRY SYSTEM: Using original prompt context")
                retry_prompt = [
                    self.system_message(f"You are a {self.dialect} expert. Please correct the SQL based on the error."),
                    self.user_message(
                        f"The following was the original prompt that generated the SQL:\n\n{self.original_prompt}\n\nHowever, there was an error when executing the generated SQL: {error_message}\n\nPlease provide a corrected SQL query that will work and answer the users question: {self.last_question}"
                    ),
                ]
                logger.debug("RETRY SYSTEM: New prompt: %s", retry_prompt)
            else:
                # Fallback if original prompt wasn't stored
                logger.debug("RETRY SYSTEM: Using fallback prompt (no original prompt)")
                retry_prompt = [
                    self.system_message(f"You are a {self.dialect} expert. Please correct the SQL based on the error."),
                    self.user_message(retry_message),
                ]
                logger.debug("RETRY SYSTEM: New prompt: %s", retry_prompt)
            if last_sql:
                logger.debug("RETRY SYSTEM: Including previous failed attempt in prompt")
                retry_prompt.append(self.assistant_message(f"My previous attempt was: {last_sql}"))
                logger.debug("RETRY SYSTEM: New prompt: %s", retry_prompt)

            self.log(title="Retry SQL Prompt", message=retry_prompt)
            logger.debug("RETRY SYSTEM: Submitting prompt to LLM")
            llm_response = self.submit_prompt(retry_prompt)
            self.log(title="Retry LLM Response", message=llm_response)

            # Extract new SQL
            new_sql = self.extract_sql(llm_response)
            logger.debug(f"RETRY SYSTEM: Generated new SQL: {new_sql[:100]}...")
            last_sql = new_sql

            # Return the new SQL for the caller to execute - don't continue retrying here
            # The caller will retry if this SQL still fails
            logger.debug("RETRY SYSTEM: Returning corrected SQL")
            return new_sql

        # This line will never be reached with current implementation
        # but keeping it as a fallback
        logger.warning("RETRY SYSTEM: Max retries reached without successful generation")
        return last_sql

    # Override generate_sql to store the question
    def generate_sql(self, question: str, *args, **kwargs):
        logger.debug(f"RETRY SETUP: Storing question for potential retry: '{question}'")
        self.last_question = question

        # Call the parent's method but capture the intermediates
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None

        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)

        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )

        # Store the original prompt for potential retries
        self.original_prompt = str(prompt)
        logger.debug("RETRY SETUP: Original prompt stored successfully")

        # Continue with the parent's implementation
        logger.debug("RETRY SETUP: Calling parent generate_sql implementation")
        return super().generate_sql(question, *args, **kwargs)


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


class MyVannaAnthropicChromaDB(ChromaDB_VectorStore, Anthropic_Chat):
    def __init__(self, config=None):
        try:
            logger.info("Using Anthropic and chromaDB")
            ChromaDB_VectorStore.__init__(self, config={"path": st.secrets["rag_model"]["chroma_path"]})
            Anthropic_Chat.__init__(
                self,
                config={
                    "api_key": st.secrets["ai_keys"]["anthropic_api"],
                    "model": st.secrets["ai_keys"]["anthropic_model"],
                },
            )
        except Exception as e:
            logger.exception("Error Configuring MyVannaAnthropicChromaDB: %s", e)


class MyVannaOllama(RetryCapableMixin, VannaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        try:
            logger.info("Using Ollama and VannaDB")
            # Ensure RetryCapableMixin is initialized first
            RetryCapableMixin.__init__(self)
            VannaDB_VectorStore.__init__(
                self,
                vanna_model=st.secrets["ai_keys"]["vanna_model"],
                vanna_api_key=st.secrets["ai_keys"]["vanna_api"],
                config=config,
            )
            Ollama.__init__(self, config={"model": st.secrets["ai_keys"]["ollama_model"]})
        except Exception as e:
            logger.exception("Error Configuring MyVannaOllama: %s", e)

    # override the log function to stop vannas annoying print statements
    def log(self, message: str, title: str = "Info"):
        logger.debug("%s: %s", title, message)


class MyVannaOllamaChromaDB(RetryCapableMixin, ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        try:
            logger.info("Using Ollama and ChromaDB")
            # Ensure RetryCapableMixin is initialized first
            RetryCapableMixin.__init__(self, config={"path": st.secrets["rag_model"]["chroma_path"]})
            ChromaDB_VectorStore.__init__(self, config={"path": st.secrets["rag_model"]["chroma_path"]})
            Ollama.__init__(self, config={"model": st.secrets["ai_keys"]["ollama_model"]})
        except Exception as e:
            logger.exception("Error Configuring MyVannaOllamaChromaDB: %s", e)

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
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls._initialize_instance()
        return cls._instance

    @classmethod
    @st.cache_resource(ttl=3600)
    def _initialize_instance(cls):
        """Initialize the VannaService instance with appropriate configuration."""
        instance = cls()
        instance._setup_vanna()
        return instance

    def __init__(self):
        """Private constructor for singleton pattern."""
        self.vn = None

    def _setup_vanna(self):
        """Setup Vanna with appropriate configuration."""
        try:
            if "ollama_host" in st.secrets.ai_keys and "ollama_model" in st.secrets.ai_keys:
                if "chroma_path" in st.secrets.rag_model:
                    self.vn = MyVannaOllamaChromaDB()
                elif "vanna_api" in st.secrets.ai_keys and "vanna_model" in st.secrets.ai_keys:
                    self.vn = MyVannaOllama()
                else:
                    raise ValueError("Missing ollama Configuration Values")
            elif "anthropic_api" in st.secrets.ai_keys and "anthropic_model" in st.secrets.ai_keys:
                if "chroma_path" in st.secrets.rag_model:
                    self.vn = MyVannaAnthropicChromaDB()
                elif "vanna_api" in st.secrets.ai_keys and "vanna_model" in st.secrets.ai_keys:
                    self.vn = MyVannaAnthropic()
                else:
                    raise ValueError("Missing anthropic Configuration Values")
            else:
                logger.info("Using Default")
                self.vn = VannaDefault(
                    api_key=st.secrets["ai_keys"]["vanna_api"],
                    model=st.secrets["ai_keys"]["vanna_model"],
                )

            self.vn.connect_to_postgres(
                host=st.secrets["postgres"]["host"],
                dbname=st.secrets["postgres"]["database"],
                user=st.secrets["postgres"]["user"],
                password=st.secrets["postgres"]["password"],
                port=st.secrets["postgres"]["port"],
            )
        except Exception as e:
            st.error(f"Error setting up Vanna: {e}")
            logger.exception("%s", e)
            raise

    @st.cache_data(show_spinner="Generating sample questions ...")
    def generate_questions(_self):
        """Generate sample questions using Vanna."""
        try:
            questions = _self.vn.generate_questions()
        except Exception as e:
            st.error(f"Error generating questions: {e}")
            logger.exception("%s", e)
            return None
        else:
            return questions

    @st.cache_data(show_spinner="Generating SQL query ...")
    def generate_sql(_self, question: str):
        """Generate SQL from a natural language question."""
        try:
            start_time = time.perf_counter()

            if "allow_llm_to_see_data" in st.secrets.security and bool(st.secrets.security["allow_llm_to_see_data"]):
                logging.info("Allowing LLM to see data")
                response = check_references(_self.vn.generate_sql(question=question, allow_llm_to_see_data=True))
            else:
                logging.info("NOT allowing LLM to see data")
                response = check_references(_self.vn.generate_sql(question=question))

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
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

    @st.cache_data(show_spinner="Running SQL query ...", ttl=60)
    def run_sql(_self, sql: str, bypass_cache: bool = False) -> DataFrame | None:
        """
        Run SQL query and return results as DataFrame.

        Args:
            sql (str): The SQL query to execute
            bypass_cache (bool): If True, bypasses Streamlit caching
        """
        # If bypass_cache is True, add a random suffix to make the cache miss
        # This avoids cached errors during retry operations
        if bypass_cache:
            pass
            # # Add a random comment to force a cache miss
            # if "/*" not in sql and "*/" not in sql:  # Ensure we're not breaking any existing comments
            #     # For PostgreSQL, add a simple trailing comment with timestamp to ensure uniqueness
            #     random_id = f"{time.time()}"
            #     sql = f"{sql} /* bypass_cache_{random_id} */"
            #     logger.info(f"CACHE BYPASS: Added unique comment to SQL query with ID {random_id}")

        # Log the SQL but don't catch exceptions - let them propagate up to the
        # retry loop in the chat_bot.py file
        logger.info(f"EXECUTING SQL: {sql}...")
        df = _self.vn.run_sql(sql=sql)
        logger.info("SQL execution successful")
        return df

    def supports_retry(self) -> bool:
        """
        Check if the current Vanna implementation supports SQL retries.

        Returns:
            bool: True if retries are supported, False otherwise
        """
        return hasattr(self.vn, "retry_sql_generation")

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
            plotly = _self.vn.get_plotly_figure(plotly_code=code, df=df)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        except Exception as e:
            st.error(f"Error generating Plotly chart: {e}")
            logger.exception("%s", e)
            return None, 0
        else:
            return plotly, elapsed_time

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
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            logger.exception("Error generating summary: %s", e)
            return None, 0.0
        else:
            return response, elapsed_time

    def remove_from_training(self, entry_id):
        """Remove a training entry by ID."""
        try:
            self.vn.remove_training_data(entry_id)
        except Exception as e:
            st.error(f"Error removing training data: {e}")
            logger.exception("%s", e)
            return False
        else:
            return True

    def get_training_data(self):
        """Get all training data."""
        try:
            return self.vn.get_training_data()
        except Exception as e:
            st.error(f"Error getting training data: {e}")
            logger.exception("%s", e)
            return DataFrame()

    def train(self, question=None, sql=None, documentation=None, ddl=None, plan=None):
        """Train Vanna with various types of data."""
        try:
            if question and sql:
                self.vn.train(question=question, sql=sql)
            elif documentation:
                self.vn.train(documentation=documentation)
            elif ddl:
                self.vn.train(ddl=ddl)
            elif plan:
                self.vn.train(plan=plan)
        except Exception as e:
            st.error(f"Error training Vanna: {e}")
            logger.exception("%s", e)
            return False
        else:
            return True

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


# Backward compatibility functions that use the VannaService singleton
def setup_vanna():
    """
    Legacy function for backward compatibility.
    Returns the Vanna instance from the VannaService singleton.
    """
    return VannaService.get_instance().vn


@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    return VannaService.get_instance().generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    return VannaService.get_instance().generate_sql(question)


@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    return VannaService.get_instance().is_sql_valid(sql)


@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str) -> DataFrame:
    return VannaService.get_instance().run_sql(sql)


@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    return VannaService.get_instance().should_generate_chart(question, sql, df)


@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    return VannaService.get_instance().generate_plotly_code(question, sql, df)


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    return VannaService.get_instance().generate_plot(code, df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    return VannaService.get_instance().generate_followup_questions(question, sql, df)


@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question: str, df: DataFrame) -> tuple[str | None, float]:
    return VannaService.get_instance().generate_summary(question, df)


def remove_from_file_training(new_entry: dict):
    try:
        vanna_service = VannaService.get_instance()
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
        vanna_service = VannaService.get_instance()
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
    vanna_service = VannaService.get_instance()
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
def train_ddl():
    try:
        vanna_service = VannaService.get_instance()
        forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()

        # PostgreSQL Connection
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            port=st.secrets["postgres"]["port"],
            database=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
            cursor_factory=RealDictCursor,
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
        for row in schema_info:
            if current_table != row["table_name"]:
                if current_table is not None:
                    ddl.append(");")
                    # Train vanna with schema and queries
                    vanna_service.train(ddl=" ".join(ddl))
                    ddl = []  # reset ddl for next table
                current_table = row["table_name"]
                ddl.append(f"\nCREATE TABLE {row['table_name']} (")
            else:
                ddl.append(",")

            nullable = "NULL" if row["is_nullable"] == "YES" else "NOT NULL"
            ddl.append(f"\n    {row['column_name']} {row['data_type']} {nullable}")

        if ddl:  # Close the last table
            ddl.append(");")
            # Train vanna with schema and queries
            vanna_service.train(ddl=" ".join(ddl))
            ddl = []  # reset ddl for next table

        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error training DDL: {e}")
        logger.exception("%s", e)


# Train Vanna on database question/query pairs from file
def train_file():
    try:
        vanna_service = VannaService.get_instance()

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


def check_references(sql):
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
