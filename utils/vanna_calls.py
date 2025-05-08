import json
from pathlib import Path
import time
import psycopg2
import sqlparse
import streamlit as st
from psycopg2.extras import RealDictCursor
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import DML, Keyword
from vanna.anthropic import Anthropic_Chat
from vanna.chromadb import ChromaDB_VectorStore
from vanna.ollama import Ollama
from vanna.remote import VannaDefault
from vanna.vannadb import VannaDB_VectorStore

class MyVannaAnthropic(VannaDB_VectorStore, Anthropic_Chat):
    def __init__(self, config=None):
        try:
            print("Using Anthropic and VannaDB")
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
            print(f"Error Configuring MyVannaAnthropic: {e}")


class MyVannaAnthropicChromaDB(ChromaDB_VectorStore, Anthropic_Chat):
    def __init__(self, config=None):
        try:
            print("Using Anthropic and chromaDB")
            ChromaDB_VectorStore.__init__(self, config={"path": st.secrets["rag_model"]["chroma_path"]})
            Anthropic_Chat.__init__(
                self,
                config={
                    "api_key": st.secrets["ai_keys"]["anthropic_api"],
                    "model": st.secrets["ai_keys"]["anthropic_model"],
                },
            )
        except Exception as e:
            print(f"Error Configuring MyVannaAnthropicChromaDB: {e}")


class MyVannaOllama(VannaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        try:
            print("Using Ollama and VannaDB")
            VannaDB_VectorStore.__init__(
                self,
                vanna_model=st.secrets["ai_keys"]["vanna_model"],
                vanna_api_key=st.secrets["ai_keys"]["vanna_api"],
                config=config,
            )
            Ollama.__init__(self, config={"model": st.secrets["ai_keys"]["ollama_model"]})
        except Exception as e:
            print(f"Error Configuring MyVannaOllama: {e}")


class MyVannaOllamaChromaDB(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        try:
            print("Using Ollama and ChromaDB")
            ChromaDB_VectorStore.__init__(self, config={"path": st.secrets["rag_model"]["chroma_path"]})
            Ollama.__init__(self, config={"model": st.secrets["ai_keys"]["ollama_model"]})
        except Exception as e:
            print(f"Error Configuring MyVannaOllamaChromaDB: {e}")


def read_forbidden_from_json():
    try:
        # Path to the forbidden_references.json file
        forbidden_file_path = Path(__file__).parent / "config/forbidden_references.json"

        # Load the forbidden references
        with forbidden_file_path.open("r") as file:
            forbidden_data = json.load(file)

        forbidden_tables = forbidden_data.get("tables", [])
        forbidden_columns = forbidden_data.get("columns", [])
        return forbidden_tables, forbidden_columns
    except Exception as e:
        print(f"Error reading forbidden_references.json: {e}")
        return [], []


forbidden_tables, forbidden_columns = read_forbidden_from_json()
forbidden_tables_str = ", ".join(f"'{table}'" for table in forbidden_tables)


@st.cache_resource(ttl=3600)
def setup_vanna():
    try:
        if "ollama_host" in st.secrets.ai_keys and "ollama_model" in st.secrets.ai_keys:
            if "chroma_path" in st.secrets.rag_model:
                vn = MyVannaOllamaChromaDB()
            elif "vanna_api" in st.secrets.ai_keys and "vanna_model" in st.secrets.ai_keys:
                vn = MyVannaOllama()
            else:
                raise ValueError("Missing ollama Configuration Values")
        elif "anthropic_api" in st.secrets.ai_keys and "anthropic_model" in st.secrets.ai_keys:
            if "chroma_path" in st.secrets.rag_model:
                vn = MyVannaAnthropicChromaDB()
            elif "vanna_api" in st.secrets.ai_keys and "vanna_model" in st.secrets.ai_keys:
                vn = MyVannaAnthropic()
            else:
                raise ValueError("Missing anthropic Configuration Values")
        else:
            print("Using Default")
            vn = VannaDefault(api_key=st.secrets["ai_keys"]["vanna_api"], model=st.secrets["ai_keys"]["vanna_model"])

        vn.connect_to_postgres(
            host=st.secrets["postgres"]["host"],
            dbname=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
            port=st.secrets["postgres"]["port"],
        )
        return vn
    except Exception as e:
        st.error(f"Error setting up Vanna: {e}")
        print(e)


@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    try:
        vn = setup_vanna()
        return vn.generate_questions()
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        print(e)


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    try:
        start_time = time.time()
        vn = setup_vanna()

        if (
            "allow_llm_to_see_data" in st.secrets.security
            and bool(st.secrets.security["allow_llm_to_see_data"]) == True
        ):
            print("Allowing LLM to see data")
            response = check_references(vn.generate_sql(question=question, allow_llm_to_see_data=True))
            end_time = time.time()
            elapsed_time = end_time - start_time
            return response, elapsed_time
        else:
            print("NOT allowing LLM to see data")
            response = check_references(vn.generate_sql(question=question))
            end_time = time.time()
            elapsed_time = end_time - start_time
            return response, elapsed_time
    except Exception as e:
        st.error(f"Error generating SQL: {e}")
        print(e)


@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    try:
        vn = setup_vanna()
        return vn.is_sql_valid(sql=sql)
    except Exception as e:
        st.error(f"Error checking SQL validity: {e}")
        print(e)


@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    try:
        vn = setup_vanna()
        return vn.run_sql(sql=sql)
    except Exception as e:
        st.error(f"Error running SQL: {e}")
        print(e)


@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    try:
        vn = setup_vanna()
        return vn.should_generate_chart(df=df)
    except Exception as e:
        st.error(f"Error checking if we should generate a chart: {e}")
        print(e)


@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    try:
        start_time = time.time()
        vn = setup_vanna()
        code = vn.generate_plotly_code(question=question, sql=sql, df=df)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return code, elapsed_time
    except Exception as e:
        st.error(f"Error generating Plotly code: {e}")
        print(e)


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    try:
        start_time = time.time()
        vn = setup_vanna()
        plotly = vn.get_plotly_figure(plotly_code=code, df=df)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return plotly, elapsed_time
    except Exception as e:
        st.error(f"Error generating Plotly chart: {e}")
        print(e)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    try:
        vn = setup_vanna()
        return vn.generate_followup_questions(question=question, sql=sql, df=df)
    except Exception as e:
        st.error(f"Error generating followup questions: {e}")
        print(e)


@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    try:
        start_time = time.time()
        vn = setup_vanna()
        response = vn.generate_summary(question=question, df=df)
        end_time = time.time()

        elapsed_time = end_time - start_time
        return response, elapsed_time
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        print(e)


def remove_from_file_training(new_entry: dict):
    try:
        vn = setup_vanna()
        training_data = vn.get_training_data()
        for index, row in training_data.iterrows():
            if row["question"] and row["question"] == new_entry["question"]:
                vn.remove_training_data(row["id"])

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

            print("Entry removed from training_data.json")
        else:
            print("Entry not found, nothing removed from training_data.json")
    except Exception as e:
        st.error(f"Error removing entry from training_data.json: {e}")


def write_to_file_and_training(new_entry: dict):
    try:
        vn = setup_vanna()

        vn.train(question=new_entry["question"], sql=new_entry["query"])

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

            print("New entry added to training_data.json")
        else:
            print("Duplicate entry found. No new entry added.")
    except Exception as e:
        st.error(f"Error writing to training_data.json: {e}")
        print(e)


def training_plan():
    vn = setup_vanna()

    # The information schema query may need some tweaking depending on your database. This is a good starting point.
    df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

    # This will break up the information schema into bite-sized chunks that can be referenced by the LLM
    plan = vn.get_training_plan_generic(df_information_schema)
    plan

    # If you like the plan, then uncomment this and run it to train
    vn.train(plan=plan)


# Train Vanna on database schema
def train_ddl():
    try:
        vn = setup_vanna()

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
                    vn.train(ddl=" ".join(ddl))
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
            vn.train(ddl=" ".join(ddl))
            ddl = []  # reset ddl for next table

        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error training DDL: {e}")
        print(e)


# Train Vanna on database question/query pairs from file
def train_file():
    try:
        vn = setup_vanna()

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
                vn.train(question=question, sql=query)

        # Extract the sample documents
        sample_documents = training_data.get("sample_documents", [])

        # Iterate over the sample documents and send the documentation to vn.train()
        for doc in sample_documents:
            documentation = doc.get("documentation")
            if documentation:
                vn.train(documentation=documentation)
    except Exception as e:
        st.error(f"Error training from file: {e}")
        print(e)


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
        print(e)


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
        return tables, columns
    except Exception as e:
        st.error(f"Error getting identifiers: {e}")
        print(e)


def check_references(sql):
    try:
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
        return sql
    except Exception as e:
        st.error(f"Error checking references: {e}")
        print(e)
