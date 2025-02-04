import streamlit as st
from vanna.remote import VannaDefault
from utils.langsec import validate
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path

@st.cache_resource(ttl=3600)
def setup_vanna():
    vn = VannaDefault(api_key=st.secrets["ai_keys"]["vanna_api"], model=st.secrets["ai_keys"]["vanna_model"])
    vn.connect_to_postgres(
        host=st.secrets["postgres"]["host"],
        dbname=st.secrets["postgres"]["database"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"], 
        port=st.secrets["postgres"]["port"]
    )
    return vn

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_vanna()
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    vn = setup_vanna()
    # Dont send data to LLM
    # return vn.generate_sql(question=question, allow_llm_to_see_data=True)
    return validate(vn.generate_sql(question=question))

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    vn = setup_vanna()
    return vn.is_sql_valid(sql=sql)

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    vn = setup_vanna()
    return vn.run_sql(sql=sql)

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    vn = setup_vanna()
    return vn.should_generate_chart(df=df)

@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    vn = setup_vanna()
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    vn = setup_vanna()
    return vn.get_plotly_figure(plotly_code=code, df=df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    vn = setup_vanna()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    vn = setup_vanna()
    return vn.generate_summary(question=question, df=df)

#TODO: Convert this to a ChromaDB implementation?

def write_to_file(new_entry: dict):
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

        print('New entry added to training_data.json')
    else:
        print('Duplicate entry found. No new entry added.')

# Train Vanna on database schema
@st.cache_resource
def train():
    vn = setup_vanna()

    # PostgreSQL Connection
    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        database=st.secrets["postgres"]["database"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        cursor_factory=RealDictCursor
    )

    # Get database schema
    cursor = conn.cursor()
    cursor.execute("""
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
        ORDER BY 
            table_schema, table_name, ordinal_position;
    """)
    schema_info = cursor.fetchall()
    
    # Format schema for training
    ddl = []
    current_table = None
    for row in schema_info:
        if current_table != row['table_name']:
            if current_table is not None:
                ddl.append(');')
            current_table = row['table_name']
            ddl.append(f"\nCREATE TABLE {row['table_name']} (")
        else:
            ddl.append(',')
        
        nullable = "NULL" if row['is_nullable'] == 'YES' else "NOT NULL"
        ddl.append(f"\n    {row['column_name']} {row['data_type']} {nullable}")
    
    if ddl:  # Close the last table
        ddl.append(');')

    # Train vanna with schema and queries
    vn.train('\n'.join(ddl), "select * from {currentTable}")
    cursor.close()
    
    # Load training queries from JSON
    training_file = Path(__file__).parent / 'config' / 'training_data.json'
    with open(training_file, 'r') as f:
        training_data = json.load(f)
    
    # Extract the sample queries
    sample_queries = training_data.get("sample_queries", [])

    # Iterate over the sample queries and send the question and sql to vn.train()
    for query in sample_queries:
        question = query.get("question")
        sql = query.get("sql")
        if question and sql:
            vn.train(question, sql)
