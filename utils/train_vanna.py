import streamlit as st
import psycopg2
import json
from psycopg2.extras import RealDictCursor
from pathlib import Path
from utils.vanna_calls import (
    setup_vanna
)
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
