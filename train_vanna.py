import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor

from vanna.remote import VannaDefault
from vanna_calls import (
    setup_vanna
)
#TODO: Convert this to a ChromaDB implementation?

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
    
    # Sample SQL query for training
    sample_sql_query = "select count(*) from public.patients; select * from medical_records;"

    vn.train('\n'.join(ddl), sample_sql_query)
    cursor.close()