# import streamlit as st
# from langsec import SQLSecurityGuard
# from langsec.schema.defaults import low_security_config
# from langsec.schema.security_schema import (SecuritySchema, TableSchema)
# import psycopg2
# from psycopg2.extras import RealDictCursor

# # https://docs.lang-sec.com/quick-start/

# def get_security_guard():
#     # PostgreSQL Connection
#     conn = psycopg2.connect(
#         host=st.secrets["postgres"]["host"],
#         port=st.secrets["postgres"]["port"],
#         database=st.secrets["postgres"]["database"],
#         user=st.secrets["postgres"]["user"],
#         password=st.secrets["postgres"]["password"],
#         cursor_factory=RealDictCursor
#     )

#     # Get database schema
#     cursor = conn.cursor()
#     cursor.execute("""
#         SELECT tablename 
#         FROM pg_tables 
#         WHERE schemaname = 'public' 
#         AND tablename NOT IN ('thrive_user', 'thrive_message', 'user_role');
#     """)
#     schema_info = cursor.fetchall()

#     tables = {}
#     for table in schema_info:
#         tablename = table['tablename']
#         tables[tablename] = TableSchema()

#     # print(tables)

#     schema = SecuritySchema(
#         tables=tables,
#         allow_temp_tables=False,        # Enable/disable temporary tables
#         max_joins=5,                    # Maximum number of joins allowed
#         allow_subqueries=True,          # Enable/disable subqueries
#         max_query_length=5000,          # Maximum query length
#         sql_injection_protection=True,   # Enable basic SQL injection protection
#         forbidden_keywords={            # SQL keywords to block
#             "DROP", "DELETE", "TRUNCATE",
#             "ALTER", "GRANT", "REVOKE",
#             "EXECUTE", "EXEC",
#             "SYSADMIN", "DBADMIN"
#         },
#         # Default schemas for tables/columns not explicitly defined
#         # default_table_security_schema=TableSchema(...),
#         # default_column_security_schema=ColumnSchema(...)
#     )
#     return SQLSecurityGuard(schema = schema)

# def validate(sql):
#     # return sql
#     try:
#         # Validate the SQL query using the guard
#         guard.validate_query(sql)
#         return sql
#     except Exception as e:
#         # Handle validation errors
#         return f"Query validation failed: {e}"

# # guard = get_security_guard()
# guard = SQLSecurityGuard(schema = low_security_config)