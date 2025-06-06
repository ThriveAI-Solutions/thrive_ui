from typing import Any
from utils.vanna_calls import run_sql_cached
import plotly.express as px
from utils.enums import MessageType, RoleType
from orm.models import Message
import time
import difflib
from utils.vanna_calls import read_forbidden_from_json

def get_all_table_names():
    forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()

    # Example for PostgreSQL; adjust for your DB
    sql = f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND  table_name NOT IN ({forbidden_tables_str});"
    df = run_sql_cached(sql)
    return df['table_name'].tolist()

def find_closest_table_name(input_name):
    table_names = get_all_table_names()
    matches = difflib.get_close_matches(input_name, table_names, n=1, cutoff=0.6)

    return matches[0] if matches else None

def is_magic_do_magic(question):
    for key, func in MAGIC_RENDERERS.items():
        if key in question:
            user_table_name = question.partition(key)[2].strip()
            actual_table_name = find_closest_table_name(user_table_name)
            response = func(question, actual_table_name)
            return response
    return False


def _table_heatmap(question, table_name):
    forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()

    start_time = time.perf_counter()
    # sql = f"SELECT * FROM {table_name} TABLESAMPLE BERNOULLI(50);"
    sql = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1000;"
    df = run_sql_cached(sql)

    # Compute the correlation matrix
    corr = df.corr(numeric_only=True)
    # Create a heatmap of the correlation matrix
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title=f"Correlation Heatmap for {table_name}"
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, None, elapsed_time)

# def _table_wordcloud(question, table_name):
#     forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()

#     start_time = time.perf_counter()
#     # sql = f"SELECT * FROM {table_name} TABLESAMPLE BERNOULLI(50);"
#     sql = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1000;"
#     df = run_sql_cached(sql)

#     # TODO: Generate a word cloud from the text data in the DataFrame
    
#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time

#     return Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, None, elapsed_time)


MAGIC_RENDERERS = {
    "generate heatmap for ": _table_heatmap,
    "generate a heatmap for ": _table_heatmap,
    # "generate wordcloud for ": _table_wordcloud,
    #mind map
}