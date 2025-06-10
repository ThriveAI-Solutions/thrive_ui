from utils.vanna_calls import run_sql_cached
import plotly.express as px
from utils.enums import MessageType, RoleType
from orm.models import Message
import time
import difflib
from wordcloud import WordCloud
from utils.vanna_calls import read_forbidden_from_json
from utils.chat_bot_helper import add_message

def get_all_table_names():
    try:
        forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()

        # Example for PostgreSQL; adjust for your DB
        sql = f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND  table_name NOT IN ({forbidden_tables_str});"
        df = run_sql_cached(sql)
        if df.empty:
            raise Exception("No tables found in the database.")
        
        return df['table_name'].tolist()
    except Exception as e:
        raise

def find_closest_table_name(table_name):
    try:
        table_names = get_all_table_names()
        if table_names is None:    
            raise Exception("No table names found in the database.")
        
        matches = difflib.get_close_matches(table_name, table_names, n=1, cutoff=0.6)

        if not matches:
            raise Exception(f"Could not find table similar to '{table_name}'")
        
        return matches[0]
    except Exception as e:
        raise


def find_closest_column_name(table_name, column_name):
    try:
        forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()
        forbidden_columns_str = ", ".join(f"'{column}'" for column in forbidden_columns)
        # Query all column names for the given table
        sql = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND column_name NOT IN ({forbidden_columns_str})
            AND table_name = '{table_name}';
        """
        df = run_sql_cached(sql)
        column_names = df['column_name'].tolist() if not df.empty else []
        matches = difflib.get_close_matches(column_name, column_names, n=1, cutoff=0.6)

        if not matches:
            raise Exception(f"Could not find column similar to  {column_name} on '{table_name}'")
        
        return matches[0]
    except Exception as e:
        raise

def is_magic_do_magic(question):
    try:
        for key, meta in MAGIC_RENDERERS.items():
            if key in question:
                end_of_question = question.partition(key)[2].strip()
                response = meta["func"](question, end_of_question)
                return True
        return False
    except Exception as e:
        add_message(
            Message(RoleType.ASSISTANT, f"Error processing magic command: {str(e)}", MessageType.ERROR)
        )
        return False


def _help(question, end_of_question):
    try:
        help_lines = ["MAGIC COMMANDS", "=" * 50, "", "Usage: /<command> [arguments]", "", "Available commands:", ""]
        # Find the longest usage string for alignment
        max_usage_len = max(len(meta["usage"]) for meta in MAGIC_RENDERERS.values()) if MAGIC_RENDERERS else 0

        for key, meta in MAGIC_RENDERERS.items():
            # Format: "  /command <args>    Description here"
            help_lines.append(f"  {meta["usage"]:<{max_usage_len + 2}} {meta["description"]}")

        help_lines.append("")
        help_lines.append("Examples:")

        for key, meta in MAGIC_RENDERERS.items():
            help_lines.append(f"  {meta["example"]:<{max_usage_len + 2}}")

        add_message(Message(RoleType.ASSISTANT, "\n".join(help_lines), MessageType.PYTHON, None, question, None, 0))
    except Exception as e:
        add_message(
            Message(RoleType.ASSISTANT, f"Error generating help message: {str(e)}", MessageType.ERROR)
        )


def _table_heatmap(question, end_of_question):
    try:
        start_time = time.perf_counter()
        forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()
        table_name = find_closest_table_name(end_of_question)
        # sql = f"SELECT * FROM {table_name} TABLESAMPLE BERNOULLI(50);"
        sql = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1000;"
        df = run_sql_cached(sql)
        if( df is None or df.empty):
            add_message(Message(RoleType.ASSISTANT, f"No data found for table '{table_name}'", MessageType.ERROR))
            return

        # Compute the correlation matrix
        corr = df.corr(numeric_only=True)
        # Create a heatmap of the correlation matrix
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title=f"Correlation Heatmap for {table_name}"
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, None, elapsed_time))
    except Exception as e:
        add_message(
            Message(RoleType.ASSISTANT, f"Error generating heatmap: {str(e)}", MessageType.ERROR)
        )

def _table_wordcloud(question, end_of_question):
    try:
        start_time = time.perf_counter()
        forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()
        words = end_of_question.split()
        table_name = find_closest_table_name(words[0])
        column_name = find_closest_column_name(table_name, words[1])
        # sql = f"SELECT * FROM {table_name} TABLESAMPLE BERNOULLI(50);"
        print(f"table_name: {table_name}, column_name: {column_name}")
        sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL;"
        df = run_sql_cached(sql)
        if( df is None or df.empty):
            add_message(Message(RoleType.ASSISTANT, f"No data found for table '{table_name}'", MessageType.ERROR))
            return

        if column_name not in df.columns:
            add_message(Message(RoleType.ASSISTANT, f"Column '{column_name}' not found in table '{table_name}'", MessageType.ERROR))
            return

        # Combine all text from the column
        text_data = df[column_name].astype(str).str.cat(sep=" ")

        if not text_data or text_data.strip() == "":
            add_message(Message(RoleType.ASSISTANT, f"No text data found in column '{column_name}' of table '{table_name}'", MessageType.ERROR))
            return

        # Generate wordcloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="viridis",
            max_words=100,
            relative_scaling=0.5,
            random_state=42,
        ).generate(text_data)

        # Convert wordcloud to image array and create plotly figure
        wordcloud_array = wordcloud.to_array()

        # Use plotly express imshow to display the wordcloud
        fig = px.imshow(wordcloud_array, title=f"Word Cloud for {table_name}.{column_name}")

        # Hide axes and ticks for clean appearance
        fig.update_layout(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            width=1200,
            height=600,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, None, elapsed_time))
    except Exception as e:
        add_message(
            Message(RoleType.ASSISTANT, f"Error generating word cloud: {str(e)}", MessageType.ERROR)
        )

#TODO: switch these to regular expressions to make them more robust
MAGIC_RENDERERS = {
    # "generate heatmap for ": {
    #     "func": _table_heatmap,
    #     "description": "Generate a correlation heatmap visualization for a table.",
    #     "usage": "generate heatmap for <table_name>",
    #     "example": "generate heatmap for wny_health"
    # },
    '/heatmap ': {
        "func": _table_heatmap,
        "description": "Generate a correlation heatmap visualization for a table.",
        "usage": "/heatmap <table_name>",
        "example": "/heatmap wny_health"
    },
    # "generate wordcloud for ": {
    #     "func": _table_wordcloud,
    #     "description": "Generate a wordcloud visualization for a table column",
    #     "usage": "generate wordcloud for <table_name> <column_name>",
    #     "example": "generate wordcloud for customers last_name"
    # },
    "/wordcloud ": {
        "func": _table_wordcloud,
        "description": "Generate a wordcloud visualization for a table column.",
        "usage": "/wordcloud <table_name> <column_name>",
        "example": "/wordcloud customers last_name"
    },
    "/help": {
        "func": _help,
        "description": "Show available magic commands",
        "usage": "/help",
        "example": "/help"
    },
    # Add more as needed...
}