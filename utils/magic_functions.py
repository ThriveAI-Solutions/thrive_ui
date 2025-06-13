import difflib
import re
from PIL import Image
import numpy as np
import time

import plotly.express as px
from wordcloud import WordCloud

from orm.models import Message
from utils.chat_bot_helper import add_message
from utils.enums import MessageType, RoleType
from utils.vanna_calls import read_forbidden_from_json, run_sql_cached


def generate_example_from_pattern(pattern, sample_values=None):
    """
    Generate an example string from a regex pattern with named groups.
    sample_values: dict of {group_name: sample_value}
    """
    if sample_values is None:
        # Default sample values for common group names
        sample_values = {"table": "my_table", "column": "my_column"}
    # Find all named groups in the pattern
    group_names = re.findall(r"\?P<(\w+)>", pattern)
    example = pattern
    # Replace regex syntax with sample values
    for group in group_names:
        value = sample_values.get(group, f"<{group}>")
        # Replace the named group with the sample value
        example = re.sub(rf"\(\?P<{group}>[^\)]+\)", value, example)
    # Remove regex anchors and escapes
    example = example.replace("^", "").replace("$", "").replace(r"\s+", " ")
    example = re.sub(r"\\", "", example)
    return example.strip()


def usage_from_pattern(pattern):
    """
    Generate a usage string from a regex pattern with named groups.
    Example: r"^/wordcloud\s+(?P<table>\w+)\s+(?P<column>\w+)$"
    -> "/wordcloud <table> <column>"
    """
    # Remove regex anchors and escapes for clarity
    usage = pattern
    usage = usage.replace("^", "").replace("$", "")
    usage = re.sub(r"\s\+", " ", usage)
    # Replace named groups with <group_name>
    usage = re.sub(r"\(\?P<(\w+)>[^\)]+\)", r"<\1>", usage)
    # Remove any remaining regex tokens
    usage = re.sub(r"\\", "", usage)
    usage = re.sub(r"\s+", " ", usage)
    return usage.strip()


def get_all_table_names():
    try:
        forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()

        # Example for PostgreSQL; adjust for your DB
        sql = f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND  table_name NOT IN ({forbidden_tables_str});"
        df = run_sql_cached(sql)
        if df.empty:
            raise Exception("No tables found in the database.")

        return df["table_name"].tolist()
    except Exception:
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
    except Exception:
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
        column_names = df["column_name"].tolist() if not df.empty else []
        matches = difflib.get_close_matches(column_name, column_names, n=1, cutoff=0.6)

        if not matches:
            raise Exception(f"Could not find column similar to  {column_name} on '{table_name}'")

        return matches[0]
    except Exception:
        raise


def is_magic_do_magic(question):
    try:
        for key, meta in MAGIC_RENDERERS.items():
            match = re.match(key, question.strip())
            if match:
                add_message(Message(RoleType.ASSISTANT, "Sounds like magic!", MessageType.TEXT))
                meta["func"](question, match.groupdict())
                return True
        return False
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error processing magic command: {str(e)}", MessageType.ERROR))
        return False


def _help(question, tuple):
    try:
        help_lines = ["MAGIC COMMANDS", "=" * 50, "", "Usage: /<command> [arguments]", "", "Available commands:", ""]
        # Find the longest usage string for alignment
        usages = [(usage_from_pattern(pattern), meta["description"]) for pattern, meta in MAGIC_RENDERERS.items()]

        max_usage_len = max(len(usage) for usage, _ in usages) if usages else 0

        for usage, description in usages:
            # Format: "  /command <args>    Description here"
            help_lines.append(f"  {usage:<{max_usage_len + 2}} {description}")

        help_lines.append("")
        help_lines.append("Examples:")

        for key, meta in MAGIC_RENDERERS.items():
            example_text = generate_example_from_pattern(key, meta["sample_values"])
            help_lines.append(f"  {example_text:<{max_usage_len + 2}}")

        add_message(Message(RoleType.ASSISTANT, "\n".join(help_lines), MessageType.PYTHON, None, question, None, 0))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating help message: {str(e)}", MessageType.ERROR))


def _generate_heatmap(question, tuple):
    try:
        start_time = time.perf_counter()
        table_name = find_closest_table_name(tuple["table"])
        # sql = f"SELECT * FROM {table_name} TABLESAMPLE BERNOULLI(50);"
        sql = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1000;"
        df = run_sql_cached(sql)
        if df is None or df.empty:
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
            title=f"Correlation Heatmap for {table_name}",
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, None, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating heatmap: {str(e)}", MessageType.ERROR))


def _generate_wordcloud_column(question, tuple):
    try:
        start_time = time.perf_counter()
        table_name = find_closest_table_name(tuple["table"])
        column_name = find_closest_column_name(table_name, tuple["column"])
        sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL;"
        
        fig = get_wordcloud(sql, table_name, column_name)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, None, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating word cloud: {str(e)}", MessageType.ERROR))


def _generate_wordcloud(question, tuple):
    try:
        start_time = time.perf_counter()
        table_name = find_closest_table_name(tuple["table"])
        # sql = f"SELECT * FROM {table_name} TABLESAMPLE BERNOULLI(50);"
        sql = f"SELECT * FROM {table_name};"
        
        fig = get_wordcloud(sql, table_name)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, None, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating word cloud: {str(e)}", MessageType.ERROR))


def get_wordcloud(sql, table_name, column_name = None):
    df = run_sql_cached(sql)
    if( df is None or df.empty):
        add_message(Message(RoleType.ASSISTANT, f"No data found for table '{table_name}'", MessageType.ERROR))
        return

    # Combine all text from the column, filtering out unwanted words
    unwanted_words = {"y", "n", "none", "unknown", "yes", "no"}
    text_data = ""

    if column_name != None:
        if column_name not in df.columns:
            add_message(Message(RoleType.ASSISTANT, f"Column '{column_name}' not found in table '{table_name}'", MessageType.ERROR))
            return
        
        text_data = df[column_name].astype(str).str.cat(sep=" ")
    else:
        string_columns = df.select_dtypes(include="object").columns
        words = []
        for col in string_columns:
            words += [w for w in df[col].astype(str).str.cat(sep=" ").split()]
        text_data = " ".join(words)

    if not text_data or text_data.strip() == "":
        add_message(Message(RoleType.ASSISTANT, f"No text data found in table '{table_name}'", MessageType.ERROR))
        return
    
    # Load the brain mask image
    mask_path = "assets/heart.png"
    img_mask = np.array(Image.open(mask_path))

    # Generate wordcloud
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=100,
        # stopwords=stopwords,
        relative_scaling=0.5,
        random_state=42,
        mask=img_mask
    ).generate(text_data)

    # Convert wordcloud to image array and create plotly figure
    wordcloud_array = wordcloud.to_array()

    # Use plotly express imshow to display the wordcloud
    fig = px.imshow(wordcloud_array, title=f"Word Cloud for {table_name}")

    # Hide axes and ticks for clean appearance
    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        width=1200,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    return fig
    

MAGIC_RENDERERS = {
    r"^/heatmap\s+(?P<table>\w+)$": {
        "func": _generate_heatmap,
        "description": "Generate a correlation heatmap visualization for a table.",
        "sample_values": {"table": "wny_health"},
    },
    # r"^generate heatmap for (?P<table>\w+)$": {
    #     "func": _generate_heatmap,
    #     "description": "Generate a correlation heatmap visualization for a table.",
    #     "sample_values": {
    #         "table": "wny_health"
    #     }
    # },
    r"^/wordcloud\s+(?P<table>\w+)$": {
        "func": _generate_wordcloud,
        "description": "Generate a wordcloud visualization for a table.",
        "sample_values": {
            "table": "wny_health",
        },
    },
    r"^/wordcloud\s+(?P<table>\w+)\.(?P<column>\w+)$": {
        "func": _generate_wordcloud_column,
        "description": "Generate a wordcloud visualization for a table column.",
        "sample_values": {"table": "wny_health", "column": "county"},
    },
    # r"^generate wordcloud for (?P<table>\w+)\s+(?P<column>\w+)$": {
    #     "func": _generate_wordcloud,
    #     "description": "Generate a wordcloud visualization for a table column",
    #     "sample_values": {
    #         "table": "wny_health",
    #         "column": "county"
    #     }
    # },
    r"^/help$": {"func": _help, "description": "Show available magic commands", "sample_values": {}},
    # Add more as needed...
}
