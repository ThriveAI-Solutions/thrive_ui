import difflib
from io import StringIO
import re
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import time
import plotly.express as px
from wordcloud import WordCloud
from orm.models import Message
from utils.chat_bot_helper import add_message, vn
from utils.enums import MessageType, RoleType
from utils.vanna_calls import read_forbidden_from_json, run_sql_cached

unwanted_words = {"y", "n", "none", "unknown", "yes", "no"}


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
    usage = re.sub(r"\\s\+", " ", usage)
    # Replace named groups with <group_name>
    usage = re.sub(r"\(\?P<(\w+)>[^\)]+\)", r"<\1>", usage)
    # Remove any remaining regex tokens
    usage = re.sub(r"\\", "", usage)
    usage = re.sub(r"\s+", " ", usage)
    return usage.strip()


def get_all_column_names(table):
    try:
        table_name = find_closest_table_name(table)

        sql = f"SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND  table_name = '{table_name}';"
        df = run_sql_cached(sql)
        if df.empty:
            raise Exception("No tables found in the database.")

        return df
    except Exception:
        raise


def get_all_table_names():
    try:
        forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()

        sql = f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND  table_name NOT IN ({forbidden_tables_str});"
        df = run_sql_cached(sql)
        if df.empty:
            raise Exception("No tables found in the database.")

        return df
    except Exception:
        raise


def find_closest_table_name(table_name):
    try:
        df = get_all_table_names()
        table_names = df["table_name"].tolist()

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


def is_magic_do_magic(question, previous_df=None):
    try:
        if question is None or question.strip() == "":
            return False

        if previous_df is not None:
            for key, meta in FOLLOW_UP_MAGIC_RENDERERS.items():
                match = re.match(key, question.strip())
                if match:
                    meta["func"](question, match.groupdict(), previous_df)
                    return True
        else:
            for key, meta in MAGIC_RENDERERS.items():
                match = re.match(key, question.strip())
                if match:
                    add_message(Message(RoleType.ASSISTANT, "Sounds like magic!", MessageType.TEXT))
                    meta["func"](question, match.groupdict(), None)
                    return True
        return False
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error processing magic command: {str(e)}", MessageType.ERROR))
        return False


def _help(question, tuple, previous_df):
    try:
        help_lines = ["MAGIC COMMANDS", "=" * 50, "", "Usage: /<command> [arguments]", "", "Available commands:"]
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


def _tables(question, tuple, previous_df):
    try:
        start_time = time.perf_counter()
        table_names = get_all_table_names()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(
            Message(RoleType.ASSISTANT, table_names, MessageType.DATAFRAME, None, question, table_names, elapsed_time)
        )
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error retrieving tables: {str(e)}", MessageType.ERROR))


def _columns(question, tuple, previous_df):
    try:
        start_time = time.perf_counter()

        column_names = get_all_column_names(tuple["table"])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(
            Message(RoleType.ASSISTANT, column_names, MessageType.DATAFRAME, None, question, column_names, elapsed_time)
        )
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error retrieving columns: {str(e)}", MessageType.ERROR))


def _head(question, tuple, previous_df):
    try:
        start_time = time.perf_counter()
        sql = ""
        if previous_df is None:
            table_name = find_closest_table_name(tuple["table"])

            sql = f"SELECT *  FROM {table_name} LIMIT 5;"
            df = run_sql_cached(sql)
        else:
            df = previous_df.head(5)

        if df.empty:
            raise Exception("No rows found in the database.")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, df, MessageType.DATAFRAME, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error retrieving tables: {str(e)}", MessageType.ERROR))


def _followup(question, tuple, previous_df):
    try:
        start_time = time.perf_counter()

        last_assistant_msg = None

        last_assistant_msg = next(
            (
                msg
                for msg in reversed(st.session_state.messages)
                if msg.role == RoleType.ASSISTANT.value and msg.elapsed_time is not None
            ),
            None,
        )

        if last_assistant_msg is None:
            add_message(
                Message(RoleType.ASSISTANT, "No previous assistant message found to follow up on.", MessageType.ERROR)
            )
            return

        command = tuple["command"].strip()

        response = None
        df = None
        type = MessageType.TEXT
        if last_assistant_msg.dataframe is not None:
            df = pd.read_json(StringIO(last_assistant_msg.dataframe))
            was_magical = is_magic_do_magic(command, df)

            if was_magical:
                return
            # TODO: trying to hack a followup feature reduction thing in here.... not succeeding /followup can you do feature reduction on this heatmap?
            # elif "heatmap" in command.lower(): #is plotly command
            #     type = MessageType.PLOTLY_CHART
            #     response, elapsed = vn.generate_plotly_code(question=command, sql=last_assistant_msg.query, df=df)
            #     print(f"Response: {response}")

        if response is None:
            response = vn.submit_prompt(command, last_assistant_msg.content)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, response, type, None, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating follow up message: {str(e)}", MessageType.ERROR))


def _generate_heatmap(question, tuple, previous_df):
    try:
        start_time = time.perf_counter()
        table_name = ""

        sql = ""
        if previous_df is None:
            table_name = find_closest_table_name(tuple["table"])
            # sql = f"SELECT * FROM {table_name} TABLESAMPLE BERNOULLI(50);"
            sql = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1000;"
            df = run_sql_cached(sql)
        else:
            table_name = "Temp Table"
            df = previous_df

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

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating heatmap: {str(e)}", MessageType.ERROR))


def _generate_wordcloud_column(question, tuple, previous_df):
    try:
        start_time = time.perf_counter()
        table_name = "Temp Table"
        column_name = tuple["column"]
        fig = None
        sql = ""
        if previous_df is None:
            table_name = find_closest_table_name(tuple["table"])
            column_name = find_closest_column_name(table_name, column_name)
            sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL;"
            # check_length( f"SELECT count({column_name}) FROM {table_name} WHERE {column_name} IS NOT NULL;")
            fig, df = get_wordcloud(sql, table_name, column_name)
        else:
            fig, df = get_wordcloud(sql, table_name, column_name, previous_df)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating word cloud column: {str(e)}", MessageType.ERROR))


def _generate_wordcloud(question, tuple, previous_df):
    try:
        start_time = time.perf_counter()
        fig = None
        table_name = "Temp Table"
        sql = ""

        if previous_df is None:
            table_name = find_closest_table_name(tuple["table"])
            # sql = f"SELECT * FROM {table_name} TABLESAMPLE BERNOULLI(50);"
            sql = f"SELECT * FROM {table_name};"
            # check_length( f"SELECT count(*) FROM {table_name}")
            fig, df = get_wordcloud(sql, table_name)
        else:
            fig, df = get_wordcloud(sql, table_name, None, previous_df)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating word cloud: {str(e)}", MessageType.ERROR))


def get_wordcloud(sql, table_name, column_name=None, previous_df=None):
    if previous_df is not None:
        df = previous_df
    else:
        df = run_sql_cached(sql)
    if df is None or df.empty:
        add_message(Message(RoleType.ASSISTANT, f"No data found for table '{table_name}'", MessageType.ERROR))
        return
    # Combine all text from the column, filtering out unwanted words
    text_data = ""

    if column_name != None:
        if column_name not in df.columns:
            add_message(
                Message(
                    RoleType.ASSISTANT, f"Column '{column_name}' not found in table '{table_name}'", MessageType.ERROR
                )
            )
            return
        text_data = df[column_name].astype(str).str.cat(sep=" ")
    else:
        string_columns = df.select_dtypes(include="object").columns
        words = []
        for col in string_columns:
            words += [w for w in df[col].astype(str).str.cat(sep=" ").split() if w.lower() not in unwanted_words]
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
        mask=img_mask,
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

    return fig, df


def _generate_pairplot(question, tuple, previous_df):
    try:
        start_time = time.perf_counter()
        table_name = "Temp Table"
        column_name = tuple["column"]
        sql = ""

        if previous_df is None:
            table_name = find_closest_table_name(tuple["table"])
            column_name = find_closest_column_name(table_name, column_name)
            sql = f"SELECT * FROM {table_name};"

            df = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(Message(RoleType.ASSISTANT, f"No data found for table '{table_name}'", MessageType.ERROR))
            return

        fig = px.scatter_matrix(
            df,
            dimensions=df.select_dtypes(include=[np.number]).columns.tolist(),
            color=column_name,
            symbol=column_name,
            title=f"Pairplot for {table_name} - {column_name}",
            width=1200,
            height=1200,
        )

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating pair plot: {str(e)}", MessageType.ERROR))


def generate_plotly(type, question, tuple, previous_df):
    try:
        start_time = time.perf_counter()
        table_name = "Temp Table"
        column_x = tuple["x"]
        column_y = tuple["y"]
        column_color = tuple["color"]
        sql = ""

        if previous_df is None:
            table_name = find_closest_table_name(tuple["table"])
            column_x = find_closest_column_name(table_name, tuple["x"])
            column_y = find_closest_column_name(table_name, tuple["y"])
            column_color = find_closest_column_name(table_name, tuple["color"])
            sql = f"SELECT {column_x}, {column_y}, {column_color} FROM {table_name};"
            df = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(Message(RoleType.ASSISTANT, f"No data found for table '{table_name}'", MessageType.ERROR))
            return

        fig = getattr(px, type)(
            df,
            x=column_x,
            y=column_y,
            color=column_color,
            width=1200,
            height=600,
        )

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating plotly: {str(e)}", MessageType.ERROR))


def _generate_scatterplot(question, tuple, previous_df):
    generate_plotly("scatter", question, tuple, previous_df)


def _generate_bar(question, tuple, previous_df):
    generate_plotly("bar", question, tuple, previous_df)


def _generate_line(question, tuple, previous_df):
    generate_plotly("line", question, tuple, previous_df)


# def check_length(sql):
#     df = run_sql_cached(sql)
#     if df is not None and not df.empty:
#         number = float(df.iloc[0, 0])
#         if number > 1:
#             return confirm_length(number)
#     return True


# @st.dialog("Confirm Length")
# def confirm_length(number):
#     st.write(f"Query is going to fetch {number} rows, are you sure you want to proceed?")
#     if st.button("Yes"):
#         return True
#     elif st.button("No"):
#         return False
#     st.stop()


FOLLOW_UP_MAGIC_RENDERERS = {
    r"^heatmap$": {
        "func": _generate_heatmap,
    },
    r"^wordcloud$": {
        "func": _generate_wordcloud,
    },
    r"^wordcloud\s+(?P<column>\w+)$": {
        "func": _generate_wordcloud_column,
    },
    r"^pairplot\s+(?P<column>\w+)$": {
        "func": _generate_pairplot,
    },
    r"^scatter\s+(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_scatterplot,
    },
    r"^bar\s+(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_bar,
    },
    r"^line\s+(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_line,
    },
    r"^head$": {
        "func": _head,
    },
}

# TODO: confusion matrix?
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
    r"^/pairplot\s+(?P<table>\w+)\.(?P<column>\w+)$": {
        "func": _generate_pairplot,
        "description": "Generate a pairplot visualization for a table column.",
        "sample_values": {"table": "wny_health", "column": "county"},
    },
    r"^/scatter\s+(?P<table>\w+)\.(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_scatterplot,
        "description": "Generate a scatterplot visualization for a table x and y axis.",
        "sample_values": {"table": "wny_health", "x": "county", "y": "obesity", "color": "age"},
    },
    r"^/bar\s+(?P<table>\w+)\.(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_bar,
        "description": "Generate a bar chart visualization for a table x and y axis.",
        "sample_values": {"table": "wny_health", "x": "county", "y": "obesity", "color": "age"},
    },
    r"^/line\s+(?P<table>\w+)\.(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_line,
        "description": "Generate a line chart visualization for a table x and y axis.",
        "sample_values": {"table": "wny_health", "x": "county", "y": "obesity", "color": "age"},
    },
    r"^/followup\s+(?P<command>.+)$": {
        "func": _followup,
        "description": "Ask a follow up question to the previous result set.  Also accepts magic commands ie: heatmap/wordcloud.",
        "sample_values": {"command": "how do these results compare to the national averages?"},
    },
    r"^/tables$": {"func": _tables, "description": "Show all available tables", "sample_values": {}},
    r"^/columns\s+(?P<table>.+)$": {
        "func": _columns,
        "description": "Show all available columns on a given table",
        "sample_values": {"table": "wny_health"},
    },
    r"^/head\s+(?P<table>.+)$": {
        "func": _head,
        "description": "Show the first 5 rows of a given table",
        "sample_values": {"table": "wny_health"},
    },
    r"^/help$": {"func": _help, "description": "Show available magic commands", "sample_values": {}},
    # Add more as needed...
}
