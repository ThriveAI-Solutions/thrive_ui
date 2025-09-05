import difflib
from io import StringIO
import random
import re
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from orm.models import Message, SessionLocal
from utils.chat_bot_helper import add_message, set_question, get_vn, normal_message_flow, add_acknowledgement
from utils.enums import MessageType, RoleType
from utils.vanna_calls import (
    read_forbidden_from_json,
    run_sql_cached,
    get_configured_schema,
    get_configured_object_type,
)

unwanted_words = {"y", "n", "none", "unknown", "yes", "no"}


def get_object_name_singular():
    """Get the singular object name (table or view) based on configuration."""
    object_type = get_configured_object_type()
    return "table" if object_type == "tables" else "view"


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
    """
    Get all column names for a given table or view.

    Uses the configured schema and object type to query the appropriate
    information_schema view for column information.

    Args:
        table (str): Name of the table or view

    Returns:
        DataFrame: DataFrame with column_name column
    """
    try:
        schema_qualified_name = find_closest_object_name(table)
        schema_name = get_configured_schema()
        object_type = get_configured_object_type()
        object_name = "table" if object_type == "tables" else "view"

        # Extract just the table name from schema.table_name format
        table_name = schema_qualified_name.split(".")[-1]

        sql = f"SELECT column_name FROM information_schema.columns WHERE table_schema = '{schema_name}' AND table_name = '{table_name}' order by column_name;"
        df, elapsed_time = run_sql_cached(sql)
        if df.empty:
            raise Exception(f"No columns found for {object_name} '{schema_qualified_name}' in schema '{schema_name}'.")

        return df
    except Exception:
        raise


def get_all_object_names():
    """
    Get all table or view names from the configured schema.

    Uses the object_type configuration from secrets.toml to determine
    whether to query tables or views from information_schema.

    Returns:
        DataFrame: DataFrame with table_name column containing object names
    """
    try:
        forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()
        schema_name = get_configured_schema()
        object_type = get_configured_object_type()

        # Build query based on configured object type (tables or views)
        if forbidden_tables_str:
            sql = f"SELECT table_name FROM information_schema.{object_type} WHERE table_schema = '{schema_name}' AND table_name NOT IN ({forbidden_tables_str}) order by table_name;"
        else:
            sql = f"SELECT table_name FROM information_schema.{object_type} WHERE table_schema = '{schema_name}' AND table_name NOT IN ({forbidden_tables_str}) order by table_name;"

        df, elapsed_time = run_sql_cached(sql)
        if df.empty:
            object_name = "table" if object_type == "tables" else "view"
            raise Exception(f"No {object_name}s found in the database.")

        return df
    except Exception:
        raise


def find_closest_object_name(object_name):
    """
    Find the closest matching table or view name using fuzzy string matching.

    Uses the object_type configuration to search within the appropriate
    database objects (tables or views).

    Args:
        object_name (str): Name to search for

    Returns:
        str: Best matching object name in schema.table_name format
    """
    try:
        df = get_all_object_names()
        table_names = df["table_name"].tolist()

        if table_names is None:
            object_type = get_configured_object_type()
            object_name_plural = "table" if object_type == "tables" else "view"
            raise Exception(f"No {object_name_plural} names found in the database.")

        matches = difflib.get_close_matches(object_name, table_names, n=1, cutoff=0.6)

        if not matches:
            object_type = get_configured_object_type()
            object_name_singular = "table" if object_type == "tables" else "view"
            raise Exception(f"Could not find {object_name_singular} similar to '{object_name}'")

        # Return schema-qualified object name
        schema_name = get_configured_schema()
        return f"{schema_name}.{matches[0]}"
    except Exception:
        raise


def find_closest_column_name(table_name, column_name):
    """
    Find the closest matching column name in a table or view using fuzzy string matching.

    Uses the configured schema and respects forbidden column restrictions.

    Args:
        table_name (str): Name of the table or view
        column_name (str): Column name to search for

    Returns:
        str: Best matching column name
    """
    try:
        forbidden_tables, forbidden_columns, forbidden_tables_str = read_forbidden_from_json()
        forbidden_columns_str = ", ".join(f"'{column}'" for column in forbidden_columns)
        schema_name = get_configured_schema()
        object_type = get_configured_object_type()
        object_name = "table" if object_type == "tables" else "view"

        # Extract just the table name from schema.table_name format if needed
        unqualified_table_name = table_name.split(".")[-1]

        # Query all column names for the given table/view
        if forbidden_columns_str:
            sql = f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = '{schema_name}'
                AND column_name NOT IN ({forbidden_columns_str})
                AND table_name = '{unqualified_table_name}';
            """
        else:
            sql = f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = '{schema_name}'
                AND table_name = '{unqualified_table_name}';
            """

        df, elapsed_time = run_sql_cached(sql)
        column_names = df["column_name"].tolist() if not df.empty else []
        matches = difflib.get_close_matches(column_name, column_names, n=1, cutoff=0.6)

        if not matches:
            raise Exception(f"Could not find column similar to '{column_name}' on {object_name} '{table_name}'")

        return matches[0]
    except Exception:
        raise


def find_closest_column_name_from_list(column_names, column_name):
    """
    Find the closest matching column name from a list of column names using fuzzy string matching.

    Args:
        column_names (list): List of available column names
        column_name (str): Column name to search for

    Returns:
        str: Best matching column name
    """
    try:
        matches = difflib.get_close_matches(column_name, column_names, n=1, cutoff=0.6)

        if not matches:
            raise Exception(f"Could not find column similar to '{column_name}' in the provided columns")

        return matches[0]
    except Exception:
        raise


# def find_closest_magic_command(question, command_dict):
#     """
#     Find the closest matching magic command using fuzzy string matching and natural language inference.

#     Args:
#         question (str): User's input question/command
#         command_dict (dict): Dictionary of magic commands (FOLLOW_UP_MAGIC_RENDERERS or MAGIC_RENDERERS)

#     Returns:
#         tuple: (pattern, metadata, match_groups) or (None, None, None) if no match found
#     """
#     try:
#         question = question.strip()

#         # First try exact regex matching (existing behavior)
#         for pattern, meta in command_dict.items():
#             match = re.match(pattern, question)
#             if match:
#                 return pattern, meta, match.groupdict()

#         # If no exact match, try fuzzy matching on command keywords
#         command_keywords = {}

#         # Extract command keywords from regex patterns
#         for pattern, meta in command_dict.items():
#             # Extract the main command word from regex pattern
#             clean_pattern = pattern.replace("^", "").replace("$", "").replace("\\s+", " ")
#             clean_pattern = re.sub(r"\(\?P<\w+>[^)]+\)", "", clean_pattern)  # Remove named groups
#             clean_pattern = re.sub(r"[\\(){}[\]|*+?.]", "", clean_pattern)  # Remove regex chars
#             clean_pattern = clean_pattern.strip()

#             # Get the first word as the command keyword
#             if clean_pattern:
#                 command_word = clean_pattern.split()[0] if clean_pattern.split() else clean_pattern
#                 if command_word:
#                     command_keywords[command_word] = (pattern, meta)

#         # Create list of command words for fuzzy matching
#         available_commands = list(command_keywords.keys())

#         # Clean the user question for matching
#         user_words = question.lower().split()

#         # Try to find fuzzy matches for each word in the user's question
#         for word in user_words:
#             matches = difflib.get_close_matches(word, available_commands, n=1, cutoff=0.6)
#             if matches:
#                 matched_command = matches[0]
#                 pattern, meta = command_keywords[matched_command]

#                 # Try to extract parameters from the original question
#                 # This is a simplified approach - for more complex commands,
#                 # we might need more sophisticated parameter extraction
#                 remaining_words = [w for w in user_words if w != word]

#                 # Create a mock match object with potential parameters
#                 mock_groups = {}

#                 # Try to infer common parameters
#                 if remaining_words:
#                     # Look for column names in the remaining words
#                     if 'column' in pattern:
#                         mock_groups['column'] = remaining_words[0] if remaining_words else ''
#                     if 'column1' in pattern and len(remaining_words) > 0:
#                         mock_groups['column1'] = remaining_words[0]
#                     if 'column2' in pattern and len(remaining_words) > 1:
#                         mock_groups['column2'] = remaining_words[1]
#                     if 'table' in pattern:
#                         mock_groups['table'] = remaining_words[0] if remaining_words else ''
#                     if 'command' in pattern:
#                         mock_groups['command'] = ' '.join(remaining_words) if remaining_words else ''
#                     if 'percentage' in pattern:
#                         # Look for numbers in remaining words
#                         for w in remaining_words:
#                             if w.isdigit():
#                                 mock_groups['percentage'] = w
#                                 break
#                     if 'operation' in pattern:
#                         mock_groups['operation'] = remaining_words[-1] if remaining_words else 'log'
#                     if 'x' in pattern and 'y' in pattern:  # For scatter plots, etc.
#                         if len(remaining_words) >= 2:
#                             mock_groups['x'] = remaining_words[0]
#                             mock_groups['y'] = remaining_words[1]
#                             mock_groups['color'] = remaining_words[2] if len(remaining_words) > 2 else ''

#                 return pattern, meta, mock_groups

#         # If no fuzzy match found, try natural language inference
#         question_lower = question.lower()

#         # Common natural language mappings
#         nlp_mappings = {
#             'show': ['head', 'describe', 'profile'],
#             'display': ['head', 'describe', 'profile'],
#             'analyze': ['describe', 'profile', 'distribution'],
#             'visualize': ['boxplot', 'heatmap', 'wordcloud'],
#             'plot': ['boxplot', 'heatmap', 'wordcloud'],
#             'chart': ['boxplot', 'heatmap', 'wordcloud'],
#             'statistics': ['describe', 'distribution'],
#             'stats': ['describe', 'distribution'],
#             'missing': ['missing'],
#             'null': ['missing'],
#             'duplicate': ['duplicates'],
#             'correlation': ['heatmap', 'correlation'],
#             'relationship': ['heatmap', 'correlation'],
#             'distribution': ['distribution', 'boxplot'],
#             'outlier': ['outliers'],
#             'anomaly': ['anomaly'],
#             'cluster': ['clusters'],
#             'group': ['clusters'],
#             'pca': ['pca'],
#             'principal': ['pca'],
#             'component': ['pca'],
#             'report': ['report'],
#             'summary': ['summary'],
#             'wordcloud': ['wordcloud'],
#             'cloud': ['wordcloud'],
#             'text': ['wordcloud'],
#             'help': ['help'],
#             'clear': ['clear'],
#             'tables': ['tables'],
#             'columns': ['columns'],
#             'sample': ['sample'],
#             'transform': ['transform'],
#             'violin': ['violin'],
#             'pair': ['pairplot'],
#             'scatter': ['scatter'],
#             'bar': ['bar'],
#             'line': ['line']
#         }

#         # Try to match natural language to commands
#         for nlp_word, possible_commands in nlp_mappings.items():
#             if nlp_word in question_lower:
#                 for cmd in possible_commands:
#                     if cmd in command_keywords:
#                         pattern, meta = command_keywords[cmd]

#                         # Extract parameters from the question
#                         mock_groups = {}
#                         words = question.split()

#                         # Simple parameter extraction
#                         if 'column' in pattern:
#                             # Look for words that might be column names
#                             for word in words:
#                                 if word.isalpha() and word.lower() not in ['show', 'display', 'analyze', 'the', 'of', 'for', 'in', 'on']:
#                                     mock_groups['column'] = word
#                                     break

#                         if 'table' in pattern:
#                             for word in words:
#                                 if word.isalpha() and word.lower() not in ['show', 'display', 'analyze', 'the', 'of', 'for', 'in', 'on']:
#                                     mock_groups['table'] = word
#                                     break

#                         return pattern, meta, mock_groups

#         return None, None, None

#     except Exception as e:
#         return None, None, None


# def is_magic_do_magic(question, previous_df=None):
#     try:
#         if question is None or question.strip() == "":
#             return False

#         if previous_df is not None:
#             # Try fuzzy matching for follow-up commands
#             pattern, meta, match_groups = find_closest_magic_command(question, FOLLOW_UP_MAGIC_RENDERERS)
#             if pattern and meta:
#                 meta["func"](question, match_groups, previous_df)
#                 return True
#         else:
#             # Try fuzzy matching for regular magic commands
#             pattern, meta, match_groups = find_closest_magic_command(question, MAGIC_RENDERERS)
#             if pattern and meta:
#                 add_message(Message(RoleType.ASSISTANT, "Sounds like magic!", MessageType.TEXT))
#                 meta["func"](question, match_groups, None)
#                 return True
#         return False
#     except Exception as e:
#         add_message(Message(RoleType.ASSISTANT, f"Error processing magic command: {str(e)}", MessageType.ERROR))
#         return False


def is_magic_do_magic(question, previous_df=None):
    try:
        if question is None or question.strip() == "":
            return False

        if previous_df is not None:
            for key, meta in FOLLOW_UP_MAGIC_RENDERERS.items():
                match = re.match(key, question.strip())
                if match:
                    add_message(Message(RoleType.ASSISTANT, "Sounds like followup magic!", MessageType.TEXT))
                    meta["func"](question, match.groupdict(), previous_df)
                    return True
        else:
            for key, meta in MAGIC_RENDERERS.items():
                match = re.match(key, question.strip())
                if match:
                    if meta["func"] != _followup and meta["func"] != _history_search:
                        add_message(Message(RoleType.ASSISTANT, "Sounds like magic!", MessageType.TEXT))
                    meta["func"](question, match.groupdict(), None)
                    return True
        return False
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error processing magic command: {str(e)}", MessageType.ERROR))
        return False


def _clear(question, tuple, previous_df):
    """
    Clear the message history in the chat window.
    """
    try:
        set_question(None)
        st.rerun()
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error clearing messages: {str(e)}", MessageType.ERROR))
        return False


def _help(question, tuple, previous_df):
    try:
        help_lines = ["MAGIC COMMANDS", "=" * 50, "", "Usage: /<command> [arguments]", ""]

        # Group commands by category
        commands_by_category = {}
        for pattern, meta in MAGIC_RENDERERS.items():
            category = meta.get("category", "Other")
            if category not in commands_by_category:
                commands_by_category[category] = []
            commands_by_category[category].append((pattern, meta))

        # Define category order
        category_order = [
            "Help & System Commands",
            "Database Exploration",
            "Data Exploration & Basic Info",
            "Data Quality & Preprocessing",
            "Statistical Analysis",
            "Visualizations",
            "Machine Learning",
            "Comprehensive Reporting",
        ]

        # Find the longest usage string for alignment
        all_usages = [(usage_from_pattern(pattern), meta["description"]) for pattern, meta in MAGIC_RENDERERS.items()]
        max_usage_len = max(len(usage) for usage, _ in all_usages) if all_usages else 0

        # Display commands by category
        for category in category_order:
            if category in commands_by_category:
                help_lines.append(f"📂 {category.upper()}")
                help_lines.append("-" * (len(category) + 4))

                for pattern, meta in commands_by_category[category]:
                    usage = usage_from_pattern(pattern)
                    description = meta["description"]
                    help_lines.append(f"  {usage:<{max_usage_len + 2}} {description}")

                help_lines.append("")

        # Add any remaining categories not in the order
        for category, commands in commands_by_category.items():
            if category not in category_order:
                help_lines.append(f"📂 {category.upper()}")
                help_lines.append("-" * (len(category) + 4))

                for pattern, meta in commands:
                    usage = usage_from_pattern(pattern)
                    description = meta["description"]
                    help_lines.append(f"  {usage:<{max_usage_len + 2}} {description}")

                help_lines.append("")

        help_lines.append("💡 EXAMPLES")
        help_lines.append("-" * 12)

        # Show examples for each category
        for category in category_order:
            if category in commands_by_category:
                # Find the first command in this category that has show_example: True
                for pattern, meta in commands_by_category[category]:
                    if meta.get("show_example", False):
                        example_text = generate_example_from_pattern(pattern, meta["sample_values"])
                        help_lines.append(f"  {example_text}")

        help_lines.append("")
        help_lines.append("For follow-up commands after running a query, use: /followuphelp")

        add_message(Message(RoleType.ASSISTANT, "\n".join(help_lines), MessageType.PYTHON, None, question, None, 0))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating help message: {str(e)}", MessageType.ERROR))


def _followup_help(question, tuple, previous_df):
    """Show available follow-up commands that can be used after running a query."""
    try:
        help_lines = ["FOLLOW-UP COMMANDS", "=" * 50, "", "Usage: /followup <command> [arguments]", ""]

        # Group follow-up commands by category
        commands_by_category = {}
        for pattern, meta in FOLLOW_UP_MAGIC_RENDERERS.items():
            category = meta.get("category", "Other")
            if category not in commands_by_category:
                commands_by_category[category] = []

            # Convert pattern to readable usage
            usage = pattern.replace("^", "").replace("$", "")
            usage = re.sub(r"\\s\+", " ", usage)
            usage = re.sub(r"\(\?P<(\w+)>[^\)]+\)", r"<\1>", usage)
            usage = re.sub(r"\\", "", usage)
            usage = re.sub(r"\s+", " ", usage)
            usage = usage.strip()

            # Get description from the main MAGIC_RENDERERS if available
            description = "Analyze data from previous result"
            for main_pattern, main_meta in MAGIC_RENDERERS.items():
                if main_meta["func"] == meta["func"] and "description" in main_meta:
                    description = main_meta["description"]
                    break

            commands_by_category[category].append((usage, description))

        # Define category order (same as main help)
        category_order = [
            "Data Exploration & Basic Info",
            "Data Quality & Preprocessing",
            "Statistical Analysis",
            "Visualizations",
            "Machine Learning",
            "Reporting",
        ]

        # Find the longest usage string for alignment
        all_usages = []
        for category_commands in commands_by_category.values():
            for usage, _ in category_commands:
                all_usages.append(usage)
        max_usage_len = max(len(usage) for usage in all_usages) if all_usages else 0

        # Display commands by category
        for category in category_order:
            if category in commands_by_category:
                help_lines.append(f"📂 {category.upper()}")
                help_lines.append("-" * (len(category) + 4))

                for usage, description in commands_by_category[category]:
                    help_lines.append(f"  {usage:<{max_usage_len + 2}} {description}")

                help_lines.append("")

        # Add any remaining categories not in the order
        for category, commands in commands_by_category.items():
            if category not in category_order:
                help_lines.append(f"📂 {category.upper()}")
                help_lines.append("-" * (len(category) + 4))

                for usage, description in commands:
                    help_lines.append(f"  {usage:<{max_usage_len + 2}} {description}")

                help_lines.append("")

        help_lines.append("ℹ️  Note: Follow-up commands work on the data from your previous query result.")

        add_message(Message(RoleType.ASSISTANT, "\n".join(help_lines), MessageType.PYTHON, None, question, None, 0))
    except Exception as e:
        add_message(
            Message(RoleType.ASSISTANT, f"Error generating follow-up help message: {str(e)}", MessageType.ERROR)
        )


def _history_search(question, match_dict, previous_df):
    """Find the most recent thumbs-up summary and recreate all messages from that group in chronological order."""
    try:    
        # Get the search sentence from the question
        search_text = match_dict.get("search_text", "").strip().lower()
        
        if not search_text:
            add_message(Message(RoleType.ASSISTANT, "Please provide a search term after /h", MessageType.ERROR))
            return
        
        # Get current user ID from session
        user_id_str = st.session_state.cookies.get("user_id")
        if not user_id_str:
            add_message(Message(RoleType.ASSISTANT, "User not authenticated", MessageType.ERROR))
            return
        
        try:
            user_id = int(user_id_str)
        except (ValueError, TypeError):
            add_message(Message(RoleType.ASSISTANT, "Invalid user ID", MessageType.ERROR))
            return
        
        # Query the database for thumbs-up summaries and find best match using fuzzy matching
        with SessionLocal() as session:
            from orm.models import Message as DBMessage
            
            # Get all thumbs-up summaries from this user
            all_thumbs_up_summaries = session.query(DBMessage).filter(
                DBMessage.user_id == user_id,
                DBMessage.role == 'assistant',
                DBMessage.type == 'summary',
                DBMessage.feedback == 'up',
                DBMessage.group_id.isnot(None)  # Must have a group_id
            ).order_by(DBMessage.created_at.desc()).all()
            
            if not all_thumbs_up_summaries:
                # add_message(Message(
                #     RoleType.ASSISTANT, 
                #     "No thumbs-up summaries found in your history", 
                #     MessageType.TEXT
                # ))
                normal_message_flow(search_text)
                return
            
            # Use fuzzy matching to find the best matching summary (90% threshold)
            matched_summaries = []
            for summary in all_thumbs_up_summaries:
                # Check similarity against the question (if available)
                if summary.question:
                    question_lower = summary.question.lower()
                    question_similarity = difflib.SequenceMatcher(None, search_text, question_lower).ratio()
                    if question_similarity >= 0.9:
                        matched_summaries.append((summary, question_similarity))
                        
                # Check similarity against the summary content
                if summary.content:
                    content_lower = summary.content.lower()
                    content_similarity = difflib.SequenceMatcher(None, search_text, content_lower).ratio()
                    if content_similarity >= 0.9:
                        matched_summaries.append((summary, content_similarity))
                
                # Check similarity against the SQL query (if available)
                if summary.query:
                    query_lower = summary.query.lower()
                    query_similarity = difflib.SequenceMatcher(None, search_text, query_lower).ratio()
                    if query_similarity >= 0.9:
                        matched_summaries.append((summary, query_similarity))
            
            if not matched_summaries:
                # add_message(Message(
                #     RoleType.ASSISTANT, 
                #     f"No thumbs-up summaries found matching '{search_text}' with ≥90% similarity", 
                #     MessageType.TEXT
                # ))
                normal_message_flow(search_text)
                return
            
            # Sort by similarity score (highest first) and get the best match
            matched_summaries.sort(key=lambda x: x[1], reverse=True)
            latest_summary, similarity_score = matched_summaries[0]
            
            # Get all messages from the same group_id, ordered by creation date ascending
            group_messages = session.query(DBMessage).filter(
                DBMessage.group_id == latest_summary.group_id,
                DBMessage.role == 'assistant'
            ).order_by(DBMessage.created_at.asc()).all()
            
            if not group_messages:
                # add_message(Message(
                #     RoleType.ASSISTANT, 
                #     "No messages found for the latest thumbs-up group", 
                #     MessageType.TEXT
                # ))
                normal_message_flow(search_text)
                return
            
            # Notify user what we're recreating
            original_question = None
            for msg in group_messages:
                if msg.role == 'user':
                    original_question = msg.content
                    break
            
            add_message(Message(
                RoleType.ASSISTANT,
                f"Recreating thumbs-up conversation ({similarity_score*100:.1f}% match){': ' + original_question if original_question else ''}...",
                MessageType.TEXT
            )) #TODO: comment this out when we are happy with the result
            
            # Create a new group for the recreated messages
            from utils.chat_bot_helper import start_new_group
            new_group_id = start_new_group()
            
            add_acknowledgement()

            # Recreate each message in chronological order with random delays
            for original_msg in group_messages:
                # Random delay between 1-3 seconds
                delay_time = random.uniform(1.0, 3.0)
                time.sleep(delay_time)
                
                # Create new message with same content but new group_id and randomized timing
                try:
                    # Convert string role back to enum
                    role_enum = RoleType.USER if original_msg.role.lower() == 'user' else RoleType.ASSISTANT
                    
                    # Convert string type back to enum
                    type_mapping = {
                        'text': MessageType.TEXT,
                        'sql': MessageType.SQL,
                        'dataframe': MessageType.DATAFRAME,
                        'summary': MessageType.SUMMARY,
                        'error': MessageType.ERROR,
                        'python': MessageType.PYTHON,
                        'plotly_chart': MessageType.PLOTLY_CHART,
                        'followup': MessageType.FOLLOWUP,
                        'st_line_chart': MessageType.ST_LINE_CHART,
                        'st_bar_chart': MessageType.ST_BAR_CHART,
                        'st_area_chart': MessageType.ST_AREA_CHART,
                        'st_scatter_chart': MessageType.ST_SCATTER_CHART,
                    }
                    
                    message_type = type_mapping.get(original_msg.type.lower(), MessageType.TEXT)
                    
                    # Parse content based on message type
                    if message_type == MessageType.DATAFRAME and original_msg.dataframe:
                        # Use the stored dataframe JSON
                        content = original_msg.dataframe
                    elif message_type in [MessageType.PLOTLY_CHART] and original_msg.content:
                        # For charts, use the original content
                        content = original_msg.content
                    else:
                        # For text, SQL, summary, etc., use content
                        content = original_msg.content
                    
                    # Create and add the new message
                    new_message = Message(
                        role=role_enum,
                        content=content,
                        type=message_type,
                        query=original_msg.query,
                        question=original_msg.question,
                        dataframe=original_msg.dataframe,
                        elapsed_time=delay_time,  # Use randomized delay as elapsed time
                        group_id=new_group_id
                    )
                    
                    add_message(new_message)
                    
                except Exception as e:
                    # If there's an error recreating a specific message, log it but continue
                    add_message(Message(
                        RoleType.ASSISTANT,
                        f"Error recreating message (type: {original_msg.type}): {str(e)}",
                        MessageType.ERROR,
                        group_id=new_group_id
                    ))
            
            # Clear the question to prevent re-execution of the magic command on rerun
            if hasattr(st.session_state, 'my_question'):
                st.session_state.my_question = None
                
            # Force Streamlit to rerun to properly register new widget event handlers
            st.rerun()
                
    except Exception as e:
        add_message(Message(
            RoleType.ASSISTANT,
            f"Error searching history: {str(e)}",
            MessageType.ERROR
        ))


def _tables(question, tuple, previous_df):
    try:
        start_time = time.perf_counter()
        table_names = get_all_object_names()

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
        count = 20  # Default to 20 rows

        if "num_rows" in tuple:
            count = int(tuple["num_rows"])
        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])

            sql = f"SELECT *  FROM {table_name} LIMIT {count};"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df.head(count)

        if df.empty:
            raise Exception("No rows found in the database.")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, df, MessageType.DATAFRAME, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error retrieving tables: {str(e)}", MessageType.ERROR))


def _tail(question, tuple, previous_df):
    try:
        start_time = time.perf_counter()
        sql = ""
        count = 20  # Default to 20 rows

        if "num_rows" in tuple:
            count = int(tuple["num_rows"])
        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])

            # For tail, we need to get the total count first, then offset appropriately
            count_sql = f"SELECT COUNT(*) as total FROM {table_name};"
            count_df = run_sql_cached(count_sql)
            total_rows = count_df.iloc[0]['total']
            
            # Calculate offset to get the last 'count' rows
            offset = max(0, total_rows - count)
            sql = f"SELECT * FROM {table_name} OFFSET {offset};"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df.tail(count)

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
            response = _followup_llm(command, last_assistant_msg.content, df)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, response, type, None, command, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating follow up message: {str(e)}", MessageType.ERROR))


def _followup_llm(command, last_content, previous_df):
    """Direct LLM query about the previous data result with enhanced context."""
    try:
        start_time = time.perf_counter()

        add_message(Message(RoleType.ASSISTANT, "Asking LLM!", MessageType.TEXT))

        if previous_df is None or previous_df.empty:
            add_message(Message(RoleType.ASSISTANT, "No previous data available for LLM analysis.", MessageType.ERROR))
            return

        if not command.strip():
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    "Please provide a question for the LLM. Usage: /followup llm <your question>",
                    MessageType.ERROR,
                )
            )
            return

        # Get comprehensive data context
        context = _get_data_context(previous_df)

        # Create rich context for the LLM
        context_lines = []
        context_lines.append(f"Dataset Overview: {context['total_rows']} rows, {context['total_columns']} columns")

        if context.get("numeric_columns"):
            context_lines.append(
                f"Numeric columns ({len(context['numeric_columns'])}): {', '.join(context['numeric_columns'][:10])}"
            )
            if len(context["numeric_columns"]) > 10:
                context_lines.append(f"... and {len(context['numeric_columns']) - 10} more numeric columns")

        if context.get("text_columns"):
            context_lines.append(
                f"Text columns ({len(context['text_columns'])}): {', '.join(context['text_columns'][:10])}"
            )
            if len(context["text_columns"]) > 10:
                context_lines.append(f"... and {len(context['text_columns']) - 10} more text columns")

        if context.get("datetime_columns"):
            context_lines.append(f"Date/time columns: {', '.join(context['datetime_columns'])}")

        if context.get("has_nulls"):
            null_counts = previous_df.isnull().sum()
            null_cols = null_counts[null_counts > 0].head(5)
            context_lines.append(
                f"Missing data detected in: {', '.join([f'{col} ({count} nulls)' for col, count in null_cols.items()])}"
            )

        # Add basic statistics for numeric columns
        if context.get("numeric_columns"):
            numeric_stats = previous_df[context["numeric_columns"][:5]].describe()
            context_lines.append(f"Sample statistics available for: {', '.join(numeric_stats.columns)}")

        # Add sample data
        sample_data = previous_df.head(3).to_string(max_cols=8, max_colwidth=50)

        # Construct enhanced prompt
        enhanced_prompt = f"""
Data Context:
{chr(10).join(context_lines)}

Sample Data (first 3 rows):
{sample_data}

Other Context or visualizations:
{last_content}

User Question: {command}

Please analyze this data and provide insights based on the user's question. Be specific and reference actual data characteristics when possible.
"""

        # Get LLM response
        vn = get_vn()
        return vn.submit_prompt(enhanced_prompt, "Data analysis request")

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error in LLM followup: {str(e)}", MessageType.ERROR))


def _get_data_context(df):
    """Analyze DataFrame to provide context for intelligent suggestions."""
    if df is None or df.empty:
        return {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    return {
        "total_columns": len(df.columns),
        "total_rows": len(df),
        "numeric_columns": numeric_cols,
        "text_columns": text_cols,
        "datetime_columns": datetime_cols,
        "has_nulls": df.isnull().any().any(),
        "memory_usage": df.memory_usage(deep=True).sum(),
    }


def _generate_heatmap(question, tuple, previous_df):
    """
    Generate an enhanced correlation heatmap with professional styling.

    Args:
        question: Original user question
        tuple: Dictionary containing table name
        previous_df: Previous dataframe if available
    """
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        sql = ""

        # Data acquisition
        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 10000;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Calculate correlation matrix for numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No numeric columns found in {get_object_name_singular()} {table_name} for correlation analysis",
                    MessageType.ERROR,
                )
            )
            return

        if len(numeric_df.columns) < 2:
            add_message(
                Message(
                    RoleType.ASSISTANT, f"Need at least 2 numeric columns for correlation heatmap", MessageType.ERROR
                )
            )
            return

        corr_matrix = numeric_df.corr()

        # Create enhanced heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title=f"Correlation Heatmap for {table_name}",
        )

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating correlation heatmap: {str(e)}", MessageType.ERROR))


def _generate_wordcloud_column(question, tuple, previous_df):
    """
    Generate a word cloud visualization for a specific column.

    Args:
        question: Original user question
        tuple: Dictionary containing table and column names
        previous_df: Previous dataframe if available
    """
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        column_name = tuple["column"]
        sql = ""

        # Data acquisition
        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            column_name = find_closest_column_name(table_name, column_name)
            sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL;"
            fig, df = get_wordcloud(sql, table_name, column_name)
        else:
            fig, df = get_wordcloud(sql, table_name, column_name, previous_df)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(
            Message(
                RoleType.ASSISTANT,
                f"Error generating word cloud for column '{tuple.get('column', 'unknown')}': {str(e)}",
                MessageType.ERROR,
            )
        )


def _generate_wordcloud(question, tuple, previous_df):
    """
    Generate a word cloud visualization for all text columns in a table.

    Args:
        question: Original user question
        tuple: Dictionary containing table name
        previous_df: Previous dataframe if available
    """
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        sql = ""

        # Data acquisition
        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name} LIMIT 10000;"  # Limit for performance
            fig, df = get_wordcloud(sql, table_name)
        else:
            fig, df = get_wordcloud(sql, table_name, None, previous_df)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating word cloud: {str(e)}", MessageType.ERROR))


def get_wordcloud(sql, table_name, column_name=None, previous_df=None):
    """
    Generate an enhanced word cloud visualization with better styling and error handling.

    Args:
        sql: SQL query string
        table_name: Name of the table
        column_name: Specific column name (optional)
        previous_df: Previous dataframe if available

    Returns:
        tuple: (plotly_figure, dataframe)
    """
    try:
        # Data acquisition
        if previous_df is not None:
            df = previous_df
        else:
            df, elapsed_time = run_sql_cached(sql)

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return None, None

        # Text data extraction
        text_data = ""

        if column_name is not None:
            # Single column word cloud
            if column_name not in df.columns:
                add_message(
                    Message(
                        RoleType.ASSISTANT,
                        f"Column '{column_name}' not found in {get_object_name_singular()} '{table_name}'",
                        MessageType.ERROR,
                    )
                )
                return None, None
            text_data = df[column_name].astype(str).str.cat(sep=" ")
        else:
            # Multi-column word cloud
            string_columns = df.select_dtypes(include="object").columns
            if len(string_columns) == 0:
                add_message(
                    Message(
                        RoleType.ASSISTANT,
                        f"No text columns found in {get_object_name_singular()} '{table_name}'",
                        MessageType.ERROR,
                    )
                )
                return None, None

            words = []
            for col in string_columns:
                col_words = df[col].astype(str).str.cat(sep=" ").split()
                words += [w for w in col_words if w.lower() not in unwanted_words and len(w) > 2]
            text_data = " ".join(words)

        if not text_data or text_data.strip() == "":
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No meaningful text data found in {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return None, None

        # Enhanced word cloud generation
        try:
            # Try to load custom mask, fallback to default if not available
            mask_path = "assets/heart.png"
            img_mask = np.array(Image.open(mask_path))
        except:
            img_mask = None

        # Generate word cloud with enhanced settings
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="viridis",
            max_words=150,
            min_font_size=8,
            max_font_size=100,
            relative_scaling=0.5,
            random_state=42,
            mask=img_mask,
            collocations=False,  # Avoid word pairs
            normalize_plurals=False,
        ).generate(text_data)

        # Convert to plotly figure
        wordcloud_array = wordcloud.to_array()

        # Create enhanced visualization
        fig = px.imshow(
            wordcloud_array,
            title=f"Word Cloud - {table_name}" + (f" ({column_name})" if column_name else ""),
            template="plotly_white",
        )

        # Enhanced styling
        fig.update_layout(
            title={
                "text": f"Word Cloud - {table_name}" + (f" ({column_name})" if column_name else ""),
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "family": "Arial, sans-serif"},
            },
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False),
            width=1200,
            height=650,
            margin=dict(l=20, r=20, t=80, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        # Add word count annotation
        word_count = len(text_data.split())
        fig.add_annotation(
            text=f"Total words analyzed: {word_count:,}",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        )

        return fig, df

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error creating word cloud: {str(e)}", MessageType.ERROR))
        return None, None


def _generate_pairplot(question, tuple, previous_df):
    """
    Generate an enhanced pairplot (scatter matrix) visualization.

    Args:
        question: Original user question
        tuple: Dictionary containing table and column names
        previous_df: Previous dataframe if available
    """
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        column_name = tuple["column"]
        sql = ""

        # Data acquisition
        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            column_name = find_closest_column_name(table_name, column_name)
            sql = f"SELECT * FROM {table_name} LIMIT 10000;"  # Limit for performance
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Get numeric columns for the scatter matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            add_message(Message(RoleType.ASSISTANT, f"Need at least 2 numeric columns for pairplot", MessageType.ERROR))
            return

        # Limit to first 8 numeric columns for readability
        if len(numeric_cols) > 8:
            numeric_cols = numeric_cols[:8]
            add_message(Message(RoleType.ASSISTANT, f"Showing first 8 numeric columns for clarity", MessageType.TEXT))

        # Create enhanced scatter matrix
        fig = px.scatter_matrix(
            df,
            dimensions=numeric_cols,
            color=column_name if column_name in df.columns else None,
            title=f"Pairplot for {table_name} - {column_name}",
            width=1200,
            height=1200,
        )

        # Calculate dynamic size based on number of dimensions
        plot_size = max(600, len(numeric_cols) * 120)

        # Enhanced styling
        fig.update_layout(
            title={
                "text": f"Pairplot Analysis - {table_name}",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "family": "Arial, sans-serif"},
            },
            width=plot_size,
            height=plot_size,
            font=dict(family="Arial, sans-serif", size=10),
            dragmode="select",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        # Update trace styling
        fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color="white")), diagonal_visible=False)

        # Add interpretation annotation
        fig.add_annotation(
            text=f"Analyzing relationships between {len(numeric_cols)} numeric variables",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=12, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        )

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating pairplot: {str(e)}", MessageType.ERROR))


def generate_plotly(chart_type, question, tuple, previous_df):
    """
    Generate enhanced Plotly visualizations with improved aesthetics and interactivity.

    Args:
        chart_type: Type of chart to generate (scatter, bar, line)
        question: Original user question
        tuple: Dictionary containing column names
        previous_df: Previous dataframe if available
    """
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        column_x = tuple["x"]
        column_y = tuple["y"]
        column_color = tuple["color"]
        sql = ""

        # Data acquisition
        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            column_x = find_closest_column_name(table_name, tuple["x"])
            column_y = find_closest_column_name(table_name, tuple["y"])
            column_color = find_closest_column_name(table_name, tuple["color"])
            sql = f"SELECT {column_x}, {column_y}, {column_color} FROM {table_name};"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Create enhanced visualization
        plot_kwargs = {
            "x": column_x,
            "y": column_y,
            "color": column_color,
            "template": "plotly_white",
            "title": f"{chart_type.title()} Plot: {column_y} vs {column_x}",
            "labels": {
                column_x: column_x.replace("_", " ").title(),
                column_y: column_y.replace("_", " ").title(),
                column_color: column_color.replace("_", " ").title(),
            },
        }

        # Only add color_continuous_scale for chart types that support it
        if chart_type in ["scatter", "density_heatmap", "density_contour"]:
            plot_kwargs["color_continuous_scale"] = "viridis"

        fig = getattr(px, chart_type)(df, **plot_kwargs)

        # Enhanced styling
        fig.update_layout(
            title={
                "text": f"{chart_type.title()} Plot: {column_y} vs {column_x}",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "family": "Arial, sans-serif"},
            },
            width=1200,
            height=700,
            showlegend=True,
            hovermode="closest",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="white",
            font=dict(family="Arial, sans-serif", size=12),
            margin=dict(l=60, r=60, t=80, b=60),
        )

        # Add grid and enhance axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
            showline=True,
            linewidth=1,
            linecolor="rgba(128,128,128,0.5)",
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
            showline=True,
            linewidth=1,
            linecolor="rgba(128,128,128,0.5)",
        )

        # Chart-specific enhancements
        if chart_type == "scatter":
            fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color="white")))
        elif chart_type == "bar":
            fig.update_traces(marker=dict(line=dict(width=1, color="white")))

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating {chart_type} plot: {str(e)}", MessageType.ERROR))


def _generate_scatterplot(question, tuple, previous_df):
    generate_plotly("scatter", question, tuple, previous_df)


def _generate_bar(question, tuple, previous_df):
    generate_plotly("bar", question, tuple, previous_df)


def _generate_line(question, tuple, previous_df):
    generate_plotly("line", question, tuple, previous_df)


def _describe_table(question, tuple, previous_df):
    """Generate comprehensive descriptive statistics for a table."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name} LIMIT 10000;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Get basic description
        desc = df.describe(include="all")

        # Add additional statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        additional_stats = {}

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                additional_stats[col] = {
                    "skewness": stats.skew(col_data),
                    "kurtosis": stats.kurtosis(col_data),
                    "variance": np.var(col_data),
                    "missing_count": df[col].isnull().sum(),
                    "missing_percent": (df[col].isnull().sum() / len(df)) * 100,
                }

        # Create a comprehensive description DataFrame
        desc_extended = desc.copy()
        for col in numeric_cols:
            if col in additional_stats:
                desc_extended.loc["skewness", col] = additional_stats[col]["skewness"]
                desc_extended.loc["kurtosis", col] = additional_stats[col]["kurtosis"]
                desc_extended.loc["variance", col] = additional_stats[col]["variance"]
                desc_extended.loc["missing_count", col] = additional_stats[col]["missing_count"]
                desc_extended.loc["missing_percent", col] = additional_stats[col]["missing_percent"]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, desc_extended, MessageType.DATAFRAME, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating description: {str(e)}", MessageType.ERROR))


def _distribution_analysis(question, tuple, previous_df):
    """Generate distribution analysis for a specific column."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        column_name = tuple["column"]
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            column_name = find_closest_column_name(table_name, column_name)
            sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df
            if column_name not in df.columns:
                add_message(Message(RoleType.ASSISTANT, f"Column '{column_name}' not found", MessageType.ERROR))
                return

        if df is None or df.empty:
            add_message(Message(RoleType.ASSISTANT, f"No data found for column '{column_name}'", MessageType.ERROR))
            return

        data = df[column_name].dropna()

        if not pd.api.types.is_numeric_dtype(data):
            add_message(Message(RoleType.ASSISTANT, f"Column '{column_name}' is not numeric", MessageType.ERROR))
            return

        # Create enhanced distribution plot
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Distribution Histogram",
                "Box Plot with Outliers",
                "Q-Q Normality Plot",
                "Statistical Summary",
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )

        # Enhanced histogram with normal overlay
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                name="Distribution",
                opacity=0.7,
                marker=dict(color="skyblue", line=dict(color="white", width=1)),
                hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add normal distribution overlay
        x_norm = np.linspace(data.min(), data.max(), 100)
        y_norm = stats.norm.pdf(x_norm, data.mean(), data.std()) * len(data) * (data.max() - data.min()) / 30
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode="lines",
                name="Normal Distribution",
                line=dict(color="red", width=2, dash="dash"),
                hovertemplate="Normal Distribution<br>Value: %{x:.2f}<br>Density: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Enhanced box plot
        fig.add_trace(
            go.Box(
                y=data,
                name="Box Plot",
                boxpoints="outliers",
                marker=dict(color="lightcoral"),
                line=dict(color="darkred"),
                fillcolor="rgba(255, 182, 193, 0.5)",
                hovertemplate="Value: %{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Enhanced Q-Q plot
        qq_data = stats.probplot(data, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode="markers",
                name="Q-Q Plot",
                marker=dict(color="darkgreen", size=6, opacity=0.6),
                hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Add reference line for Q-Q plot
        fig.add_trace(
            go.Scatter(
                x=[qq_data[0][0].min(), qq_data[0][0].max()],
                y=[qq_data[0][1].min(), qq_data[0][1].max()],
                mode="lines",
                name="Reference Line",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Enhanced statistics table
        normality_stat, normality_p = stats.shapiro(data.sample(min(5000, len(data))))

        stat_data = {
            "Statistic": [
                "Count",
                "Mean",
                "Median",
                "Std Dev",
                "Skewness",
                "Kurtosis",
                "Min",
                "Max",
                "Range",
                "Normality (p-value)",
            ],
            "Value": [
                f"{len(data):,}",
                f"{data.mean():.4f}",
                f"{data.median():.4f}",
                f"{data.std():.4f}",
                f"{stats.skew(data):.4f}",
                f"{stats.kurtosis(data):.4f}",
                f"{data.min():.4f}",
                f"{data.max():.4f}",
                f"{data.max() - data.min():.4f}",
                f"{normality_p:.4f}",
            ],
        }

        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(stat_data.keys()),
                    fill_color="lightblue",
                    align="left",
                    font=dict(color="black", size=12),
                ),
                cells=dict(
                    values=list(stat_data.values()), fill_color="white", align="left", font=dict(color="black", size=11)
                ),
            ),
            row=2,
            col=2,
        )

        # Enhanced layout
        fig.update_layout(
            title={
                "text": f"Distribution Analysis - {column_name}",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "family": "Arial, sans-serif"},
            },
            height=800,
            showlegend=False,
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=11),
        )

        # Update axes labels
        fig.update_xaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating distribution analysis: {str(e)}", MessageType.ERROR))


def _outlier_detection(question, tuple, previous_df):
    """Detect outliers in a column using multiple methods."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        column_name = tuple["column"]
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            column_name = find_closest_column_name(table_name, column_name)
            sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df
            if column_name not in df.columns:
                add_message(Message(RoleType.ASSISTANT, f"Column '{column_name}' not found", MessageType.ERROR))
                return

        if df is None or df.empty:
            add_message(Message(RoleType.ASSISTANT, f"No data found for column '{column_name}'", MessageType.ERROR))
            return

        data = df[column_name].dropna()

        if not pd.api.types.is_numeric_dtype(data):
            add_message(Message(RoleType.ASSISTANT, f"Column '{column_name}' is not numeric", MessageType.ERROR))
            return

        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]

        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        z_outliers = data[z_scores > 3]

        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
        iso_outliers = data[outlier_labels == -1]

        # Create visualization
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Box Plot with Outliers", "Histogram", "Outlier Methods Comparison", "Outlier Summary"),
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
        )

        # Box plot
        fig.add_trace(go.Box(y=data, name="Data with Outliers"), row=1, col=1)

        # Histogram with outlier boundaries
        fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Distribution"), row=1, col=2)

        # Outlier comparison
        methods = ["IQR", "Z-Score", "Isolation Forest"]
        outlier_counts = [len(iqr_outliers), len(z_outliers), len(iso_outliers)]

        fig.add_trace(go.Bar(x=methods, y=outlier_counts, name="Outlier Count by Method"), row=2, col=1)

        # Summary table
        summary_data = {
            "Method": ["IQR (1.5×IQR)", "Z-Score (|z| > 3)", "Isolation Forest"],
            "Outliers Found": [len(iqr_outliers), len(z_outliers), len(iso_outliers)],
            "Percentage": [
                f"{len(iqr_outliers) / len(data) * 100:.2f}%",
                f"{len(z_outliers) / len(data) * 100:.2f}%",
                f"{len(iso_outliers) / len(data) * 100:.2f}%",
            ],
        }

        fig.add_trace(
            go.Table(header=dict(values=list(summary_data.keys())), cells=dict(values=list(summary_data.values()))),
            row=2,
            col=2,
        )

        fig.update_layout(title=f"Outlier Detection for {column_name}", height=800, showlegend=False)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error detecting outliers: {str(e)}", MessageType.ERROR))


def _profile_table(question, tuple, previous_df):
    """Generate comprehensive data profiling report."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name} LIMIT 10000;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Create profiling report
        profile_data = []

        for col in df.columns:
            col_data = df[col]
            profile_info = {
                "Column": col,
                "Data Type": str(col_data.dtype),
                "Non-Null Count": col_data.count(),
                "Null Count": col_data.isnull().sum(),
                "Null Percentage": f"{(col_data.isnull().sum() / len(df)) * 100:.2f}%",
                "Unique Values": col_data.nunique(),
                "Unique Percentage": f"{(col_data.nunique() / len(df)) * 100:.2f}%",
            }

            if pd.api.types.is_numeric_dtype(col_data):
                profile_info.update(
                    {
                        "Min": col_data.min(),
                        "Max": col_data.max(),
                        "Mean": f"{col_data.mean():.4f}" if col_data.mean() == col_data.mean() else "N/A",
                        "Std Dev": f"{col_data.std():.4f}" if col_data.std() == col_data.std() else "N/A",
                    }
                )
            else:
                most_common = col_data.mode()
                profile_info.update(
                    {
                        "Most Common": most_common.iloc[0] if len(most_common) > 0 else "N/A",
                        "Min Length": col_data.astype(str).str.len().min(),
                        "Max Length": col_data.astype(str).str.len().max(),
                        "Avg Length": f"{col_data.astype(str).str.len().mean():.2f}",
                    }
                )

            profile_data.append(profile_info)

        profile_df = pd.DataFrame(profile_data)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, profile_df, MessageType.DATAFRAME, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating profile: {str(e)}", MessageType.ERROR))


def _missing_analysis(question, tuple, previous_df):
    """Analyze missing data patterns."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name} LIMIT 10000;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Calculate missing data
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100

        # Create missing data heatmap
        missing_matrix = df.isnull().astype(int)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Missing Data Heatmap",
                "Missing Data by Column",
                "Missing Data Pattern",
                "Missing Data Summary",
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
        )

        # Missing data heatmap
        fig.add_trace(
            go.Heatmap(
                z=missing_matrix.T.values,
                x=list(range(len(df))),
                y=df.columns.tolist(),
                colorscale="Reds",
                showscale=False,
            ),
            row=1,
            col=1,
        )

        # Missing data by column
        fig.add_trace(go.Bar(x=df.columns, y=missing_percent, name="Missing %"), row=1, col=2)

        # Missing data pattern
        pattern_counts = missing_matrix.groupby(list(missing_matrix.columns)).size().reset_index(name="Count")
        pattern_counts = pattern_counts.sort_values("Count", ascending=False).head(10)

        fig.add_trace(
            go.Bar(x=list(range(len(pattern_counts))), y=pattern_counts["Count"], name="Pattern Frequency"),
            row=2,
            col=1,
        )

        # Summary table
        summary_data = {
            "Column": df.columns.tolist(),
            "Missing Count": missing_data.tolist(),
            "Missing %": [f"{pct:.2f}%" for pct in missing_percent],
        }

        fig.add_trace(
            go.Table(header=dict(values=list(summary_data.keys())), cells=dict(values=list(summary_data.values()))),
            row=2,
            col=2,
        )

        fig.update_layout(title=f"Missing Data Analysis for {table_name}", height=800, showlegend=False)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error analyzing missing data: {str(e)}", MessageType.ERROR))


def _duplicate_analysis(question, tuple, previous_df):
    """Analyze duplicate rows in the dataset."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name} LIMIT 10000;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Find duplicates
        duplicates = df.duplicated()
        duplicate_rows = df[duplicates]

        # Analyze duplicates by column
        duplicate_analysis = []
        for col in df.columns:
            col_duplicates = df[col].duplicated()
            duplicate_analysis.append(
                {
                    "Column": col,
                    "Duplicate Count": col_duplicates.sum(),
                    "Duplicate Percentage": f"{(col_duplicates.sum() / len(df)) * 100:.2f}%",
                    "Unique Values": df[col].nunique(),
                    "Total Values": len(df),
                }
            )

        duplicate_df = pd.DataFrame(duplicate_analysis)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Return summary of duplicates
        summary_message = f"Duplicate Analysis for {table_name}:\n"
        summary_message += f"Total Rows: {len(df)}\n"
        summary_message += f"Duplicate Rows: {len(duplicate_rows)}\n"
        summary_message += f"Duplicate Percentage: {(len(duplicate_rows) / len(df)) * 100:.2f}%\n\n"
        summary_message += "Column-wise Duplicate Analysis:"

        add_message(Message(RoleType.ASSISTANT, summary_message, MessageType.TEXT, sql, question, df, elapsed_time))
        add_message(Message(RoleType.ASSISTANT, duplicate_df, MessageType.DATAFRAME, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error analyzing duplicates: {str(e)}", MessageType.ERROR))


def _boxplot_visualization(question, tuple, previous_df):
    """Create box plot visualization."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        column_name = tuple["column"]
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            column_name = find_closest_column_name(table_name, column_name)
            sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df
            if column_name not in df.columns:
                add_message(Message(RoleType.ASSISTANT, f"Column '{column_name}' not found", MessageType.ERROR))
                return

        if df is None or df.empty:
            add_message(Message(RoleType.ASSISTANT, f"No data found for column '{column_name}'", MessageType.ERROR))
            return

        data = df[column_name].dropna()

        if not pd.api.types.is_numeric_dtype(data):
            add_message(Message(RoleType.ASSISTANT, f"Column '{column_name}' is not numeric", MessageType.ERROR))
            return

        # Create enhanced box plot with statistical annotations
        fig = go.Figure()

        # Add main box plot
        fig.add_trace(
            go.Box(
                y=data,
                name=column_name.replace("_", " ").title(),
                boxpoints="outliers",
                jitter=0.3,
                pointpos=-1.8,
                boxmean=True,
                marker=dict(color="lightblue", outliercolor="red", line=dict(outliercolor="red", outlierwidth=2)),
                line=dict(color="darkblue"),
                fillcolor="rgba(173, 216, 230, 0.5)",
                hovertemplate="<b>%{y}</b><extra></extra>",
            )
        )

        # Calculate statistical measures
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        outliers = data[(data < lower_fence) | (data > upper_fence)]

        # Add statistical annotations
        stats_text = f"<b>Statistical Summary</b><br>"
        stats_text += f"Count: {len(data):,}<br>"
        stats_text += f"Mean: {data.mean():.3f}<br>"
        stats_text += f"Median: {data.median():.3f}<br>"
        stats_text += f"Q1: {Q1:.3f}<br>"
        stats_text += f"Q3: {Q3:.3f}<br>"
        stats_text += f"IQR: {IQR:.3f}<br>"
        stats_text += f"Std Dev: {data.std():.3f}<br>"
        stats_text += f"Outliers: {len(outliers)} ({len(outliers) / len(data) * 100:.1f}%)"

        fig.add_annotation(
            text=stats_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="darkblue",
            borderwidth=1,
            font=dict(size=12, family="Arial, sans-serif"),
        )

        # Add interpretation guide
        interpretation_text = f"<b>Interpretation Guide</b><br>"
        interpretation_text += f"Box: Q1 to Q3 (middle 50%)<br>"
        interpretation_text += f"Line: Median<br>"
        interpretation_text += f"◊: Mean<br>"
        interpretation_text += f"Whiskers: 1.5×IQR from box<br>"
        interpretation_text += f"Red dots: Outliers"

        fig.add_annotation(
            text=interpretation_text,
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10, family="Arial, sans-serif"),
        )

        # Enhanced layout
        fig.update_layout(
            title={
                "text": f"Box Plot Analysis - {column_name.replace('_', ' ').title()}",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "family": "Arial, sans-serif"},
            },
            yaxis_title=column_name.replace("_", " ").title(),
            height=700,
            width=900,
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="white",
        )

        # Add grid
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error creating box plot: {str(e)}", MessageType.ERROR))


def _cluster_analysis(question, tuple, previous_df):
    """Perform K-means clustering analysis."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name} LIMIT 10000;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.empty:
            add_message(Message(RoleType.ASSISTANT, f"No numeric columns found for clustering", MessageType.ERROR))
            return

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Determine optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(11, len(numeric_df)))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)

            from sklearn.metrics import silhouette_score

            silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)

        # Perform clustering with optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)

        # Create visualization
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Elbow Method", "Silhouette Scores", "PCA Cluster Visualization", "Cluster Summary"),
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
        )

        # Elbow method plot
        fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers", name="Inertia"), row=1, col=1)

        # Silhouette scores plot
        fig.add_trace(
            go.Scatter(x=list(k_range), y=silhouette_scores, mode="lines+markers", name="Silhouette Score"),
            row=1,
            col=2,
        )

        # PCA visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        fig.add_trace(
            go.Scatter(
                x=pca_data[:, 0],
                y=pca_data[:, 1],
                mode="markers",
                marker=dict(color=cluster_labels, colorscale="Viridis"),
                name="Clusters",
            ),
            row=2,
            col=1,
        )

        # Cluster summary table
        cluster_summary = pd.DataFrame(
            {
                "Cluster": range(optimal_k),
                "Size": [np.sum(cluster_labels == i) for i in range(optimal_k)],
                "Percentage": [
                    f"{(np.sum(cluster_labels == i) / len(cluster_labels)) * 100:.1f}%" for i in range(optimal_k)
                ],
            }
        )

        fig.add_trace(
            go.Table(
                header=dict(values=["Cluster", "Size", "Percentage"]),
                cells=dict(values=[cluster_summary["Cluster"], cluster_summary["Size"], cluster_summary["Percentage"]]),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=f"K-Means Clustering Analysis for {table_name} (Optimal K={optimal_k})", height=800, showlegend=False
        )

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error performing cluster analysis: {str(e)}", MessageType.ERROR))


def _pca_analysis(question, tuple, previous_df):
    """Perform Principal Component Analysis."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name} LIMIT 10000;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.empty:
            add_message(Message(RoleType.ASSISTANT, f"No numeric columns found for PCA", MessageType.ERROR))
            return

        if len(numeric_df.columns) < 2:
            add_message(Message(RoleType.ASSISTANT, f"Need at least 2 numeric columns for PCA", MessageType.ERROR))
            return

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)

        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # Create visualization
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Explained Variance", "Cumulative Variance", "PCA Biplot", "Component Loadings"),
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
        )

        # Explained variance plot
        fig.add_trace(
            go.Bar(x=list(range(1, len(explained_variance) + 1)), y=explained_variance, name="Explained Variance"),
            row=1,
            col=1,
        )

        # Cumulative variance plot
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_variance) + 1)),
                y=cumulative_variance,
                mode="lines+markers",
                name="Cumulative Variance",
            ),
            row=1,
            col=2,
        )

        # PCA biplot (first two components)
        fig.add_trace(
            go.Scatter(
                x=pca_result[:, 0], y=pca_result[:, 1], mode="markers", name="Data Points", marker=dict(opacity=0.6)
            ),
            row=2,
            col=1,
        )

        # Component loadings table
        components_df = pd.DataFrame(
            pca.components_[:4].T,  # First 4 components
            columns=[f"PC{i + 1}" for i in range(min(4, len(pca.components_)))],
            index=numeric_df.columns,
        )

        fig.add_trace(
            go.Table(
                header=dict(values=["Feature"] + [f"PC{i + 1}" for i in range(min(4, len(pca.components_)))]),
                cells=dict(
                    values=[components_df.index] + [components_df[col].round(3) for col in components_df.columns]
                ),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(title=f"PCA Analysis for {table_name}", height=800, showlegend=False)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error performing PCA analysis: {str(e)}", MessageType.ERROR))


def _confusion_matrix(question, tuple, previous_df):
    """Generate confusion matrix for classification analysis."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        true_column = tuple["true_column"]
        pred_column = tuple["pred_column"]
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            true_column = find_closest_column_name(table_name, true_column)
            pred_column = find_closest_column_name(table_name, pred_column)
            sql = f"SELECT {true_column}, {pred_column} FROM {table_name} WHERE {true_column} IS NOT NULL AND {pred_column} IS NOT NULL;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df
            if true_column not in df.columns:
                add_message(Message(RoleType.ASSISTANT, f"Column '{true_column}' not found", MessageType.ERROR))
                return
            if pred_column not in df.columns:
                add_message(Message(RoleType.ASSISTANT, f"Column '{pred_column}' not found", MessageType.ERROR))
                return

        if df is None or df.empty:
            add_message(Message(RoleType.ASSISTANT, f"No data found for confusion matrix analysis", MessageType.ERROR))
            return

        # Clean data
        df_clean = df[[true_column, pred_column]].dropna()

        if df_clean.empty:
            add_message(
                Message(RoleType.ASSISTANT, f"No valid data found after removing null values", MessageType.ERROR)
            )
            return

        y_true = df_clean[true_column]
        y_pred = df_clean[pred_column]

        # Import confusion matrix from sklearn
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true) | set(y_pred)))

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)

        # Create heatmap visualization
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                hoverongaps=False,
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Confusion Matrix: {true_column} vs {pred_column}<br>Accuracy: {accuracy:.3f}",
            xaxis_title="Predicted",
            yaxis_title="True",
            xaxis={"side": "bottom"},
            yaxis={"autorange": "reversed"},
            width=600,
            height=500,
        )

        # Generate detailed analysis
        analysis = []
        analysis.append(f"CONFUSION MATRIX ANALYSIS")
        analysis.append("=" * 40)
        analysis.append(f"Dataset: {table_name}")
        analysis.append(f"True Column: {true_column}")
        analysis.append(f"Predicted Column: {pred_column}")
        analysis.append(f"Total Samples: {len(df_clean):,}")
        analysis.append(f"Overall Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")
        analysis.append("")

        # Per-class metrics
        # analysis.append("PER-CLASS METRICS:")
        # analysis.append("-" * 20)
        # for label in labels:
        #     if str(label) in class_report:
        #         metrics = class_report[str(label)]
        #         analysis.append(f"Class '{label}':")
        #         analysis.append(f"  Precision: {metrics['precision']:.3f}")
        #         analysis.append(f"  Recall: {metrics['recall']:.3f}")
        #         analysis.append(f"  F1-Score: {metrics['f1-score']:.3f}")
        #         analysis.append(f"  Support: {int(metrics['support'])}")
        #         analysis.append("")

        # Overall metrics
        if "macro avg" in class_report:
            macro_avg = class_report["macro avg"]
            analysis.append("OVERALL METRICS:")
            analysis.append("-" * 16)
            analysis.append(f"Macro Avg Precision: {macro_avg['precision']:.3f}")
            analysis.append(f"Macro Avg Recall: {macro_avg['recall']:.3f}")
            analysis.append(f"Macro Avg F1-Score: {macro_avg['f1-score']:.3f}")

        elapsed = time.perf_counter() - start_time

        # Add messages
        add_message(Message(RoleType.ASSISTANT, "\n".join(analysis), MessageType.PYTHON, None, question, None, elapsed))
        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, None, question, sql, elapsed))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating confusion matrix: {str(e)}", MessageType.ERROR))


def _generate_report(question, tuple, previous_df):
    """Generate a comprehensive data analysis report."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name} LIMIT 10000;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Generate comprehensive report
        report_sections = []

        # 1. Dataset Overview
        report_sections.append("# COMPREHENSIVE DATA ANALYSIS REPORT")
        report_sections.append("=" * 60)
        report_sections.append(f"Dataset: {table_name}")
        report_sections.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append("")

        # 2. Basic Dataset Information
        report_sections.append("## 1. DATASET OVERVIEW")
        report_sections.append("-" * 30)
        report_sections.append(f"Total Rows: {len(df):,}")
        report_sections.append(f"Total Columns: {len(df.columns)}")
        report_sections.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report_sections.append("")

        # 3. Column Information
        report_sections.append("## 2. COLUMN INFORMATION")
        report_sections.append("-" * 30)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

        report_sections.append(
            f"Numeric Columns ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'None'}"
        )
        report_sections.append(
            f"Categorical Columns ({len(categorical_cols)}): {', '.join(categorical_cols) if categorical_cols else 'None'}"
        )
        report_sections.append(
            f"DateTime Columns ({len(datetime_cols)}): {', '.join(datetime_cols) if datetime_cols else 'None'}"
        )
        report_sections.append("")

        # 4. Data Quality Assessment
        report_sections.append("## 3. DATA QUALITY ASSESSMENT")
        report_sections.append("-" * 30)

        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100

        if missing_data.sum() > 0:
            report_sections.append("Missing Data Summary:")
            for col in df.columns:
                if missing_data[col] > 0:
                    report_sections.append(f"  {col}: {missing_data[col]:,} ({missing_percent[col]:.1f}%)")
        else:
            report_sections.append("✓ No missing data detected")

        # Duplicate analysis
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            report_sections.append(f"Duplicate Rows: {duplicates:,} ({duplicates / len(df) * 100:.1f}%)")
        else:
            report_sections.append("✓ No duplicate rows detected")

        report_sections.append("")

        # 5. Descriptive Statistics for Numeric Columns
        if numeric_cols:
            report_sections.append("## 4. DESCRIPTIVE STATISTICS (Numeric Columns)")
            report_sections.append("-" * 30)

            desc_stats = df[numeric_cols].describe()
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    report_sections.append(f"{col}:")
                    report_sections.append(f"  Count: {desc_stats.loc['count', col]:.0f}")
                    report_sections.append(f"  Mean: {desc_stats.loc['mean', col]:.4f}")
                    report_sections.append(f"  Std Dev: {desc_stats.loc['std', col]:.4f}")
                    report_sections.append(f"  Min: {desc_stats.loc['min', col]:.4f}")
                    report_sections.append(f"  25%: {desc_stats.loc['25%', col]:.4f}")
                    report_sections.append(f"  50% (Median): {desc_stats.loc['50%', col]:.4f}")
                    report_sections.append(f"  75%: {desc_stats.loc['75%', col]:.4f}")
                    report_sections.append(f"  Max: {desc_stats.loc['max', col]:.4f}")
                    report_sections.append(f"  Skewness: {stats.skew(col_data):.4f}")
                    report_sections.append(f"  Kurtosis: {stats.kurtosis(col_data):.4f}")

                    # Outlier detection using IQR method
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    report_sections.append(
                        f"  Outliers (IQR method): {len(outliers)} ({len(outliers) / len(col_data) * 100:.1f}%)"
                    )
                    report_sections.append("")

        # 6. Categorical Data Analysis
        if categorical_cols:
            report_sections.append("## 5. CATEGORICAL DATA ANALYSIS")
            report_sections.append("-" * 30)

            for col in categorical_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    value_counts = col_data.value_counts()
                    report_sections.append(f"{col}:")
                    report_sections.append(f"  Unique Values: {col_data.nunique()}")
                    report_sections.append(
                        f"  Most Common: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)"
                    )
                    if len(value_counts) > 1:
                        report_sections.append(
                            f"  Second Most Common: {value_counts.index[1]} ({value_counts.iloc[1]} occurrences)"
                        )
                    report_sections.append(
                        f"  Least Common: {value_counts.index[-1]} ({value_counts.iloc[-1]} occurrences)"
                    )
                    report_sections.append("")

        # 7. Correlation Analysis (if multiple numeric columns)
        if len(numeric_cols) > 1:
            report_sections.append("## 6. CORRELATION ANALYSIS")
            report_sections.append("-" * 30)

            corr_matrix = df[numeric_cols].corr()

            # Find strongest positive and negative correlations
            correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    corr_val = corr_matrix.loc[col1, col2]
                    if not pd.isna(corr_val):
                        correlations.append((col1, col2, corr_val))

            if correlations:
                # Sort by absolute correlation value
                correlations.sort(key=lambda x: abs(x[2]), reverse=True)

                report_sections.append("Strongest Correlations:")
                for col1, col2, corr_val in correlations[:5]:  # Top 5
                    strength = "Strong" if abs(corr_val) >= 0.7 else "Moderate" if abs(corr_val) >= 0.3 else "Weak"
                    direction = "Positive" if corr_val > 0 else "Negative"
                    report_sections.append(f"  {col1} ↔ {col2}: {corr_val:.4f} ({strength} {direction})")
                report_sections.append("")

        # 8. Data Distribution Insights
        if numeric_cols:
            report_sections.append("## 7. DISTRIBUTION INSIGHTS")
            report_sections.append("-" * 30)

            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
                    if len(col_data) <= 5000:
                        _, normality_p = stats.shapiro(col_data.sample(min(5000, len(col_data))))
                        test_name = "Shapiro-Wilk"
                    else:
                        normality_p = 0.0  # For large samples, assume non-normal
                        test_name = "Large Sample"

                    is_normal = normality_p > 0.05
                    skewness = stats.skew(col_data)

                    report_sections.append(f"{col}:")
                    report_sections.append(
                        f"  Distribution: {'Normal' if is_normal else 'Non-Normal'} ({test_name} test)"
                    )
                    if abs(skewness) > 0.5:
                        skew_direction = "Right" if skewness > 0 else "Left"
                        report_sections.append(f"  Skewness: {skewness:.4f} ({skew_direction}-skewed)")
                    else:
                        report_sections.append(f"  Skewness: {skewness:.4f} (Approximately symmetric)")
                    report_sections.append("")

        # 9. Recommendations
        report_sections.append("## 8. RECOMMENDATIONS")
        report_sections.append("-" * 30)

        recommendations = []

        # Data quality recommendations
        if missing_data.sum() > 0:
            high_missing_cols = missing_percent[missing_percent > 50].index.tolist()
            if high_missing_cols:
                recommendations.append(
                    f"⚠️  Consider dropping columns with >50% missing data: {', '.join(high_missing_cols)}"
                )

            medium_missing_cols = missing_percent[(missing_percent > 20) & (missing_percent <= 50)].index.tolist()
            if medium_missing_cols:
                recommendations.append(f"📋 Consider imputation strategies for: {', '.join(medium_missing_cols)}")

        if duplicates > 0:
            recommendations.append(f"🔄 Consider removing {duplicates:,} duplicate rows")

        # Statistical recommendations
        if numeric_cols:
            highly_skewed = []
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0 and abs(stats.skew(col_data)) > 2:
                    highly_skewed.append(col)

            if highly_skewed:
                recommendations.append(
                    f"📊 Consider log transformation for highly skewed columns: {', '.join(highly_skewed)}"
                )

        # Correlation recommendations
        if len(numeric_cols) > 1:
            high_corr_pairs = [(col1, col2) for col1, col2, corr_val in correlations if abs(corr_val) > 0.9]
            if high_corr_pairs:
                recommendations.append(
                    f"🔗 High correlation detected - consider multicollinearity: {', '.join([f'{c1}↔{c2}' for c1, c2 in high_corr_pairs])}"
                )

        # Feature engineering recommendations
        if categorical_cols:
            high_cardinality_cols = []
            for col in categorical_cols:
                if df[col].nunique() > len(df) * 0.5:  # More than 50% unique values
                    high_cardinality_cols.append(col)

            if high_cardinality_cols:
                recommendations.append(
                    f"🏷️  High cardinality categorical columns may need encoding: {', '.join(high_cardinality_cols)}"
                )

        if recommendations:
            for rec in recommendations:
                report_sections.append(f"  {rec}")
        else:
            report_sections.append("  ✅ Data quality looks good - no major issues detected")

        report_sections.append("")

        # 10. Summary
        report_sections.append("## 9. EXECUTIVE SUMMARY")
        report_sections.append("-" * 30)
        report_sections.append(f"This dataset contains {len(df):,} records with {len(df.columns)} features.")

        quality_score = 100
        if missing_data.sum() > 0:
            quality_score -= min(30, (missing_data.sum() / (len(df) * len(df.columns))) * 100)
        if duplicates > 0:
            quality_score -= min(20, (duplicates / len(df)) * 100)

        report_sections.append(f"Overall data quality score: {quality_score:.1f}/100")

        if numeric_cols:
            report_sections.append(
                f"The dataset has {len(numeric_cols)} numeric features suitable for statistical analysis."
            )
        if categorical_cols:
            report_sections.append(
                f"There are {len(categorical_cols)} categorical features for grouping and segmentation."
            )

        report_sections.append("")
        report_sections.append("For detailed visualizations, use specific magic commands:")
        report_sections.append("  /heatmap <table> - Correlation heatmap")
        report_sections.append("  /clusters <table> - Clustering analysis")
        report_sections.append("  /pca <table> - Principal component analysis")
        report_sections.append("  /missing <table> - Missing data visualization")

        # Combine all sections
        full_report = "\n".join(report_sections)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, full_report, MessageType.PYTHON, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating report: {str(e)}", MessageType.ERROR))


def _generate_summary(question, tuple, previous_df):
    """Generate an executive summary of key findings and insights."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name} LIMIT 10000;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Generate executive summary
        summary_sections = []

        # Header
        summary_sections.append("# EXECUTIVE SUMMARY")
        summary_sections.append("=" * 40)
        summary_sections.append(f"Dataset: {table_name}")
        summary_sections.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_sections.append("")

        # Key metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        total_outliers = 0  # Initialize outlier counter

        # 1. Dataset Overview
        summary_sections.append("## 📊 DATASET OVERVIEW")
        summary_sections.append(f"• **{len(df):,} records** across **{len(df.columns)} features**")
        summary_sections.append(
            f"• **{len(numeric_cols)} numeric** and **{len(categorical_cols)} categorical** variables"
        )
        summary_sections.append(f"• Dataset size: **{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB**")
        summary_sections.append("")

        # 2. Data Quality Score
        missing_data = df.isnull().sum()
        duplicates = df.duplicated().sum()

        quality_score = 100
        quality_issues = []

        if missing_data.sum() > 0:
            missing_pct = (missing_data.sum() / (len(df) * len(df.columns))) * 100
            quality_score -= min(30, missing_pct)
            quality_issues.append(f"{missing_data.sum():,} missing values ({missing_pct:.1f}%)")

        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            quality_score -= min(20, dup_pct)
            quality_issues.append(f"{duplicates:,} duplicate rows ({dup_pct:.1f}%)")

        summary_sections.append("## 🎯 DATA QUALITY SCORE")
        if quality_score >= 90:
            quality_emoji = "🟢"
            quality_status = "Excellent"
        elif quality_score >= 75:
            quality_emoji = "🟡"
            quality_status = "Good"
        elif quality_score >= 50:
            quality_emoji = "🟠"
            quality_status = "Fair"
        else:
            quality_emoji = "🔴"
            quality_status = "Poor"

        summary_sections.append(f"{quality_emoji} **{quality_score:.1f}/100** ({quality_status})")

        if quality_issues:
            summary_sections.append("**Issues identified:**")
            for issue in quality_issues:
                summary_sections.append(f"• {issue}")
        else:
            summary_sections.append("• No significant data quality issues detected")
        summary_sections.append("")

        # 3. Key Statistical Insights
        if numeric_cols:
            summary_sections.append("## 📈 KEY STATISTICAL INSIGHTS")

            insights = []

            # Find most variable columns
            cv_data = []  # Coefficient of variation
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0 and col_data.mean() != 0:
                    cv = col_data.std() / abs(col_data.mean())
                    cv_data.append((col, cv))

            if cv_data:
                cv_data.sort(key=lambda x: x[1], reverse=True)
                most_variable = cv_data[0][0]
                insights.append(
                    f"**Most variable feature:** {most_variable} (coefficient of variation: {cv_data[0][1]:.2f})"
                )

            # Find columns with extreme values
            extreme_cols = []
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    skewness = abs(stats.skew(col_data))
                    if skewness > 2:  # Highly skewed
                        extreme_cols.append((col, skewness))

            if extreme_cols:
                extreme_cols.sort(key=lambda x: x[1], reverse=True)
                insights.append(f"**Highly skewed data:** {extreme_cols[0][0]} (skewness: {extreme_cols[0][1]:.2f})")

            # Outlier summary
            outlier_cols = []
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    if len(outliers) > 0:
                        total_outliers += len(outliers)
                        outlier_cols.append((col, len(outliers)))

            if total_outliers > 0:
                outlier_cols.sort(key=lambda x: x[1], reverse=True)
                insights.append(f"**Outliers detected:** {total_outliers:,} across {len(outlier_cols)} columns")
                insights.append(f"  └ Most outliers in: {outlier_cols[0][0]} ({outlier_cols[0][1]:,} outliers)")

            for insight in insights:
                summary_sections.append(f"• {insight}")

            summary_sections.append("")

        # 4. Correlation Insights
        if len(numeric_cols) > 1:
            summary_sections.append("## 🔗 CORRELATION INSIGHTS")

            corr_matrix = df[numeric_cols].corr()
            correlations = []

            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    corr_val = corr_matrix.loc[col1, col2]
                    if not pd.isna(corr_val):
                        correlations.append((col1, col2, corr_val))

            if correlations:
                correlations.sort(key=lambda x: abs(x[2]), reverse=True)

                # Strong correlations
                strong_corr = [c for c in correlations if abs(c[2]) >= 0.7]
                if strong_corr:
                    summary_sections.append(f"• **{len(strong_corr)} strong correlations** found (|r| ≥ 0.7)")
                    top_corr = strong_corr[0]
                    direction = "positive" if top_corr[2] > 0 else "negative"
                    summary_sections.append(
                        f"  └ Strongest: {top_corr[0]} ↔ {top_corr[1]} (r = {top_corr[2]:.3f}, {direction})"
                    )

                # Multicollinearity warning
                very_high_corr = [c for c in correlations if abs(c[2]) >= 0.9]
                if very_high_corr:
                    summary_sections.append(
                        f"⚠️  **Multicollinearity warning:** {len(very_high_corr)} very high correlations (|r| ≥ 0.9)"
                    )

                if not strong_corr:
                    summary_sections.append("• No strong correlations detected between numeric variables")

            summary_sections.append("")

        # 5. Categorical Insights
        if categorical_cols:
            summary_sections.append("## 🏷️ CATEGORICAL INSIGHTS")

            cat_insights = []

            for col in categorical_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    unique_count = col_data.nunique()
                    total_count = len(col_data)

                    # High cardinality detection
                    if unique_count > total_count * 0.5:
                        cat_insights.append(f"**High cardinality:** {col} ({unique_count:,} unique values)")

                    # Dominant category detection
                    value_counts = col_data.value_counts()
                    if len(value_counts) > 1:
                        dominant_pct = (value_counts.iloc[0] / len(col_data)) * 100
                        if dominant_pct > 80:
                            cat_insights.append(
                                f"**Dominant category:** {col} - '{value_counts.index[0]}' ({dominant_pct:.1f}%)"
                            )

                    # Rare categories
                    rare_count = sum(1 for count in value_counts if count == 1)
                    if rare_count > 0:
                        rare_pct = (rare_count / len(value_counts)) * 100
                        if rare_pct > 20:
                            cat_insights.append(f"**Many rare categories:** {col} ({rare_count} singleton categories)")

            if cat_insights:
                for insight in cat_insights:
                    summary_sections.append(f"• {insight}")
            else:
                summary_sections.append("• Categorical variables show balanced distributions")

            summary_sections.append("")

        # 6. Key Recommendations
        summary_sections.append("## 💡 KEY RECOMMENDATIONS")

        recommendations = []

        # Data quality recommendations
        if missing_data.sum() > 0:
            high_missing = missing_data[missing_data > len(df) * 0.5]
            if len(high_missing) > 0:
                recommendations.append(f"🚨 **Critical:** Drop {len(high_missing)} columns with >50% missing data")
            else:
                recommendations.append("📋 **Data Quality:** Implement imputation strategy for missing values")

        if duplicates > 0:
            recommendations.append(f"🔄 **Data Cleaning:** Remove {duplicates:,} duplicate records")

        # Statistical recommendations
        if numeric_cols:
            highly_skewed = []
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0 and abs(stats.skew(col_data)) > 2:
                    highly_skewed.append(col)

            if highly_skewed:
                recommendations.append(
                    f"📊 **Statistical:** Apply transformations to {len(highly_skewed)} highly skewed columns"
                )

        # Correlation recommendations
        if len(numeric_cols) > 1:
            very_high_corr = [c for c in correlations if abs(c[2]) >= 0.9]
            if very_high_corr:
                recommendations.append("🔗 **Modeling:** Address multicollinearity before predictive modeling")

        # Feature engineering recommendations
        if categorical_cols:
            high_cardinality = []
            for col in categorical_cols:
                if df[col].nunique() > len(df) * 0.1:
                    high_cardinality.append(col)

            if high_cardinality:
                recommendations.append(
                    f"🛠️ **Feature Engineering:** Consider encoding strategies for {len(high_cardinality)} high-cardinality columns"
                )

        # Analysis recommendations
        if len(numeric_cols) >= 3:
            recommendations.append("🔍 **Analysis:** Consider dimensionality reduction (PCA) or clustering analysis")

        if not recommendations:
            recommendations.append("✅ **Status:** Dataset is analysis-ready with good quality")

        for rec in recommendations:
            summary_sections.append(f"• {rec}")

        summary_sections.append("")

        # 7. Next Steps
        summary_sections.append("## 🚀 SUGGESTED NEXT STEPS")

        next_steps = []

        if quality_score < 75:
            next_steps.append(
                "1. **Data Cleaning:** `/missing <table>` and `/duplicates <table>` for detailed analysis"
            )

        if len(numeric_cols) > 1:
            next_steps.append(
                f"2. **Correlation Analysis:** `/heatmap <{get_object_name_singular()}>` to visualize relationships"
            )

        if len(numeric_cols) >= 3:
            next_steps.append(
                f"3. **Advanced Analysis:** `/clusters <{get_object_name_singular()}>` or `/pca <{get_object_name_singular()}>` for pattern discovery"
            )

        if numeric_cols:
            next_steps.append(
                f"4. **Distribution Analysis:** `/distribution <{get_object_name_singular()}>.<column>` for specific variables"
            )

        if total_outliers > 0:
            next_steps.append("5. **Outlier Investigation:** `/outliers <table>.<column>` for anomaly detection")

        next_steps.append("6. **Detailed Report:** `/report <table>` for comprehensive analysis")

        for step in next_steps:
            summary_sections.append(step)

        summary_sections.append("")
        summary_sections.append("---")
        summary_sections.append(
            "💼 **Business Impact:** This summary provides actionable insights for data-driven decision making."
        )

        # Combine all sections
        full_summary = "\n".join(summary_sections)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, full_summary, MessageType.PYTHON, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating summary: {str(e)}", MessageType.ERROR))


def _correlation_analysis(question, tuple, previous_df):
    """Perform detailed correlation analysis between two columns with p-values and confidence intervals."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        column1_name = tuple["column1"]
        column2_name = tuple["column2"]
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            column1_name = find_closest_column_name(table_name, column1_name)
            column2_name = find_closest_column_name(table_name, column2_name)
            sql = f"SELECT {column1_name}, {column2_name} FROM {table_name} WHERE {column1_name} IS NOT NULL AND {column2_name} IS NOT NULL;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df
            if column1_name not in df.columns:
                add_message(Message(RoleType.ASSISTANT, f"Column '{column1_name}' not found", MessageType.ERROR))
                return
            if column2_name not in df.columns:
                add_message(Message(RoleType.ASSISTANT, f"Column '{column2_name}' not found", MessageType.ERROR))
                return

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for columns '{column1_name}' and '{column2_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Get clean data for both columns
        clean_df = df[[column1_name, column2_name]].dropna()

        if clean_df.empty:
            add_message(Message(RoleType.ASSISTANT, f"No valid data found for correlation analysis", MessageType.ERROR))
            return

        # Check if both columns are numeric
        if not pd.api.types.is_numeric_dtype(clean_df[column1_name]):
            add_message(Message(RoleType.ASSISTANT, f"Column '{column1_name}' is not numeric", MessageType.ERROR))
            return

        if not pd.api.types.is_numeric_dtype(clean_df[column2_name]):
            add_message(Message(RoleType.ASSISTANT, f"Column '{column2_name}' is not numeric", MessageType.ERROR))
            return

        data1 = clean_df[column1_name].values
        data2 = clean_df[column2_name].values

        # Calculate different correlation coefficients
        pearson_corr, pearson_p = stats.pearsonr(data1, data2)
        spearman_corr, spearman_p = stats.spearmanr(data1, data2)
        kendall_corr, kendall_p = stats.kendalltau(data1, data2)

        # Calculate confidence intervals for Pearson correlation
        def correlation_ci(r, n, alpha=0.05):
            """Calculate confidence interval for correlation coefficient using Fisher's z-transformation."""
            if abs(r) >= 1:
                return (r, r)  # Perfect correlation

            # Fisher's z-transformation
            z = 0.5 * np.log((1 + r) / (1 - r))
            z_critical = stats.norm.ppf(1 - alpha / 2)
            z_se = 1 / np.sqrt(n - 3)

            z_lower = z - z_critical * z_se
            z_upper = z + z_critical * z_se

            # Transform back to correlation scale
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

            return (r_lower, r_upper)

        pearson_ci = correlation_ci(pearson_corr, len(data1))

        # Create visualization
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Scatter Plot with Regression Line",
                "Correlation Comparison",
                "Residual Plot",
                "Correlation Statistics",
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
        )

        # Scatter plot with regression line
        fig.add_trace(
            go.Scatter(x=data1, y=data2, mode="markers", name="Data Points", marker=dict(opacity=0.6, size=6)),
            row=1,
            col=1,
        )

        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(data1, data2)
        line_x = np.linspace(data1.min(), data1.max(), 100)
        line_y = slope * line_x + intercept

        fig.add_trace(
            go.Scatter(x=line_x, y=line_y, mode="lines", name="Regression Line", line=dict(color="red", width=2)),
            row=1,
            col=1,
        )

        # Correlation comparison
        correlations = ["Pearson", "Spearman", "Kendall"]
        corr_values = [pearson_corr, spearman_corr, kendall_corr]
        p_values = [pearson_p, spearman_p, kendall_p]

        colors = ["green" if p < 0.05 else "orange" if p < 0.1 else "red" for p in p_values]

        fig.add_trace(
            go.Bar(
                x=correlations,
                y=corr_values,
                name="Correlation Coefficients",
                marker=dict(color=colors),
                text=[f"{val:.3f}" for val in corr_values],
                textposition="auto",
            ),
            row=1,
            col=2,
        )

        # Residual plot
        predicted = slope * data1 + intercept
        residuals = data2 - predicted

        fig.add_trace(
            go.Scatter(x=predicted, y=residuals, mode="markers", name="Residuals", marker=dict(opacity=0.6, size=6)),
            row=2,
            col=1,
        )

        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

        # Statistics table
        def interpret_correlation(r):
            """Interpret correlation strength."""
            abs_r = abs(r)
            if abs_r >= 0.7:
                return "Strong"
            elif abs_r >= 0.3:
                return "Moderate"
            elif abs_r >= 0.1:
                return "Weak"
            else:
                return "Very Weak"

        def significance_level(p):
            """Determine significance level."""
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            elif p < 0.1:
                return "."
            else:
                return ""

        stats_data = {
            "Metric": [
                "Pearson Correlation",
                "Pearson P-value",
                "Pearson 95% CI",
                "Spearman Correlation",
                "Spearman P-value",
                "Kendall Correlation",
                "Kendall P-value",
                "Sample Size",
                "Interpretation",
                "R-squared",
                "Regression Equation",
            ],
            "Value": [
                f"{pearson_corr:.4f} {significance_level(pearson_p)}",
                f"{pearson_p:.4f}",
                f"[{pearson_ci[0]:.4f}, {pearson_ci[1]:.4f}]",
                f"{spearman_corr:.4f} {significance_level(spearman_p)}",
                f"{spearman_p:.4f}",
                f"{kendall_corr:.4f} {significance_level(kendall_p)}",
                f"{kendall_p:.4f}",
                f"{len(data1)}",
                f"{interpret_correlation(pearson_corr)} {'Positive' if pearson_corr > 0 else 'Negative' if pearson_corr < 0 else 'No'} Correlation",
                f"{r_value**2:.4f}",
                f"y = {slope:.4f}x + {intercept:.4f}",
            ],
        }

        fig.add_trace(
            go.Table(header=dict(values=list(stats_data.keys())), cells=dict(values=list(stats_data.values()))),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_xaxes(title_text=column1_name, row=1, col=1)
        fig.update_yaxes(title_text=column2_name, row=1, col=1)
        fig.update_yaxes(title_text="Correlation Coefficient", row=1, col=2)
        fig.update_xaxes(title_text="Predicted Values", row=2, col=1)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)

        fig.update_layout(title=f"Correlation Analysis: {column1_name} vs {column2_name}", height=800, showlegend=False)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))
    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error performing correlation analysis: {str(e)}", MessageType.ERROR))


# def check_length(sql):
#     df, elapsed_time = run_sql_cached(sql)
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


def _analyze_datatypes(question, tuple, previous_df):
    """Analyze and suggest optimal data types for columns."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name} LIMIT 10000;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Analyze each column
        analysis_results = []

        for col in df.columns:
            current_dtype = str(df[col].dtype)
            non_null_data = df[col].dropna()

            if len(non_null_data) == 0:
                analysis_results.append(
                    {
                        "Column": col,
                        "Current Type": current_dtype,
                        "Suggested Type": "object (all null)",
                        "Memory Usage (bytes)": df[col].memory_usage(deep=True),
                        "Reason": "All values are null",
                    }
                )
                continue

            # Memory usage analysis
            current_memory = df[col].memory_usage(deep=True)

            # Analyze the data
            suggested_type = current_dtype
            reason = "Current type is optimal"
            potential_memory = current_memory

            # For object columns, try to optimize
            if df[col].dtype == "object":
                # Try numeric conversion
                try:
                    numeric_converted = pd.to_numeric(non_null_data, errors="coerce")
                    if not numeric_converted.isna().all():
                        # Check if integers
                        if numeric_converted.dropna().apply(lambda x: x.is_integer()).all():
                            min_val = numeric_converted.min()
                            max_val = numeric_converted.max()

                            if min_val >= 0:
                                if max_val <= 255:
                                    suggested_type = "uint8"
                                    potential_memory = len(df) * 1
                                elif max_val <= 65535:
                                    suggested_type = "uint16"
                                    potential_memory = len(df) * 2
                                elif max_val <= 4294967295:
                                    suggested_type = "uint32"
                                    potential_memory = len(df) * 4
                                else:
                                    suggested_type = "uint64"
                                    potential_memory = len(df) * 8
                            else:
                                if min_val >= -128 and max_val <= 127:
                                    suggested_type = "int8"
                                    potential_memory = len(df) * 1
                                elif min_val >= -32768 and max_val <= 32767:
                                    suggested_type = "int16"
                                    potential_memory = len(df) * 2
                                elif min_val >= -2147483648 and max_val <= 2147483647:
                                    suggested_type = "int32"
                                    potential_memory = len(df) * 4
                                else:
                                    suggested_type = "int64"
                                    potential_memory = len(df) * 8
                            reason = f"Contains only integers in range [{min_val}, {max_val}]"
                        else:
                            suggested_type = "float32" if abs(numeric_converted).max() < 3.4e38 else "float64"
                            potential_memory = len(df) * (4 if suggested_type == "float32" else 8)
                            reason = "Contains floating point numbers"
                except:
                    pass

                # Try datetime conversion
                if suggested_type == current_dtype:
                    try:
                        date_converted = pd.to_datetime(non_null_data, errors="coerce")
                        if not date_converted.isna().all():
                            suggested_type = "datetime64[ns]"
                            potential_memory = len(df) * 8
                            reason = "Contains datetime-like values"
                    except:
                        pass

                # Try boolean conversion
                if suggested_type == current_dtype:
                    unique_vals = set(str(v).lower() for v in non_null_data.unique())
                    bool_vals = {"true", "false", "1", "0", "yes", "no", "t", "f", "y", "n"}
                    if unique_vals.issubset(bool_vals) and len(unique_vals) <= 2:
                        suggested_type = "bool"
                        potential_memory = len(df) * 1
                        reason = "Contains only boolean-like values"

                # Check for categorical
                if suggested_type == current_dtype:
                    unique_count = non_null_data.nunique()
                    total_count = len(non_null_data)
                    if unique_count < total_count * 0.5:  # Less than 50% unique values
                        categorical_memory = unique_count * 8 + len(df) * 4  # Rough estimate
                        if categorical_memory < current_memory:
                            suggested_type = "category"
                            potential_memory = categorical_memory
                            reason = f"Only {unique_count} unique values out of {total_count} ({unique_count / total_count * 100:.1f}%)"

            # For numeric columns, check if we can downcast
            elif df[col].dtype in ["int64", "float64"]:
                if df[col].dtype == "int64":
                    min_val = non_null_data.min()
                    max_val = non_null_data.max()

                    if min_val >= 0:
                        if max_val <= 255:
                            suggested_type = "uint8"
                            potential_memory = len(df) * 1
                        elif max_val <= 65535:
                            suggested_type = "uint16"
                            potential_memory = len(df) * 2
                        elif max_val <= 4294967295:
                            suggested_type = "uint32"
                            potential_memory = len(df) * 4
                    else:
                        if min_val >= -128 and max_val <= 127:
                            suggested_type = "int8"
                            potential_memory = len(df) * 1
                        elif min_val >= -32768 and max_val <= 32767:
                            suggested_type = "int16"
                            potential_memory = len(df) * 2
                        elif min_val >= -2147483648 and max_val <= 2147483647:
                            suggested_type = "int32"
                            potential_memory = len(df) * 4

                    if suggested_type != current_dtype:
                        reason = f"Integer values fit in smaller type range [{min_val}, {max_val}]"

                elif df[col].dtype == "float64":
                    # Check if all values can fit in float32
                    if abs(non_null_data).max() < 3.4e38:
                        suggested_type = "float32"
                        potential_memory = len(df) * 4
                        reason = "Values fit in float32 precision"

            memory_savings = current_memory - potential_memory
            savings_pct = (memory_savings / current_memory * 100) if current_memory > 0 else 0

            analysis_results.append(
                {
                    "Column": col,
                    "Current Type": current_dtype,
                    "Suggested Type": suggested_type,
                    "Current Memory (bytes)": current_memory,
                    "Optimized Memory (bytes)": potential_memory,
                    "Memory Savings (bytes)": memory_savings,
                    "Savings %": f"{savings_pct:.1f}%",
                    "Reason": reason,
                }
            )

        # Create results DataFrame
        results_df = pd.DataFrame(analysis_results)

        # Create visualization
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Memory Usage by Column",
                "Potential Memory Savings",
                "Data Type Distribution",
                "Optimization Opportunities",
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"type": "pie"}, {"type": "table"}]],
        )

        # Memory usage comparison
        cols_for_plot = results_df["Column"].tolist()
        current_memory = results_df["Current Memory (bytes)"].tolist()
        optimized_memory = results_df["Optimized Memory (bytes)"].tolist()

        fig.add_trace(
            go.Bar(name="Current", x=cols_for_plot, y=current_memory, marker_color="lightcoral"), row=1, col=1
        )
        fig.add_trace(
            go.Bar(name="Optimized", x=cols_for_plot, y=optimized_memory, marker_color="lightgreen"), row=1, col=1
        )

        # Memory savings
        savings = results_df["Memory Savings (bytes)"].tolist()
        fig.add_trace(go.Bar(x=cols_for_plot, y=savings, marker_color="gold", name="Savings"), row=1, col=2)

        # Data type distribution
        type_counts = results_df["Current Type"].value_counts()
        fig.add_trace(go.Pie(labels=type_counts.index, values=type_counts.values, name="Data Types"), row=2, col=1)

        # Summary table
        summary_data = results_df[["Column", "Current Type", "Suggested Type", "Savings %", "Reason"]]
        fig.add_trace(
            go.Table(
                header=dict(values=list(summary_data.columns), fill_color="paleturquoise", align="left"),
                cells=dict(
                    values=[summary_data[col] for col in summary_data.columns], fill_color="lavender", align="left"
                ),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text=f"Data Type Analysis - {table_name}", showlegend=True, height=800, template="plotly_white"
        )

        # Summary statistics
        total_current = results_df["Current Memory (bytes)"].sum()
        total_optimized = results_df["Optimized Memory (bytes)"].sum()
        total_savings = total_current - total_optimized
        total_savings_pct = (total_savings / total_current * 100) if total_current > 0 else 0

        summary_text = f"""
        📊 **Data Type Analysis Summary**
        
        • **Total Columns Analyzed:** {len(results_df)}
        • **Current Memory Usage:** {total_current:,.0f} bytes ({total_current / 1024 / 1024:.2f} MB)
        • **Optimized Memory Usage:** {total_optimized:,.0f} bytes ({total_optimized / 1024 / 1024:.2f} MB)
        • **Potential Savings:** {total_savings:,.0f} bytes ({total_savings_pct:.1f}%)
        
        **Optimization Recommendations:**
        """

        optimizable = results_df[results_df["Memory Savings (bytes)"] > 0]
        if len(optimizable) > 0:
            summary_text += f"\n• {len(optimizable)} columns can be optimized"
            for _, row in optimizable.head(5).iterrows():
                summary_text += f"\n  - {row['Column']}: {row['Current Type']} → {row['Suggested Type']} ({row['Savings %']} savings)"
        else:
            summary_text += "\n• All columns are already optimally typed"

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, summary_text, MessageType.TEXT))
        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error analyzing data types: {str(e)}", MessageType.ERROR))


def _violin_plot(question, tuple, previous_df):
    """Generate violin plots showing distribution shapes."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        column_name = tuple["column"]
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            column_name = find_closest_column_name(table_name, column_name)
            sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df
            if column_name not in df.columns:
                column_name = find_closest_column_name_from_list(df.columns.tolist(), column_name)

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        if column_name not in df.columns:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"Column '{column_name}' not found in {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Prepare data
        data = df[column_name].dropna()

        if len(data) == 0:
            add_message(
                Message(RoleType.ASSISTANT, f"No non-null data found in column '{column_name}'", MessageType.ERROR)
            )
            return

        # Check if numeric
        if not pd.api.types.is_numeric_dtype(data):
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"Column '{column_name}' is not numeric. Violin plots require numeric data.",
                    MessageType.ERROR,
                )
            )
            return

        # Create violin plot with additional statistical overlays
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Violin Plot - {column_name}",
                "Box Plot Comparison",
                "Distribution Statistics",
                "Kernel Density Estimation",
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"type": "table"}, {"secondary_y": False}]],
        )

        # Main violin plot
        fig.add_trace(
            go.Violin(
                y=data,
                name=column_name,
                box_visible=True,
                meanline_visible=True,
                fillcolor="lightblue",
                opacity=0.6,
                line_color="black",
            ),
            row=1,
            col=1,
        )

        # Box plot for comparison
        fig.add_trace(go.Box(y=data, name=column_name, fillcolor="lightgreen", line_color="darkgreen"), row=1, col=2)

        # Statistical summary
        stats = data.describe()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        # Additional statistics
        from scipy import stats as scipy_stats

        skewness = scipy_stats.skew(data)
        kurtosis = scipy_stats.kurtosis(data)

        # Shapiro-Wilk test for normality (if sample size is reasonable)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = scipy_stats.shapiro(data)
        else:
            # Use Kolmogorov-Smirnov test for larger samples
            ks_stat, shapiro_p = scipy_stats.kstest(data, "norm", args=(data.mean(), data.std()))
            shapiro_stat = ks_stat

        stats_data = {
            "Statistic": [
                "Count",
                "Mean",
                "Std Dev",
                "Min",
                "Q1 (25%)",
                "Median (50%)",
                "Q3 (75%)",
                "Max",
                "IQR",
                "Skewness",
                "Kurtosis",
                "Normality Test Stat",
                "Normality P-value",
                "Distribution Shape",
            ],
            "Value": [
                f"{len(data):,.0f}",
                f"{data.mean():.4f}",
                f"{data.std():.4f}",
                f"{data.min():.4f}",
                f"{q1:.4f}",
                f"{data.median():.4f}",
                f"{q3:.4f}",
                f"{data.max():.4f}",
                f"{iqr:.4f}",
                f"{skewness:.4f}",
                f"{kurtosis:.4f}",
                f"{shapiro_stat:.4f}",
                f"{shapiro_p:.4e}",
                "Right-skewed" if skewness > 0.5 else "Left-skewed" if skewness < -0.5 else "Approximately symmetric",
            ],
        }

        fig.add_trace(
            go.Table(
                header=dict(values=list(stats_data.keys()), fill_color="paleturquoise", align="left"),
                cells=dict(values=list(stats_data.values()), fill_color="lavender", align="left"),
            ),
            row=2,
            col=1,
        )

        # KDE plot
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            kde_values = kde(x_range)

            fig.add_trace(
                go.Scatter(x=x_range, y=kde_values, mode="lines", fill="tozeroy", name="KDE", line_color="purple"),
                row=2,
                col=2,
            )
        except Exception:
            # Fallback to histogram if KDE fails
            fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Histogram", opacity=0.7), row=2, col=2)

        # Update layout
        fig.update_layout(
            title_text=f"Violin Plot Analysis - {column_name}", showlegend=False, height=800, template="plotly_white"
        )

        # Interpretation
        interpretation = f"""
        🎻 **Violin Plot Analysis for {column_name}**
        
        **Distribution Shape:** {stats_data["Value"][13]}
        **Skewness:** {skewness:.3f} ({"Right" if skewness > 0 else "Left"} skewed)
        **Kurtosis:** {kurtosis:.3f} ({"Heavy" if kurtosis > 0 else "Light"} tails)
        **Normality:** {"Likely normal" if shapiro_p > 0.05 else "Not normal"} (p={shapiro_p:.3e})
        
        **Key Insights:**
        • The violin plot shows the full distribution shape, including multi-modality
        • Box plot overlay highlights quartiles and outliers
        • Width of violin indicates density of values at each level
        """

        if abs(skewness) > 1:
            interpretation += f"\n• **High skewness** ({skewness:.2f}) indicates significant asymmetry"

        if abs(kurtosis) > 1:
            interpretation += (
                f"\n• **Excess kurtosis** ({kurtosis:.2f}) indicates {'heavy' if kurtosis > 0 else 'light'} tails"
            )

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, interpretation, MessageType.TEXT))
        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating violin plot: {str(e)}", MessageType.ERROR))


def _anomaly_detection(question, tuple, previous_df):
    """Detect anomalies using statistical and ML methods."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        column_name = tuple["column"]
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            column_name = find_closest_column_name(table_name, column_name)
            sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df
            if column_name not in df.columns:
                column_name = find_closest_column_name_from_list(df.columns.tolist(), column_name)

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        if column_name not in df.columns:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"Column '{column_name}' not found in {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Prepare data
        data = df[column_name].dropna()

        if len(data) == 0:
            add_message(
                Message(RoleType.ASSISTANT, f"No non-null data found in column '{column_name}'", MessageType.ERROR)
            )
            return

        # Check if numeric
        if not pd.api.types.is_numeric_dtype(data):
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"Column '{column_name}' is not numeric. Anomaly detection requires numeric data.",
                    MessageType.ERROR,
                )
            )
            return

        if len(data) < 10:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"Insufficient data points ({len(data)}) for reliable anomaly detection.",
                    MessageType.ERROR,
                )
            )
            return

        # Method 1: Z-Score (statistical)
        z_scores = np.abs((data - data.mean()) / data.std())
        z_threshold = 3
        z_anomalies = data[z_scores > z_threshold]

        # Method 2: IQR Method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_anomalies = data[(data < iqr_lower) | (data > iqr_upper)]

        # Method 3: Modified Z-Score (using median)
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        modified_z_threshold = 3.5
        modified_z_anomalies = data[np.abs(modified_z_scores) > modified_z_threshold]

        # Method 4: Isolation Forest (ML method)
        try:
            from sklearn.ensemble import IsolationForest

            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            data_reshaped = data.values.reshape(-1, 1)
            isolation_predictions = isolation_forest.fit_predict(data_reshaped)
            isolation_anomalies = data[isolation_predictions == -1]
        except ImportError:
            isolation_anomalies = pd.Series([], dtype=data.dtype)

        # Method 5: Local Outlier Factor
        try:
            from sklearn.neighbors import LocalOutlierFactor

            lof = LocalOutlierFactor(n_neighbors=min(20, len(data) // 2), contamination=0.1)
            lof_predictions = lof.fit_predict(data_reshaped)
            lof_anomalies = data[lof_predictions == -1]
        except (ImportError, NameError):
            lof_anomalies = pd.Series([], dtype=data.dtype)

        # Create visualization
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Data Distribution with Anomalies",
                "Anomaly Detection Methods Comparison",
                "Z-Score Analysis",
                "IQR Method Visualization",
                "Anomaly Summary Statistics",
                "Method Effectiveness",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"type": "table"}, {"type": "bar"}],
            ],
        )

        # 1. Main distribution with anomalies highlighted
        fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Normal Data", opacity=0.7), row=1, col=1)

        # Add anomaly markers
        if len(z_anomalies) > 0:
            fig.add_trace(
                go.Scatter(
                    x=z_anomalies,
                    y=[1] * len(z_anomalies),
                    mode="markers",
                    name="Z-Score Anomalies",
                    marker=dict(color="red", size=10, symbol="x"),
                ),
                row=1,
                col=1,
            )

        # 2. Box plot with all methods
        fig.add_trace(go.Box(y=data, name="All Data", fillcolor="lightblue"), row=1, col=2)

        # 3. Z-score plot
        fig.add_trace(
            go.Scatter(
                x=list(range(len(data))), y=z_scores, mode="markers", name="Z-Scores", marker=dict(color="blue")
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=z_threshold, line_dash="dash", line_color="red", annotation_text="Z-Score Threshold (3)", row=2, col=1
        )

        # 4. IQR visualization
        fig.add_trace(
            go.Scatter(
                x=list(range(len(data))), y=data, mode="markers", name="Data Points", marker=dict(color="green")
            ),
            row=2,
            col=2,
        )
        fig.add_hline(y=iqr_upper, line_dash="dash", line_color="red", annotation_text="Upper Fence", row=2, col=2)
        fig.add_hline(y=iqr_lower, line_dash="dash", line_color="red", annotation_text="Lower Fence", row=2, col=2)

        # 5. Summary statistics table
        methods_summary = {
            "Method": [
                "Z-Score (μ±3σ)",
                "IQR (Q1-1.5*IQR, Q3+1.5*IQR)",
                "Modified Z-Score",
                "Isolation Forest",
                "Local Outlier Factor",
            ],
            "Anomalies Found": [
                len(z_anomalies),
                len(iqr_anomalies),
                len(modified_z_anomalies),
                len(isolation_anomalies),
                len(lof_anomalies),
            ],
            "Percentage": [
                f"{len(z_anomalies) / len(data) * 100:.2f}%",
                f"{len(iqr_anomalies) / len(data) * 100:.2f}%",
                f"{len(modified_z_anomalies) / len(data) * 100:.2f}%",
                f"{len(isolation_anomalies) / len(data) * 100:.2f}%",
                f"{len(lof_anomalies) / len(data) * 100:.2f}%",
            ],
        }

        fig.add_trace(
            go.Table(
                header=dict(values=list(methods_summary.keys()), fill_color="paleturquoise", align="left"),
                cells=dict(values=list(methods_summary.values()), fill_color="lavender", align="left"),
            ),
            row=3,
            col=1,
        )

        # 6. Method comparison bar chart
        fig.add_trace(
            go.Bar(
                x=methods_summary["Method"],
                y=methods_summary["Anomalies Found"],
                name="Anomalies Count",
                marker_color="coral",
            ),
            row=3,
            col=2,
        )

        fig.update_layout(
            title_text=f"Anomaly Detection Analysis - {column_name}",
            showlegend=True,
            height=1000,
            template="plotly_white",
        )

        # Consensus anomalies (detected by multiple methods)
        all_anomalies = set()
        if len(z_anomalies) > 0:
            all_anomalies.update(z_anomalies.index)
        if len(iqr_anomalies) > 0:
            all_anomalies.update(iqr_anomalies.index)
        if len(modified_z_anomalies) > 0:
            all_anomalies.update(modified_z_anomalies.index)
        if len(isolation_anomalies) > 0:
            all_anomalies.update(isolation_anomalies.index)
        if len(lof_anomalies) > 0:
            all_anomalies.update(lof_anomalies.index)

        # Create detailed report
        report = f"""
        🚨 **Anomaly Detection Report for {column_name}**
        
        **Dataset Summary:**
        • Total data points: {len(data):,}
        • Mean: {data.mean():.4f}
        • Standard deviation: {data.std():.4f}
        • Range: [{data.min():.4f}, {data.max():.4f}]
        
        **Method Results:**
        """

        for i, method in enumerate(methods_summary["Method"]):
            count = methods_summary["Anomalies Found"][i]
            pct = methods_summary["Percentage"][i]
            report += f"\n• **{method}**: {count} anomalies ({pct})"

        if len(all_anomalies) > 0:
            consensus_count = len(all_anomalies)
            report += f"\n\n**Consensus Analysis:**\n• {consensus_count} unique data points flagged as anomalies"
            report += f"\n• This represents {consensus_count / len(data) * 100:.2f}% of the dataset"

            # Show some example anomalies
            example_anomalies = list(all_anomalies)[:5]
            report += f"\n\n**Example Anomalous Values:**"
            for idx in example_anomalies:
                if idx < len(data):
                    value = data.iloc[idx] if hasattr(data, "iloc") else data[idx]
                    report += f"\n• Index {idx}: {value:.4f}"
        else:
            report += f"\n\n**No significant anomalies detected** by any method."

        report += f"\n\n**Recommendations:**"
        if len(all_anomalies) / len(data) > 0.1:
            report += f"\n• High anomaly rate (>{len(all_anomalies) / len(data) * 100:.1f}%) - investigate data quality"
        if len(z_anomalies) != len(iqr_anomalies):
            report += f"\n• Methods show different results - consider domain knowledge for validation"
        report += f"\n• Use multiple methods for robust anomaly detection"
        report += f"\n• Investigate flagged points for data entry errors or genuine outliers"

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, report, MessageType.TEXT))
        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error in anomaly detection: {str(e)}", MessageType.ERROR))


def _smart_sample(question, tuple, previous_df):
    """Generate smart samples with stratification options."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        percentage = float(tuple["percentage"])
        sql = ""

        if percentage <= 0 or percentage > 100:
            add_message(Message(RoleType.ASSISTANT, "Percentage must be between 0 and 100", MessageType.ERROR))
            return

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            sql = f"SELECT * FROM {table_name};"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        sample_size = int(len(df) * percentage / 100)
        if sample_size == 0:
            sample_size = 1

        # Analyze data for stratification opportunities
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        sampling_methods = {}

        # Method 1: Simple Random Sampling
        simple_sample = df.sample(n=sample_size, random_state=42)
        sampling_methods["Simple Random"] = simple_sample

        # Method 2: Systematic Sampling
        step = len(df) // sample_size
        if step > 0:
            systematic_indices = list(range(0, len(df), step))[:sample_size]
            systematic_sample = df.iloc[systematic_indices]
            sampling_methods["Systematic"] = systematic_sample

        # Method 3: Stratified Sampling (if categorical columns exist)
        if categorical_cols:
            best_stratify_col = None
            best_stratify_sample = None
            min_groups = 2

            for col in categorical_cols:
                value_counts = df[col].value_counts()
                if len(value_counts) >= min_groups and len(value_counts) <= 20:  # Reasonable number of groups
                    try:
                        stratified_sample = (
                            df.groupby(col, group_keys=False)
                            .apply(lambda x: x.sample(n=max(1, int(len(x) * percentage / 100)), random_state=42))
                            .reset_index(drop=True)
                        )

                        if len(stratified_sample) > 0:
                            best_stratify_col = col
                            best_stratify_sample = stratified_sample
                            break
                    except:
                        continue

            if best_stratify_sample is not None:
                sampling_methods[f"Stratified ({best_stratify_col})"] = best_stratify_sample

        # Method 4: Cluster Sampling (using numeric features if available)
        if len(numeric_cols) >= 2:
            try:
                from sklearn.cluster import KMeans

                # Use first 2 numeric columns for clustering
                cluster_data = df[numeric_cols[:2]].dropna()
                if len(cluster_data) > 10:
                    n_clusters = min(10, max(2, sample_size // 5))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(cluster_data)

                    # Add cluster labels to dataframe
                    df_with_clusters = df.copy()
                    df_with_clusters["_cluster"] = -1
                    df_with_clusters.loc[cluster_data.index, "_cluster"] = cluster_labels

                    # Sample from each cluster
                    cluster_sample = (
                        df_with_clusters.groupby("_cluster", group_keys=False)
                        .apply(lambda x: x.sample(n=max(1, len(x) // n_clusters), random_state=42))
                        .drop("_cluster", axis=1)
                        .reset_index(drop=True)
                    )

                    if len(cluster_sample) > 0:
                        sampling_methods["Cluster-based"] = cluster_sample.head(sample_size)
            except ImportError:
                pass

        # Method 5: Balanced Sampling (for imbalanced datasets)
        if categorical_cols:
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                if len(value_counts) >= 2 and value_counts.min() / value_counts.max() < 0.3:  # Imbalanced
                    try:
                        min_samples_per_class = max(1, sample_size // len(value_counts))
                        balanced_sample = (
                            df.groupby(col, group_keys=False)
                            .apply(lambda x: x.sample(n=min(len(x), min_samples_per_class), random_state=42))
                            .reset_index(drop=True)
                        )

                        if len(balanced_sample) > 0:
                            sampling_methods[f"Balanced ({col})"] = balanced_sample
                            break
                    except:
                        continue

        # Create comparison visualization
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Sampling Methods Comparison",
                "Sample Size Distribution",
                "Method Effectiveness",
                "Sample Statistics",
            ),
            specs=[[{"type": "bar"}, {"type": "pie"}], [{"type": "table"}, {"type": "bar"}]],
        )

        # Sample sizes comparison
        method_names = list(sampling_methods.keys())
        sample_sizes = [len(sample) for sample in sampling_methods.values()]

        fig.add_trace(
            go.Bar(x=method_names, y=sample_sizes, name="Sample Sizes", marker_color="lightblue"), row=1, col=1
        )

        # Sample size distribution pie chart
        fig.add_trace(go.Pie(labels=method_names, values=sample_sizes, name="Sample Distribution"), row=1, col=2)

        # Method comparison table
        method_stats = []
        for method_name, sample_df in sampling_methods.items():
            stats = {
                "Method": method_name,
                "Sample Size": len(sample_df),
                "Coverage %": f"{len(sample_df) / len(df) * 100:.2f}%",
                "Efficiency": "High"
                if len(sample_df) >= sample_size * 0.8
                else "Medium"
                if len(sample_df) >= sample_size * 0.5
                else "Low",
            }

            # Add representativeness score if categorical columns exist
            if categorical_cols:
                representativeness_scores = []
                for col in categorical_cols[:3]:  # Check up to 3 categorical columns
                    if col in sample_df.columns:
                        original_dist = df[col].value_counts(normalize=True).sort_index()
                        sample_dist = (
                            sample_df[col].value_counts(normalize=True).reindex(original_dist.index, fill_value=0)
                        )
                        # Calculate Chi-square-like similarity score
                        score = 1 - np.sum(np.abs(original_dist - sample_dist)) / 2
                        representativeness_scores.append(score)

                if representativeness_scores:
                    avg_repr = np.mean(representativeness_scores)
                    stats["Representativeness"] = f"{avg_repr:.3f}"
                else:
                    stats["Representativeness"] = "N/A"
            else:
                stats["Representativeness"] = "N/A"

            method_stats.append(stats)

        method_df = pd.DataFrame(method_stats)

        fig.add_trace(
            go.Table(
                header=dict(values=list(method_df.columns), fill_color="paleturquoise", align="left"),
                cells=dict(values=[method_df[col] for col in method_df.columns], fill_color="lavender", align="left"),
            ),
            row=2,
            col=1,
        )

        # Representativeness scores (if available)
        if "Representativeness" in method_df.columns and method_df["Representativeness"].str.contains(
            "N/A"
        ).sum() < len(method_df):
            repr_scores = []
            for score in method_df["Representativeness"]:
                try:
                    repr_scores.append(float(score))
                except:
                    repr_scores.append(0)

            fig.add_trace(
                go.Bar(x=method_names, y=repr_scores, name="Representativeness Score", marker_color="coral"),
                row=2,
                col=2,
            )

        fig.update_layout(
            title_text=f"Smart Sampling Analysis - {table_name} ({percentage}% sample)",
            showlegend=False,
            height=800,
            template="plotly_white",
        )

        # Generate comprehensive report
        report = f"""
        📊 **Smart Sampling Report**
        
        **Original Dataset:**
        • Total records: {len(df):,}
        • Target sample size: {sample_size:,} ({percentage}%)
        • Columns: {len(df.columns)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)
        
        **Sampling Methods Applied:**
        """

        best_method = None
        best_score = 0

        for i, (method_name, sample_df) in enumerate(sampling_methods.items()):
            actual_size = len(sample_df)
            coverage = actual_size / len(df) * 100

            report += f"\n• **{method_name}**: {actual_size:,} records ({coverage:.2f}% coverage)"

            # Calculate overall score
            size_score = min(1, actual_size / sample_size)
            efficiency_score = (
                1 if actual_size >= sample_size * 0.8 else 0.7 if actual_size >= sample_size * 0.5 else 0.3
            )

            try:
                repr_score = (
                    float(method_df.iloc[i]["Representativeness"])
                    if method_df.iloc[i]["Representativeness"] != "N/A"
                    else 0.5
                )
            except:
                repr_score = 0.5

            overall_score = (size_score + efficiency_score + repr_score) / 3

            if overall_score > best_score:
                best_score = overall_score
                best_method = method_name

        report += f"\n\n**Recommendation:**"
        if best_method:
            report += f"\n• **Best method: {best_method}** (score: {best_score:.3f})"
            best_sample = sampling_methods[best_method]
            report += f"\n• Recommended sample size: {len(best_sample):,} records"

            # Provide usage advice
            if "Stratified" in best_method:
                report += f"\n• Maintains proportional representation across categories"
            elif "Balanced" in best_method:
                report += f"\n• Provides balanced representation for imbalanced classes"
            elif "Cluster" in best_method:
                report += f"\n• Captures data structure through clustering"
            elif best_method == "Systematic":
                report += f"\n• Ensures even coverage across the dataset"
            else:
                report += f"\n• Simple random sampling provides unbiased selection"

        report += f"\n\n**Implementation Note:**"
        report += f"\n• The recommended sample has been prepared and can be used for analysis"
        report += f"\n• Consider your specific use case when choosing the sampling method"
        report += f"\n• For ML training, stratified or balanced sampling often works best"
        report += f"\n• For general analysis, systematic or simple random sampling is sufficient"

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Return the best sample as the result
        if best_method and best_method in sampling_methods:
            result_df = sampling_methods[best_method]
            add_message(Message(RoleType.ASSISTANT, report, MessageType.TEXT))
            add_message(
                Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, result_df, elapsed_time)
            )
        else:
            add_message(Message(RoleType.ASSISTANT, report, MessageType.TEXT))
            add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error in smart sampling: {str(e)}", MessageType.ERROR))


def _transform_data(question, tuple, previous_df):
    """Apply data transformation operations (log, sqrt, normalize, etc.)."""
    try:
        start_time = time.perf_counter()
        table_name = "Previous Result"
        column_name = tuple["column"]
        operation = tuple["operation"].lower()
        sql = ""

        if previous_df is None:
            table_name = find_closest_object_name(tuple["table"])
            column_name = find_closest_column_name(table_name, column_name)
            sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL;"
            df, elapsed_time = run_sql_cached(sql)
        else:
            df = previous_df
            if column_name not in df.columns:
                column_name = find_closest_column_name_from_list(df.columns.tolist(), column_name)

        if df is None or df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"No data found for {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        if column_name not in df.columns:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    f"Column '{column_name}' not found in {get_object_name_singular()} '{table_name}'",
                    MessageType.ERROR,
                )
            )
            return

        # Prepare original data
        original_data = df[column_name].dropna()

        if len(original_data) == 0:
            add_message(
                Message(RoleType.ASSISTANT, f"No non-null data found in column '{column_name}'", MessageType.ERROR)
            )
            return

        # Check if numeric for most operations
        is_numeric = pd.api.types.is_numeric_dtype(original_data)

        # Available transformations
        transformations = {}
        transformation_info = {}

        # Numeric transformations
        if is_numeric:
            # Log transformation (natural log)
            if operation in ["log", "ln", "natural_log"]:
                if (original_data > 0).all():
                    transformations["Log (ln)"] = np.log(original_data)
                    transformation_info["Log (ln)"] = "Natural logarithm - reduces right skewness"
                else:
                    # Log1p for data with zeros
                    transformations["Log1p (ln(x+1))"] = np.log1p(original_data)
                    transformation_info["Log1p (ln(x+1))"] = "Log(x+1) - handles zeros and negative values"

            # Log10 transformation
            if operation in ["log10", "log_10"]:
                if (original_data > 0).all():
                    transformations["Log10"] = np.log10(original_data)
                    transformation_info["Log10"] = "Base-10 logarithm - interpretable scale reduction"
                else:
                    transformations["Log10 (with offset)"] = np.log10(original_data + 1 - original_data.min())
                    transformation_info["Log10 (with offset)"] = "Log10 with offset to handle non-positive values"

            # Square root transformation
            if operation in ["sqrt", "square_root"]:
                if (original_data >= 0).all():
                    transformations["Square Root"] = np.sqrt(original_data)
                    transformation_info["Square Root"] = "Square root - moderate skewness reduction"
                else:
                    transformations["Square Root (shifted)"] = np.sqrt(original_data - original_data.min())
                    transformation_info["Square Root (shifted)"] = "Square root with shift for negative values"

            # Power transformations
            if operation in ["square", "power2"]:
                transformations["Square (x²)"] = np.power(original_data, 2)
                transformation_info["Square (x²)"] = "Square transformation - increases right skewness"

            if operation in ["cube", "power3"]:
                transformations["Cube (x³)"] = np.power(original_data, 3)
                transformation_info["Cube (x³)"] = "Cube transformation - strong skewness increase"

            # Reciprocal transformation
            if operation in ["reciprocal", "inverse"]:
                if (original_data != 0).all():
                    transformations["Reciprocal (1/x)"] = 1 / original_data
                    transformation_info["Reciprocal (1/x)"] = "Reciprocal - reverses data order and distribution"

            # Normalization transformations
            if operation in ["normalize", "standard", "zscore", "standardize"]:
                transformations["Z-Score Normalization"] = (original_data - original_data.mean()) / original_data.std()
                transformation_info["Z-Score Normalization"] = "Standard normalization - mean=0, std=1"

            if operation in ["minmax", "min_max"]:
                min_val = original_data.min()
                max_val = original_data.max()
                if max_val != min_val:
                    transformations["Min-Max Scaling"] = (original_data - min_val) / (max_val - min_val)
                    transformation_info["Min-Max Scaling"] = "Scale to [0,1] range"

            if operation in ["robust", "robust_scale"]:
                median_val = original_data.median()
                mad = np.median(np.abs(original_data - median_val))
                if mad != 0:
                    transformations["Robust Scaling"] = (original_data - median_val) / mad
                    transformation_info["Robust Scaling"] = "Median and MAD-based scaling - robust to outliers"

            # Box-Cox transformation
            if operation in ["boxcox", "box_cox"]:
                try:
                    from scipy import stats

                    if (original_data > 0).all():
                        transformed_data, lambda_param = stats.boxcox(original_data)
                        transformations[f"Box-Cox (λ={lambda_param:.3f})"] = transformed_data
                        transformation_info[f"Box-Cox (λ={lambda_param:.3f})"] = (
                            f"Box-Cox transformation optimized for normality"
                        )
                except ImportError:
                    transformations["Box-Cox (unavailable)"] = original_data
                    transformation_info["Box-Cox (unavailable)"] = "Requires scipy library"

            # Yeo-Johnson transformation
            if operation in ["yeojohnson", "yeo_johnson"]:
                try:
                    from scipy import stats

                    transformed_data, lambda_param = stats.yeojohnson(original_data)
                    transformations[f"Yeo-Johnson (λ={lambda_param:.3f})"] = transformed_data
                    transformation_info[f"Yeo-Johnson (λ={lambda_param:.3f})"] = "Yeo-Johnson - handles negative values"
                except ImportError:
                    transformations["Yeo-Johnson (unavailable)"] = original_data
                    transformation_info["Yeo-Johnson (unavailable)"] = "Requires scipy library"

        # Text transformations (for string data)
        if not is_numeric:
            if operation in ["upper", "uppercase"]:
                transformations["Uppercase"] = original_data.str.upper()
                transformation_info["Uppercase"] = "Convert all text to uppercase"

            if operation in ["lower", "lowercase"]:
                transformations["Lowercase"] = original_data.str.lower()
                transformation_info["Lowercase"] = "Convert all text to lowercase"

            if operation in ["title", "title_case"]:
                transformations["Title Case"] = original_data.str.title()
                transformation_info["Title Case"] = "Convert to title case"

            if operation in ["length", "len"]:
                transformations["Text Length"] = original_data.str.len()
                transformation_info["Text Length"] = "Length of each text string"
                is_numeric = True  # Result is numeric

        # If operation doesn't match or no transformations found, provide suggestions
        if not transformations:
            available_ops = [
                "log",
                "log10",
                "sqrt",
                "square",
                "cube",
                "reciprocal",
                "normalize",
                "minmax",
                "robust",
                "boxcox",
                "yeojohnson",
            ]
            if not is_numeric:
                available_ops.extend(["upper", "lower", "title", "length"])

            error_msg = f"Operation '{operation}' not recognized or applicable. Available operations: {', '.join(available_ops)}"
            add_message(Message(RoleType.ASSISTANT, error_msg, MessageType.ERROR))
            return

        # Apply all relevant transformations for comparison
        if len(transformations) == 1:
            # If only one transformation matches the operation, apply additional common ones for comparison
            if is_numeric:
                if "Log" not in list(transformations.keys())[0] and (original_data > 0).all():
                    transformations["Log (comparison)"] = np.log(original_data)
                    transformation_info["Log (comparison)"] = "For comparison"

                if "Square Root" not in list(transformations.keys())[0] and (original_data >= 0).all():
                    transformations["Square Root (comparison)"] = np.sqrt(original_data)
                    transformation_info["Square Root (comparison)"] = "For comparison"

                if "Z-Score" not in list(transformations.keys())[0]:
                    transformations["Z-Score (comparison)"] = (
                        original_data - original_data.mean()
                    ) / original_data.std()
                    transformation_info["Z-Score (comparison)"] = "For comparison"

        # Create comprehensive visualization
        n_transforms = len(transformations)
        rows = (n_transforms + 1) // 2 + 1  # +1 for summary row

        fig = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=["Original Distribution"] + list(transformations.keys()) + ["Transformation Summary"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}] for _ in range(rows - 1)]
            + [[{"type": "table", "colspan": 2}, None]],
        )

        # Original distribution
        fig.add_trace(
            go.Histogram(x=original_data, nbinsx=30, name="Original", opacity=0.7, marker_color="lightblue"),
            row=1,
            col=1,
        )

        # Add transformed distributions
        colors = ["lightcoral", "lightgreen", "lightsalmon", "lightpink", "lightgray", "lightyellow"]
        row, col = 1, 2

        transform_stats = []

        for i, (transform_name, transformed_data) in enumerate(transformations.items()):
            if pd.api.types.is_numeric_dtype(transformed_data):
                # Add histogram
                fig.add_trace(
                    go.Histogram(
                        x=transformed_data,
                        nbinsx=30,
                        name=transform_name,
                        opacity=0.7,
                        marker_color=colors[i % len(colors)],
                    ),
                    row=row,
                    col=col,
                )

                # Calculate statistics
                original_skew = original_data.skew() if is_numeric else np.nan
                transformed_skew = transformed_data.skew()
                skew_improvement = abs(original_skew) - abs(transformed_skew) if not np.isnan(original_skew) else np.nan

                transform_stats.append(
                    {
                        "Transformation": transform_name,
                        "Mean": f"{transformed_data.mean():.4f}",
                        "Std Dev": f"{transformed_data.std():.4f}",
                        "Skewness": f"{transformed_skew:.4f}",
                        "Skew Improvement": f"{skew_improvement:.4f}" if not np.isnan(skew_improvement) else "N/A",
                        "Description": transformation_info.get(transform_name, ""),
                    }
                )

            # Move to next subplot
            col += 1
            if col > 2:
                col = 1
                row += 1

        # Add summary table
        if transform_stats:
            stats_df = pd.DataFrame(transform_stats)
            fig.add_trace(
                go.Table(
                    header=dict(values=list(stats_df.columns), fill_color="paleturquoise", align="left"),
                    cells=dict(values=[stats_df[col] for col in stats_df.columns], fill_color="lavender", align="left"),
                ),
                row=rows,
                col=1,
            )

        fig.update_layout(
            title_text=f"Data Transformation Analysis - {column_name}",
            showlegend=False,
            height=200 * rows,
            template="plotly_white",
        )

        # Generate recommendations
        report = f"""
        🔄 **Data Transformation Report for {column_name}**
        
        **Original Data Characteristics:**
        • Data type: {"Numeric" if is_numeric else "Text/Categorical"}
        • Sample size: {len(original_data):,}
        """

        if is_numeric:
            original_skew = original_data.skew()
            original_kurtosis = original_data.kurtosis()

            report += f"""
        • Mean: {original_data.mean():.4f}
        • Standard deviation: {original_data.std():.4f}
        • Skewness: {original_skew:.4f} ({"Right" if original_skew > 0 else "Left"} skewed)
        • Kurtosis: {original_kurtosis:.4f}
        """

        report += f"\n\n**Applied Transformation: {operation.upper()}**"

        if transformations:
            # Find the best transformation based on skewness reduction
            best_transform = None
            best_skew_improvement = -float("inf")

            main_transform_name = list(transformations.keys())[0]  # First one is usually the requested one
            main_transformed = transformations[main_transform_name]

            report += f"\n\n**Primary Result: {main_transform_name}**"
            if pd.api.types.is_numeric_dtype(main_transformed):
                main_skew = main_transformed.skew()
                report += f"\n• New skewness: {main_skew:.4f}"
                if is_numeric:
                    skew_change = abs(original_data.skew()) - abs(main_skew)
                    report += f"\n• Skewness improvement: {skew_change:.4f}"
                    if skew_change > 0:
                        report += " ✅ (Improved)"
                    else:
                        report += " ⚠️ (No improvement)"

            # Provide usage recommendations
            report += f"\n\n**Recommendations:**"
            transform_desc = transformation_info.get(main_transform_name, "")
            if transform_desc:
                report += f"\n• {transform_desc}"

            if is_numeric:
                if abs(original_data.skew()) > 1:
                    report += f"\n• Original data is highly skewed - transformation recommended"
                if main_transform_name.startswith("Log") and (original_data <= 0).any():
                    report += f"\n• ⚠️ Log transformation requires positive values - consider log1p instead"
                if "normalize" in operation.lower():
                    report += f"\n• Normalization is useful for machine learning algorithms"

            # Create result DataFrame with transformed column
            result_df = df.copy()
            result_df[f"{column_name}_transformed"] = main_transformed

            report += f"\n\n**Result:**"
            report += f"\n• Added column: '{column_name}_transformed'"
            report += f"\n• Use this transformed data for further analysis"

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        add_message(Message(RoleType.ASSISTANT, report, MessageType.TEXT))
        add_message(Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, question, result_df, elapsed_time))

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error in data transformation: {str(e)}", MessageType.ERROR))


def _suggestions(question, tuple, previous_df):
    """Generate intelligent suggestions for next analysis steps based on data characteristics."""
    try:
        if previous_df is None or previous_df.empty:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    "No data available for suggestions. Please run a query first.",
                    MessageType.ERROR,
                )
            )
            return

        df = previous_df
        suggestion_commands = []

        # Analyze data characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        # Data shape and quality suggestions
        if len(df) > 1000:
            suggestion_commands.append("📊 Data Quality: /followup sample 10")

        if df.isnull().sum().sum() > 0:
            suggestion_commands.append("🔍 Missing Data Analysis: /followup missing")

        if len(df) != len(df.drop_duplicates()):
            suggestion_commands.append("🔄 Duplicate Analysis: /followup duplicates")

        # Statistical analysis suggestions
        if len(numeric_cols) > 0:
            suggestion_commands.append("📈 Statistical Overview: /followup describe")

            if len(numeric_cols) > 1:
                suggestion_commands.append("🔗 Correlation Analysis: /followup heatmap")

            # Suggest specific column analysis for interesting numeric columns
            for col in numeric_cols[:2]:  # Limit to first 2 columns
                if df[col].nunique() > 10:  # Skip if too few unique values
                    suggestion_commands.append(f"📊 Distribution of {col}: /followup distribution {col}")
                    suggestion_commands.append(f"🎯 Outliers in {col}: /followup outliers {col}")

        # Visualization suggestions
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            suggestion_commands.append(f"📦 Boxplot of {col}: /followup boxplot {col}")

        if len(categorical_cols) > 0:
            suggestion_commands.append("☁️ Text Analysis: /followup wordcloud")

        # Advanced analysis suggestions
        if len(numeric_cols) >= 2:
            suggestion_commands.append("🎯 Clustering Analysis: /followup clusters")
            suggestion_commands.append("📉 PCA Analysis: /followup pca")

        # Data profiling suggestions
        if len(df.columns) > 5:
            suggestion_commands.append("🔍 Data Profiling: /followup profile")

        # Reporting suggestions
        if len(suggestion_commands) > 3:
            suggestion_commands.append("📋 Comprehensive Report: /followup report")
            suggestion_commands.append("📄 Executive Summary: /followup summary")

        # Display suggestions
        if suggestion_commands:
            # Display as text message first
            # add_message(Message(RoleType.ASSISTANT, "🔮 **Suggested Next Steps**\n\nClick any button below to run the suggested analysis:", MessageType.TEXT))

            # Create clickable buttons using FOLLOWUP message type
            add_message(Message(RoleType.ASSISTANT, str(suggestion_commands), MessageType.FOLLOWUP))
        else:
            add_message(
                Message(RoleType.ASSISTANT, "🤔 No specific suggestions available for this dataset.", MessageType.TEXT)
            )

    except Exception as e:
        add_message(Message(RoleType.ASSISTANT, f"Error generating suggestions: {str(e)}", MessageType.ERROR))


FOLLOW_UP_MAGIC_RENDERERS = {
    # ==================== DATA EXPLORATION & BASIC INFO ====================
    r"^head\s+(?P<num_rows>\d+)$": {
        "func": _head,
        "category": "Data Exploration & Basic Info",
    },
    r"^head$": {
        "func": _head,
        "category": "Data Exploration & Basic Info",
    },
    r"^tail\s+(?P<num_rows>\d+)$": {
        "func": _tail,
        "category": "Data Exploration & Basic Info",
    },
    r"^tail$": {
        "func": _tail,
        "category": "Data Exploration & Basic Info",
    },
    r"^describe$": {
        "func": _describe_table,
        "category": "Data Exploration & Basic Info",
    },
    r"^profile$": {
        "func": _profile_table,
        "category": "Data Exploration & Basic Info",
    },
    r"^datatypes$": {
        "func": _analyze_datatypes,
        "category": "Data Exploration & Basic Info",
    },
    # ==================== DATA QUALITY & PREPROCESSING ====================
    r"^missing$": {
        "func": _missing_analysis,
        "category": "Data Quality & Preprocessing",
    },
    r"^duplicates$": {
        "func": _duplicate_analysis,
        "category": "Data Quality & Preprocessing",
    },
    r"^outliers\s+(?P<column>\w+)$": {
        "func": _outlier_detection,
        "category": "Data Quality & Preprocessing",
    },
    r"^anomaly\s+(?P<column>\w+)$": {
        "func": _anomaly_detection,
        "category": "Data Quality & Preprocessing",
    },
    r"^sample\s+(?P<percentage>\d+)$": {
        "func": _smart_sample,
        "category": "Data Quality & Preprocessing",
    },
    r"^transform\s+(?P<column>\w+)\.(?P<operation>\w+)$": {
        "func": _transform_data,
        "category": "Data Quality & Preprocessing",
    },
    # ==================== STATISTICAL ANALYSIS ====================
    r"^distribution\s+(?P<column>\w+)$": {
        "func": _distribution_analysis,
        "category": "Statistical Analysis",
    },
    r"^correlation\s+(?P<column1>\w+)\.(?P<column2>\w+)$": {
        "func": _correlation_analysis,
        "category": "Statistical Analysis",
    },
    # ==================== VISUALIZATIONS ====================
    # Basic Plots
    r"^boxplot\s+(?P<column>\w+)$": {
        "func": _boxplot_visualization,
        "category": "Visualizations",
    },
    r"^violin\s+(?P<column>\w+)$": {
        "func": _violin_plot,
        "category": "Visualizations",
    },
    r"^heatmap$": {
        "func": _generate_heatmap,
        "category": "Visualizations",
    },
    r"^wordcloud$": {
        "func": _generate_wordcloud,
        "category": "Visualizations",
    },
    r"^wordcloud\s+(?P<column>\w+)$": {
        "func": _generate_wordcloud_column,
        "category": "Visualizations",
    },
    # Multi-variable Plots
    r"^pairplot\s+(?P<column>\w+)$": {
        "func": _generate_pairplot,
        "category": "Visualizations",
    },
    r"^scatter\s+(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_scatterplot,
        "category": "Visualizations",
    },
    r"^bar\s+(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_bar,
        "category": "Visualizations",
    },
    r"^line\s+(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_line,
        "category": "Visualizations",
    },
    # ==================== MACHINE LEARNING ====================
    r"^clusters$": {
        "func": _cluster_analysis,
        "category": "Machine Learning",
    },
    r"^pca$": {
        "func": _pca_analysis,
        "category": "Machine Learning",
    },
    # ==================== REPORTING ====================
    r"^report$": {
        "func": _generate_report,
        "category": "Reporting",
    },
    r"^summary$": {
        "func": _generate_summary,
        "category": "Reporting",
    },
    # ==================== SUGGESTIONS ====================
    r"^suggestions$": {
        "func": _suggestions,
        "category": "Help & System Commands",
    },
}

MAGIC_RENDERERS = {
    # ==================== HELP & SYSTEM COMMANDS ====================
    r"^/clear$": {
        "func": _clear,
        "description": "Clear message history in window",
        "sample_values": {},
        "category": "Help & System Commands",
        "show_example": False,
    },
    r"^/help$": {
        "func": _help,
        "description": "Show available magic commands",
        "sample_values": {},
        "category": "Help & System Commands",
        "show_example": False,
    },
    r"^/followuphelp$": {
        "func": _followup_help,
        "description": "Show available follow-up commands for use after queries",
        "sample_values": {},
        "category": "Help & System Commands",
        "show_example": False,
    },
    r"^/followup\s+(?P<command>.+)$": {
        "func": _followup,
        "description": "Ask a follow up question to the previous result set.  Also accepts magic commands ie: heatmap/wordcloud.",
        "sample_values": {"command": "how do these results compare to the national averages?"},
        "category": "Help & System Commands",
        "show_example": True,
    },
    r"^/h\s+(?P<search_text>.+)$": {
        "func": _history_search,
        "description": "Find and recreate a thumbs-up conversation matching your search (≥90% similarity)",
        "sample_values": {"search_text": "diabetes by county"},
        "category": "Help & System Commands",
        "show_example": True,
    },
    # ==================== DATABASE EXPLORATION ====================
    r"^/tables$": {
        "func": _tables,
        "description": "Show all available tables",
        "sample_values": {},
        "category": "Database Exploration",
        "show_example": False,
    },
    r"^/columns\s+(?P<table>.+)$": {
        "func": _columns,
        "description": "Show all available columns on a given table",
        "sample_values": {"table": "wny_health"},
        "category": "Database Exploration",
        "show_example": False,
    },
    r"^/head\s+(?P<table>\w+)\.(?P<num_rows>\d+)$": {
        "func": _head,
        "description": "Show the first {num_rows} rows of a given table",
        "sample_values": {"table": "wny_health", "num_rows": 50},
        "category": "Database Exploration",
        "show_example": False,
    },
    r"^/head\s+(?P<table>.+)$": {
        "func": _head,
        "description": "Show the first 20 rows of a given table",
        "sample_values": {"table": "wny_health"},
        "category": "Database Exploration",
        "show_example": False,
    },
    r"^/tail\s+(?P<table>\w+)\.(?P<num_rows>\d+)$": {
        "func": _tail,
        "description": "Show the last {num_rows} rows of a given table",
        "sample_values": {"table": "wny_health", "num_rows": 50},
        "category": "Database Exploration",
        "show_example": False,
    },
    r"^/tail\s+(?P<table>.+)$": {
        "func": _tail,
        "description": "Show the last 20 rows of a given table",
        "sample_values": {"table": "wny_health"},
        "category": "Database Exploration",
        "show_example": False,
    },
    # ==================== DATA EXPLORATION & BASIC INFO ====================
    r"^/describe\s+(?P<table>\w+)$": {
        "func": _describe_table,
        "description": "Generate comprehensive descriptive statistics for a table",
        "sample_values": {"table": "wny_health"},
        "category": "Data Exploration & Basic Info",
        "show_example": True,
    },
    r"^/profile\s+(?P<table>\w+)$": {
        "func": _profile_table,
        "description": "Generate comprehensive data profiling report",
        "sample_values": {"table": "wny_health"},
        "category": "Data Exploration & Basic Info",
        "show_example": True,
    },
    r"^/datatypes\s+(?P<table>\w+)$": {
        "func": _analyze_datatypes,
        "description": "Analyze and suggest optimal data types for columns",
        "sample_values": {"table": "wny_health"},
        "category": "Data Exploration & Basic Info",
        "show_example": True,
    },
    # ==================== DATA QUALITY & PREPROCESSING ====================
    r"^/missing\s+(?P<table>\w+)$": {
        "func": _missing_analysis,
        "description": "Analyze missing data patterns and create visualizations",
        "sample_values": {"table": "wny_health"},
        "category": "Data Quality & Preprocessing",
        "show_example": True,
    },
    r"^/duplicates\s+(?P<table>\w+)$": {
        "func": _duplicate_analysis,
        "description": "Analyze duplicate rows and values in the dataset",
        "sample_values": {"table": "wny_health"},
        "category": "Data Quality & Preprocessing",
        "show_example": True,
    },
    r"^/outliers\s+(?P<table>\w+)\.(?P<column>\w+)$": {
        "func": _outlier_detection,
        "description": "Detect outliers using multiple statistical methods",
        "sample_values": {"table": "wny_health", "column": "age"},
        "category": "Data Quality & Preprocessing",
        "show_example": True,
    },
    r"^/anomaly\s+(?P<table>\w+)\.(?P<column>\w+)$": {
        "func": _anomaly_detection,
        "description": "Anomaly detection using statistical and ML methods",
        "sample_values": {"table": "wny_health", "column": "age"},
        "category": "Data Quality & Preprocessing",
        "show_example": True,
    },
    r"^/sample\s+(?P<table>\w+)\.(?P<percentage>\d+)$": {
        "func": _smart_sample,
        "description": "Smart sampling with stratification options",
        "sample_values": {"table": "wny_health", "percentage": "10"},
        "category": "Data Quality & Preprocessing",
        "show_example": True,
    },
    r"^/transform\s+(?P<table>\w+)\.(?P<column>\w+)\.(?P<operation>\w+)$": {
        "func": _transform_data,
        "description": "Data transformation operations (log, sqrt, normalize, etc.)",
        "sample_values": {"table": "wny_health", "column": "age", "operation": "log"},
        "category": "Data Quality & Preprocessing",
        "show_example": True,
    },
    # ==================== STATISTICAL ANALYSIS ====================
    r"^/distribution\s+(?P<table>\w+)\.(?P<column>\w+)$": {
        "func": _distribution_analysis,
        "description": "Analyze distribution of a specific column with statistical tests",
        "sample_values": {"table": "wny_health", "column": "age"},
        "category": "Statistical Analysis",
        "show_example": True,
    },
    r"^/correlation\s+(?P<table>\w+)\.(?P<column1>\w+)\.(?P<column2>\w+)$": {
        "func": _correlation_analysis,
        "description": "Detailed correlation analysis with p-values and confidence intervals",
        "sample_values": {"table": "wny_health", "column1": "age", "column2": "zip_code"},
        "category": "Statistical Analysis",
        "show_example": True,
    },
    # ==================== VISUALIZATIONS ====================
    # Basic Single-Variable Plots
    r"^/boxplot\s+(?P<table>\w+)\.(?P<column>\w+)$": {
        "func": _boxplot_visualization,
        "description": "Create box plot with statistical annotations",
        "sample_values": {"table": "wny_health", "column": "obesity"},
        "category": "Visualizations",
        "show_example": True,
    },
    r"^/violin\s+(?P<table>\w+)\.(?P<column>\w+)$": {
        "func": _violin_plot,
        "description": "Violin plots showing distribution shapes",
        "sample_values": {"table": "wny_health", "column": "age"},
        "category": "Visualizations",
        "show_example": True,
    },
    # Multi-Variable & Correlation Plots
    r"^/heatmap\s+(?P<table>\w+)$": {
        "func": _generate_heatmap,
        "description": "Generate a correlation heatmap visualization for a table.",
        "sample_values": {"table": "wny_health"},
        "category": "Visualizations",
        "show_example": True,
    },
    r"^/pairplot\s+(?P<table>\w+)\.(?P<column>\w+)$": {
        "func": _generate_pairplot,
        "description": "Generate a pairplot visualization for a table column.",
        "sample_values": {"table": "wny_health", "column": "county"},
        "category": "Visualizations",
        "show_example": True,
    },
    r"^/scatter\s+(?P<table>\w+)\.(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_scatterplot,
        "description": "Generate a scatterplot visualization for a table x and y axis.",
        "sample_values": {"table": "wny_health", "x": "county", "y": "obesity", "color": "age"},
        "category": "Visualizations",
        "show_example": True,
    },
    r"^/bar\s+(?P<table>\w+)\.(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_bar,
        "description": "Generate a bar chart visualization for a table x and y axis.",
        "sample_values": {"table": "wny_health", "x": "county", "y": "obesity", "color": "age"},
        "category": "Visualizations",
        "show_example": True,
    },
    r"^/line\s+(?P<table>\w+)\.(?P<x>\w+)\.(?P<y>\w+)\.(?P<color>\w+)$": {
        "func": _generate_line,
        "description": "Generate a line chart visualization for a table x and y axis.",
        "sample_values": {"table": "wny_health", "x": "county", "y": "obesity", "color": "age"},
        "category": "Visualizations",
        "show_example": True,
    },
    # Text & Word Analysis
    r"^/wordcloud\s+(?P<table>\w+)$": {
        "func": _generate_wordcloud,
        "description": "Generate a wordcloud visualization for a table.",
        "sample_values": {
            "table": "wny_health",
        },
        "category": "Visualizations",
        "show_example": True,
    },
    r"^/wordcloud\s+(?P<table>\w+)\.(?P<column>\w+)$": {
        "func": _generate_wordcloud_column,
        "description": "Generate a wordcloud visualization for a table column.",
        "sample_values": {"table": "wny_health", "column": "county"},
        "category": "Visualizations",
        "show_example": False,
    },
    # ==================== MACHINE LEARNING ====================
    r"^/clusters\s+(?P<table>\w+)$": {
        "func": _cluster_analysis,
        "description": "Perform K-means clustering analysis with optimal cluster selection",
        "sample_values": {"table": "wny_health"},
        "category": "Machine Learning",
        "show_example": True,
    },
    r"^/pca\s+(?P<table>\w+)$": {
        "func": _pca_analysis,
        "description": "Perform Principal Component Analysis and visualize results",
        "sample_values": {"table": "wny_health"},
        "category": "Machine Learning",
        "show_example": True,
    },
    r"^/confusion\s+(?P<table>\w+)\.(?P<true_column>\w+)\.(?P<pred_column>\w+)$": {
        "func": _confusion_matrix,
        "description": "Generate confusion matrix for classification analysis with detailed metrics",
        "sample_values": {"table": "wny_health", "true_column": "race_level_1", "pred_column": "diabetes"},
        "category": "Machine Learning",
        "show_example": True,
    },
    # ==================== COMPREHENSIVE REPORTING ====================
    r"^/report\s+(?P<table>\w+)$": {
        "func": _generate_report,
        "description": "Generate comprehensive data analysis report",
        "sample_values": {"table": "wny_health"},
        "category": "Comprehensive Reporting",
        "show_example": True,
    },
    r"^/summary\s+(?P<table>\w+)$": {
        "func": _generate_summary,
        "description": "Executive summary of key findings and insights",
        "sample_values": {"table": "wny_health"},
        "category": "Comprehensive Reporting",
        "show_example": True,
    },
    # Add more as needed...
}
