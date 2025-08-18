"""Configuration helper utilities for accessing settings."""

import os
import re
import streamlit as st
import sqlparse
from typing import Union


def get_max_display_rows() -> int:
    """
    Get the maximum number of rows to display and store in DataFrames.
    
    Checks in order:
    1. Environment variable MAX_DISPLAY_ROWS
    2. Streamlit secrets display.max_display_rows
    3. Default value of 1000
    
    Returns:
        int: Maximum number of rows to display
    """
    # Check environment variable first
    env_value = os.getenv('MAX_DISPLAY_ROWS')
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    
    # Check Streamlit secrets
    try:
        if hasattr(st, 'secrets') and 'display' in st.secrets and 'max_display_rows' in st.secrets['display']:
            return int(st.secrets['display']['max_display_rows'])
    except:
        pass
    
    # Default value
    return 1000


def get_max_session_messages() -> int:
    """
    Get the maximum number of messages to keep in session state for performance.
    
    Checks in order:
    1. Environment variable MAX_SESSION_MESSAGES
    2. Streamlit secrets session.max_session_messages
    3. Default value of 20
    
    Returns:
        int: Maximum number of messages to keep in session state
    """
    # Check environment variable first
    env_value = os.getenv('MAX_SESSION_MESSAGES')
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    
    # Check Streamlit secrets
    try:
        if hasattr(st, 'secrets') and 'session' in st.secrets and 'max_session_messages' in st.secrets['session']:
            return int(st.secrets['session']['max_session_messages'])
    except:
        pass
    
    # Default value
    return 20


def ensure_query_has_limit(sql: str, max_rows: Union[int, None] = None) -> str:
    """
    Ensure a SQL query has a LIMIT clause. If it doesn't, add one.
    If it already has a LIMIT clause with a higher value than max_rows, reduce it.
    
    Args:
        sql: SQL query string
        max_rows: Maximum number of rows (if None, uses get_max_display_rows())
    
    Returns:
        str: SQL query with appropriate LIMIT clause
    """
    if not sql or not sql.strip():
        return sql
    
    if max_rows is None:
        max_rows = get_max_display_rows()
    
    # Clean and normalize the SQL
    sql = sql.strip()
    if sql.endswith(';'):
        sql = sql[:-1].strip()
    
    try:
        # Parse the SQL to check for existing LIMIT clause
        parsed = sqlparse.parse(sql)
        if not parsed:
            return sql
        
        statement = parsed[0]
        
        # Look for existing LIMIT clause
        has_limit = False
        existing_limit = None
        
        for token in statement.flatten():
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'LIMIT':
                has_limit = True
            elif has_limit and token.ttype is sqlparse.tokens.Number.Integer:
                existing_limit = int(token.value)
                break
        
        # If there's already a LIMIT clause
        if has_limit and existing_limit is not None:
            if existing_limit <= max_rows:
                # Existing limit is already appropriate
                return sql
            else:
                # Replace the existing limit with our max_rows
                # Use regex to replace the LIMIT value
                limit_pattern = r'\bLIMIT\s+\d+\b'
                return re.sub(limit_pattern, f'LIMIT {max_rows}', sql, flags=re.IGNORECASE)
        
        # No LIMIT clause exists, add one
        return f"{sql} LIMIT {max_rows}"
        
    except Exception as e:
        # If parsing fails, try a simple regex approach
        # Check if LIMIT already exists (case-insensitive)
        limit_match = re.search(r'\bLIMIT\s+(\d+)\b', sql, re.IGNORECASE)
        
        if limit_match:
            existing_limit = int(limit_match.group(1))
            if existing_limit <= max_rows:
                return sql
            else:
                # Replace existing limit
                return re.sub(r'\bLIMIT\s+\d+\b', f'LIMIT {max_rows}', sql, flags=re.IGNORECASE)
        else:
            # No limit exists, add one
            return f"{sql} LIMIT {max_rows}"