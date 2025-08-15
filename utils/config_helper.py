"""Configuration helper utilities for accessing settings."""

import os
import streamlit as st
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


def truncate_dataframe(df, max_rows: Union[int, None] = None) -> tuple:
    """
    Truncate a DataFrame to the maximum allowed rows.
    
    Args:
        df: pandas DataFrame to truncate
        max_rows: Maximum number of rows (if None, uses get_max_display_rows())
    
    Returns:
        tuple: (truncated_dataframe, was_truncated: bool, original_row_count: int)
    """
    if df is None:
        return df, False, 0
        
    if max_rows is None:
        max_rows = get_max_display_rows()
    
    original_count = len(df)
    
    if original_count <= max_rows:
        return df, False, original_count
    
    truncated_df = df.head(max_rows).copy()
    return truncated_df, True, original_count


def apply_dataframe_limit(df):
    """
    Apply the configured row limit to a DataFrame, without showing warnings.
    This is for internal use in magic functions and other places where we want
    to silently truncate results.
    
    Args:
        df: pandas DataFrame to truncate
    
    Returns:
        pandas DataFrame: Truncated DataFrame
    """
    if df is None:
        return df
    
    max_rows = get_max_display_rows()
    if len(df) > max_rows:
        return df.head(max_rows).copy()
    
    return df