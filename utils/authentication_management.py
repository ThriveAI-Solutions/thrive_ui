import pandas as pd
# import pickle 
from pathlib import Path 
import sqlite3
# import streamlit_authenticator as stauth 

def get_user_list_excel(file_name = "./config/user_list.xlsx"):
    """
    returns a dataframe of users. Reads Excel File into a Dataframe 
    """
    try:
        df_user_list = pd.read_excel(file_name) # Reads Excel File into a Dataframe
        if len(df_user_list) > 1:
                print(f"✅ Read File Successfully : {df_user_list.shape[0]} ")
        return df_user_list
    except Exception as e:
        print(f"❌ File Read failed: {e}")
        return False

def save_user_list_excel(df, file_name="user_list.xlsx"):
    """Save DataFrame to Excel file."""
    df.to_excel(file_name, index=False)
    print(f"✅ Saved {len(df):,} rows to {file_name}")
    return file_name

def save_user_list_parquet(df, file_name="user_list.parquet"):
    """Save DataFrame to Parquet file."""
    df.to_parquet(file_name, index=False)
    print(f"✅ Saved {len(df):,} rows to {file_name}")
    return file_name

def save_user_list_sqlite(df, db_file="db.sqlite3.db", table_name="user_list"):
    """Save DataFrame to SQLite database table."""
    try:
        with sqlite3.connect(db_file) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"✅ Saved {len(df):,} rows to {db_file} (table: {table_name})")
        return db_file
    except Exception as e:
        print(f"❌ Failed to save to SQLite: {e}")
        return None

def authenticate_user(user_id,password):
    if length(password) > 1:
        authenticated = 1 
    else: 
        authenticated = 0 
    return authenticated


def save_user_list_json_array(df, file_name="user_list.json"):
    """
    Save the entire DataFrame as a single JSON array of objects.
    """
    df_out = df.where(pd.notnull(df), None)  # NaN -> null
    df_out.to_json(
        file_name,
        orient="records",
        indent=2,
        date_format="iso",
        force_ascii=False
    )
    print(f"✅ Saved {len(df_out):,} rows to {file_name} (JSON array)")
    return file_name


def persist_user_list(df):
    """
    Save the entire DataFrame as a single JSON array of objects.
    """
    print(save_user_list_excel(df) )    
    print(save_user_list_parquet(df) )    
    print(save_user_list_sqlite(df) )       
    print(save_user_list_json_array(df) )
    print(f"✅ Saved {len(df):,} rows to everything")
    return file_name




