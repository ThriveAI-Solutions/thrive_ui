"""Migration script to add feedback dashboard columns to thrive_message table.

This script adds the following columns:
- training_status: VARCHAR(20) - 'pending', 'approved', 'rejected', or NULL
- reviewed_by: INTEGER - Foreign key to thrive_user.id
- reviewed_at: TIMESTAMP - When the review happened

Run with: uv run python scripts/migrate_feedback_columns.py
"""

import sqlite3
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st


def get_database_path():
    """Get the database path from Streamlit secrets."""
    try:
        db_settings = st.secrets.get("sqlite", {"database": "./pgDatabase/db.sqlite3"})
        return db_settings["database"]
    except Exception:
        return "./pgDatabase/db.sqlite3"


def migrate_database(db_path: str):
    """Add new columns to thrive_message table."""
    print(f"Connecting to database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check existing columns
    cursor.execute("PRAGMA table_info(thrive_message)")
    columns = {row[1] for row in cursor.fetchall()}
    print(f"Existing columns: {columns}")

    columns_to_add = []

    # Add training_status column if it doesn't exist
    if "training_status" not in columns:
        columns_to_add.append(("training_status", "VARCHAR(20)"))

    # Add reviewed_by column if it doesn't exist
    if "reviewed_by" not in columns:
        columns_to_add.append(("reviewed_by", "INTEGER REFERENCES thrive_user(id)"))

    # Add reviewed_at column if it doesn't exist
    if "reviewed_at" not in columns:
        columns_to_add.append(("reviewed_at", "TIMESTAMP"))

    if not columns_to_add:
        print("All columns already exist. No migration needed.")
        conn.close()
        return

    # Add each column
    for column_name, column_type in columns_to_add:
        print(f"Adding column: {column_name} {column_type}")
        try:
            cursor.execute(f"ALTER TABLE thrive_message ADD COLUMN {column_name} {column_type}")
            print(f"  Successfully added {column_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print(f"  Column {column_name} already exists, skipping")
            else:
                raise

    conn.commit()
    print("Migration completed successfully!")

    # Verify columns were added
    cursor.execute("PRAGMA table_info(thrive_message)")
    new_columns = {row[1] for row in cursor.fetchall()}
    print(f"Updated columns: {new_columns}")

    conn.close()


if __name__ == "__main__":
    db_path = get_database_path()
    migrate_database(db_path)
