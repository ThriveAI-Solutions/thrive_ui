#!/usr/bin/env python3
"""
Migration script to add LLM preference columns to the thrive_user table in existing SQLite database.
Run this script to update your existing database with the new selected_llm_provider and selected_llm_model columns.

These columns enable per-user LLM selection via the LLM Registry feature.
"""

import os
import sqlite3
import sys


def migrate_database():
    """Add selected_llm_provider and selected_llm_model columns to thrive_user table."""

    # Path to the SQLite database
    db_path = "./pgDatabase/db.sqlite3"

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Database file not found at: {db_path}")
        print("Make sure you're running this script from the thrive_ui directory")
        return False

    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if columns already exist
        cursor.execute("PRAGMA table_info(thrive_user)")
        columns = [column[1] for column in cursor.fetchall()]

        columns_to_add = []
        if "selected_llm_provider" not in columns:
            columns_to_add.append(("selected_llm_provider", "VARCHAR(50)"))
        if "selected_llm_model" not in columns:
            columns_to_add.append(("selected_llm_model", "VARCHAR(100)"))

        if not columns_to_add:
            print("✅ LLM preference columns already exist in thrive_user table")
            conn.close()
            return True

        # Add the missing columns
        for column_name, column_type in columns_to_add:
            print(f"🔄 Adding {column_name} column to thrive_user table...")
            cursor.execute(f"ALTER TABLE thrive_user ADD COLUMN {column_name} {column_type}")

        # Commit the changes
        conn.commit()

        # Verify the columns were added
        cursor.execute("PRAGMA table_info(thrive_user)")
        columns = [column[1] for column in cursor.fetchall()]

        all_added = all(col_name in columns for col_name, _ in columns_to_add)

        if all_added:
            print("✅ Successfully added LLM preference columns to thrive_user table:")
            for column_name, _ in columns_to_add:
                print(f"   - {column_name}")

            # Show current user count
            cursor.execute("SELECT COUNT(*) FROM thrive_user")
            user_count = cursor.fetchone()[0]
            print(f"📊 Table now has {user_count} users with the new columns (initially NULL)")
            print("ℹ️  NULL values will use LLM settings from secrets.toml (backward compatible)")

            success = True
        else:
            print("❌ Failed to add one or more LLM preference columns")
            success = False

        # Close the connection
        conn.close()
        return success

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Preferences Migration Script")
    print("=" * 60)
    print()

    success = migrate_database()

    print()
    if success:
        print("✅ Migration completed successfully!")
        print()
        print("Next steps:")
        print("1. Restart your Streamlit app")
        print("2. Go to the sidebar and expand '🤖 LLM Selection'")
        print("3. Choose your preferred LLM provider and model")
        print("4. Click 'Apply LLM Selection' to save your choice")
        sys.exit(0)
    else:
        print("❌ Migration failed. Please fix the errors and try again.")
        sys.exit(1)
