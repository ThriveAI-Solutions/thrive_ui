#!/usr/bin/env python3
"""
Migration script to add feedback_comment column to the thrive_message table.
Run this script to update your existing database with the new feedback_comment column.

This column stores optional user feedback when they click the thumbs down button,
helping to understand why responses were marked as unhelpful.
"""

import os
import sqlite3
import sys


def migrate_database():
    """Add feedback_comment column to thrive_message table."""

    # Path to the SQLite database
    db_path = "./pgDatabase/db.sqlite3"

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Database file not found at: {db_path}")
        print("Make sure you're running this script from the thrive_ui directory")
        return False

    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if feedback_comment column already exists
        cursor.execute("PRAGMA table_info(thrive_message)")
        columns = [column[1] for column in cursor.fetchall()]

        if "feedback_comment" in columns:
            print("feedback_comment column already exists in thrive_message table")
            conn.close()
            return True

        # Add the feedback_comment column
        print("Adding feedback_comment column to thrive_message table...")
        cursor.execute("ALTER TABLE thrive_message ADD COLUMN feedback_comment VARCHAR(500)")

        # Commit the changes
        conn.commit()

        # Verify the column was added
        cursor.execute("PRAGMA table_info(thrive_message)")
        columns = [column[1] for column in cursor.fetchall()]

        if "feedback_comment" in columns:
            print("Successfully added feedback_comment column to thrive_message table")

            # Show current row count
            cursor.execute("SELECT COUNT(*) FROM thrive_message")
            row_count = cursor.fetchone()[0]
            print(f"Table now has {row_count} rows with the new feedback_comment column (initially NULL)")

            success = True
        else:
            print("Failed to add feedback_comment column")
            success = False

        conn.close()
        return success

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def main():
    print("Starting migration to add feedback_comment column...")
    print("=" * 50)

    success = migrate_database()

    print("=" * 50)
    if success:
        print("Migration completed successfully!")
        print("\nNext steps:")
        print("1. Restart your Streamlit application")
        print("2. Users can now provide feedback comments when clicking thumbs down")
    else:
        print("Migration failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
