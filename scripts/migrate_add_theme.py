#!/usr/bin/env python3
"""
Migration script to add theme column to the thrive_user table in existing SQLite database.
Run this script to update your existing database with the new theme column.
"""

import os
import sqlite3
import sys


def migrate_database():
    """Add theme column to thrive_user table."""

    # Path to the SQLite database
    db_path = "./pgDatabase/db.sqlite3"

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"[ERROR] Database file not found at: {db_path}")
        print("Make sure you're running this script from the thrive_ui directory")
        return False

    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if theme column already exists
        cursor.execute("PRAGMA table_info(thrive_user)")
        columns = [column[1] for column in cursor.fetchall()]

        if "theme" in columns:
            print("[OK] theme column already exists in thrive_user table")
            conn.close()
            return True

        # Add the theme column with default value
        print("[INFO] Adding theme column to thrive_user table...")
        cursor.execute("ALTER TABLE thrive_user ADD COLUMN theme VARCHAR(50) DEFAULT 'HEALTHeLINK'")

        # Update existing rows to have the default theme value
        print("[INFO] Setting default theme for existing users...")
        cursor.execute("UPDATE thrive_user SET theme = 'HEALTHeLINK' WHERE theme IS NULL")

        # Commit the changes
        conn.commit()

        # Verify the column was added
        cursor.execute("PRAGMA table_info(thrive_user)")
        columns = [column[1] for column in cursor.fetchall()]

        if "theme" in columns:
            print("[OK] Successfully added theme column to thrive_user table")

            # Show current row count
            cursor.execute("SELECT COUNT(*) FROM thrive_user")
            row_count = cursor.fetchone()[0]
            print(f"[INFO] Table now has {row_count} users with the theme column set to 'HEALTHeLINK'")

            success = True
        else:
            print("[ERROR] Failed to add theme column")
            success = False

        conn.close()
        return success

    except sqlite3.Error as e:
        print(f"[ERROR] SQLite error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


def main():
    print("Starting migration to add theme column...")
    print("=" * 50)

    success = migrate_database()

    print("=" * 50)
    if success:
        print("[OK] Migration completed successfully!")
        print("\nNext steps:")
        print("1. Restart your Streamlit application")
        print("2. Users can now select their preferred theme")
        print("3. All existing users have been set to the default 'HEALTHeLINK' theme")
    else:
        print("[ERROR] Migration failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
