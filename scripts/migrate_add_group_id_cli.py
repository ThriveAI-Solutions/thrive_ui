#!/usr/bin/env python3
"""
Command-line migration script to add group_id column to the thrive_message table in existing SQLite database.
Usage: python migrate_add_group_id_cli.py [database_path]
"""

import argparse
import os
import sqlite3
import sys


def migrate_database(db_path):
    """Add group_id column to thrive_message table."""
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Database file not found at: {db_path}")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if group_id column already exists
        cursor.execute("PRAGMA table_info(thrive_message)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'group_id' in columns:
            print("✅ group_id column already exists in thrive_message table")
            conn.close()
            return True
        
        # Add the group_id column
        print("🔄 Adding group_id column to thrive_message table...")
        cursor.execute("ALTER TABLE thrive_message ADD COLUMN group_id VARCHAR(50)")
        
        # Commit the changes
        conn.commit()
        
        # Verify the column was added
        cursor.execute("PRAGMA table_info(thrive_message)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'group_id' in columns:
            print("✅ Successfully added group_id column to thrive_message table")
            
            # Show current row count
            cursor.execute("SELECT COUNT(*) FROM thrive_message")
            row_count = cursor.fetchone()[0]
            print(f"📊 Table now has {row_count} rows with the new group_id column (initially NULL)")
            
            success = True
        else:
            print("❌ Failed to add group_id column")
            success = False
            
        conn.close()
        return success
        
    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add group_id column to thrive_message table in SQLite database"
    )
    parser.add_argument(
        "database_path", 
        nargs='?', 
        default="./pgDatabase/db.sqlite3",
        help="Path to the SQLite database file (default: ./pgDatabase/db.sqlite3)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting migration to add group_id column...")
    print("=" * 50)
    print(f"Database path: {args.database_path}")
    print("=" * 50)
    
    success = migrate_database(args.database_path)
    
    print("=" * 50)
    if success:
        print("✅ Migration completed successfully!")
        print("\n📝 Next steps:")
        print("1. Restart your application")
        print("2. New messages will automatically get group_ids")
        print("3. Existing messages will have NULL group_ids (this is expected)")
    else:
        print("❌ Migration failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()