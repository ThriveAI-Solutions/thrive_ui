#!/usr/bin/env python3
"""
Migration script to add the thrive_conversation table and conversation_id FK
to the thrive_message table in existing SQLite database.

Assigns existing messages to a default "General" conversation per user.

Run this script to update your existing database:
    python scripts/migrate_add_conversations.py
"""

import os
import sqlite3
import sys
import uuid


def migrate_database():
    """Add thrive_conversation table and conversation_id column to thrive_message."""

    # Path to the SQLite database
    db_path = "./pgDatabase/db.sqlite3"

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Database file not found at: {db_path}")
        print("Make sure you're running this script from the thrive_ui directory")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # --- Step 1: Create thrive_conversation table if it doesn't exist ---
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='thrive_conversation'"
        )
        if cursor.fetchone() is None:
            print("Creating thrive_conversation table...")
            cursor.execute("""
                CREATE TABLE thrive_conversation (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES thrive_user(id),
                    title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
                    is_archived BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ix_thrive_conversation_user_id ON thrive_conversation(user_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ix_thrive_conversation_updated_at ON thrive_conversation(updated_at)"
            )
            print("Successfully created thrive_conversation table")
        else:
            print("thrive_conversation table already exists")

        # --- Step 2: Add conversation_id column to thrive_message if missing ---
        cursor.execute("PRAGMA table_info(thrive_message)")
        columns = [column[1] for column in cursor.fetchall()]

        if "conversation_id" not in columns:
            print("Adding conversation_id column to thrive_message...")
            cursor.execute(
                "ALTER TABLE thrive_message ADD COLUMN conversation_id VARCHAR(36) REFERENCES thrive_conversation(id)"
            )
            print("Successfully added conversation_id column")
        else:
            print("conversation_id column already exists in thrive_message")

        # --- Step 3: Assign existing messages to a default "General" conversation per user ---
        # Find users who have messages without a conversation_id
        cursor.execute("""
            SELECT DISTINCT user_id FROM thrive_message
            WHERE conversation_id IS NULL AND user_id IS NOT NULL
        """)
        users_needing_default = cursor.fetchall()

        if users_needing_default:
            print(f"Assigning orphan messages for {len(users_needing_default)} user(s) to default conversations...")
            for (user_id,) in users_needing_default:
                # Check if user already has a "General" conversation
                cursor.execute(
                    "SELECT id FROM thrive_conversation WHERE user_id = ? AND title = 'General' LIMIT 1",
                    (user_id,),
                )
                row = cursor.fetchone()
                if row:
                    conv_id = row[0]
                else:
                    conv_id = str(uuid.uuid4())
                    cursor.execute(
                        """INSERT INTO thrive_conversation (id, user_id, title, is_archived, created_at, updated_at)
                           VALUES (?, ?, 'General', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                        (conv_id, user_id),
                    )

                # Assign all orphan messages for this user to the default conversation
                cursor.execute(
                    "UPDATE thrive_message SET conversation_id = ? WHERE user_id = ? AND conversation_id IS NULL",
                    (conv_id, user_id),
                )
                count = cursor.rowcount
                print(f"  User {user_id}: assigned {count} message(s) to 'General' conversation ({conv_id[:8]}...)")
        else:
            print("No orphan messages to assign")

        conn.commit()
        conn.close()
        return True

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def main():
    print("Starting migration to add conversations support...")
    print("=" * 60)

    success = migrate_database()

    print("=" * 60)
    if success:
        print("Migration completed successfully!")
        print("\nNext steps:")
        print("1. Restart your Streamlit application")
        print("2. Users will see a conversation sidebar with their existing messages in a 'General' thread")
        print("3. Users can create new conversation threads from the sidebar")
    else:
        print("Migration failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
