#!/usr/bin/env python3
"""
Database Migration Script for Thrive UI
Adds the salt column to the existing user table.

Usage:
    python scripts/migrate_database.py
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_database_path():
    """Get the database path from Streamlit secrets."""
    try:
        # Mock streamlit secrets for the migration
        class MockSecrets:
            def get(self, key, default=None):
                if key == "sqlite":
                    return {"database": "./pgDatabase/db.sqlite3"}
                return default
        
        # Use the same logic as in models.py
        db_settings = MockSecrets().get("sqlite", {"database": "./pgDatabase/db.sqlite3"})
        db_path = db_settings['database']
        
        # Convert to absolute path
        if not Path(db_path).is_absolute():
            db_path = Path(__file__).parent.parent / db_path
        
        return str(db_path)
    except Exception as e:
        logger.error(f"Error getting database path: {e}")
        return "./pgDatabase/db.sqlite3"


def migrate_database():
    """Add the salt column to the thrive_user table."""
    db_path = get_database_path()
    
    if not Path(db_path).exists():
        logger.error(f"Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if salt column already exists
        cursor.execute("PRAGMA table_info(thrive_user)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'salt' in columns:
            logger.info("Salt column already exists in thrive_user table")
            return True
        
        # Add the salt column
        logger.info("Adding salt column to thrive_user table...")
        cursor.execute("ALTER TABLE thrive_user ADD COLUMN salt BLOB")
        
        conn.commit()
        logger.info("Successfully added salt column to thrive_user table")
        
        # Verify the column was added
        cursor.execute("PRAGMA table_info(thrive_user)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'salt' in columns:
            logger.info("Migration verified: salt column exists")
            return True
        else:
            logger.error("Migration failed: salt column not found after adding")
            return False
            
    except sqlite3.Error as e:
        logger.error(f"SQLite error during migration: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        return False
    finally:
        if conn:
            conn.close()


def main():
    logger.info("Starting database migration...")
    
    if migrate_database():
        logger.info("Database migration completed successfully")
        return True
    else:
        logger.error("Database migration failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)