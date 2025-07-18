#!/usr/bin/env python3
"""
Password Migration Script for Thrive UI
Migrates existing SHA-256 passwords to secure PBKDF2 hashes with salt.

Usage:
    python scripts/migrate_passwords.py [--default-password PASSWORD]
    
Options:
    --default-password: Default password to use for migration (default: "hello")
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from orm.models import SessionLocal, User
from orm.functions import hash_password
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_user_passwords(default_password: str = "hello"):
    """
    Migrate existing SHA-256 passwords to secure PBKDF2 hashes.
    
    Args:
        default_password: The default password that was used for seeding users
    """
    try:
        with SessionLocal() as session:
            # Get all users with old-style passwords (64 character hex strings)
            users = session.query(User).filter(
                User.password.isnot(None),
                User.salt.is_(None)  # Users without salt need migration
            ).all()
            
            migrated_count = 0
            
            for user in users:
                # Check if this looks like an old SHA-256 hash
                if len(user.password) == 64 and user.password.isalnum():
                    # Check if it matches the default password hash
                    default_hash = hashlib.sha256(default_password.encode()).hexdigest()
                    
                    if user.password == default_hash:
                        # Migrate to new secure hash
                        new_hash, salt = hash_password(default_password)
                        user.password = new_hash
                        user.salt = salt
                        migrated_count += 1
                        logger.info(f"Migrated password for user: {user.username}")
                    else:
                        logger.warning(f"User {user.username} has non-default password - skipping migration")
                        logger.warning(f"You may need to reset their password manually")
            
            if migrated_count > 0:
                session.commit()
                logger.info(f"Successfully migrated {migrated_count} user passwords")
            else:
                logger.info("No passwords needed migration")
                
    except Exception as e:
        logger.error(f"Error during password migration: {e}")
        raise


def reset_user_password(username: str, new_password: str):
    """
    Reset a specific user's password to a new secure hash.
    
    Args:
        username: The username to reset
        new_password: The new password to set
    """
    try:
        with SessionLocal() as session:
            user = session.query(User).filter(User.username == username).first()
            
            if not user:
                logger.error(f"User {username} not found")
                return False
            
            # Set new secure password
            new_hash, salt = hash_password(new_password)
            user.password = new_hash
            user.salt = salt
            
            session.commit()
            logger.info(f"Reset password for user: {username}")
            return True
            
    except Exception as e:
        logger.error(f"Error resetting password for {username}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Migrate passwords to secure hashing')
    parser.add_argument('--default-password', default='hello', 
                       help='Default password used for seeding (default: hello)')
    parser.add_argument('--reset-user', help='Reset password for specific user')
    parser.add_argument('--new-password', help='New password when resetting user')
    
    args = parser.parse_args()
    
    if args.reset_user:
        if not args.new_password:
            logger.error("--new-password is required when using --reset-user")
            sys.exit(1)
        
        success = reset_user_password(args.reset_user, args.new_password)
        sys.exit(0 if success else 1)
    else:
        try:
            migrate_user_passwords(args.default_password)
            logger.info("Password migration completed successfully")
        except Exception as e:
            logger.error(f"Password migration failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()