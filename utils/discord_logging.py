"""
utils.discord_logging
Discord webhook integration for logging notifications.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Optional

import requests
import streamlit as st

class DiscordHandler(logging.Handler):
    """Custom logging handler that sends log messages to Discord via webhook."""

    current_user = "Unknown"
    
    def __init__(self, webhook_url: str, username: str = "ThriveAI Bot"):
        super().__init__()
        self.webhook_url = webhook_url
        self.username = username
        self._session = requests.Session()
        
    def emit(self, record: logging.LogRecord) -> None:
        """Send log record to Discord channel."""
        try:
            # Format the message
            log_message = self.format(record)
            
            # Create Discord embed
            embed = {
                "title": f"{record.levelname} - {record.name}",
                "description": log_message,
                "color": self._get_color_for_level(record.levelname),
                "fields": [
                    {
                        "name": "User",
                        "value": DiscordHandler.current_user,
                        "inline": True
                    },
                    {
                        "name": "Module",
                        "value": record.name,
                        "inline": True
                    }
                ],
                "timestamp": datetime.fromtimestamp(record.created).isoformat()
            }
            
            # Prepare webhook payload
            payload = {
                "username": self.username,
                "embeds": [embed]
            }
            
            # Send to Discord in a separate thread to avoid blocking
            threading.Thread(
                target=self._send_to_discord,
                args=(payload,),
                daemon=True
            ).start()
            
        except Exception as e:
            # Silently handle Discord handler errors to avoid spam
            pass
    
    def _send_to_discord(self, payload: dict) -> None:
        """Send payload to Discord webhook."""
        try:
            response = self._session.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
        except Exception:
            # Silently handle webhook errors to avoid spam
            pass
    
    def _get_color_for_level(self, level: str) -> int:
        """Get Discord embed color based on log level."""
        colors = {
            "DEBUG": 0x808080,    # Gray
            "INFO": 0x00FF00,     # Green
            "WARNING": 0xFFFF00,  # Yellow
            "ERROR": 0xFF0000,    # Red
            "CRITICAL": 0x8B0000  # Dark Red
        }
        return colors.get(level, 0x808080)


def get_discord_webhook_url() -> Optional[str]:
    """Get Discord webhook URL from Streamlit secrets if configured."""
    try:
        if hasattr(st, 'secrets') and 'discord' in st.secrets:
            webhook_url = st.secrets.discord.get('webhook_url')
            if webhook_url and webhook_url.strip():
                return webhook_url
    except Exception:
        pass
    return None


def add_discord_handler_if_configured(logger: logging.Logger) -> None:
    """Add Discord handler to logger if webhook URL is configured."""
    # Check if this logger already has a Discord handler to avoid duplicates
    for handler in logger.handlers:
        if isinstance(handler, DiscordHandler):
            return
    
    webhook_url = get_discord_webhook_url()
    if webhook_url:
        discord_handler = DiscordHandler(webhook_url)
        discord_handler.setLevel(logging.WARNING)  # Only send WARNING and above to Discord
        
        # Use the same formatter as console handler
        formatter = logging.Formatter(
            "{asctime} {levelname:<8} {name}: {message}",
            style="{",
            datefmt="%H:%M:%S"
        )
        discord_handler.setFormatter(formatter)
        
        logger.addHandler(discord_handler)


def initialize_discord_logging_after_streamlit(username: str):
    """Call this function after Streamlit is fully initialized to ensure Discord logging works."""
    # Get all existing loggers and add Discord handlers
    import logging
    loggers_to_update = [logging.getLogger()]  # Start with root logger

    DiscordHandler.current_user = username
    
    # Add specific module loggers
    module_names = [
        'utils.vanna_calls',
        'views.chat_bot', 
        'utils.auth',
        'orm.models',
        'orm.functions',
        'utils.chat_bot_helper',
        'utils.communicate',
        'app'
    ]
    
    for name in module_names:
        loggers_to_update.append(logging.getLogger(name))
    
    for logger in loggers_to_update:
        add_discord_handler_if_configured(logger)