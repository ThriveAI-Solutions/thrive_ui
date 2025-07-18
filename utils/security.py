"""
Security utilities for the Thrive UI application.
Provides additional security measures including headers and content security.
"""

import logging
import re
from typing import Dict, Any
import streamlit as st

logger = logging.getLogger(__name__)


def apply_security_headers():
    """
    Apply security headers to the Streamlit application.
    Note: Streamlit has limited control over HTTP headers, but we can add what's possible.
    """
    # Content Security Policy via meta tag
    st.markdown(
        """
        <meta http-equiv="Content-Security-Policy" content="
            default-src 'self';
            script-src 'self' 'unsafe-inline' 'unsafe-eval';
            style-src 'self' 'unsafe-inline';
            img-src 'self' data: https:;
            font-src 'self';
            connect-src 'self';
            frame-ancestors 'none';
        ">
        <meta http-equiv="X-Content-Type-Options" content="nosniff">
        <meta http-equiv="X-Frame-Options" content="DENY">
        <meta http-equiv="Referrer-Policy" content="strict-origin-when-cross-origin">
        """,
        unsafe_allow_html=True,
    )


def sanitize_html_content(content: str) -> str:
    """
    Sanitize HTML content to prevent XSS attacks.
    """
    if not content:
        return ""
    
    # Remove potentially dangerous HTML tags and attributes
    dangerous_tags = [
        'script', 'iframe', 'object', 'embed', 'form', 'input', 'textarea', 
        'button', 'select', 'option', 'link', 'meta', 'style', 'base'
    ]
    
    dangerous_attributes = [
        'onclick', 'onload', 'onmouseover', 'onmouseout', 'onfocus', 'onblur',
        'onchange', 'onsubmit', 'onreset', 'onselect', 'onabort', 'onerror',
        'onresize', 'onscroll', 'onunload', 'javascript:', 'vbscript:', 'data:'
    ]
    
    sanitized = content
    
    # Remove dangerous tags
    for tag in dangerous_tags:
        pattern = rf'<{tag}[^>]*>.*?</{tag}>'
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        pattern = rf'<{tag}[^>]*/?>'
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    # Remove dangerous attributes
    for attr in dangerous_attributes:
        pattern = rf'{attr}=["\'][^"\']*["\']'
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    return sanitized


def validate_sql_query(query: str) -> tuple[bool, str]:
    """
    Basic SQL injection prevention - validate that queries don't contain dangerous patterns.
    """
    if not query:
        return True, ""
    
    # Convert to lowercase for checking
    query_lower = query.lower().strip()
    
    # Dangerous SQL patterns that should be blocked
    dangerous_patterns = [
        # SQL injection patterns
        r';\s*drop\s+',
        r';\s*delete\s+',
        r';\s*insert\s+',
        r';\s*update\s+',
        r';\s*alter\s+',
        r';\s*create\s+',
        r';\s*truncate\s+',
        r'union\s+select',
        r'exec\s*\(',
        r'execute\s*\(',
        r'xp_cmdshell',
        r'sp_executesql',
        
        # System access patterns
        r'information_schema',
        r'pg_catalog',
        r'mysql\.',
        r'sys\.',
        r'master\.',
        
        # File system access
        r'load_file',
        r'into\s+outfile',
        r'into\s+dumpfile',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query_lower):
            return False, f"Query contains potentially dangerous pattern: {pattern}"
    
    return True, ""


def log_security_event(event_type: str, details: Dict[str, Any], user_id: str = None):
    """
    Log security-related events for monitoring and auditing.
    """
    log_entry = {
        'event_type': event_type,
        'details': details,
        'user_id': user_id,
        'timestamp': str(st.session_state.get('timestamp', 'unknown'))
    }
    
    # Log at appropriate level based on event type
    if event_type in ['failed_login', 'rate_limit_exceeded', 'csrf_token_invalid']:
        logger.warning(f"Security event: {event_type} - {details}")
    elif event_type in ['sql_injection_attempt', 'xss_attempt']:
        logger.error(f"Security event: {event_type} - {details}")
    else:
        logger.info(f"Security event: {event_type} - {details}")


def check_content_security(content: str, content_type: str = 'text') -> tuple[bool, str]:
    """
    Check content for security issues before processing.
    """
    if not content:
        return True, ""
    
    # Check for potential XSS
    xss_patterns = [
        r'<script[^>]*>',
        r'javascript:',
        r'vbscript:',
        r'data:text/html',
        r'onload\s*=',
        r'onerror\s*=',
        r'onclick\s*=',
    ]
    
    for pattern in xss_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            log_security_event('xss_attempt', {'pattern': pattern, 'content_preview': content[:100]})
            return False, f"Content contains potential XSS: {pattern}"
    
    # Check for SQL injection if content type suggests it might be a query
    if content_type in ['sql', 'query']:
        is_safe, message = validate_sql_query(content)
        if not is_safe:
            log_security_event('sql_injection_attempt', {'content_preview': content[:100]})
            return False, message
    
    return True, ""


def create_audit_log_entry(action: str, user_id: str, details: Dict[str, Any] = None):
    """
    Create an audit log entry for important user actions.
    """
    audit_entry = {
        'action': action,
        'user_id': user_id,
        'details': details or {},
        'timestamp': str(st.session_state.get('timestamp', 'unknown')),
        'session_id': st.session_state.get('session_id', 'unknown')
    }
    
    logger.info(f"Audit log: {audit_entry}")


class SecurityMiddleware:
    """
    Security middleware for additional protection measures.
    """
    
    @staticmethod
    def validate_user_input(input_data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate user input for security issues.
        """
        for key, value in input_data.items():
            if isinstance(value, str):
                is_safe, message = check_content_security(value)
                if not is_safe:
                    return False, f"Invalid input in field '{key}': {message}"
        
        return True, ""
    
    @staticmethod
    def check_rate_limit(user_id: str, action: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """
        Simple rate limiting check (in production, use Redis or similar).
        """
        # This is a simplified implementation
        # In production, use a proper rate limiting solution
        return True
    
    @staticmethod
    def generate_session_id() -> str:
        """
        Generate a unique session ID.
        """
        import secrets
        return secrets.token_urlsafe(32)