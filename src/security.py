"""
Security utilities for the Credit Card Delinquency Prediction System
"""
import hashlib
import secrets
import re
from functools import wraps
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Security validation utilities"""
    
    @staticmethod
    def validate_customer_id(customer_id: str) -> bool:
        """
        Validate customer ID format
        
        Args:
            customer_id: Customer ID to validate
        
        Returns:
            True if valid, False otherwise
        """
        if not customer_id or not isinstance(customer_id, str):
            return False
        
        # Customer ID should match pattern C followed by digits
        pattern = r'^C\d{3,}$'
        return bool(re.match(pattern, customer_id))
    
    @staticmethod
    def validate_numeric_range(value: float, min_val: float, max_val: float) -> bool:
        """
        Validate numeric value is within range
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
        
        Returns:
            True if valid, False otherwise
        """
        try:
            num_value = float(value)
            return min_val <= num_value <= max_val
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            input_str: Input string to sanitize
        
        Returns:
            Sanitized string
        """
        if not isinstance(input_str, str):
            return str(input_str)
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';]', '', input_str)
        return sanitized.strip()
    
    @staticmethod
    def validate_feature_inputs(features: dict) -> tuple[bool, Optional[str]]:
        """
        Validate all feature inputs
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Define validation rules
        validation_rules = {
            'Utilisation %': (0, 100),
            'Avg Payment Ratio': (0, 1),
            'Min Due Paid Frequency': (0, 100),
            'Merchant Mix Index': (0, 1),
            'Cash Withdrawal %': (0, 100),
            'Recent Spend Change %': (-100, 100)
        }
        
        for feature_name, (min_val, max_val) in validation_rules.items():
            if feature_name in features:
                if not SecurityValidator.validate_numeric_range(
                    features[feature_name], min_val, max_val
                ):
                    return False, f"Invalid {feature_name}: must be between {min_val} and {max_val}"
        
        return True, None


class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed in time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {client_id: [timestamps]}
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed
        
        Args:
            client_id: Client identifier
        
        Returns:
            True if allowed, False if rate limited
        """
        import time
        current_time = time.time()
        
        # Clean old requests outside window
        if client_id in self.requests:
            self.requests[client_id] = [
                ts for ts in self.requests[client_id]
                if current_time - ts < self.window_seconds
            ]
        else:
            self.requests[client_id] = []
        
        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for client {client_id}")
            return False
        
        # Add current request
        self.requests[client_id].append(current_time)
        return True


def require_api_key(api_key: Optional[str] = None):
    """
    Decorator to require API key for API endpoints
    
    Args:
        api_key: Expected API key (can be from environment variable)
    
    Returns:
        Decorator function
    """
    if api_key is None:
        import os
        api_key = os.getenv('API_KEY', 'default-secret-key-change-in-production')
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # In Flask/FastAPI, get API key from request headers
            # This is a placeholder - implement based on your framework
            request_api_key = kwargs.get('api_key') or args[0].headers.get('X-API-Key', '')
            
            if request_api_key != api_key:
                logger.warning("Invalid API key provided")
                return {'error': 'Invalid API key'}, 401
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def hash_sensitive_data(data: str) -> str:
    """
    Hash sensitive data for logging/storage
    
    Args:
        data: Data to hash
    
    Returns:
        Hashed string
    """
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def generate_api_key() -> str:
    """
    Generate a secure API key
    
    Returns:
        Random API key string
    """
    return secrets.token_urlsafe(32)

