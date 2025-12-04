"""
Logging configuration for the Credit Card Delinquency Prediction System
"""
import logging
import sys
from datetime import datetime
import os


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up structured logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs only to console
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger
    logger = logging.getLogger('credit_risk_system')
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name=None):
    """
    Get a logger instance for a specific module
    
    Args:
        name: Module name (defaults to caller's module)
    
    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        name = inspect.getmodule(inspect.stack()[1][0]).__name__
    
    return logging.getLogger(f'credit_risk_system.{name}')


# Default logger instance
default_logger = setup_logging(
    log_level=logging.INFO,
    log_file='logs/credit_risk_system.log' if os.path.exists('logs') else None
)

