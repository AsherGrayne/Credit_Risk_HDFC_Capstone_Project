"""
Configuration management for the Credit Card Delinquency Prediction System
"""
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration"""
    n_estimators: int = 100
    max_depth: int = 5
    min_samples_split: int = 10
    random_state: int = 42
    test_size: float = 0.2


@dataclass
class SecurityConfig:
    """Security configuration"""
    api_key: Optional[str] = None
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    enable_encryption: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_file_logging: bool = True


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    batch_size: int = 1000
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    max_workers: int = 4


class Config:
    """Main configuration class"""
    
    def __init__(self):
        """Initialize configuration from environment variables"""
        # Model configuration
        self.model = ModelConfig(
            n_estimators=int(os.getenv('MODEL_N_ESTIMATORS', '100')),
            max_depth=int(os.getenv('MODEL_MAX_DEPTH', '5')),
            min_samples_split=int(os.getenv('MODEL_MIN_SAMPLES_SPLIT', '10')),
            random_state=int(os.getenv('MODEL_RANDOM_STATE', '42')),
            test_size=float(os.getenv('MODEL_TEST_SIZE', '0.2'))
        )
        
        # Security configuration
        self.security = SecurityConfig(
            api_key=os.getenv('API_KEY'),
            rate_limit_requests=int(os.getenv('RATE_LIMIT_REQUESTS', '100')),
            rate_limit_window=int(os.getenv('RATE_LIMIT_WINDOW', '60')),
            enable_encryption=os.getenv('ENABLE_ENCRYPTION', 'false').lower() == 'true'
        )
        
        # Logging configuration
        self.logging = LoggingConfig(
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE', 'logs/credit_risk_system.log'),
            enable_file_logging=os.getenv('ENABLE_FILE_LOGGING', 'true').lower() == 'true'
        )
        
        # Performance configuration
        self.performance = PerformanceConfig(
            batch_size=int(os.getenv('BATCH_SIZE', '1000')),
            enable_caching=os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
            cache_ttl=int(os.getenv('CACHE_TTL', '3600')),
            max_workers=int(os.getenv('MAX_WORKERS', '4'))
        )
        
        # Data paths
        self.data_dir = os.getenv('DATA_DIR', 'data')
        self.models_dir = os.getenv('MODELS_DIR', 'models')
        self.visualizations_dir = os.getenv('VISUALIZATIONS_DIR', 'visualizations')
        self.logs_dir = os.getenv('LOGS_DIR', 'logs')


# Global configuration instance
config = Config()

