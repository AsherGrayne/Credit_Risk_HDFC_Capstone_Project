"""
Monitoring and metrics collection for the Credit Card Delinquency Prediction System
"""
import time
import functools
from collections import defaultdict
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and track system metrics"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.timers = {}
    
    def increment_counter(self, metric_name: str, value: int = 1):
        """
        Increment a counter metric
        
        Args:
            metric_name: Name of the metric
            value: Value to increment by
        """
        self.counters[metric_name] += value
        logger.debug(f"Counter {metric_name} incremented by {value}, total: {self.counters[metric_name]}")
    
    def record_value(self, metric_name: str, value: float):
        """
        Record a value metric
        
        Args:
            metric_name: Name of the metric
            value: Value to record
        """
        self.metrics[metric_name].append(value)
        logger.debug(f"Recorded {metric_name}: {value}")
    
    def start_timer(self, metric_name: str) -> str:
        """
        Start a timer
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Timer ID
        """
        timer_id = f"{metric_name}_{time.time()}"
        self.timers[timer_id] = time.time()
        return timer_id
    
    def stop_timer(self, timer_id: str, metric_name: str):
        """
        Stop a timer and record duration
        
        Args:
            timer_id: Timer ID returned from start_timer
            metric_name: Name of the metric
        """
        if timer_id in self.timers:
            duration = time.time() - self.timers[timer_id]
            self.record_value(metric_name, duration)
            del self.timers[timer_id]
            logger.debug(f"Timer {metric_name} completed in {duration:.3f}s")
    
    def get_metric_summary(self, metric_name: str) -> Optional[Dict]:
        """
        Get summary statistics for a metric
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Dictionary with summary statistics or None
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = self.metrics[metric_name]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'p50': self._percentile(values, 50),
            'p95': self._percentile(values, 95),
            'p99': self._percentile(values, 99)
        }
    
    def get_all_metrics(self) -> Dict:
        """
        Get all collected metrics
        
        Returns:
            Dictionary with all metrics
        """
        return {
            'counters': dict(self.counters),
            'metrics': {
                name: self.get_metric_summary(name)
                for name in self.metrics.keys()
            }
        }
    
    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        """Calculate percentile"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


# Global metrics collector instance
metrics_collector = MetricsCollector()


def track_performance(metric_name: str):
    """
    Decorator to track function performance
    
    Args:
        metric_name: Name of the metric to track
    
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_id = metrics_collector.start_timer(f"{metric_name}_duration")
            try:
                result = func(*args, **kwargs)
                metrics_collector.increment_counter(f"{metric_name}_success")
                return result
            except Exception as e:
                metrics_collector.increment_counter(f"{metric_name}_errors")
                logger.error(f"Error in {metric_name}: {e}")
                raise
            finally:
                metrics_collector.stop_timer(timer_id, f"{metric_name}_duration")
        return wrapper
    return decorator


class HealthChecker:
    """Health check utilities"""
    
    def __init__(self):
        """Initialize health checker"""
        self.checks = {}
    
    def register_check(self, name: str, check_func):
        """
        Register a health check
        
        Args:
            name: Name of the check
            check_func: Function that returns (is_healthy, message)
        """
        self.checks[name] = check_func
    
    def check_health(self) -> Dict:
        """
        Run all health checks
        
        Returns:
            Dictionary with health status
        """
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                is_healthy, message = check_func()
                results[name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'message': message
                }
                if not is_healthy:
                    overall_healthy = False
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'message': str(e)
                }
                overall_healthy = False
        
        return {
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'checks': results,
            'timestamp': time.time()
        }


# Global health checker instance
health_checker = HealthChecker()

