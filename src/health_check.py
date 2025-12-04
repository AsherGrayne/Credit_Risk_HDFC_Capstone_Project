"""
Health check endpoints for the Credit Card Delinquency Prediction System
"""
from flask import Flask, jsonify
from src.monitoring import health_checker, metrics_collector
from src.config import config
import os


def create_health_check_app():
    """Create Flask app with health check endpoints"""
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health():
        """Basic health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'service': 'credit-card-delinquency-prediction',
            'version': '1.0.0'
        }), 200
    
    @app.route('/health/ready', methods=['GET'])
    def readiness():
        """Readiness probe - checks if service is ready to accept traffic"""
        checks = {}
        is_ready = True
        
        # Check if model files exist
        model_files = [
            'models/random_forest_model.joblib',
            'models/logistic_regression_model.joblib'
        ]
        
        for model_file in model_files:
            exists = os.path.exists(model_file)
            checks[f'model_{model_file}'] = 'ready' if exists else 'not_ready'
            if not exists:
                is_ready = False
        
        # Check if data directory exists
        data_dir_exists = os.path.exists(config.data_dir)
        checks['data_directory'] = 'ready' if data_dir_exists else 'not_ready'
        if not data_dir_exists:
            is_ready = False
        
        status_code = 200 if is_ready else 503
        return jsonify({
            'status': 'ready' if is_ready else 'not_ready',
            'checks': checks
        }), status_code
    
    @app.route('/health/live', methods=['GET'])
    def liveness():
        """Liveness probe - checks if service is alive"""
        return jsonify({
            'status': 'alive'
        }), 200
    
    @app.route('/metrics', methods=['GET'])
    def metrics():
        """Metrics endpoint"""
        return jsonify(metrics_collector.get_all_metrics()), 200
    
    @app.route('/health/detailed', methods=['GET'])
    def detailed_health():
        """Detailed health check with all registered checks"""
        health_status = health_checker.check_health()
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code
    
    return app


if __name__ == '__main__':
    app = create_health_check_app()
    app.run(host='0.0.0.0', port=5001, debug=False)

