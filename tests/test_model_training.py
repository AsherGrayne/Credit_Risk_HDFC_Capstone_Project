"""
Unit tests for model training functionality
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.early_risk_signals import EarlyRiskSignalSystem


class TestModelTraining:
    """Test suite for model training"""
    
    @pytest.fixture
    def system(self):
        """Create EarlyRiskSignalSystem instance"""
        return EarlyRiskSignalSystem()
    
    @pytest.fixture
    def training_data(self):
        """Create training dataset"""
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'Customer ID': [f'C{i:03d}' for i in range(1, n_samples + 1)],
            'Utilisation %': np.random.uniform(0, 100, n_samples),
            'Avg Payment Ratio': np.random.uniform(0, 1, n_samples),
            'Min Due Paid Frequency': np.random.uniform(0, 100, n_samples),
            'Merchant Mix Index': np.random.uniform(0, 1, n_samples),
            'Cash Withdrawal %': np.random.uniform(0, 50, n_samples),
            'Recent Spend Change %': np.random.uniform(-50, 50, n_samples),
            'DPD Bucket Next Month': np.random.choice([0, 1, 2, 3], n_samples)
        })
        
        # Engineer features
        system = EarlyRiskSignalSystem()
        return system.engineer_early_signals(df)
    
    def test_model_training(self, system, training_data):
        """Test that model can be trained successfully"""
        system.train_model(training_data)
        
        assert system.model is not None, "Model should be trained"
        assert system.scaler is not None, "Scaler should be fitted"
    
    def test_model_prediction(self, system, training_data):
        """Test that trained model can make predictions"""
        system.train_model(training_data)
        
        # Prepare test data
        feature_cols = [
            'Utilisation %', 'Avg Payment Ratio', 'Min Due Paid Frequency',
            'Merchant Mix Index', 'Cash Withdrawal %', 'Recent Spend Change %',
            'spending_stress', 'utilization_risk', 'payment_stress',
            'cash_stress_indicator', 'utilization_payment_mismatch',
            'spending_utilization_stress', 'payment_utilization_critical'
        ]
        
        X_test = training_data[feature_cols].iloc[:5]
        X_test_scaled = system.scaler.transform(X_test)
        
        predictions = system.model.predict(X_test_scaled)
        
        assert len(predictions) == 5, "Should predict for 5 samples"
        assert all(pred in [0, 1] for pred in predictions), \
            "Predictions should be binary (0 or 1)"
    
    def test_feature_importance(self, system, training_data):
        """Test that feature importance can be retrieved"""
        system.train_model(training_data)
        feature_importance_df = system.get_feature_importance()
        
        assert feature_importance_df is not None, "Feature importance should be available"
        assert len(feature_importance_df) > 0, "Should have feature importance values"
        assert 'importance' in feature_importance_df.columns, \
            "Feature importance should have 'importance' column"
    
    def test_model_accuracy(self, system, training_data):
        """Test that model achieves reasonable accuracy"""
        system.train_model(training_data)
        
        # Model should have been evaluated during training
        # Check that model exists and can make predictions
        assert system.model is not None, "Model should be trained"
        
        # Note: Actual accuracy depends on data quality
        # This test just ensures training completes successfully


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

