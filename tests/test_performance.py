"""
Performance tests for system components
"""
import pytest
import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.early_risk_signals import EarlyRiskSignalSystem


class TestPerformance:
    """Performance tests"""
    
    @pytest.fixture
    def system(self):
        """Create EarlyRiskSignalSystem instance"""
        return EarlyRiskSignalSystem()
    
    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing"""
        np.random.seed(42)
        n_samples = 10000
        
        df = pd.DataFrame({
            'Customer ID': [f'C{i:05d}' for i in range(1, n_samples + 1)],
            'Credit Limit': np.random.uniform(10000, 200000, n_samples),
            'Utilisation %': np.random.uniform(0, 100, n_samples),
            'Avg Payment Ratio': np.random.uniform(0, 1, n_samples),
            'Min Due Paid Frequency': np.random.uniform(0, 100, n_samples),
            'Merchant Mix Index': np.random.uniform(0, 1, n_samples),
            'Cash Withdrawal %': np.random.uniform(0, 50, n_samples),
            'Recent Spend Change %': np.random.uniform(-50, 50, n_samples),
            'DPD Bucket Next Month': np.random.choice([0, 1, 2, 3], n_samples)
        })
        return df
    
    def test_feature_engineering_performance(self, system, large_dataset):
        """Test feature engineering performance on large dataset"""
        start_time = time.time()
        df_engineered = system.engineer_early_signals(large_dataset)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed_time < 10.0, \
            f"Feature engineering took {elapsed_time:.2f}s, should be < 10s"
        
        print(f"\nFeature engineering: {elapsed_time:.2f}s for {len(large_dataset)} records")
    
    def test_risk_flag_generation_performance(self, system, large_dataset):
        """Test risk flag generation performance"""
        df_engineered = system.engineer_early_signals(large_dataset)
        
        start_time = time.time()
        risk_flags_df = system.identify_risk_flags(df_engineered)
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 5.0, \
            f"Risk flag generation took {elapsed_time:.2f}s, should be < 5s"
        
        print(f"\nRisk flag generation: {elapsed_time:.2f}s for {len(df_engineered)} records")
    
    def test_model_prediction_performance(self, system):
        """Test model prediction performance"""
        # Load and prepare data
        df = system.load_data('data/Sample.csv')
        df_engineered = system.engineer_early_signals(df)
        system.train_model(df_engineered)
        
        # Prepare test data
        feature_cols = [
            'Utilisation %', 'Avg Payment Ratio', 'Min Due Paid Frequency',
            'Merchant Mix Index', 'Cash Withdrawal %', 'Recent Spend Change %',
            'spending_stress', 'utilization_risk', 'payment_stress',
            'cash_stress_indicator', 'utilization_payment_mismatch',
            'spending_utilization_stress', 'payment_utilization_critical'
        ]
        
        X_test = df_engineered[feature_cols].iloc[:100]
        X_test_scaled = system.scaler.transform(X_test)
        
        # Test prediction time
        start_time = time.time()
        predictions = system.model.predict(X_test_scaled)
        elapsed_time = time.time() - start_time
        
        # Should predict 100 samples in < 100ms
        assert elapsed_time < 0.1, \
            f"Prediction took {elapsed_time*1000:.2f}ms, should be < 100ms"
        
        avg_time_per_prediction = (elapsed_time / len(X_test)) * 1000
        print(f"\nAverage prediction time: {avg_time_per_prediction:.2f}ms per prediction")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

