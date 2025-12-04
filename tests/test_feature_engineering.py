"""
Unit tests for feature engineering functions
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.early_risk_signals import EarlyRiskSignalSystem


class TestFeatureEngineering:
    """Test suite for feature engineering functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe for testing"""
        return pd.DataFrame({
            'Customer ID': ['C001', 'C002', 'C003'],
            'Credit Limit': [100000, 50000, 75000],
            'Utilisation %': [85, 45, 95],
            'Avg Payment Ratio': [0.8, 0.9, 0.5],
            'Min Due Paid Frequency': [25, 60, 15],
            'Merchant Mix Index': [0.7, 0.5, 0.8],
            'Cash Withdrawal %': [15, 5, 25],
            'Recent Spend Change %': [-25, 5, -30],
            'DPD Bucket Next Month': [0, 0, 3]
        })
    
    @pytest.fixture
    def system(self):
        """Create EarlyRiskSignalSystem instance"""
        return EarlyRiskSignalSystem()
    
    def test_spending_decline_flag(self, system, sample_data):
        """Test spending decline flag generation"""
        df_engineered = system.engineer_early_signals(sample_data)
        
        # Customer with -25% spending should have flag = 1
        assert df_engineered.loc[0, 'spending_decline_flag'] == 1
        # Customer with +5% spending should have flag = 0
        assert df_engineered.loc[1, 'spending_decline_flag'] == 0
    
    def test_spending_stress(self, system, sample_data):
        """Test spending stress indicator"""
        df_engineered = system.engineer_early_signals(sample_data)
        
        # Customer with -25% spending should have stress = 2 (severe)
        assert df_engineered.loc[0, 'spending_stress'] == 2
        # Customer with +5% spending should have stress = 0 (normal)
        assert df_engineered.loc[1, 'spending_stress'] == 0
    
    def test_utilization_risk(self, system, sample_data):
        """Test utilization risk indicator"""
        df_engineered = system.engineer_early_signals(sample_data)
        
        # Customer with 85% utilization should have risk = 2 (high)
        assert df_engineered.loc[0, 'utilization_risk'] == 2
        # Customer with 45% utilization should have risk = 0 (low)
        assert df_engineered.loc[1, 'utilization_risk'] == 0
        # Customer with 95% utilization should have risk = 3 (critical)
        assert df_engineered.loc[2, 'utilization_risk'] == 3
    
    def test_payment_stress(self, system, sample_data):
        """Test payment stress indicator"""
        df_engineered = system.engineer_early_signals(sample_data)
        
        # Customer with 25% payment frequency should have stress = 1 (moderate)
        assert df_engineered.loc[0, 'payment_stress'] == 1
        # Customer with 15% payment frequency should have stress = 2 (severe)
        assert df_engineered.loc[2, 'payment_stress'] == 2
    
    def test_early_risk_score_range(self, system, sample_data):
        """Test that early risk score is between 0 and 1"""
        df_engineered = system.engineer_early_signals(sample_data)
        
        risk_scores = df_engineered['early_risk_score']
        assert all(risk_scores >= 0) and all(risk_scores <= 1), \
            "Risk scores should be between 0 and 1"
    
    def test_no_data_loss(self, system, sample_data):
        """Test that original data is preserved"""
        df_engineered = system.engineer_early_signals(sample_data)
        
        # Check that all original columns are present
        for col in sample_data.columns:
            assert col in df_engineered.columns, f"Original column {col} missing"
        
        # Check that number of rows is preserved
        assert len(df_engineered) == len(sample_data)
    
    def test_new_features_created(self, system, sample_data):
        """Test that new features are created"""
        df_engineered = system.engineer_early_signals(sample_data)
        
        expected_features = [
            'spending_decline_flag', 'spending_stress', 'utilization_risk',
            'payment_stress', 'early_risk_score'
        ]
        
        for feature in expected_features:
            assert feature in df_engineered.columns, f"Feature {feature} not created"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

