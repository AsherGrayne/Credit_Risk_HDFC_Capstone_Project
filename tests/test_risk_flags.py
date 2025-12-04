"""
Unit tests for risk flag generation
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.early_risk_signals import EarlyRiskSignalSystem


class TestRiskFlags:
    """Test suite for risk flag generation"""
    
    @pytest.fixture
    def system(self):
        """Create EarlyRiskSignalSystem instance"""
        return EarlyRiskSignalSystem()
    
    @pytest.fixture
    def engineered_data(self):
        """Create engineered dataframe"""
        df = pd.DataFrame({
            'Customer ID': ['C001', 'C002', 'C003', 'C004'],
            'Utilisation %': [95, 45, 85, 30],
            'Recent Spend Change %': [-25, 5, -20, 10],
            'Min Due Paid Frequency': [15, 60, 25, 50],
            'Cash Withdrawal %': [30, 5, 15, 8],
            'spending_stress': [2, 0, 2, 0],
            'utilization_risk': [3, 0, 2, 0],
            'payment_stress': [2, 0, 1, 0],
            'early_risk_score': [0.9, 0.1, 0.7, 0.2]
        })
        return df
    
    def test_critical_risk_flags(self, system, engineered_data):
        """Test that critical risk flags are generated correctly"""
        risk_flags_df = system.identify_risk_flags(engineered_data)
        
        # Customer with high utilization and low payment should be CRITICAL
        critical_customers = risk_flags_df[risk_flags_df['risk_level'] == 'CRITICAL']
        assert len(critical_customers) > 0, "Should have at least one CRITICAL customer"
    
    def test_flag_count(self, system, engineered_data):
        """Test that flag count is calculated correctly"""
        risk_flags_df = system.identify_risk_flags(engineered_data)
        
        # All customers should have flag_count >= 0
        assert all(risk_flags_df['flag_count'] >= 0), "Flag count should be non-negative"
    
    def test_risk_score_range(self, system, engineered_data):
        """Test that risk scores are in valid range"""
        risk_flags_df = system.identify_risk_flags(engineered_data)
        
        risk_scores = risk_flags_df['risk_score']
        assert all(risk_scores >= 0) and all(risk_scores <= 1), \
            "Risk scores should be between 0 and 1"
    
    def test_risk_levels(self, system, engineered_data):
        """Test that risk levels are valid"""
        risk_flags_df = system.identify_risk_flags(engineered_data)
        
        valid_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        assert all(risk_flags_df['risk_level'].isin(valid_levels)), \
            "All risk levels should be valid"
    
    def test_customer_id_preserved(self, system, engineered_data):
        """Test that customer IDs are preserved"""
        risk_flags_df = system.identify_risk_flags(engineered_data)
        
        original_ids = set(engineered_data['Customer ID'])
        flag_ids = set(risk_flags_df['customer_id'])
        
        assert original_ids == flag_ids, "All customer IDs should be preserved"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

