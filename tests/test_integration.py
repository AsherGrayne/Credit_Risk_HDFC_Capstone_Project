"""
Integration tests for end-to-end workflow
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.early_risk_signals import EarlyRiskSignalSystem


class TestIntegration:
    """Integration tests for complete workflow"""
    
    @pytest.fixture
    def system(self):
        """Create EarlyRiskSignalSystem instance"""
        return EarlyRiskSignalSystem()
    
    def test_end_to_end_workflow(self, system):
        """Test complete workflow from data loading to strategy generation"""
        # Step 1: Load data
        df = system.load_data('data/Sample.csv')
        assert len(df) > 0, "Should load data successfully"
        
        # Step 2: Engineer features
        df_engineered = system.engineer_early_signals(df)
        assert len(df_engineered.columns) > len(df.columns), \
            "Should have more columns after feature engineering"
        
        # Step 3: Generate risk flags
        risk_flags_df = system.identify_risk_flags(df_engineered)
        assert len(risk_flags_df) == len(df), \
            "Should have risk flags for all customers"
        
        # Step 4: Train model
        system.train_model(df_engineered)
        assert system.model is not None, "Model should be trained"
        
        # Step 5: Generate outreach strategies
        strategies_df = system.generate_outreach_strategies(risk_flags_df)
        assert len(strategies_df) == len(df), \
            "Should have strategies for all customers"
    
    def test_data_consistency(self, system):
        """Test that data remains consistent throughout workflow"""
        df = system.load_data('data/Sample.csv')
        original_customer_ids = set(df['Customer ID'])
        
        df_engineered = system.engineer_early_signals(df)
        risk_flags_df = system.identify_risk_flags(df_engineered)
        strategies_df = system.generate_outreach_strategies(risk_flags_df)
        
        # Check customer IDs are preserved
        flag_ids = set(risk_flags_df['customer_id'])
        strategy_ids = set(strategies_df['customer_id'])
        
        assert original_customer_ids == flag_ids == strategy_ids, \
            "Customer IDs should be consistent throughout workflow"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

