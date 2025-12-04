"""
Batch CSV Prediction Script
Predicts DPD Bucket Next Month for all customers in an uploaded CSV file
Uses the best performing model (Random Forest) for predictions
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import sys

def predict_batch_csv(csv_filepath, model_name='random_forest'):
    """
    Predict DPD Bucket for all customers in CSV file
    
    Parameters:
    -----------
    csv_filepath : str
        Path to CSV file with customer data
    model_name : str
        Name of model to use (default: 'random_forest')
        
    Returns:
    --------
    results : dict
        Dictionary with predictions and categorized customers
    """
    try:
        # Load CSV
        df = pd.read_csv(csv_filepath)
        
        # Required columns
        required_cols = [
            'Customer ID', 'Credit Limit', 'Utilisation %', 'Avg Payment Ratio',
            'Min Due Paid Frequency', 'Merchant Mix Index', 'Cash Withdrawal %',
            'Recent Spend Change %'
        ]
        
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {
                'error': f'Missing required columns: {", ".join(missing_cols)}',
                'success': False
            }
        
        # Load model and scaler
        model_path = f'models/{model_name}_model.joblib'
        scaler_path = f'models/{model_name}_scaler.joblib'
        
        if not os.path.exists(model_path):
            return {
                'error': f'Model file not found: {model_path}',
                'success': False
            }
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        # Prepare features (exclude Customer ID and DPD Bucket Next Month if present)
        feature_cols = [
            'Credit Limit', 'Utilisation %', 'Avg Payment Ratio',
            'Min Due Paid Frequency', 'Merchant Mix Index',
            'Cash Withdrawal %', 'Recent Spend Change %'
        ]
        
        X = df[feature_cols].values
        
        # Scale features if scaler exists
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)
        
        # Create results DataFrame
        results_df = df[['Customer ID']].copy()
        results_df['Predicted DPD Bucket'] = predictions
        
        # Map DPD Bucket to Risk Level
        risk_mapping = {
            0: 'No Risk',
            1: 'Low Risk',
            2: 'Medium Risk',
            3: 'High Risk'
        }
        results_df['Risk Level'] = results_df['Predicted DPD Bucket'].map(risk_mapping)
        
        # Add probabilities if available
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                results_df[f'Probability_DPD_{i}'] = probabilities[:, i]
        
        # Categorize customers by risk level
        categorized = {
            'No Risk': results_df[results_df['Risk Level'] == 'No Risk']['Customer ID'].tolist(),
            'Low Risk': results_df[results_df['Risk Level'] == 'Low Risk']['Customer ID'].tolist(),
            'Medium Risk': results_df[results_df['Risk Level'] == 'Medium Risk']['Customer ID'].tolist(),
            'High Risk': results_df[results_df['Risk Level'] == 'High Risk']['Customer ID'].tolist()
        }
        
        # Count customers per risk level
        risk_counts = {
            'No Risk': len(categorized['No Risk']),
            'Low Risk': len(categorized['Low Risk']),
            'Medium Risk': len(categorized['Medium Risk']),
            'High Risk': len(categorized['High Risk'])
        }
        
        # Convert DataFrame to dict for JSON serialization
        results_dict = results_df.to_dict('records')
        
        return {
            'success': True,
            'predictions': results_dict,
            'categorized': categorized,
            'risk_counts': risk_counts,
            'total_customers': len(df)
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

if __name__ == '__main__':
    # Command line usage
    if len(sys.argv) < 2:
        print(json.dumps({
            'error': 'Please provide CSV file path',
            'success': False
        }))
        sys.exit(1)
    
    csv_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'random_forest'
    
    result = predict_batch_csv(csv_path, model_name)
    print(json.dumps(result, indent=2))

