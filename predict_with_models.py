"""
Prediction Script using Saved Joblib Models
This script demonstrates how to use the saved models to predict DPD Bucket values
"""

import pandas as pd
import numpy as np
import joblib
import os

def predict_dpd_bucket(model_name, features):
    """
    Predict DPD Bucket using a saved model
    
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., 'Random Forest', 'Logistic Regression')
    features : dict or pd.DataFrame
        Feature values for prediction
        
    Returns:
    --------
    prediction : int
        Predicted DPD Bucket (0=No Risk, 1=Low Risk, 2=Medium Risk, 3=High Risk)
    """
    # Load model
    model_filename = f"models/{model_name.lower().replace(' ', '_')}_model.joblib"
    
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    
    model = joblib.load(model_filename)
    
    # Check if scaler is needed
    scaler_filename = f"models/{model_name.lower().replace(' ', '_')}_scaler.joblib"
    scaler = None
    if os.path.exists(scaler_filename):
        scaler = joblib.load(scaler_filename)
    
    # Prepare features
    if isinstance(features, dict):
        # Convert dict to DataFrame
        feature_order = [
            'Credit Limit', 'Utilisation %', 'Avg Payment Ratio', 
            'Min Due Paid Frequency', 'Merchant Mix Index', 
            'Cash Withdrawal %', 'Recent Spend Change %'
        ]
        feature_array = np.array([[features.get(col, 0) for col in feature_order]])
    else:
        feature_array = features.values if hasattr(features, 'values') else features
    
    # Scale if needed
    if scaler is not None:
        feature_array = scaler.transform(feature_array)
    
    # Predict
    prediction = model.predict(feature_array)[0]
    
    return prediction

def predict_batch(model_name, df):
    """
    Predict DPD Bucket for a batch of customers
    
    Parameters:
    -----------
    model_name : str
        Name of the model to use
    df : pd.DataFrame
        DataFrame with feature columns
        
    Returns:
    --------
    predictions : np.array
        Array of predicted DPD Buckets
    """
    # Load model
    model_filename = f"models/{model_name.lower().replace(' ', '_')}_model.joblib"
    
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    
    model = joblib.load(model_filename)
    
    # Check if scaler is needed
    scaler_filename = f"models/{model_name.lower().replace(' ', '_')}_scaler.joblib"
    scaler = None
    if os.path.exists(scaler_filename):
        scaler = joblib.load(scaler_filename)
    
    # Prepare features (remove Customer ID if present)
    feature_cols = [
        'Credit Limit', 'Utilisation %', 'Avg Payment Ratio', 
        'Min Due Paid Frequency', 'Merchant Mix Index', 
        'Cash Withdrawal %', 'Recent Spend Change %'
    ]
    
    X = df[feature_cols].values
    
    # Scale if needed
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X)
    
    return predictions

def get_risk_label(dpd_bucket):
    """Convert DPD Bucket to risk label"""
    risk_labels = {
        0: "No Risk",
        1: "Low Risk",
        2: "Medium Risk",
        3: "High Risk"
    }
    return risk_labels.get(dpd_bucket, "Unknown")

# Example usage
if __name__ == "__main__":
    print("="*80)
    print("DPD BUCKET PREDICTION USING SAVED MODELS")
    print("="*80)
    
    # Example 1: Single prediction
    print("\nExample 1: Single Customer Prediction")
    print("-"*80)
    
    # Sample customer features
    customer_features = {
        'Credit Limit': 150000,
        'Utilisation %': 85,
        'Avg Payment Ratio': 45,
        'Min Due Paid Frequency': 25,
        'Merchant Mix Index': 0.35,
        'Cash Withdrawal %': 18,
        'Recent Spend Change %': -22
    }
    
    print("\nCustomer Features:")
    for key, value in customer_features.items():
        print(f"  {key}: {value}")
    
    # Predict using different models
    models_to_test = ['Random Forest', 'Logistic Regression', 'Gradient Boosting']
    
    print("\nPredictions:")
    for model_name in models_to_test:
        try:
            prediction = predict_dpd_bucket(model_name, customer_features)
            risk_label = get_risk_label(prediction)
            print(f"  {model_name:25s}: DPD Bucket {prediction} ({risk_label})")
        except Exception as e:
            print(f"  {model_name:25s}: Error - {e}")
    
    # Example 2: Batch prediction
    print("\n" + "="*80)
    print("Example 2: Batch Prediction")
    print("-"*80)
    
    # Load sample data
    try:
        df = pd.read_csv('data/Sample.csv')
        print(f"\nLoaded {len(df)} customers from Sample.csv")
        
        # Use first 5 customers for demonstration
        df_sample = df.head(5).copy()
        
        # Predict using Random Forest
        predictions = predict_batch('Random Forest', df_sample)
        
        print("\nPredictions for first 5 customers:")
        print("-"*80)
        for idx, (_, row) in enumerate(df_sample.iterrows()):
            customer_id = row.get('Customer ID', f'Customer {idx+1}')
            actual = row.get('DPD Bucket Next Month', 'N/A')
            predicted = predictions[idx]
            predicted_label = get_risk_label(predicted)
            actual_label = get_risk_label(actual) if actual != 'N/A' else 'N/A'
            
            print(f"\n{customer_id}:")
            print(f"  Actual:   DPD Bucket {actual} ({actual_label})")
            print(f"  Predicted: DPD Bucket {predicted} ({predicted_label})")
            print(f"  Match:    {'✓' if actual == predicted else '✗'}")
        
    except Exception as e:
        print(f"Error in batch prediction: {e}")
    
    print("\n" + "="*80)
    print("Prediction examples completed!")
    print("="*80)

