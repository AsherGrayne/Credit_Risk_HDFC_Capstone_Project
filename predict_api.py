"""
Prediction API for Early Risk Signals
This script provides an API endpoint for risk prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from early_risk_signals import EarlyRiskSignalSystem

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the system
system = EarlyRiskSignalSystem()

# Load the trained model if available
model_loaded = False
try:
    if os.path.exists('trained_model.pkl'):
        with open('trained_model.pkl', 'rb') as f:
            system.model = pickle.load(f)
        model_loaded = True
except Exception as e:
    print(f"Could not load saved model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict risk for a single customer
    Expected JSON:
    {
        "customer_id": "C001",
        "credit_limit": 165000,
        "utilisation": 12,
        "avg_payment_ratio": 32,
        "min_due_frequency": 66,
        "merchant_mix": 0.73,
        "cash_withdrawal": 12,
        "spend_change": -21,
        "dpd_bucket": 3 (optional)
    }
    """
    try:
        data = request.json
        
        # Create a DataFrame with single row
        customer_data = pd.DataFrame([{
            'Customer ID': data.get('customer_id', 'C001'),
            'Credit Limit': data.get('credit_limit', 0),
            'Utilisation %': data.get('utilisation', 0),
            'Avg Payment Ratio': data.get('avg_payment_ratio', 0),
            'Min Due Paid Frequency': data.get('min_due_frequency', 0),
            'Merchant Mix Index': data.get('merchant_mix', 0),
            'Cash Withdrawal %': data.get('cash_withdrawal', 0),
            'Recent Spend Change %': data.get('spend_change', 0),
            'DPD Bucket Next Month': data.get('dpd_bucket', 0)
        }])
        
        # Engineer early signals
        df_engineered = system.engineer_early_signals(customer_data)
        
        # Generate risk flags
        risk_flags_df = system.identify_risk_flags(df_engineered)
        
        # Get risk level and score
        risk_level = risk_flags_df.iloc[0]['risk_level']
        risk_score = float(risk_flags_df.iloc[0]['risk_score'])
        flags = risk_flags_df.iloc[0]['flags']
        
        # Predict using model if available
        prediction = "Not At-Risk"
        probability = 0.0
        
        if model_loaded and system.model is not None:
            try:
                feature_cols = [
                    'Utilisation %', 'Avg Payment Ratio', 'Min Due Paid Frequency',
                    'Merchant Mix Index', 'Cash Withdrawal %', 'Recent Spend Change %',
                    'spending_stress', 'utilization_risk', 'payment_stress',
                    'cash_stress_indicator', 'utilization_payment_mismatch',
                    'spending_utilization_stress', 'payment_utilization_critical'
                ]
                
                X = df_engineered[feature_cols]
                X_scaled = system.scaler.transform(X)
                
                y_pred = system.model.predict(X_scaled)[0]
                y_pred_proba = system.model.predict_proba(X_scaled)[0]
                
                prediction = "At-Risk" if y_pred == 1 else "Not At-Risk"
                probability = float(y_pred_proba[1]) * 100
            except Exception as e:
                print(f"Model prediction error: {e}")
                # Fallback to risk score based prediction
                probability = risk_score * 100
                prediction = "At-Risk" if risk_score > 0.5 else "Not At-Risk"
        else:
            # Fallback prediction based on risk score
            probability = risk_score * 100
            prediction = "At-Risk" if risk_score > 0.5 else "Not At-Risk"
        
        # Format flags for JSON response
        formatted_flags = []
        for flag in flags:
            formatted_flags.append({
                'flag': flag['flag'],
                'severity': flag['severity'],
                'message': flag['message']
            })
        
        return jsonify({
            'risk_level': risk_level,
            'risk_score': round(risk_score, 3),
            'prediction': prediction,
            'probability': round(probability, 1),
            'flags': formatted_flags,
            'flag_count': len(formatted_flags)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': model_loaded})

if __name__ == '__main__':
    # Train model if not loaded
    if not model_loaded:
        print("Training model...")
        try:
            df = system.load_data('Sample.csv')
            df_engineered = system.engineer_early_signals(df)
            system.train_model(df_engineered)
            
            # Save model
            with open('trained_model.pkl', 'wb') as f:
                pickle.dump(system.model, f)
            print("Model trained and saved!")
        except Exception as e:
            print(f"Error training model: {e}")
    
    print("Starting prediction API server...")
    print("API available at: http://localhost:5000/predict")
    app.run(debug=True, host='0.0.0.0', port=5000)

