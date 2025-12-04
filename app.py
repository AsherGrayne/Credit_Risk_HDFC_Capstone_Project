"""
Flask API for Batch CSV Prediction
Provides endpoint for predicting DPD Bucket for uploaded CSV files
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MODEL_NAME = 'random_forest'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict DPD Bucket for all customers in uploaded CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided', 'success': False}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file', 'success': False}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Load CSV
            df = pd.read_csv(filepath)
            
            # Required columns
            required_cols = [
                'Customer ID', 'Credit Limit', 'Utilisation %', 'Avg Payment Ratio',
                'Min Due Paid Frequency', 'Merchant Mix Index', 'Cash Withdrawal %',
                'Recent Spend Change %'
            ]
            
            # Check if all required columns exist
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return jsonify({
                    'error': f'Missing required columns: {", ".join(missing_cols)}',
                    'success': False
                }), 400
            
            # Load model and scaler
            model_path = f'models/{MODEL_NAME}_model.joblib'
            scaler_path = f'models/{MODEL_NAME}_scaler.joblib'
            
            if not os.path.exists(model_path):
                return jsonify({
                    'error': f'Model file not found: {model_path}',
                    'success': False
                }), 500
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            
            # Prepare features
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
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'predictions': results_dict,
                'categorized': categorized,
                'risk_counts': risk_counts,
                'total_customers': len(df)
            })
            
        except Exception as e:
            # Clean up uploaded file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'error': str(e),
                'success': False
            }), 500
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'API is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

