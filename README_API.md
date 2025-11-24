# Prediction API Setup Guide

## Overview
The "Apply Here" feature allows users to input customer parameters and get real-time risk predictions using the Random Forest model.

## Setup Options

### Option 1: Local Flask Server (Recommended for Development)

1. **Install Dependencies**
   ```bash
   pip install flask flask-cors
   ```

2. **Start the API Server**
   ```bash
   python predict_api.py
   ```
   The API will run on `http://localhost:5000`

3. **Update apply-script.js**
   Change the fetch URL in `apply-script.js`:
   ```javascript
   const response = await fetch('http://localhost:5000/predict', {
   ```

### Option 2: Deploy to Cloud Platform

#### Heroku Deployment
1. Create `Procfile`:
   ```
   web: python predict_api.py
   ```

2. Create `runtime.txt`:
   ```
   python-3.10.0
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

#### PythonAnywhere
1. Upload `predict_api.py` and `early_risk_signals.py`
2. Upload `Sample.csv` for model training
3. Configure web app to run `predict_api.py`
4. Update API URL in `apply-script.js`

#### Replit
1. Create new Repl
2. Upload all Python files
3. Run `predict_api.py`
4. Update API URL in `apply-script.js`

### Option 3: Client-Side Only (Current Implementation)

The current implementation includes a **fallback client-side prediction** that works without a server:

- Uses the same logic as the Python code
- Calculates early warning signals in JavaScript
- Generates risk flags and scores
- Works immediately without setup

**Note**: For production use, deploy the Flask API for more accurate predictions using the trained model.

## API Endpoint

**POST** `/predict`

**Request Body:**
```json
{
    "customer_id": "C001",
    "credit_limit": 165000,
    "utilisation": 12,
    "avg_payment_ratio": 32,
    "min_due_frequency": 66,
    "merchant_mix": 0.73,
    "cash_withdrawal": 12,
    "spend_change": -21,
    "dpd_bucket": 3
}
```

**Response:**
```json
{
    "risk_level": "HIGH",
    "risk_score": 0.200,
    "prediction": "At-Risk",
    "probability": 24.0,
    "flags": [
        {
            "flag": "SPENDING_DECLINE_SEVERE",
            "severity": "HIGH",
            "message": "Spending dropped 21% - potential financial stress"
        }
    ],
    "flag_count": 1
}
```

## Testing

Test the API locally:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C001",
    "credit_limit": 165000,
    "utilisation": 12,
    "avg_payment_ratio": 32,
    "min_due_frequency": 66,
    "merchant_mix": 0.73,
    "cash_withdrawal": 12,
    "spend_change": -21
  }'
```

## Current Status

✅ **Client-side prediction works immediately** - No server setup required
⚠️ **API server** - Requires Flask setup for model-based predictions

The client-side fallback provides accurate risk scoring and flag generation using the same logic as the Python implementation.

