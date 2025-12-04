# Batch CSV Prediction Guide

## Overview
The "See your output" tab now has two partitions:
1. **Upload the entire CSV** - Upload a CSV file and get batch predictions for all customers
2. **See individual customer** - Enter individual customer data for single prediction

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Flask API Server
The batch CSV prediction requires the Flask API to be running:

```bash
python app.py
```

The API will start on `http://localhost:5000`

### 3. Open the Website
Open `index.html` in your web browser.

## Usage

### Upload CSV File
1. Click on "See your output" tab
2. Click "Upload the entire CSV" partition (default)
3. Click "Choose CSV File" and select your CSV file
4. The CSV file should have these columns:
   - Customer ID
   - Credit Limit
   - Utilisation %
   - Avg Payment Ratio
   - Min Due Paid Frequency
   - Merchant Mix Index
   - Cash Withdrawal %
   - Recent Spend Change %
   - DPD Bucket Next Month (optional)

5. Click "Predict DPD Bucket" button
6. Results will show:
   - Pie chart showing risk level distribution
   - Categorized customer lists by risk level (No Risk, Low Risk, Medium Risk, High Risk)

### Individual Customer Prediction
1. Click "See individual customer" partition
2. Fill in the form with customer data
3. Click "Predict Risk"
4. View the prediction results

## API Endpoints

### POST /predict_batch
Upload a CSV file and get batch predictions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: CSV file (form field name: `file`)

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "Customer ID": "C001",
      "Predicted DPD Bucket": 3,
      "Risk Level": "High Risk"
    }
  ],
  "categorized": {
    "No Risk": ["C002", "C003"],
    "Low Risk": ["C004"],
    "Medium Risk": ["C005"],
    "High Risk": ["C001"]
  },
  "risk_counts": {
    "No Risk": 2,
    "Low Risk": 1,
    "Medium Risk": 1,
    "High Risk": 1
  },
  "total_customers": 5
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running"
}
```

## Model Used
The batch prediction uses the **Random Forest** model (saved as `models/random_forest_model.joblib`).

## Fallback Behavior
If the Flask API is not available, the system will attempt a client-side rule-based prediction as a fallback. However, for accurate predictions using the trained ML models, the Flask API must be running.

## Troubleshooting

### API Not Running
- Make sure Flask is installed: `pip install flask flask-cors`
- Start the API: `python app.py`
- Check that port 5000 is not in use

### CSV Format Errors
- Ensure all required columns are present
- Check that column names match exactly (case-sensitive)
- Verify data types are correct (numbers should be numeric)

### Model Files Missing
- Ensure `models/random_forest_model.joblib` and `models/random_forest_scaler.joblib` exist
- Run `ml_model_training.py` to generate model files if missing

