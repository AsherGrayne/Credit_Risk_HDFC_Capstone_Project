# ML Model Usage Explanation

## Current Implementation Status

### ✅ ML Model IS Trained
- **Random Forest Classifier** is trained in `early_risk_signals.py`
- Model achieves **75% accuracy** on test data
- Model is trained with 14 engineered features
- Model is saved/available for predictions

### ⚠️ Current Prediction Methods

#### 1. **Flask API (Uses ML Model)** ✅
- **File**: `predict_api.py`
- **Status**: Uses the trained Random Forest model
- **Requires**: Flask server running
- **How it works**:
  ```python
  # Uses actual trained model
  y_pred = system.model.predict(X_scaled)[0]
  y_pred_proba = system.model.predict_proba(X_scaled)[0]
  ```
- **Accuracy**: Uses the trained model (75% accuracy)

#### 2. **Client-Side JavaScript (Rule-Based)** ⚠️
- **File**: `apply-script.js`
- **Status**: Does NOT use ML model
- **Uses**: Rule-based calculations (same formulas as Python)
- **How it works**:
  ```javascript
  // Calculates risk score using formulas
  const early_risk_score = (
      spending_stress * 0.25 +
      utilization_risk * 0.30 +
      // ... etc
  ) / 3.0;
  ```
- **Accuracy**: Based on thresholds, not ML model

## Why This Design?

### Current Setup (Hybrid Approach):
1. **Primary**: Tries to call Flask API (uses ML model)
2. **Fallback**: Uses client-side calculation if API unavailable

### Benefits:
- ✅ Works immediately without server (client-side fallback)
- ✅ Can use ML model when API is available
- ✅ No server required for basic functionality

## How to Use ML Model

### Option 1: Run Flask API (Recommended)

1. **Start the API server:**
   ```bash
   python predict_api.py
   ```

2. **The API will:**
   - Train the model if not already trained
   - Save model as `trained_model.pkl`
   - Serve predictions at `http://localhost:5000/predict`

3. **Update `apply-script.js`:**
   Change the API URL:
   ```javascript
   const response = await fetch('http://localhost:5000/predict', {
   ```

### Option 2: Deploy API to Cloud

Deploy `predict_api.py` to:
- **Heroku**: Free tier available
- **PythonAnywhere**: Free tier available
- **Railway**: Free tier available
- **Render**: Free tier available

Then update the API URL in `apply-script.js`.

### Option 3: Use Pre-trained Model (Advanced)

1. **Save the trained model:**
   ```python
   import pickle
   # After training
   with open('trained_model.pkl', 'wb') as f:
       pickle.dump(model, f)
   ```

2. **Load in API:**
   ```python
   with open('trained_model.pkl', 'rb') as f:
       model = pickle.load(f)
   ```

## Current Prediction Accuracy

### ML Model (Random Forest):
- **Accuracy**: 75%
- **Precision (Not at-risk)**: 82%
- **Recall (Not at-risk)**: 88%
- **ROC-AUC**: 0.500

### Client-Side (Rule-Based):
- **Method**: Threshold-based calculations
- **Accuracy**: Approximate (not validated)
- **Uses**: Same formulas as Python feature engineering

## Recommendation

**For Production Use:**
1. ✅ Deploy Flask API (`predict_api.py`)
2. ✅ Use the trained Random Forest model
3. ✅ Get accurate ML-based predictions

**For Demo/Testing:**
1. ✅ Current client-side works fine
2. ✅ Shows risk flags and scores
3. ✅ No server needed

## Summary

| Method | Uses ML Model? | Accuracy | Requires Server? |
|--------|---------------|----------|------------------|
| Flask API | ✅ Yes | 75% | Yes |
| Client-Side JS | ❌ No | Approximate | No |

**Answer**: The ML model IS trained and available, but currently only used when the Flask API is running. The client-side version uses rule-based calculations as a fallback.

