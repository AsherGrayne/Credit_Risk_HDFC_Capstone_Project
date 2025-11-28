"""
Predict Credit Card Delinquency for Real Dataset
This script loads the real dataset, trains models, and predicts which customers will be delinquent.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from early_risk_signals import EarlyRiskSignalSystem

def predict_delinquency():
    """
    Main function to predict delinquency for all customers in the real dataset.
    """
    print("="*80)
    print("CREDIT CARD DELINQUENCY PREDICTION - REAL DATASET")
    print("="*80)
    
    # Initialize system
    system = EarlyRiskSignalSystem()
    
    # Step 1: Load real dataset
    print("\n[1/5] Loading real dataset (Sample.csv)...")
    df = system.load_data('data/Sample.csv')
    print(f"✓ Loaded {len(df)} customer records")
    
    # Step 2: Engineer features
    print("\n[2/5] Engineering early warning signals...")
    df_engineered = system.engineer_early_signals(df)
    print(f"✓ Created {len(df_engineered.columns) - len(df.columns)} new features")
    
    # Step 3: Train model
    print("\n[3/5] Training machine learning models...")
    system.train_model(df_engineered)
    print("✓ Models trained successfully")
    
    # Step 4: Make predictions for all customers
    print("\n[4/5] Making predictions for all customers...")
    
    # Prepare features
    feature_cols = [
        'Utilisation %', 'Avg Payment Ratio', 'Min Due Paid Frequency',
        'Merchant Mix Index', 'Cash Withdrawal %', 'Recent Spend Change %',
        'spending_stress', 'utilization_risk', 'payment_stress',
        'cash_stress_indicator', 'utilization_payment_mismatch',
        'spending_utilization_stress', 'payment_utilization_critical'
    ]
    
    X = df_engineered[feature_cols]
    
    # Scale features using the scaler from training
    # The scaler was already fitted during model training
    X_scaled = system.scaler.transform(X)
    
    # Make predictions using all three models
    predictions = {}
    
    # Random Forest predictions
    rf_model = system.model  # Already trained Random Forest
    rf_predictions = rf_model.predict(X_scaled)
    rf_probabilities = rf_model.predict_proba(X_scaled)[:, 1]
    
    predictions['Random Forest'] = {
        'predictions': rf_predictions,
        'probabilities': rf_probabilities
    }
    
    # Get other models from model_results if available
    if hasattr(system, 'model_results'):
        for model_name, model_data in system.model_results.items():
            if model_name != 'Random Forest':
                model = model_data['model']
                pred = model.predict(X_scaled)
                proba = model.predict_proba(X_scaled)[:, 1]
                predictions[model_name] = {
                    'predictions': pred,
                    'probabilities': proba
                }
    
    # Step 5: Create prediction results dataframe
    print("\n[5/5] Creating prediction results...")
    
    results = []
    
    for idx, row in df_engineered.iterrows():
        customer_id = row['Customer ID']
        actual_dpd = row['DPD Bucket Next Month']
        actual_status = 'Delinquent' if actual_dpd > 0 else 'Not Delinquent'
        
        # Get predictions from Random Forest (primary model)
        rf_pred = rf_predictions[idx]
        rf_prob = rf_probabilities[idx]
        rf_status = 'Predicted Delinquent' if rf_pred == 1 else 'Predicted Not Delinquent'
        
        # Get risk flags
        risk_flags = system.identify_risk_flags(df_engineered.iloc[[idx]])
        risk_level = risk_flags.iloc[0]['risk_level'] if len(risk_flags) > 0 else 'LOW'
        flag_count = risk_flags.iloc[0]['flag_count'] if len(risk_flags) > 0 else 0
        
        # Calculate early risk score
        early_risk_score = row['early_risk_score']
        
        result = {
            'Customer ID': customer_id,
            'Actual DPD Bucket': actual_dpd,
            'Actual Status': actual_status,
            'Predicted Status (RF)': rf_status,
            'Prediction Probability (RF)': rf_prob,
            'Risk Level': risk_level,
            'Early Risk Score': early_risk_score,
            'Flag Count': flag_count,
            'Utilisation %': row['Utilisation %'],
            'Min Due Paid Frequency': row['Min Due Paid Frequency'],
            'Recent Spend Change %': row['Recent Spend Change %'],
            'Avg Payment Ratio': row['Avg Payment Ratio']
        }
        
        # Add predictions from other models if available
        for model_name, model_preds in predictions.items():
            if model_name != 'Random Forest':
                pred = model_preds['predictions'][idx]
                prob = model_preds['probabilities'][idx]
                result[f'Predicted Status ({model_name})'] = 'Predicted Delinquent' if pred == 1 else 'Predicted Not Delinquent'
                result[f'Prediction Probability ({model_name})'] = prob
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Step 6: Analyze predictions
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    
    # Count predictions
    predicted_delinquent = (rf_predictions == 1).sum()
    predicted_not_delinquent = (rf_predictions == 0).sum()
    actual_delinquent = (df_engineered['DPD Bucket Next Month'] > 0).sum()
    actual_not_delinquent = (df_engineered['DPD Bucket Next Month'] == 0).sum()
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Customers: {len(df)}")
    print(f"  Actual Delinquent: {actual_delinquent} ({actual_delinquent/len(df)*100:.1f}%)")
    print(f"  Actual Not Delinquent: {actual_not_delinquent} ({actual_not_delinquent/len(df)*100:.1f}%)")
    
    print(f"\n🤖 Model Predictions (Random Forest):")
    print(f"  Predicted Delinquent: {predicted_delinquent} ({predicted_delinquent/len(df)*100:.1f}%)")
    print(f"  Predicted Not Delinquent: {predicted_not_delinquent} ({predicted_not_delinquent/len(df)*100:.1f}%)")
    
    # Calculate accuracy
    correct_predictions = (rf_predictions == (df_engineered['DPD Bucket Next Month'] > 0).astype(int)).sum()
    accuracy = correct_predictions / len(df) * 100
    
    print(f"\n✅ Prediction Accuracy:")
    print(f"  Correct Predictions: {correct_predictions}/{len(df)}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Confusion matrix
    true_positives = ((rf_predictions == 1) & (df_engineered['DPD Bucket Next Month'] > 0)).sum()
    true_negatives = ((rf_predictions == 0) & (df_engineered['DPD Bucket Next Month'] == 0)).sum()
    false_positives = ((rf_predictions == 1) & (df_engineered['DPD Bucket Next Month'] == 0)).sum()
    false_negatives = ((rf_predictions == 0) & (df_engineered['DPD Bucket Next Month'] > 0)).sum()
    
    print(f"\n📈 Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Delinquent  Not Delinquent")
    print(f"  Actual Delinquent    {true_positives:3d}         {false_negatives:3d}")
    print(f"  Actual Not Delinquent {false_positives:3d}         {true_negatives:3d}")
    
    # Precision and Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print(f"\n📊 Model Performance Metrics:")
    print(f"  Precision (Delinquent): {precision:.3f} ({precision*100:.1f}%)")
    print(f"  Recall (Delinquent): {recall:.3f} ({recall*100:.1f}%)")
    
    # List predicted delinquent customers
    predicted_delinquent_df = results_df[results_df['Predicted Status (RF)'] == 'Predicted Delinquent'].copy()
    predicted_delinquent_df = predicted_delinquent_df.sort_values('Prediction Probability (RF)', ascending=False)
    
    print(f"\n🚨 Customers Predicted as Delinquent ({len(predicted_delinquent_df)}):")
    print("="*80)
    print(f"{'Customer ID':<15} {'Probability':<15} {'Risk Level':<15} {'Actual Status':<20}")
    print("-"*80)
    for idx, row in predicted_delinquent_df.iterrows():
        print(f"{row['Customer ID']:<15} {row['Prediction Probability (RF)']*100:>6.1f}%       {row['Risk Level']:<15} {row['Actual Status']:<20}")
    
    # Save results
    output_file = 'data/delinquency_predictions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved predictions to: {output_file}")
    
    # Save predicted delinquent customers separately
    high_risk_file = 'data/predicted_delinquent_customers.csv'
    predicted_delinquent_df.to_csv(high_risk_file, index=False)
    print(f"💾 Saved predicted delinquent customers to: {high_risk_file}")
    
    print("\n" + "="*80)
    print("✅ Prediction Complete!")
    print("="*80)
    
    return results_df, predicted_delinquent_df

if __name__ == "__main__":
    results_df, predicted_delinquent_df = predict_delinquency()
    
    # Display top 10 highest risk customers
    print("\n" + "="*80)
    print("TOP 10 HIGHEST RISK CUSTOMERS")
    print("="*80)
    top_10 = predicted_delinquent_df.head(10) if len(predicted_delinquent_df) > 0 else results_df.nlargest(10, 'Prediction Probability (RF)')
    
    for idx, row in top_10.iterrows():
        print(f"\nCustomer ID: {row['Customer ID']}")
        print(f"  Prediction Probability: {row['Prediction Probability (RF)']*100:.1f}%")
        print(f"  Risk Level: {row['Risk Level']}")
        print(f"  Early Risk Score: {row['Early Risk Score']:.3f}")
        print(f"  Flag Count: {row['Flag Count']}")
        print(f"  Utilisation: {row['Utilisation %']:.1f}%")
        print(f"  Payment Frequency: {row['Min Due Paid Frequency']:.1f}%")
        print(f"  Spend Change: {row['Recent Spend Change %']:.1f}%")

