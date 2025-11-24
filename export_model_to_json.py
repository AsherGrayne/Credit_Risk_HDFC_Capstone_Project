"""
Export Random Forest model to JSON format for JavaScript use
This allows the trained ML model to be used client-side without a server
"""

import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from early_risk_signals import EarlyRiskSignalSystem

def export_tree_to_dict(tree, feature_names):
    """Recursively convert a decision tree to a dictionary"""
    def recurse(node, depth=0):
        if tree.children_left[node] == tree.children_right[node]:  # Leaf node
            return {
                'leaf': True,
                'value': tree.value[node][0].tolist(),  # [class_0_count, class_1_count]
                'samples': int(tree.n_node_samples[node])
            }
        else:
            return {
                'leaf': False,
                'feature': feature_names[tree.feature[node]],
                'threshold': float(tree.threshold[node]),
                'left': recurse(tree.children_left[node], depth + 1),
                'right': recurse(tree.children_right[node], depth + 1),
                'samples': int(tree.n_node_samples[node])
            }
    
    return recurse(0)

def export_model_to_json():
    """Export trained Random Forest model to JSON"""
    
    print("Loading and training model...")
    
    # Initialize system and load data
    system = EarlyRiskSignalSystem()
    df = system.load_data('Sample.csv')
    df_engineered = system.engineer_early_signals(df)
    
    # Train model
    model = system.train_model(df_engineered)
    
    # Get feature columns
    feature_cols = [
        'Utilisation %', 'Avg Payment Ratio', 'Min Due Paid Frequency',
        'Merchant Mix Index', 'Cash Withdrawal %', 'Recent Spend Change %',
        'spending_stress', 'utilization_risk', 'payment_stress',
        'cash_stress_indicator', 'utilization_payment_mismatch',
        'spending_utilization_stress', 'payment_utilization_critical'
    ]
    
    # Prepare data for scaler
    X = df_engineered[feature_cols]
    y = (df_engineered['DPD Bucket Next Month'] > 0).astype(int)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Export model structure
    model_data = {
        'n_estimators': len(model.estimators_),
        'feature_names': feature_cols,
        'scaler': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        },
        'trees': []
    }
    
    print(f"Exporting {len(model.estimators_)} trees...")
    
    # Export each tree
    for i, tree in enumerate(model.estimators_):
        if i % 10 == 0:
            print(f"  Exporting tree {i+1}/{len(model.estimators_)}...")
        tree_dict = export_tree_to_dict(tree.tree_, feature_cols)
        model_data['trees'].append(tree_dict)
    
    # Save to JSON
    output_file = 'model.json'
    with open(output_file, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"\n✓ Model exported to {output_file}")
    print(f"  File size: {len(json.dumps(model_data)) / 1024:.2f} KB")
    print(f"  Number of trees: {len(model.estimators_)}")
    print(f"  Features: {len(feature_cols)}")
    
    return model_data

if __name__ == '__main__':
    export_model_to_json()

