"""
Feature Importance Analysis
Analyzes and visualizes feature importance from trained ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FeatureImportanceAnalyzer:
    """Analyze feature importance from trained ML models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            'Credit Limit', 'Utilisation %', 'Avg Payment Ratio', 
            'Min Due Paid Frequency', 'Merchant Mix Index', 
            'Cash Withdrawal %', 'Recent Spend Change %'
        ]
        self.importance_results = {}
        
    def load_models_and_data(self):
        """Load trained models and prepare data"""
        print("="*80)
        print("LOADING MODELS AND DATA")
        print("="*80)
        
        # Load data for permutation importance
        try:
            df = pd.read_csv('data/Credit_Card_Delinquency_Watch.csv')
            if len(df) < 10 or 'Field Name' in df.columns:
                df = pd.read_csv('data/Sample.csv')
        except:
            df = pd.read_csv('data/Sample.csv')
        
        print(f"✓ Loaded {len(df)} rows of data")
        
        # Preprocess data
        df_processed = df.copy()
        if 'Customer ID' in df_processed.columns:
            df_processed = df_processed.drop('Customer ID', axis=1)
        
        # Get target column
        target_col = 'DPD Bucket Next Month'
        if target_col not in df_processed.columns:
            target_col = [col for col in df_processed.columns if 'DPD' in col.upper()][0]
        
        # Prepare features and target
        X = df_processed[self.feature_names]
        y = df_processed[target_col]
        
        # Load models
        model_files = {
            'Random Forest': 'models/random_forest_model.joblib',
            'Gradient Boosting': 'models/gradient_boosting_model.joblib',
            'Decision Tree': 'models/decision_tree_model.joblib',
            'AdaBoost': 'models/adaboost_model.joblib',
            'Logistic Regression': 'models/logistic_regression_model.joblib',
            'SVM': 'models/svm_model.joblib',
            'Naive Bayes': 'models/naive_bayes_model.joblib',
            'K-Nearest Neighbors': 'models/k-nearest_neighbors_model.joblib'
        }
        
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"✓ Loaded {name}")
                
                # Load scaler if exists
                scaler_file = filepath.replace('_model.joblib', '_scaler.joblib')
                if os.path.exists(scaler_file):
                    self.scalers[name] = joblib.load(scaler_file)
        
        print(f"\n✓ Loaded {len(self.models)} models")
        
        return X, y
    
    def get_feature_importance(self, model_name, model, X, y):
        """Extract feature importance from a model"""
        importance_dict = {}
        
        # Models with built-in feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances))
            method = 'Built-in'
        
        # Logistic Regression - use coefficients
        elif hasattr(model, 'coef_'):
            # For multi-class, take mean of absolute coefficients
            if len(model.coef_.shape) > 1:
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_[0])
            # Normalize to sum to 1
            importances = importances / importances.sum()
            importance_dict = dict(zip(self.feature_names, importances))
            method = 'Coefficients'
        
        # Use permutation importance for other models
        else:
            try:
                # Prepare data
                if model_name in self.scalers:
                    X_scaled = self.scalers[model_name].transform(X)
                else:
                    X_scaled = X.values
                
                # Calculate permutation importance
                perm_importance = permutation_importance(
                    model, X_scaled, y, 
                    n_repeats=10, 
                    random_state=42,
                    n_jobs=-1
                )
                importances = perm_importance.importances_mean
                
                # Handle negative values by shifting to positive
                if importances.min() < 0:
                    importances = importances - importances.min()
                
                # Normalize to sum to 1 (or use absolute values if sum is too small)
                if importances.sum() > 0:
                    importances = importances / importances.sum()
                else:
                    # If all zeros or very small, use absolute values normalized
                    importances = np.abs(importances)
                    if importances.sum() > 0:
                        importances = importances / importances.sum()
                    else:
                        # Equal importance if all zero
                        importances = np.ones(len(importances)) / len(importances)
                
                importance_dict = dict(zip(self.feature_names, importances))
                method = 'Permutation'
            except Exception as e:
                print(f"  ⚠️  Could not calculate importance for {model_name}: {e}")
                return None, None
        
        return importance_dict, method
    
    def analyze_all_models(self, X, y):
        """Analyze feature importance for all models"""
        print("\n" + "="*80)
        print("ANALYZING FEATURE IMPORTANCE")
        print("="*80)
        
        for model_name, model in self.models.items():
            print(f"\nAnalyzing {model_name}...")
            
            importance_dict, method = self.get_feature_importance(model_name, model, X, y)
            
            if importance_dict:
                # Convert to DataFrame and sort
                importance_df = pd.DataFrame({
                    'feature': list(importance_dict.keys()),
                    'importance': list(importance_dict.values())
                }).sort_values('importance', ascending=False)
                
                self.importance_results[model_name] = {
                    'importance': importance_df,
                    'method': method
                }
                
                print(f"  ✓ Calculated using {method} method")
                print(f"  Top 3 features:")
                for idx, row in importance_df.head(3).iterrows():
                    print(f"    {row['feature']}: {row['importance']:.4f}")
    
    def plot_individual_feature_importance(self):
        """Plot feature importance for each model individually"""
        print("\n" + "="*80)
        print("GENERATING INDIVIDUAL FEATURE IMPORTANCE PLOTS")
        print("="*80)
        
        os.makedirs('visualizations/new_visualization', exist_ok=True)
        
        for model_name, result in self.importance_results.items():
            importance_df = result['importance']
            method = result['method']
            
            plt.figure(figsize=(12, 8))
            
            # Create horizontal bar plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            bars = plt.barh(range(len(importance_df)), 
                           importance_df['importance'].values, 
                           color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            plt.yticks(range(len(importance_df)), importance_df['feature'].values)
            plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
            plt.title(f'{model_name} - Feature Importance\n(Method: {method})', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (idx, row) in enumerate(importance_df.iterrows()):
                plt.text(row['importance'] + 0.005, i, f'{row["importance"]:.4f}', 
                        va='center', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            
            filename = f"visualizations/new_visualization/{model_name.lower().replace(' ', '_')}_feature_importance.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved {model_name} feature importance to {filename}")
            plt.close()
    
    def plot_combined_feature_importance(self):
        """Plot combined feature importance comparison"""
        print("\n" + "="*80)
        print("GENERATING COMBINED FEATURE IMPORTANCE COMPARISON")
        print("="*80)
        
        # Create a matrix of feature importance across all models
        importance_matrix = []
        model_names = []
        
        for model_name, result in self.importance_results.items():
            importance_df = result['importance']
            # Reorder to match feature_names order
            importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
            importance_values = [importance_dict.get(f, 0) for f in self.feature_names]
            importance_matrix.append(importance_values)
            model_names.append(model_name)
        
        importance_matrix = np.array(importance_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(importance_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   xticklabels=self.feature_names,
                   yticklabels=model_names,
                   cbar_kws={'label': 'Feature Importance'},
                   linewidths=0.5,
                   linecolor='gray')
        
        plt.title('Feature Importance Comparison Across All Models', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Features', fontsize=12, fontweight='bold')
        plt.ylabel('Models', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = 'visualizations/new_visualization/all_models_feature_importance_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved combined comparison to {filename}")
        plt.close()
    
    def plot_average_feature_importance(self):
        """Plot average feature importance across all models"""
        print("\n" + "="*80)
        print("GENERATING AVERAGE FEATURE IMPORTANCE")
        print("="*80)
        
        # Calculate average importance across all models
        avg_importance = {}
        
        for feature in self.feature_names:
            importances = []
            for model_name, result in self.importance_results.items():
                importance_df = result['importance']
                feature_importance = importance_df[importance_df['feature'] == feature]['importance'].values
                if len(feature_importance) > 0:
                    importances.append(feature_importance[0])
            
            if importances:
                avg_importance[feature] = np.mean(importances)
        
        # Convert to DataFrame and sort
        avg_df = pd.DataFrame({
            'feature': list(avg_importance.keys()),
            'avg_importance': list(avg_importance.values())
        }).sort_values('avg_importance', ascending=False)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(avg_df)))
        bars = plt.barh(range(len(avg_df)), 
                       avg_df['avg_importance'].values, 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        plt.yticks(range(len(avg_df)), avg_df['feature'].values)
        plt.xlabel('Average Feature Importance', fontsize=12, fontweight='bold')
        plt.title('Average Feature Importance Across All Models', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(avg_df.iterrows()):
            plt.text(row['avg_importance'] + 0.005, i, f'{row["avg_importance"]:.4f}', 
                    va='center', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        
        filename = 'visualizations/new_visualization/average_feature_importance.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved average feature importance to {filename}")
        plt.close()
        
        # Print summary
        print("\n" + "-"*80)
        print("AVERAGE FEATURE IMPORTANCE RANKING")
        print("-"*80)
        for rank, (idx, row) in enumerate(avg_df.iterrows(), 1):
            print(f"{rank}. {row['feature']:30s}: {row['avg_importance']:.4f}")
    
    def save_importance_data(self):
        """Save feature importance data to CSV files"""
        print("\n" + "="*80)
        print("SAVING FEATURE IMPORTANCE DATA")
        print("="*80)
        
        os.makedirs('results', exist_ok=True)
        
        # Save individual model importance
        for model_name, result in self.importance_results.items():
            filename = f"results/{model_name.lower().replace(' ', '_')}_feature_importance.csv"
            result['importance'].to_csv(filename, index=False)
            print(f"✓ Saved {model_name} importance to {filename}")
        
        # Save average importance
        avg_importance = {}
        for feature in self.feature_names:
            importances = []
            for model_name, result in self.importance_results.items():
                importance_df = result['importance']
                feature_importance = importance_df[importance_df['feature'] == feature]['importance'].values
                if len(feature_importance) > 0:
                    importances.append(feature_importance[0])
            if importances:
                avg_importance[feature] = np.mean(importances)
        
        avg_df = pd.DataFrame({
            'feature': list(avg_importance.keys()),
            'avg_importance': list(avg_importance.values())
        }).sort_values('avg_importance', ascending=False)
        
        filename = 'results/average_feature_importance.csv'
        avg_df.to_csv(filename, index=False)
        print(f"✓ Saved average importance to {filename}")


def main():
    """Main execution function"""
    analyzer = FeatureImportanceAnalyzer()
    
    # Load models and data
    X, y = analyzer.load_models_and_data()
    
    # Analyze all models
    analyzer.analyze_all_models(X, y)
    
    # Generate visualizations
    analyzer.plot_individual_feature_importance()
    analyzer.plot_combined_feature_importance()
    analyzer.plot_average_feature_importance()
    
    # Save data
    analyzer.save_importance_data()
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  - Individual feature importance plots: visualizations/new_visualization/*_feature_importance.png")
    print("  - Combined comparison: visualizations/new_visualization/all_models_feature_importance_comparison.png")
    print("  - Average importance: visualizations/new_visualization/average_feature_importance.png")
    print("  - CSV data files: results/*_feature_importance.csv")


if __name__ == "__main__":
    main()

