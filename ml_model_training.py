"""
Machine Learning Model Training and Evaluation
Trains multiple classification models to predict DPD Bucket Next Month
DPD Bucket: 0=No Risk, 1=Low Risk, 2=Medium Risk, 3=High Risk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MLModelTrainer:
    """Train and evaluate multiple ML models for DPD Bucket prediction"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the dataset"""
        print("="*80)
        print("LOADING AND PREPROCESSING DATA")
        print("="*80)
        
        # Try to load the specified file
        try:
            df = pd.read_csv(filepath)
            print(f"\n✓ Loaded {len(df)} rows from {filepath}")
            
            # Check if file only has metadata (like Credit_Card_Delinquency_Watch.csv)
            if len(df) < 10 or 'Field Name' in df.columns or 'Description' in df.columns:
                print("⚠️  File appears to contain only metadata. Using Sample.csv instead...")
                df = pd.read_csv('data/Sample.csv')
                print(f"✓ Loaded {len(df)} rows from Sample.csv")
        except Exception as e:
            print(f"⚠️  Error loading {filepath}: {e}")
            print("Using Sample.csv instead...")
            df = pd.read_csv('data/Sample.csv')
            print(f"✓ Loaded {len(df)} rows from Sample.csv")
        
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display basic info
        print(f"\nTarget variable distribution:")
        if 'DPD Bucket Next Month' in df.columns:
            target_col = 'DPD Bucket Next Month'
        elif 'Delinquency_Flag_Next_Month (DPD_Bucket)' in df.columns:
            target_col = 'Delinquency_Flag_Next_Month (DPD_Bucket)'
        else:
            # Try to find DPD related column
            target_col = [col for col in df.columns if 'DPD' in col.upper()][0]
        
        print(df[target_col].value_counts().sort_index())
        
        # Data preprocessing
        print("\n" + "-"*80)
        print("DATA PREPROCESSING")
        print("-"*80)
        
        # Create a copy for processing
        df_processed = df.copy()
        
        # Remove Customer ID if present (not a feature)
        if 'Customer ID' in df_processed.columns:
            df_processed = df_processed.drop('Customer ID', axis=1)
        elif 'Customer_ID' in df_processed.columns:
            df_processed = df_processed.drop('Customer_ID', axis=1)
        
        # Handle missing values
        print(f"\nMissing values per column:")
        missing = df_processed.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values found")
        
        if missing.sum() > 0:
            df_processed = df_processed.fillna(df_processed.median())
            print("✓ Filled missing values with median")
        
        # Separate features and target
        feature_cols = [col for col in df_processed.columns if col != target_col]
        X = df_processed[feature_cols]
        y = df_processed[target_col]
        
        print(f"\nFeatures used: {feature_cols}")
        print(f"Target variable: {target_col}")
        print(f"Target classes: {sorted(y.unique())}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✓ Data preprocessing completed")
        
        return feature_cols
    
    def initialize_models(self):
        """Initialize multiple ML models"""
        print("\n" + "="*80)
        print("INITIALIZING MODELS")
        print("="*80)
        
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial'),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        print(f"\n✓ Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\n" + "="*80)
        print("TRAINING AND EVALUATING MODELS")
        print("="*80)
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*80}")
            print(f"Training {name}...")
            print(f"{'='*80}")
            
            try:
                # Determine if model needs scaled data
                if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors', 'Naive Bayes']:
                    X_train_use = self.X_train_scaled
                    X_test_use = self.X_test_scaled
                else:
                    X_train_use = self.X_train.values
                    X_test_use = self.X_test.values
                
                # Train model
                model.fit(X_train_use, self.y_train)
                
                # Predictions
                y_pred = model.predict(X_test_use)
                
                # Calculate accuracy
                accuracy = accuracy_score(self.y_test, y_pred)
                
                # Confusion matrix
                cm = confusion_matrix(self.y_test, y_pred)
                
                # Classification report
                report = classification_report(self.y_test, y_pred, output_dict=True)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'confusion_matrix': cm,
                    'predictions': y_pred,
                    'classification_report': report
                }
                
                print(f"✓ {name} trained successfully")
                print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                # Save model
                model_filename = f"models/{name.lower().replace(' ', '_')}_model.joblib"
                os.makedirs('models', exist_ok=True)
                joblib.dump(model, model_filename)
                print(f"✓ Model saved to {model_filename}")
                
                # Save scaler if needed
                if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors', 'Naive Bayes']:
                    scaler_filename = f"models/{name.lower().replace(' ', '_')}_scaler.joblib"
                    joblib.dump(self.scaler, scaler_filename)
                    print(f"✓ Scaler saved to {scaler_filename}")
                
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
                continue
        
        print(f"\n{'='*80}")
        print(f"Training completed for {len(self.results)} models")
        print(f"{'='*80}")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrix for each model"""
        print("\n" + "="*80)
        print("GENERATING CONFUSION MATRICES")
        print("="*80)
        
        os.makedirs('visualizations/new_visualization', exist_ok=True)
        
        n_models = len(self.results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            accuracy = result['accuracy']
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['No Risk', 'Low Risk', 'Medium Risk', 'High Risk'],
                       yticklabels=['No Risk', 'Low Risk', 'Medium Risk', 'High Risk'])
            axes[idx].set_title(f'{name}\nAccuracy: {accuracy:.4f}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        cm_filename = 'visualizations/new_visualization/all_confusion_matrices.png'
        plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved all confusion matrices to {cm_filename}")
        plt.close()
        
        # Individual confusion matrices
        for name, result in self.results.items():
            cm = result['confusion_matrix']
            accuracy = result['accuracy']
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Risk', 'Low Risk', 'Medium Risk', 'High Risk'],
                       yticklabels=['No Risk', 'Low Risk', 'Medium Risk', 'High Risk'],
                       cbar_kws={'label': 'Count'})
            plt.title(f'{name} - Confusion Matrix\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Predicted DPD Bucket', fontsize=12)
            plt.ylabel('Actual DPD Bucket', fontsize=12)
            plt.tight_layout()
            
            cm_filename = f"visualizations/new_visualization/{name.lower().replace(' ', '_')}_confusion_matrix.png"
            plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved {name} confusion matrix to {cm_filename}")
            plt.close()
    
    def plot_accuracy_comparison(self):
        """Plot accuracy comparison of all models"""
        print("\n" + "="*80)
        print("GENERATING ACCURACY COMPARISON")
        print("="*80)
        
        os.makedirs('visualizations/new_visualization', exist_ok=True)
        
        # Extract accuracies
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = ax1.barh(model_names, accuracies, color=colors)
        ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlim([0, 1])
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax1.text(acc + 0.01, i, f'{acc:.4f}\n({acc*100:.2f}%)', 
                    va='center', fontsize=10, fontweight='bold')
        
        # Horizontal bar plot sorted
        sorted_data = sorted(zip(model_names, accuracies), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_accs = zip(*sorted_data)
        
        bars2 = ax2.barh(sorted_names, sorted_accs, color=colors)
        ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Model Accuracy Comparison (Sorted)', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlim([0, 1])
        ax2.grid(axis='x', alpha=0.3)
        
        for i, (bar, acc) in enumerate(zip(bars2, sorted_accs)):
            ax2.text(acc + 0.01, i, f'{acc:.4f}\n({acc*100:.2f}%)', 
                    va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        acc_filename = 'visualizations/new_visualization/model_accuracy_comparison.png'
        plt.savefig(acc_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved accuracy comparison to {acc_filename}")
        plt.close()
        
        # Print summary
        print("\n" + "-"*80)
        print("ACCURACY SUMMARY")
        print("-"*80)
        for name, acc in sorted_data:
            print(f"{name:30s}: {acc:.4f} ({acc*100:.2f}%)")
        
        best_model = sorted_data[0][0]
        best_accuracy = sorted_data[0][1]
        print(f"\n🏆 Best Model: {best_model} with accuracy {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    def save_results_summary(self):
        """Save results summary to a text file"""
        print("\n" + "="*80)
        print("SAVING RESULTS SUMMARY")
        print("="*80)
        
        os.makedirs('results', exist_ok=True)
        
        with open('results/model_comparison_summary.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("MACHINE LEARNING MODELS - ACCURACY COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            f.write("DPD Bucket Classification:\n")
            f.write("  0 = No Risk\n")
            f.write("  1 = Low Risk\n")
            f.write("  2 = Medium Risk\n")
            f.write("  3 = High Risk\n\n")
            
            f.write("-"*80 + "\n")
            f.write("ACCURACY RESULTS\n")
            f.write("-"*80 + "\n\n")
            
            sorted_results = sorted(self.results.items(), 
                                  key=lambda x: x[1]['accuracy'], 
                                  reverse=True)
            
            for rank, (name, result) in enumerate(sorted_results, 1):
                accuracy = result['accuracy']
                f.write(f"{rank}. {name}\n")
                f.write(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
                f.write(f"   Model file: models/{name.lower().replace(' ', '_')}_model.joblib\n\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("DETAILED CLASSIFICATION REPORTS\n")
            f.write("-"*80 + "\n\n")
            
            for name, result in sorted_results:
                f.write(f"\n{name}:\n")
                f.write("-"*40 + "\n")
                report = result['classification_report']
                
                # Write per-class metrics
                for class_label in ['0', '1', '2', '3']:
                    if class_label in report:
                        metrics = report[class_label]
                        f.write(f"Class {class_label}:\n")
                        f.write(f"  Precision: {metrics['precision']:.4f}\n")
                        f.write(f"  Recall: {metrics['recall']:.4f}\n")
                        f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                        f.write(f"  Support: {metrics['support']}\n\n")
                
                # Write overall metrics
                f.write(f"Overall:\n")
                f.write(f"  Accuracy: {report['accuracy']:.4f}\n")
                f.write(f"  Macro Avg F1: {report['macro avg']['f1-score']:.4f}\n")
                f.write(f"  Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}\n\n")
        
        print("✓ Saved results summary to results/model_comparison_summary.txt")


def main():
    """Main execution function"""
    trainer = MLModelTrainer()
    
    # Load and preprocess data
    feature_cols = trainer.load_and_preprocess_data('data/Credit_Card_Delinquency_Watch.csv')
    
    # Initialize models
    trainer.initialize_models()
    
    # Train and evaluate models
    trainer.train_and_evaluate_models()
    
    # Generate visualizations
    trainer.plot_confusion_matrices()
    trainer.plot_accuracy_comparison()
    
    # Save results summary
    trainer.save_results_summary()
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  - Model files: models/*_model.joblib")
    print("  - Scaler files: models/*_scaler.joblib (for models requiring scaling)")
    print("  - Confusion matrices: visualizations/new_visualization/*_confusion_matrix.png")
    print("  - All confusion matrices: visualizations/new_visualization/all_confusion_matrices.png")
    print("  - Accuracy comparison: visualizations/new_visualization/model_accuracy_comparison.png")
    print("  - Results summary: results/model_comparison_summary.txt")


if __name__ == "__main__":
    main()

