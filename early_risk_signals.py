"""
Early Risk Signals – Credit Card Delinquency Watch
A lightweight framework for identifying behavioral patterns that precede delinquency
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class EarlyRiskSignalSystem:
    """
    Lightweight framework to identify early behavioral signals of credit card delinquency.
    Focuses on leading indicators rather than lag indicators.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = None
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
    def load_data(self, filepath):
        """Load and prepare the dataset"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} customer records")
        return df
    
    def engineer_early_signals(self, df):
        """
        Create early warning signals from behavioral patterns.
        These are LEADING indicators, not lag indicators.
        """
        df_engineered = df.copy()
        
        # 1. SPENDING BEHAVIOR SIGNALS (Early Warning)
        # Sudden drop in spending often precedes payment issues
        df_engineered['spending_decline_flag'] = (df_engineered['Recent Spend Change %'] < -15).astype(int)
        df_engineered['spending_stress'] = np.where(
            df_engineered['Recent Spend Change %'] < -20, 2,
            np.where(df_engineered['Recent Spend Change %'] < -10, 1, 0)
        )
        
        # 2. UTILIZATION PATTERNS (Early Warning)
        # High utilization without corresponding payment behavior
        df_engineered['high_utilization_flag'] = (df_engineered['Utilisation %'] >= 80).astype(int)
        df_engineered['utilization_risk'] = np.where(
            df_engineered['Utilisation %'] >= 90, 3,
            np.where(df_engineered['Utilisation %'] >= 70, 2,
            np.where(df_engineered['Utilisation %'] >= 50, 1, 0))
        )
        
        # 3. PAYMENT BEHAVIOR SIGNALS (Early Warning)
        # Low payment frequency relative to utilization
        df_engineered['payment_risk_ratio'] = df_engineered['Min Due Paid Frequency'] / (df_engineered['Utilisation %'] + 1)
        df_engineered['low_payment_frequency'] = (df_engineered['Min Due Paid Frequency'] < 30).astype(int)
        df_engineered['payment_stress'] = np.where(
            df_engineered['Min Due Paid Frequency'] < 20, 2,
            np.where(df_engineered['Min Due Paid Frequency'] < 40, 1, 0)
        )
        
        # 4. CASH WITHDRAWAL PATTERNS (Early Warning)
        # Increased cash withdrawals can indicate financial stress
        df_engineered['high_cash_withdrawal'] = (df_engineered['Cash Withdrawal %'] >= 15).astype(int)
        df_engineered['cash_stress_indicator'] = np.where(
            df_engineered['Cash Withdrawal %'] >= 20, 2,
            np.where(df_engineered['Cash Withdrawal %'] >= 10, 1, 0)
        )
        
        # 5. MERCHANT MIX CHANGES (Early Warning)
        # Narrow merchant mix might indicate financial constraints
        df_engineered['narrow_merchant_mix'] = (df_engineered['Merchant Mix Index'] < 0.4).astype(int)
        
        # 6. COMPOSITE RISK SIGNALS (Combining multiple indicators)
        # Credit limit utilization vs payment ratio mismatch
        df_engineered['utilization_payment_mismatch'] = np.where(
            (df_engineered['Utilisation %'] > 70) & 
            (df_engineered['Avg Payment Ratio'] < 60), 1, 0
        )
        
        # Spending decline + high utilization = high risk
        df_engineered['spending_utilization_stress'] = np.where(
            (df_engineered['Recent Spend Change %'] < -15) & 
            (df_engineered['Utilisation %'] > 60), 1, 0
        )
        
        # Low payment frequency + high utilization = critical
        df_engineered['payment_utilization_critical'] = np.where(
            (df_engineered['Min Due Paid Frequency'] < 30) & 
            (df_engineered['Utilisation %'] > 70), 1, 0
        )
        
        # 7. RISK SCORE CALCULATION (Weighted early signals)
        df_engineered['early_risk_score'] = (
            df_engineered['spending_stress'] * 0.25 +
            df_engineered['utilization_risk'] * 0.30 +
            df_engineered['payment_stress'] * 0.25 +
            df_engineered['cash_stress_indicator'] * 0.10 +
            df_engineered['narrow_merchant_mix'] * 0.10
        )
        
        # Normalize risk score to 0-1
        df_engineered['early_risk_score'] = df_engineered['early_risk_score'] / 3.0
        
        return df_engineered
    
    def identify_risk_flags(self, df):
        """
        Create lightweight risk flags that balance sensitivity and false alarms.
        Flags are interpretable and actionable.
        """
        flags = []
        
        for idx, row in df.iterrows():
            customer_flags = []
            risk_level = 'LOW'
            
            # Flag 1: Spending Decline Alert
            if row['Recent Spend Change %'] < -20:
                customer_flags.append({
                    'flag': 'SPENDING_DECLINE_SEVERE',
                    'severity': 'HIGH',
                    'message': f"Spending dropped {abs(row['Recent Spend Change %'])}% - potential financial stress",
                    'action': 'Proactive payment plan discussion'
                })
            elif row['Recent Spend Change %'] < -15:
                customer_flags.append({
                    'flag': 'SPENDING_DECLINE_MODERATE',
                    'severity': 'MEDIUM',
                    'message': f"Spending declined {abs(row['Recent Spend Change %'])}%",
                    'action': 'Monitor spending patterns'
                })
            
            # Flag 2: High Utilization Alert
            if row['Utilisation %'] >= 90:
                customer_flags.append({
                    'flag': 'CRITICAL_UTILIZATION',
                    'severity': 'HIGH',
                    'message': f"Credit utilization at {row['Utilisation %']}% - near limit",
                    'action': 'Immediate credit limit review and payment reminder'
                })
            elif row['Utilisation %'] >= 80:
                customer_flags.append({
                    'flag': 'HIGH_UTILIZATION',
                    'severity': 'MEDIUM',
                    'message': f"Credit utilization at {row['Utilisation %']}%",
                    'action': 'Payment behavior monitoring'
                })
            
            # Flag 3: Payment Frequency Alert
            if row['Min Due Paid Frequency'] < 20:
                customer_flags.append({
                    'flag': 'LOW_PAYMENT_FREQUENCY',
                    'severity': 'HIGH',
                    'message': f"Only {row['Min Due Paid Frequency']}% minimum due payments made",
                    'action': 'Payment assistance program offer'
                })
            elif row['Min Due Paid Frequency'] < 40:
                customer_flags.append({
                    'flag': 'MODERATE_PAYMENT_FREQUENCY',
                    'severity': 'MEDIUM',
                    'message': f"Payment frequency at {row['Min Due Paid Frequency']}%",
                    'action': 'Payment reminder and education'
                })
            
            # Flag 4: Cash Withdrawal Pattern
            if row['Cash Withdrawal %'] >= 20:
                customer_flags.append({
                    'flag': 'HIGH_CASH_WITHDRAWAL',
                    'severity': 'MEDIUM',
                    'message': f"Cash withdrawals at {row['Cash Withdrawal %']}% of spending",
                    'action': 'Financial wellness check-in'
                })
            
            # Flag 5: Composite Risk Signals
            if row['utilization_payment_mismatch'] == 1:
                customer_flags.append({
                    'flag': 'UTILIZATION_PAYMENT_MISMATCH',
                    'severity': 'HIGH',
                    'message': 'High utilization with low payment ratio',
                    'action': 'Structured payment plan discussion'
                })
            
            if row['spending_utilization_stress'] == 1:
                customer_flags.append({
                    'flag': 'SPENDING_UTILIZATION_STRESS',
                    'severity': 'HIGH',
                    'message': 'Declining spending with high utilization',
                    'action': 'Financial counseling referral'
                })
            
            if row['payment_utilization_critical'] == 1:
                customer_flags.append({
                    'flag': 'PAYMENT_UTILIZATION_CRITICAL',
                    'severity': 'CRITICAL',
                    'message': 'Low payment frequency with high utilization',
                    'action': 'Immediate intervention required'
                })
            
            # Determine overall risk level
            if any(f['severity'] == 'CRITICAL' for f in customer_flags):
                risk_level = 'CRITICAL'
            elif any(f['severity'] == 'HIGH' for f in customer_flags):
                risk_level = 'HIGH'
            elif any(f['severity'] == 'MEDIUM' for f in customer_flags):
                risk_level = 'MEDIUM'
            
            flags.append({
                'customer_id': row['Customer ID'],
                'risk_level': risk_level,
                'risk_score': row['early_risk_score'],
                'flags': customer_flags,
                'flag_count': len(customer_flags)
            })
        
        return pd.DataFrame(flags)
    
    def train_model(self, df, target_col='DPD Bucket Next Month'):
        """
        Train multiple lightweight predictive models to identify at-risk customers.
        Uses early signals as features.
        """
        # Prepare features (use engineered early signals)
        feature_cols = [
            'Utilisation %', 'Avg Payment Ratio', 'Min Due Paid Frequency',
            'Merchant Mix Index', 'Cash Withdrawal %', 'Recent Spend Change %',
            'spending_stress', 'utilization_risk', 'payment_stress',
            'cash_stress_indicator', 'utilization_payment_mismatch',
            'spending_utilization_stress', 'payment_utilization_critical'
        ]
        
        X = df[feature_cols]
        y = (df[target_col] > 0).astype(int)  # Binary: at-risk (1) vs not at-risk (0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                random_state=42,
                class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        model_results = {}
        model_predictions = {}
        
        print("\n" + "="*60)
        print("TRAINING MULTIPLE MODELS")
        print("="*60)
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            model_results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            model_predictions[model_name] = y_pred
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  ROC-AUC: {roc_auc:.3f}")
        
        # Set the best model as the main model (Random Forest by default)
        self.model = models['Random Forest']
        self.model.fit(X_train_scaled, y_train)
        
        # Get feature importance from Random Forest
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store model results for comparison
        self.model_results = model_results
        
        # Print detailed results for Random Forest
        print("\n" + "="*60)
        print("DETAILED PERFORMANCE - RANDOM FOREST")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(y_test, model_predictions['Random Forest']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, model_predictions['Random Forest']))
        
        print("\n" + "="*60)
        print("TOP FEATURE IMPORTANCE")
        print("="*60)
        print(self.feature_importance.head(10).to_string(index=False))
        
        return self.model
    
    def generate_outreach_strategies(self, risk_flags_df):
        """
        Suggest simple, actionable outreach strategies for at-risk customers.
        """
        strategies = []
        
        for idx, row in risk_flags_df.iterrows():
            customer_strategies = []
            
            # Get all flags for this customer
            flags = row['flags']
            
            # Strategy based on risk level
            if row['risk_level'] == 'CRITICAL':
                customer_strategies.append({
                    'priority': 1,
                    'channel': 'Phone Call',
                    'timing': 'Immediate (within 24 hours)',
                    'message': 'Urgent: We noticed some concerning patterns in your account. Let\'s discuss payment options.',
                    'offer': 'Payment plan, hardship program, or credit limit adjustment'
                })
            elif row['risk_level'] == 'HIGH':
                customer_strategies.append({
                    'priority': 2,
                    'channel': 'Phone Call or Email',
                    'timing': 'Within 48 hours',
                    'message': 'We\'re here to help. Let\'s review your account and explore options.',
                    'offer': 'Payment plan or financial counseling'
                })
            elif row['risk_level'] == 'MEDIUM':
                customer_strategies.append({
                    'priority': 3,
                    'channel': 'Email or SMS',
                    'timing': 'Within 1 week',
                    'message': 'Tips for managing your credit card account effectively.',
                    'offer': 'Educational resources and payment reminders'
                })
            
            # Specific strategies based on flags
            for flag in flags:
                if flag['flag'] == 'SPENDING_DECLINE_SEVERE':
                    customer_strategies.append({
                        'priority': 1,
                        'channel': 'Phone Call',
                        'timing': 'Within 48 hours',
                        'message': 'We noticed a significant change in your spending patterns.',
                        'offer': 'Financial wellness check-in and budgeting tools'
                    })
                
                if flag['flag'] == 'CRITICAL_UTILIZATION':
                    customer_strategies.append({
                        'priority': 1,
                        'channel': 'Phone Call',
                        'timing': 'Immediate',
                        'message': 'Your credit utilization is very high. Let\'s discuss options.',
                        'offer': 'Payment plan or credit limit increase (if qualified)'
                    })
                
                if flag['flag'] == 'LOW_PAYMENT_FREQUENCY':
                    customer_strategies.append({
                        'priority': 1,
                        'channel': 'Phone Call',
                        'timing': 'Within 24 hours',
                        'message': 'We can help you get back on track with payments.',
                        'offer': 'Autopay setup, payment reminders, or payment assistance'
                    })
            
            strategies.append({
                'customer_id': row['customer_id'],
                'risk_level': row['risk_level'],
                'strategies': customer_strategies
            })
        
        return pd.DataFrame(strategies)
    
    def generate_insights_report(self, df, risk_flags_df, strategies_df):
        """
        Generate comprehensive insights report on early risk signals.
        """
        print("\n" + "="*80)
        print("EARLY RISK SIGNALS - COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        # Overall statistics
        total_customers = len(df)
        at_risk_customers = len(risk_flags_df[risk_flags_df['risk_level'].isin(['HIGH', 'CRITICAL'])])
        
        print(f"\n📊 PORTFOLIO OVERVIEW")
        print(f"Total Customers: {total_customers}")
        print(f"At-Risk Customers (High/Critical): {at_risk_customers} ({at_risk_customers/total_customers*100:.1f}%)")
        
        # Risk distribution
        print(f"\n📈 RISK DISTRIBUTION")
        risk_dist = risk_flags_df['risk_level'].value_counts()
        for level, count in risk_dist.items():
            print(f"  {level}: {count} customers ({count/total_customers*100:.1f}%)")
        
        # Top risk signals
        print(f"\n🚨 TOP EARLY WARNING SIGNALS")
        all_flags = []
        for flags in risk_flags_df['flags']:
            all_flags.extend([f['flag'] for f in flags])
        
        flag_counts = pd.Series(all_flags).value_counts()
        for flag, count in flag_counts.head(10).items():
            print(f"  {flag}: {count} customers")
        
        # Behavioral patterns
        print(f"\n🔍 KEY BEHAVIORAL PATTERNS")
        
        # Spending decline pattern
        spending_decline = len(df[df['Recent Spend Change %'] < -15])
        print(f"  Customers with spending decline >15%: {spending_decline} ({spending_decline/total_customers*100:.1f}%)")
        
        # High utilization pattern
        high_util = len(df[df['Utilisation %'] >= 80])
        print(f"  Customers with utilization ≥80%: {high_util} ({high_util/total_customers*100:.1f}%)")
        
        # Payment frequency pattern
        low_payment = len(df[df['Min Due Paid Frequency'] < 30])
        print(f"  Customers with payment frequency <30%: {low_payment} ({low_payment/total_customers*100:.1f}%)")
        
        # Composite patterns
        composite_risk = len(df[df['payment_utilization_critical'] == 1])
        print(f"  Critical: Low payment + High utilization: {composite_risk} ({composite_risk/total_customers*100:.1f}%)")
        
        # Outreach summary
        print(f"\n📞 OUTREACH STRATEGY SUMMARY")
        critical = len(risk_flags_df[risk_flags_df['risk_level'] == 'CRITICAL'])
        high = len(risk_flags_df[risk_flags_df['risk_level'] == 'HIGH'])
        medium = len(risk_flags_df[risk_flags_df['risk_level'] == 'MEDIUM'])
        
        print(f"  Immediate Phone Calls (Critical): {critical} customers")
        print(f"  Priority Calls (High): {high} customers")
        print(f"  Email/SMS Outreach (Medium): {medium} customers")
        
        print("\n" + "="*80)
    
    def get_feature_importance(self):
        """Return feature importance dataframe"""
        return self.feature_importance
    
    def get_model_results(self):
        """Return model comparison results"""
        return self.model_results
    
    def generate_synthetic_dataset(self, original_df, target_size=50000, random_state=42):
        """
        Generate synthetic dataset by resampling and adding noise to preserve statistical properties.
        """
        np.random.seed(random_state)
        
        print(f"\n{'='*60}")
        print(f"GENERATING SYNTHETIC DATASET ({target_size:,} records)")
        print(f"{'='*60}")
        print(f"Original dataset size: {len(original_df):,} records")
        
        # Calculate how many times we need to resample
        n_samples_needed = target_size
        synthetic_records = []
        
        # Get statistical properties of original data
        numeric_cols = ['Credit Limit', 'Utilisation %', 'Avg Payment Ratio', 
                       'Min Due Paid Frequency', 'Merchant Mix Index', 
                       'Cash Withdrawal %', 'Recent Spend Change %']
        
        # Calculate means and stds for adding controlled noise
        means = original_df[numeric_cols].mean()
        stds = original_df[numeric_cols].std()
        
        # Generate synthetic data
        for i in range(n_samples_needed):
            # Randomly select a row from original data
            base_idx = np.random.randint(0, len(original_df))
            base_row = original_df.iloc[base_idx].copy()
            
            # Add controlled noise to numeric features (preserve relationships)
            noise_factor = 0.1  # 10% noise to maintain realism
            
            for col in numeric_cols:
                # Add Gaussian noise scaled by standard deviation
                noise = np.random.normal(0, stds[col] * noise_factor)
                base_row[col] = base_row[col] + noise
                
                # Apply bounds to keep values realistic
                if col == 'Credit Limit':
                    base_row[col] = max(10000, min(300000, base_row[col]))
                elif col == 'Utilisation %':
                    base_row[col] = max(0, min(100, base_row[col]))
                elif col == 'Avg Payment Ratio':
                    base_row[col] = max(0, min(100, base_row[col]))
                elif col == 'Min Due Paid Frequency':
                    base_row[col] = max(0, min(100, base_row[col]))
                elif col == 'Merchant Mix Index':
                    base_row[col] = max(0, min(1, base_row[col]))
                elif col == 'Cash Withdrawal %':
                    base_row[col] = max(0, min(100, base_row[col]))
                elif col == 'Recent Spend Change %':
                    base_row[col] = max(-50, min(50, base_row[col]))
            
            # Generate new Customer ID
            base_row['Customer ID'] = f'C{i+1:06d}'
            
            # Preserve DPD Bucket distribution (resample based on original distribution)
            # This maintains the class distribution
            dpd_probs = original_df['DPD Bucket Next Month'].value_counts(normalize=True)
            base_row['DPD Bucket Next Month'] = np.random.choice(
                dpd_probs.index, 
                p=dpd_probs.values
            )
            
            synthetic_records.append(base_row)
        
        synthetic_df = pd.DataFrame(synthetic_records)
        
        print(f"✓ Generated synthetic dataset: {len(synthetic_df):,} records")
        print(f"  Original DPD distribution: {original_df['DPD Bucket Next Month'].value_counts().to_dict()}")
        print(f"  Synthetic DPD distribution: {synthetic_df['DPD Bucket Next Month'].value_counts().to_dict()}")
        
        return synthetic_df
    
    def compare_datasets_performance(self, original_df, synthetic_df, save_path='dataset_comparison.png'):
        """
        Train models on both datasets and compare performance.
        """
        try:
            import matplotlib.pyplot as plt
            
            print(f"\n{'='*60}")
            print("COMPARING DATASET PERFORMANCE")
            print(f"{'='*60}")
            
            # Prepare both datasets
            feature_cols = [
                'Utilisation %', 'Avg Payment Ratio', 'Min Due Paid Frequency',
                'Merchant Mix Index', 'Cash Withdrawal %', 'Recent Spend Change %',
                'spending_stress', 'utilization_risk', 'payment_stress',
                'cash_stress_indicator', 'utilization_payment_mismatch',
                'spending_utilization_stress', 'payment_utilization_critical'
            ]
            
            results = {}
            
            for dataset_name, df in [('Original (100)', original_df), ('Synthetic (50,000)', synthetic_df)]:
                print(f"\n📊 Processing {dataset_name} dataset...")
                
                # Engineer features
                df_eng = self.engineer_early_signals(df)
                
                # Prepare features
                X = df_eng[feature_cols]
                y = (df_eng['DPD Bucket Next Month'] > 0).astype(int)
                
                # Split data (80/20)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train models
                models = {
                    'Random Forest': RandomForestClassifier(
                        n_estimators=100, max_depth=5, min_samples_split=10,
                        random_state=42, class_weight='balanced'
                    ),
                    'Logistic Regression': LogisticRegression(
                        random_state=42, class_weight='balanced', max_iter=1000
                    ),
                    'Gradient Boosting': GradientBoostingClassifier(
                        n_estimators=100, max_depth=3, random_state=42
                    )
                }
                
                dataset_results = {}
                
                for model_name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    dataset_results[model_name] = accuracy
                    print(f"  {model_name}: Accuracy = {accuracy:.4f}")
                
                results[dataset_name] = dataset_results
            
            # Create comparison visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Prepare data for plotting
            model_names = list(results['Original (100)'].keys())
            original_accs = [results['Original (100)'][m] for m in model_names]
            synthetic_accs = [results['Synthetic (50,000)'][m] for m in model_names]
            
            x_pos = np.arange(len(model_names))
            width = 0.35
            
            # Plot 1: Side-by-side bar comparison
            bars1 = axes[0].bar(x_pos - width/2, original_accs, width, 
                               label='Original (100)', color='#4169E1', alpha=0.8, edgecolor='black')
            bars2 = axes[0].bar(x_pos + width/2, synthetic_accs, width,
                               label='Synthetic (50,000)', color='#32CD32', alpha=0.8, edgecolor='black')
            
            axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            axes[0].set_title('Model Accuracy: Original vs Synthetic Dataset', fontsize=14, fontweight='bold')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(model_names, rotation=0, ha='center')
            axes[0].set_ylim([0, 1])
            axes[0].grid(axis='y', alpha=0.3, linestyle='--')
            axes[0].legend(fontsize=11, loc='best')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Plot 2: Line graph comparison
            x_positions = range(len(model_names))
            axes[1].plot(x_positions, original_accs, marker='o', linewidth=3, markersize=10,
                        color='#4169E1', markerfacecolor='#4169E1', markeredgecolor='black',
                        markeredgewidth=2, label='Original (100)', alpha=0.8)
            axes[1].plot(x_positions, synthetic_accs, marker='s', linewidth=3, markersize=10,
                        color='#32CD32', markerfacecolor='#32CD32', markeredgecolor='black',
                        markeredgewidth=2, label='Synthetic (50,000)', alpha=0.8)
            
            axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            axes[1].set_title('Accuracy Comparison: Line Graph', fontsize=14, fontweight='bold')
            axes[1].set_xticks(x_positions)
            axes[1].set_xticklabels(model_names, rotation=0, ha='center')
            axes[1].set_ylim([0, 1])
            axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=1)
            axes[1].legend(fontsize=11, loc='best')
            
            # Add value labels
            for i, (orig, synth) in enumerate(zip(original_accs, synthetic_accs)):
                axes[1].text(i, orig + 0.03, f'{orig:.3f}', ha='center', va='bottom',
                           fontweight='bold', fontsize=9, color='#4169E1',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
                axes[1].text(i, synth + 0.03, f'{synth:.3f}', ha='center', va='bottom',
                           fontweight='bold', fontsize=9, color='#32CD32',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved dataset comparison: {save_path}")
            plt.close()
            
            # Print comparison table
            print(f"\n{'='*60}")
            print("DATASET COMPARISON SUMMARY")
            print(f"{'='*60}")
            comparison_data = {
                'Model': model_names,
                'Original (100)': original_accs,
                'Synthetic (50,000)': synthetic_accs,
                'Improvement': [f"{(s-o)*100:+.2f}%" for o, s in zip(original_accs, synthetic_accs)]
            }
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False))
            
            # Calculate average improvement
            avg_improvement = np.mean([(s-o) for o, s in zip(original_accs, synthetic_accs)]) * 100
            print(f"\n📈 Average Accuracy Improvement: {avg_improvement:+.2f}%")
            
            return results
            
        except Exception as e:
            print(f"⚠️  Error in dataset comparison: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_model_comparison(self, save_path='model_comparison.png'):
        """Plot accuracy comparison of all models using line graphs"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not hasattr(self, 'model_results') or self.model_results is None:
                print("⚠️  No model results available. Train models first.")
                return
            
            # Prepare data
            model_names = list(self.model_results.keys())
            accuracies = [self.model_results[m]['accuracy'] for m in model_names]
            roc_aucs = [self.model_results[m]['roc_auc'] for m in model_names]
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Create x-axis positions
            x_positions = range(len(model_names))
            
            # Plot 1: Accuracy Comparison - Line Graph
            axes[0].plot(x_positions, accuracies, marker='o', linewidth=3, markersize=10, 
                        color='#4169E1', markerfacecolor='#4169E1', markeredgecolor='black', 
                        markeredgewidth=2, label='Accuracy', alpha=0.8)
            axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            axes[0].set_xticks(x_positions)
            axes[0].set_xticklabels(model_names, rotation=0, ha='center')
            axes[0].set_ylim([0, 1])
            axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=1)
            axes[0].legend(fontsize=11, loc='best')
            
            # Add value labels on points
            for i, (x, acc) in enumerate(zip(x_positions, accuracies)):
                axes[0].text(x, acc + 0.03, f'{acc:.3f}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=11, bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', alpha=0.7, edgecolor='gray'))
            
            # Plot 2: ROC-AUC Comparison - Line Graph
            axes[1].plot(x_positions, roc_aucs, marker='s', linewidth=3, markersize=10, 
                        color='#32CD32', markerfacecolor='#32CD32', markeredgecolor='black', 
                        markeredgewidth=2, label='ROC-AUC', alpha=0.8)
            axes[1].set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
            axes[1].set_title('Model ROC-AUC Comparison', fontsize=14, fontweight='bold')
            axes[1].set_xticks(x_positions)
            axes[1].set_xticklabels(model_names, rotation=0, ha='center')
            axes[1].set_ylim([0, 1])
            axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=1)
            axes[1].legend(fontsize=11, loc='best')
            
            # Add value labels on points
            for i, (x, auc) in enumerate(zip(x_positions, roc_aucs)):
                axes[1].text(x, auc + 0.03, f'{auc:.3f}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=11, bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', alpha=0.7, edgecolor='gray'))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved model comparison: {save_path}")
            plt.close()
            
            # Print comparison table
            print("\n" + "="*60)
            print("MODEL COMPARISON SUMMARY")
            print("="*60)
            comparison_df = pd.DataFrame({
                'Model': model_names,
                'Accuracy': accuracies,
                'ROC-AUC': roc_aucs
            })
            print(comparison_df.to_string(index=False))
            
            # Find best model
            best_acc_idx = accuracies.index(max(accuracies))
            best_auc_idx = roc_aucs.index(max(roc_aucs))
            print(f"\n🏆 Best Accuracy: {model_names[best_acc_idx]} ({max(accuracies):.3f})")
            print(f"🏆 Best ROC-AUC: {model_names[best_auc_idx]} ({max(roc_aucs):.3f})")
            
        except ImportError:
            print("⚠️  matplotlib not available. Skipping visualization.")
        except Exception as e:
            print(f"⚠️  Error creating comparison plot: {e}")


def main():
    """
    Main execution function for the Early Risk Signal System.
    """
    print("="*80)
    print("EARLY RISK SIGNALS – CREDIT CARD DELINQUENCY WATCH")
    print("="*80)
    
    # Initialize system
    system = EarlyRiskSignalSystem()
    
    # Load data
    print("\n[1/6] Loading data...")
    df = system.load_data('Sample.csv')
    
    # Engineer early signals
    print("\n[2/6] Engineering early warning signals...")
    df_engineered = system.engineer_early_signals(df)
    
    # Identify risk flags
    print("\n[3/6] Identifying risk flags...")
    risk_flags_df = system.identify_risk_flags(df_engineered)
    
    # Train predictive model
    print("\n[4/6] Training predictive model...")
    system.train_model(df_engineered)
    
    # Generate outreach strategies
    print("\n[5/6] Generating outreach strategies...")
    strategies_df = system.generate_outreach_strategies(risk_flags_df)
    
    # Generate insights report
    print("\n[6/7] Generating insights report...")
    system.generate_insights_report(df_engineered, risk_flags_df, strategies_df)
    
    # Generate visualizations
    print("\n[7/9] Generating visualizations...")
    try:
        from visualization_dashboard import RiskVisualizationDashboard
        viz = RiskVisualizationDashboard()
        feature_importance_df = system.get_feature_importance()
        viz.generate_all_visualizations(df_engineered, risk_flags_df, strategies_df, feature_importance_df)
    except Exception as e:
        print(f"⚠️  Visualization generation skipped: {e}")
    
    # Generate model comparison plot
    print("\n[8/9] Generating model comparison...")
    try:
        system.plot_model_comparison()
    except Exception as e:
        print(f"⚠️  Model comparison skipped: {e}")
    
    # Generate workflow diagram
    print("\n[9/10] Generating workflow diagram...")
    try:
        from workflow_diagram import create_workflow_diagram
        create_workflow_diagram()
    except Exception as e:
        print(f"⚠️  Workflow diagram generation skipped: {e}")
    
    # Generate synthetic dataset and compare performance
    print("\n[10/10] Generating synthetic dataset and comparing performance...")
    try:
        synthetic_df = system.generate_synthetic_dataset(df, target_size=50000)
        synthetic_df_engineered = system.engineer_early_signals(synthetic_df)
        comparison_results = system.compare_datasets_performance(df_engineered, synthetic_df_engineered)
        
        # Save synthetic dataset
        synthetic_df.to_csv('synthetic_dataset_50000.csv', index=False)
        synthetic_df_engineered.to_csv('synthetic_dataset_with_signals.csv', index=False)
        print("\n💾 Saved synthetic datasets:")
        print("  - synthetic_dataset_50000.csv")
        print("  - synthetic_dataset_with_signals.csv")
    except Exception as e:
        print(f"⚠️  Synthetic dataset generation skipped: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    print("\n💾 Saving results...")
    risk_flags_df.to_csv('risk_flags_output.csv', index=False)
    strategies_df.to_csv('outreach_strategies.csv', index=False)
    try:
        df_engineered.to_csv('data_with_early_signals.csv', index=False)
    except PermissionError:
        print("⚠️  Could not save data_with_early_signals.csv (file may be open)")
    
    print("\n✅ Analysis complete! Output files saved:")
    print("  - risk_flags_output.csv")
    print("  - outreach_strategies.csv")
    print("  - data_with_early_signals.csv")
    print("  - Visualization PNG files (if matplotlib available)")
    
    return system, df_engineered, risk_flags_df, strategies_df


if __name__ == "__main__":
    system, df_engineered, risk_flags_df, strategies_df = main()

