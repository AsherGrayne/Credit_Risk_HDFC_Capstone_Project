"""
Visualization Dashboard for Early Risk Signals
Creates comprehensive visualizations for risk monitoring and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

class RiskVisualizationDashboard:
    """Create visualizations for early risk signal monitoring"""
    
    def __init__(self):
        self.colors = {
            'CRITICAL': '#DC143C',
            'HIGH': '#FF6347',
            'MEDIUM': '#FFA500',
            'LOW': '#32CD32',
            'background': '#F5F5F5'
        }
    
    def plot_risk_distribution(self, risk_flags_df, save_path='risk_distribution.png'):
        """Plot distribution of risk levels across portfolio"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Risk level distribution
        risk_counts = risk_flags_df['risk_level'].value_counts()
        colors_list = [self.colors.get(level, '#808080') for level in risk_counts.index]
        
        axes[0].bar(risk_counts.index, risk_counts.values, color=colors_list, alpha=0.8, edgecolor='black')
        axes[0].set_title('Risk Level Distribution Across Portfolio', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Risk Level', fontsize=12)
        axes[0].set_ylabel('Number of Customers', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(risk_counts.values):
            axes[0].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Risk score distribution
        axes[1].hist(risk_flags_df['risk_score'], bins=20, color='#4169E1', alpha=0.7, edgecolor='black')
        axes[1].axvline(risk_flags_df['risk_score'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {risk_flags_df["risk_score"].mean():.2f}')
        axes[1].set_title('Early Risk Score Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Risk Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_behavioral_patterns(self, df, save_path='behavioral_patterns.png'):
        """Visualize key behavioral patterns that precede delinquency"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Spending Change vs Utilization
        scatter = axes[0, 0].scatter(
            df['Utilisation %'], 
            df['Recent Spend Change %'],
            c=df['DPD Bucket Next Month'],
            cmap='RdYlGn_r',
            alpha=0.6,
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
        axes[0, 0].axhline(-15, color='red', linestyle='--', alpha=0.5, label='Spending Decline Threshold')
        axes[0, 0].axvline(80, color='orange', linestyle='--', alpha=0.5, label='High Utilization Threshold')
        axes[0, 0].set_xlabel('Credit Utilization (%)', fontsize=12)
        axes[0, 0].set_ylabel('Recent Spend Change (%)', fontsize=12)
        axes[0, 0].set_title('Spending Decline vs Utilization\n(Early Warning Signal)', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='DPD Bucket')
        
        # 2. Payment Frequency vs Utilization
        scatter2 = axes[0, 1].scatter(
            df['Utilisation %'],
            df['Min Due Paid Frequency'],
            c=df['DPD Bucket Next Month'],
            cmap='RdYlGn_r',
            alpha=0.6,
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
        axes[0, 1].axhline(30, color='red', linestyle='--', alpha=0.5, label='Low Payment Threshold')
        axes[0, 1].axvline(70, color='orange', linestyle='--', alpha=0.5, label='High Utilization Threshold')
        axes[0, 1].set_xlabel('Credit Utilization (%)', fontsize=12)
        axes[0, 1].set_ylabel('Min Due Paid Frequency (%)', fontsize=12)
        axes[0, 1].set_title('Payment Frequency vs Utilization\n(Critical Risk Indicator)', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        plt.colorbar(scatter2, ax=axes[0, 1], label='DPD Bucket')
        
        # 3. Cash Withdrawal Pattern
        cash_risk = df.groupby('DPD Bucket Next Month')['Cash Withdrawal %'].mean()
        axes[1, 0].bar(cash_risk.index, cash_risk.values, color=['green', 'orange', 'red', 'darkred'], 
                      alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('DPD Bucket', fontsize=12)
        axes[1, 0].set_ylabel('Average Cash Withdrawal %', fontsize=12)
        axes[1, 0].set_title('Cash Withdrawal Pattern by Risk Level', fontsize=13, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(cash_risk.values):
            axes[1, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Early Risk Score by DPD Bucket
        risk_by_dpd = df.groupby('DPD Bucket Next Month')['early_risk_score'].mean()
        axes[1, 1].bar(risk_by_dpd.index, risk_by_dpd.values, color=['green', 'orange', 'red', 'darkred'],
                       alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('DPD Bucket', fontsize=12)
        axes[1, 1].set_ylabel('Average Early Risk Score', fontsize=12)
        axes[1, 1].set_title('Early Risk Score Predictive Power', fontsize=13, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(risk_by_dpd.values):
            axes[1, 1].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_flag_frequency(self, risk_flags_df, save_path='flag_frequency.png'):
        """Visualize frequency of different risk flags"""
        # Extract all flags
        all_flags = []
        for flags in risk_flags_df['flags']:
            all_flags.extend([f['flag'] for f in flags])
        
        flag_counts = pd.Series(all_flags).value_counts()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(flag_counts)))
        bars = ax.barh(range(len(flag_counts)), flag_counts.values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(flag_counts)))
        ax.set_yticklabels(flag_counts.index, fontsize=10)
        ax.set_xlabel('Number of Customers Flagged', fontsize=12)
        ax.set_title('Top Early Warning Flags Frequency', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(flag_counts.values):
            ax.text(v + 0.5, i, str(v), va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_feature_importance(self, feature_importance_df, save_path='feature_importance.png'):
        """Plot feature importance from the predictive model"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_features = feature_importance_df.head(10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'].values, 
                      color=colors, alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values, fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Top 10 Most Important Early Warning Features', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(top_features['importance'].values):
            ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_outreach_strategy(self, strategies_df, save_path='outreach_strategy.png'):
        """Visualize outreach strategy distribution"""
        # Count strategies by channel and priority
        channel_counts = {}
        priority_counts = {1: 0, 2: 0, 3: 0}
        
        for strategies in strategies_df['strategies']:
            for strategy in strategies:
                channel = strategy['channel']
                priority = strategy['priority']
                
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Channel distribution
        channel_df = pd.Series(channel_counts)
        axes[0].bar(channel_df.index, channel_df.values, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], 
                   alpha=0.8, edgecolor='black')
        axes[0].set_title('Outreach Channel Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Interventions', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(channel_df.values):
            axes[0].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Priority distribution
        priority_df = pd.Series(priority_counts)
        axes[1].bar(priority_df.index.astype(str), priority_df.values, 
                   color=['#DC143C', '#FF6347', '#FFA500'], alpha=0.8, edgecolor='black')
        axes[1].set_title('Intervention Priority Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Priority Level', fontsize=12)
        axes[1].set_ylabel('Number of Interventions', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(priority_df.values):
            axes[1].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def create_risk_heatmap(self, df, save_path='risk_heatmap.png'):
        """Create correlation heatmap of risk factors"""
        # Select key features for correlation
        corr_features = [
            'Utilisation %', 'Avg Payment Ratio', 'Min Due Paid Frequency',
            'Cash Withdrawal %', 'Recent Spend Change %', 'DPD Bucket Next Month',
            'early_risk_score', 'spending_stress', 'utilization_risk', 'payment_stress'
        ]
        
        corr_matrix = df[corr_features].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        ax.set_title('Risk Factors Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def generate_all_visualizations(self, df, risk_flags_df, strategies_df, feature_importance_df):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATION DASHBOARD")
        print("="*60)
        
        self.plot_risk_distribution(risk_flags_df)
        self.plot_behavioral_patterns(df)
        self.plot_flag_frequency(risk_flags_df)
        self.plot_feature_importance(feature_importance_df)
        self.plot_outreach_strategy(strategies_df)
        self.create_risk_heatmap(df)
        
        print("\n✅ All visualizations generated successfully!")


if __name__ == "__main__":
    # This will be called from the main script
    pass

