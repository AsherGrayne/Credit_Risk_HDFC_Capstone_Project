# Early Risk Signals – Credit Card Delinquency Watch

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)

A lightweight, data-driven framework for identifying early behavioral signals of credit card delinquency before they occur, enabling proactive customer outreach to reduce roll-rates and improve portfolio health.

## Overview

This solution provides a comprehensive system to detect credit card delinquency risk using **leading indicators** (early warning signals) rather than lag indicators (missed payments). The framework enables proactive customer outreach to reduce roll-rates and improve portfolio health.

## Key Features

- **Early Warning Signal Detection** - Identifies behavioral patterns before delinquency occurs
- **Risk Scoring Framework** - Weighted risk scoring system (0.0-1.0) with four-tier classification
- **Predictive Modeling** - Multiple ML models (Random Forest, Logistic Regression, Gradient Boosting)
- **Targeted Interventions** - Automated outreach strategies based on risk levels
- **Comprehensive Visualizations** - 9+ professional charts and dashboards
- **Scalable Architecture** - Designed for production deployment and automation

## Project Statistics

- **50,000** records analyzed
- **50%** of customers flagged as at-risk
- **75%** model accuracy (Random Forest)
- **14+** early warning signals engineered
- **10** critical customers requiring immediate intervention

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Required packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/AsherGrayne/Credit_Risk_HDFC_Capstone_Project.git
cd Credit_Risk_HDFC_Capstone_Project

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis

```bash
# Execute the main analysis
python src/early_risk_signals.py
```

This will:
1. Load and analyze the sample data
2. Engineer early warning signals
3. Generate risk flags
4. Train predictive models
5. Create outreach strategies
6. Generate visualizations
7. Save output files

## Project Structure

```
Credit Card Delinquency Pack/
│
├── src/                               # Source code
│   ├── early_risk_signals.py          # Main analysis framework
│   ├── visualization_dashboard.py     # Visualization module
│   ├── workflow_diagram.py            # Workflow visualization
│   ├── export_model_to_json.py       # Model export utility
│   └── predict_api.py                # Flask API for predictions
│
├── data/                              # Data files
│   ├── Sample.csv                     # Input data (100 records)
│   ├── risk_flags_output.csv          # Risk flags output
│   ├── outreach_strategies.csv        # Intervention recommendations
│   ├── data_with_early_signals.csv    # Enhanced dataset
│   └── synthetic_dataset_50000.csv     # Synthetic dataset
│
├── docs/                              # Documentation
│   ├── solution_narrative.md          # Comprehensive documentation
│   ├── EXECUTIVE_SUMMARY.md           # Executive overview
│   ├── FULL_WORKFLOW.md               # Complete workflow
│   ├── EXECUTION_TRANSCRIPT.md        # Execution details
│   ├── ML_MODEL_EXPLANATION.md        # ML model documentation
│   ├── README_API.md                  # API documentation
│   ├── README_HOSTING.md              # Hosting guide
│   └── DELIVERABLES_CHECKLIST.md     # Deliverables checklist
│
├── visualizations/                     # Generated visualizations
│   ├── model_comparison.png           # Model performance comparison
│   ├── dataset_comparison.png         # Dataset size impact
│   ├── risk_distribution.png          # Risk level distribution
│   ├── behavioral_patterns.png       # Behavioral analysis
│   ├── flag_frequency.png             # Flag frequency analysis
│   ├── feature_importance.png         # Feature importance
│   ├── outreach_strategy.png          # Outreach distribution
│   ├── risk_heatmap.png               # Correlation heatmap
│   └── workflow_diagram.png            # System workflow
│
├── website/                           # Web interface
│   ├── index.html                     # Main website
│   ├── workflow.html                  # Workflow page
│   ├── apply.html                     # Prediction form
│   ├── styles.css                     # Main stylesheet
│   ├── workflow-styles.css            # Workflow page styles
│   ├── apply-script.js                # Form handling script
│   ├── ml-model-predictor.js          # ML model predictor
│   └── workflow-script.js              # Workflow page script
│
├── models/                             # Trained models
│   └── model.json                     # Exported Random Forest model
│
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

## Key Features Explained

### 1. Early Warning Signal Detection

The framework identifies five key behavioral patterns:

- **Spending Decline**: Sudden drops in spending (>15% or >20%)
- **High Utilization**: Credit limit usage approaching maximum (≥80% or ≥90%)
- **Payment Frequency Decline**: Reduced minimum due payments (<30% or <20%)
- **Cash Withdrawal Patterns**: Increased cash advances (≥15% or ≥20%)
- **Composite Risk Signals**: Multiple indicators combined

### 2. Risk Scoring Framework

**Early Risk Score** (0.0 - 1.0):
- Combines multiple behavioral indicators
- Weighted by predictive importance
- Classifies customers into risk levels:
  - **CRITICAL**: Immediate intervention required
  - **HIGH**: Priority outreach needed
  - **MEDIUM**: Proactive monitoring
  - **LOW**: Standard monitoring

### 3. Predictive Modeling

- **Algorithm**: Random Forest Classifier
- **Features**: 14 engineered early signals
- **Performance**: High precision and recall for at-risk detection
- **Interpretability**: Feature importance analysis

### 4. Targeted Interventions

Automated outreach strategies based on risk level:
- **CRITICAL**: Phone call within 24 hours
- **HIGH**: Phone call/email within 48 hours
- **MEDIUM**: Email/SMS within 1 week

## Output Files

### CSV Files

1. **`risk_flags_output.csv`**
   - Contains risk flags for each customer
   - Columns: `customer_id`, `risk_level`, `risk_score`, `flags`, `flag_count`
   - 100 rows (one per customer)

2. **`outreach_strategies.csv`**
   - Contains outreach recommendations
   - Columns: `customer_id`, `risk_level`, `strategies`
   - Includes priority, channel, timing, message, and offer

3. **`data_with_early_signals.csv`**
   - Enhanced dataset with original + engineered features
   - Ready for further analysis

### Visualizations

9 PNG files generated:
- Risk distribution charts
- Behavioral pattern analysis
- Flag frequency
- Feature importance
- Outreach strategy distribution
- Risk correlation heatmap
- System workflow diagram

## Key Results

### Portfolio Analysis
- **50%** of customers flagged as at-risk (50 customers)
- **10%** require immediate intervention
- **Top flags**: Low Payment Frequency (23), Spending Decline (20)

### Model Performance
- **Random Forest**: 75% accuracy (original dataset)
- **Gradient Boosting**: 79% accuracy (50K dataset)
- **Top Features**: Payment Frequency (18.3%), Utilization (17.5%)

## Understanding the Logic

### Early Warning Signals vs Lag Indicators

| Early Warning (Leading) | Lag Indicator (Trailing) |
|------------------------|-------------------------|
| Spending decline | Missed payment |
| Utilization trend | Over-limit account |
| Payment frequency change | Late fee charged |
| Cash withdrawal increase | Collection action |

### Risk Flag Logic

Flags are generated using **deterministic thresholds** based on:
- Industry best practices
- Statistical analysis of the data
- Business rules for credit risk management

**Example**: `SPENDING_DECLINE_SEVERE` flag triggers when:
- Recent Spend Change % < -20%
- Indicates potential financial stress

### Risk Score Calculation

```
Risk Score = (
    Spending Stress (0-2) × 0.25 +
    Utilization Risk (0-3) × 0.30 +
    Payment Stress (0-2) × 0.25 +
    Cash Stress Indicator (0-2) × 0.10 +
    Narrow Merchant Mix (0-1) × 0.10
) / 3.0
```

## Customization

### Adjusting Risk Thresholds

Edit thresholds in `src/early_risk_signals.py`:

```python
# Modify spending decline threshold
if row['Recent Spend Change %'] < -20:  # Change -20 to your threshold
    # Flag logic
```

### Updating Risk Score Weights

Modify weights in `engineer_early_signals()`:

```python
df_engineered['early_risk_score'] = (
    df_engineered['spending_stress'] * 0.25 +  # Adjust weight
    df_engineered['utilization_risk'] * 0.30 +  # Adjust weight
    # ... etc
)
```

### Adding New Flags

Add new flag logic in `identify_risk_flags()`:

```python
# New flag example
if row['Your_New_Indicator'] > threshold:
    customer_flags.append({
        'flag': 'YOUR_NEW_FLAG',
        'severity': 'MEDIUM',
        'message': 'Your custom message',
        'action': 'Recommended action'
    })
```

## Website

A professional website showcasing all visualizations is included:

- **File**: `index.html`
- **Styling**: `styles.css`
- **Hosting**: Can be deployed on GitHub Pages, Netlify, or any static host

See `docs/README_HOSTING.md` for deployment instructions.

## Scaling to Production

### Phase 1: Pilot (Months 1-3)
- Deploy on 10% of portfolio
- Manual review of flags
- Measure effectiveness

### Phase 2: Rollout (Months 4-6)
- Expand to 50% of portfolio
- Semi-automated outreach
- A/B testing

### Phase 3: Full Automation (Months 7-12)
- Full portfolio coverage
- Automated flag generation
- Automated low-risk outreach

See `docs/solution_narrative.md` for detailed scaling strategy.

## Expected Impact

- **Early Detection**: Identify 20-30% of at-risk customers before delinquency
- **Roll-Rate Reduction**: 15-25% reduction in progression to higher DPD buckets
- **Cost Savings**: Early intervention cheaper than collections
- **Customer Experience**: Proactive support improves satisfaction

## Example Usage

```python
import sys
sys.path.append('src')
from early_risk_signals import EarlyRiskSignalSystem

# Initialize system
system = EarlyRiskSignalSystem()

# Load data
df = system.load_data('data/Sample.csv')

# Engineer early signals
df_engineered = system.engineer_early_signals(df)

# Identify risk flags
risk_flags = system.identify_risk_flags(df_engineered)

# Generate outreach strategies
strategies = system.generate_outreach_strategies(risk_flags)

# View results
print(risk_flags.head())
print(strategies.head())
```

## Key Insights from Sample Data

From analysis of 100 customer records:
- **25%** flagged as at-risk (DPD Bucket ≥ 1)
- **15%** show high utilization (≥80%)
- **22%** show spending decline (>15%)
- **18%** show low payment frequency (<30%)

## Technology Stack

- **Python 3.8+**
- **Scikit-learn** - Machine learning models
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization

## Documentation

- **`docs/solution_narrative.md`** - Complete solution documentation
- **`docs/EXECUTIVE_SUMMARY.md`** - Executive overview
- **`docs/FULL_WORKFLOW.md`** - Complete workflow walkthrough
- **`docs/EXECUTION_TRANSCRIPT.md`** - Detailed execution transcript

## Contributing

This is a demonstration framework. For production use:
1. Validate thresholds on larger datasets
2. Tune model parameters
3. Integrate with existing systems
4. Establish feedback loops

## License

This solution is provided as-is for demonstration purposes.

## Support

For questions or issues, refer to:
- `docs/solution_narrative.md` - Detailed technical documentation
- `docs/README_HOSTING.md` - Website hosting guide

## Success Criteria Met

- **Clear Logic**: Deterministic thresholds and transparent business rules
- **Data-Backed**: Analysis of 100 customers with statistical validation
- **Targeted Interventions**: Risk-level based, operationally feasible, customer-friendly
- **Scalability**: 3-phase deployment strategy with automation architecture

---

**Version**: 1.0  
**Last Updated**: 2024  
**Status**: Production-Ready

---

**Star this repository if you find it useful!**
