// Interactive Dashboard Script
let riskChart = null;
let featureChart = null;
let updateTimeout = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupSliders();
    updatePrediction();
});

function setupSliders() {
    const sliders = ['utilisation', 'payment-ratio', 'frequency', 'cash', 'spend', 'merchant'];
    
    sliders.forEach(sliderId => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(sliderId + '-value');
        
        slider.addEventListener('input', function() {
            // Update value display
            let displayValue = parseFloat(this.value);
            if (sliderId === 'merchant') {
                valueDisplay.textContent = displayValue.toFixed(2);
            } else if (sliderId === 'spend') {
                valueDisplay.textContent = displayValue.toFixed(1) + '%';
            } else {
                valueDisplay.textContent = displayValue.toFixed(1) + (sliderId === 'utilisation' || sliderId === 'payment-ratio' || sliderId === 'frequency' || sliderId === 'cash' ? '%' : '');
            }
            
            // Debounce prediction update
            clearTimeout(updateTimeout);
            updateTimeout = setTimeout(updatePrediction, 100);
        });
    });
}

async function updatePrediction() {
    // Get slider values
    const data = {
        utilisation: parseFloat(document.getElementById('utilisation').value),
        avg_payment_ratio: parseFloat(document.getElementById('payment-ratio').value),
        min_due_frequency: parseFloat(document.getElementById('frequency').value),
        cash_withdrawal: parseFloat(document.getElementById('cash').value),
        spend_change: parseFloat(document.getElementById('spend').value),
        merchant_mix: parseFloat(document.getElementById('merchant').value)
    };
    
    try {
        // Use ML model for prediction
        const mlResult = await predictWithMLModel(data);
        const probability = parseFloat(mlResult.probability);
        
        // Generate flags
        const flags = generateRiskFlags(data);
        
        // Calculate risk level
        const risk_score = probability;
        const risk_level = determineRiskLevel(risk_score, flags);
        
        // Update display
        updateRiskDisplay(risk_level, risk_score, probability * 100);
        updateRiskChart(risk_score, probability);
        updateFeatureChart(data);
        updateFlags(flags);
        
    } catch (error) {
        console.error('Prediction error:', error);
        // Fallback to rule-based
        const result = predictClientSide(data);
        updateRiskDisplay(result.risk_level, parseFloat(result.risk_score), parseFloat(result.probability));
        updateRiskChart(parseFloat(result.risk_score), parseFloat(result.probability) / 100);
        updateFeatureChart(data);
        updateFlags(result.flags);
    }
}

function updateRiskDisplay(level, score, probability) {
    const riskDisplay = document.getElementById('risk-display');
    const riskLevel = document.getElementById('risk-level');
    const riskScore = document.getElementById('risk-score');
    const riskProb = document.getElementById('risk-probability');
    
    // Remove all classes
    riskDisplay.className = 'risk-level-display';
    riskLevel.className = 'risk-level';
    
    // Add appropriate class
    riskDisplay.classList.add(level.toLowerCase());
    riskLevel.classList.add(level.toLowerCase());
    
    // Update text
    riskLevel.textContent = level;
    riskScore.textContent = score.toFixed(3);
    riskProb.textContent = probability.toFixed(1) + '% At-Risk Probability';
}

function initializeCharts() {
    // Risk Score Chart
    const riskCtx = document.getElementById('riskChart').getContext('2d');
    riskChart = new Chart(riskCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Risk Score',
                data: [],
                borderColor: '#8b0000',
                backgroundColor: 'rgba(139, 0, 0, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    grid: {
                        color: '#1e293b'
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                },
                x: {
                    grid: {
                        color: '#1e293b'
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                }
            }
        }
    });
    
    // Feature Importance Chart
    const featureCtx = document.getElementById('featureChart').getContext('2d');
    featureChart = new Chart(featureCtx, {
        type: 'bar',
        data: {
            labels: ['Utilisation', 'Payment Ratio', 'Frequency', 'Cash', 'Spend Change', 'Merchant Mix'],
            datasets: [{
                label: 'Impact',
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: [
                    'rgba(139, 0, 0, 0.8)',
                    'rgba(220, 20, 60, 0.8)',
                    'rgba(255, 99, 71, 0.8)',
                    'rgba(255, 165, 0, 0.8)',
                    'rgba(50, 205, 50, 0.8)',
                    'rgba(100, 149, 237, 0.8)'
                ],
                borderColor: [
                    '#8b0000',
                    '#dc143c',
                    '#ff6347',
                    '#ffa500',
                    '#32cd32',
                    '#6495ed'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#1e293b'
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                },
                x: {
                    grid: {
                        color: '#1e293b'
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                }
            }
        }
    });
}

let riskHistory = [];
let historyIndex = 0;

function updateRiskChart(score, probability) {
    riskHistory.push({
        index: historyIndex++,
        score: score,
        probability: probability
    });
    
    // Keep only last 20 points
    if (riskHistory.length > 20) {
        riskHistory.shift();
    }
    
    riskChart.data.labels = riskHistory.map(d => d.index);
    riskChart.data.datasets[0].data = riskHistory.map(d => d.score);
    riskChart.update('none');
}

function updateFeatureChart(data) {
    // Calculate feature impacts
    const impacts = calculateFeatureImpacts(data);
    
    featureChart.data.datasets[0].data = [
        impacts.utilisation,
        impacts.paymentRatio,
        impacts.frequency,
        impacts.cash,
        impacts.spend,
        impacts.merchant
    ];
    featureChart.update('none');
}

function calculateFeatureImpacts(data) {
    // Calculate how much each feature contributes to risk
    const spending_stress = data.spend_change < -20 ? 2 : (data.spend_change < -10 ? 1 : 0);
    const utilization_risk = data.utilisation >= 90 ? 3 : (data.utilisation >= 70 ? 2 : (data.utilisation >= 50 ? 1 : 0));
    const payment_stress = data.min_due_frequency < 20 ? 2 : (data.min_due_frequency < 40 ? 1 : 0);
    const cash_stress = data.cash_withdrawal >= 20 ? 2 : (data.cash_withdrawal >= 10 ? 1 : 0);
    
    return {
        utilisation: (utilization_risk / 3) * 0.30,
        paymentRatio: (data.avg_payment_ratio < 60 ? 1 : 0) * 0.20,
        frequency: (payment_stress / 2) * 0.25,
        cash: (cash_stress / 2) * 0.10,
        spend: (spending_stress / 2) * 0.25,
        merchant: (data.merchant_mix < 0.4 ? 1 : 0) * 0.10
    };
}

function updateFlags(flags) {
    const container = document.getElementById('flags-container');
    
    if (!flags || flags.length === 0) {
        container.innerHTML = '<p style="color: #9ca3af; text-align: center; padding: 2rem;">No risk flags detected. Customer appears to be low risk.</p>';
        return;
    }
    
    container.innerHTML = flags.map(flag => `
        <div class="flag-item">
            <span class="flag-severity ${flag.severity.toLowerCase()}">${flag.severity}</span>
            <strong>${flag.flag}</strong>
            <div style="margin-top: 0.5rem; color: #9ca3af; font-size: 0.875rem;">${flag.message}</div>
        </div>
    `).join('');
}

// Import functions from apply-script.js (they should be available globally)
function generateRiskFlags(data) {
    const flags = [];
    
    if (data.spend_change < -20) {
        flags.push({
            flag: 'SPENDING_DECLINE_SEVERE',
            severity: 'HIGH',
            message: `Spending dropped ${Math.abs(data.spend_change)}% - potential financial stress`
        });
    } else if (data.spend_change < -15) {
        flags.push({
            flag: 'SPENDING_DECLINE_MODERATE',
            severity: 'MEDIUM',
            message: `Spending declined ${Math.abs(data.spend_change)}%`
        });
    }
    
    if (data.utilisation >= 90) {
        flags.push({
            flag: 'CRITICAL_UTILIZATION',
            severity: 'HIGH',
            message: `Credit utilization at ${data.utilisation}% - near limit`
        });
    } else if (data.utilisation >= 80) {
        flags.push({
            flag: 'HIGH_UTILIZATION',
            severity: 'MEDIUM',
            message: `Credit utilization at ${data.utilisation}%`
        });
    }
    
    if (data.min_due_frequency < 20) {
        flags.push({
            flag: 'LOW_PAYMENT_FREQUENCY',
            severity: 'HIGH',
            message: `Only ${data.min_due_frequency}% minimum due payments made`
        });
    } else if (data.min_due_frequency < 40) {
        flags.push({
            flag: 'MODERATE_PAYMENT_FREQUENCY',
            severity: 'MEDIUM',
            message: `Payment frequency at ${data.min_due_frequency}%`
        });
    }
    
    if (data.cash_withdrawal >= 20) {
        flags.push({
            flag: 'HIGH_CASH_WITHDRAWAL',
            severity: 'MEDIUM',
            message: `Cash withdrawals at ${data.cash_withdrawal}% of spending`
        });
    }
    
    if (data.utilisation > 70 && data.avg_payment_ratio < 60) {
        flags.push({
            flag: 'UTILIZATION_PAYMENT_MISMATCH',
            severity: 'HIGH',
            message: 'High utilization with low payment ratio'
        });
    }
    
    if (data.spend_change < -15 && data.utilisation > 60) {
        flags.push({
            flag: 'SPENDING_UTILIZATION_STRESS',
            severity: 'HIGH',
            message: 'Declining spending with high utilization'
        });
    }
    
    if (data.min_due_frequency < 30 && data.utilisation > 70) {
        flags.push({
            flag: 'PAYMENT_UTILIZATION_CRITICAL',
            severity: 'CRITICAL',
            message: 'Low payment frequency with high utilization'
        });
    }
    
    return flags;
}

function determineRiskLevel(probability, flags) {
    if (probability >= 0.8 || flags.some(f => f.severity === 'CRITICAL')) {
        return 'CRITICAL';
    } else if (probability >= 0.6 || flags.some(f => f.severity === 'HIGH')) {
        return 'HIGH';
    } else if (probability >= 0.3 || flags.length > 0) {
        return 'MEDIUM';
    }
    return 'LOW';
}

function predictClientSide(data) {
    const spending_stress = data.spend_change < -20 ? 2 : (data.spend_change < -10 ? 1 : 0);
    const utilization_risk = data.utilisation >= 90 ? 3 : (data.utilisation >= 70 ? 2 : (data.utilisation >= 50 ? 1 : 0));
    const payment_stress = data.min_due_frequency < 20 ? 2 : (data.min_due_frequency < 40 ? 1 : 0);
    const cash_stress = data.cash_withdrawal >= 20 ? 2 : (data.cash_withdrawal >= 10 ? 1 : 0);
    
    const early_risk_score = (
        spending_stress * 0.25 +
        utilization_risk * 0.30 +
        payment_stress * 0.25 +
        cash_stress * 0.10
    ) / 3.0;
    
    const flags = generateRiskFlags(data);
    const risk_level = determineRiskLevel(early_risk_score, flags);
    const at_risk_probability = Math.min(0.95, Math.max(0.05, early_risk_score * 1.2));
    
    return {
        risk_level: risk_level,
        risk_score: early_risk_score.toFixed(3),
        prediction: at_risk_probability > 0.5 ? 'At-Risk' : 'Not At-Risk',
        probability: (at_risk_probability * 100).toFixed(1) + '%',
        flags: flags,
        flag_count: flags.length
    };
}

