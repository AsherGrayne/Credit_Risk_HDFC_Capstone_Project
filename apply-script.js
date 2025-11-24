// Apply Here - Prediction Script

document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Hide previous results and errors
    document.getElementById('resultContainer').classList.remove('show');
    document.getElementById('errorMessage').classList.remove('show');
    
    // Show loading
    document.getElementById('loading').classList.add('show');
    document.getElementById('submitBtn').disabled = true;
    
    // Get form data
    const formData = {
        customer_id: document.getElementById('customer_id').value,
        credit_limit: parseFloat(document.getElementById('credit_limit').value),
        utilisation: parseFloat(document.getElementById('utilisation').value),
        avg_payment_ratio: parseFloat(document.getElementById('avg_payment_ratio').value),
        min_due_frequency: parseFloat(document.getElementById('min_due_frequency').value),
        merchant_mix: parseFloat(document.getElementById('merchant_mix').value),
        cash_withdrawal: parseFloat(document.getElementById('cash_withdrawal').value),
        spend_change: parseFloat(document.getElementById('spend_change').value),
        dpd_bucket: document.getElementById('dpd_bucket').value ? parseInt(document.getElementById('dpd_bucket').value) : null
    };
    
    try {
        // Call prediction API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed. Please try again.');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        // If API call fails, use client-side prediction (fallback)
        console.log('API not available, using client-side prediction...');
        const result = predictClientSide(formData);
        displayResults(result);
    } finally {
        document.getElementById('loading').classList.remove('show');
        document.getElementById('submitBtn').disabled = false;
    }
});

function predictClientSide(data) {
    // Calculate early warning signals (same logic as Python)
    const spending_stress = data.spend_change < -20 ? 2 : (data.spend_change < -10 ? 1 : 0);
    const utilization_risk = data.utilisation >= 90 ? 3 : (data.utilisation >= 70 ? 2 : (data.utilisation >= 50 ? 1 : 0));
    const payment_stress = data.min_due_frequency < 20 ? 2 : (data.min_due_frequency < 40 ? 1 : 0);
    const cash_stress = data.cash_withdrawal >= 20 ? 2 : (data.cash_withdrawal >= 10 ? 1 : 0);
    const narrow_merchant = data.merchant_mix < 0.4 ? 1 : 0;
    
    // Calculate early risk score
    const early_risk_score = (
        spending_stress * 0.25 +
        utilization_risk * 0.30 +
        payment_stress * 0.25 +
        cash_stress * 0.10 +
        narrow_merchant * 0.10
    ) / 3.0;
    
    // Generate flags
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
    
    // Determine risk level
    let risk_level = 'LOW';
    if (early_risk_score >= 0.8 || flags.some(f => f.severity === 'CRITICAL')) {
        risk_level = 'CRITICAL';
    } else if (early_risk_score >= 0.6 || flags.some(f => f.severity === 'HIGH')) {
        risk_level = 'HIGH';
    } else if (early_risk_score >= 0.3 || flags.length > 0) {
        risk_level = 'MEDIUM';
    }
    
    // Simple prediction based on risk score and flags
    // This is a simplified version - in production, use the actual trained model
    const at_risk_probability = Math.min(0.95, Math.max(0.05, early_risk_score * 1.2));
    const prediction = at_risk_probability > 0.5 ? 'At-Risk' : 'Not At-Risk';
    
    return {
        risk_level: risk_level,
        risk_score: early_risk_score.toFixed(3),
        prediction: prediction,
        probability: (at_risk_probability * 100).toFixed(1) + '%',
        flags: flags,
        flag_count: flags.length
    };
}

function displayResults(result) {
    // Update risk level
    const riskLevelCard = document.getElementById('riskLevelCard');
    riskLevelCard.className = 'result-card ' + result.risk_level.toLowerCase();
    document.getElementById('riskLevel').textContent = result.risk_level;
    
    // Update other values
    document.getElementById('riskScore').textContent = result.risk_score;
    document.getElementById('prediction').textContent = result.prediction;
    document.getElementById('probability').textContent = result.probability;
    
    // Display flags
    const flagsList = document.getElementById('flagsList');
    flagsList.innerHTML = '';
    
    if (result.flags && result.flags.length > 0) {
        result.flags.forEach(flag => {
            const flagDiv = document.createElement('div');
            flagDiv.className = 'flag-item';
            flagDiv.innerHTML = `
                <strong>${flag.flag}</strong>
                <span>${flag.message}</span>
            `;
            flagsList.appendChild(flagDiv);
        });
    } else {
        flagsList.innerHTML = '<p style="color: #9ca3af;">No risk flags detected. Customer appears to be low risk.</p>';
    }
    
    // Show results
    document.getElementById('resultContainer').classList.add('show');
    
    // Scroll to results
    document.getElementById('resultContainer').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

