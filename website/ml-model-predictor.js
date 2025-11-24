// ML Model Predictor - Client-side Random Forest prediction using JSON model
// This allows the trained ML model to run in the browser without a server

let modelData = null;
let modelLoaded = false;

// Load model from JSON file
async function loadModel() {
    if (modelLoaded) return true;
    
    try {
        const response = await fetch('../models/model.json');
        if (!response.ok) {
            throw new Error('Failed to load model.json');
        }
        modelData = await response.json();
        modelLoaded = true;
        console.log('✓ ML Model loaded successfully');
        console.log(`  Trees: ${modelData.trees.length}`);
        console.log(`  Features: ${modelData.feature_names.length}`);
        return true;
    } catch (error) {
        console.error('Error loading model:', error);
        return false;
    }
}

// Scale features using the same scaler as Python
function scaleFeatures(features) {
    const scaled = [];
    for (let i = 0; i < features.length; i++) {
        const mean = modelData.scaler.mean[i];
        const scale = modelData.scaler.scale[i];
        scaled.push((features[i] - mean) / scale);
    }
    return scaled;
}

// Predict using a single tree
function predictTree(tree, features) {
    function traverse(node) {
        if (node.leaf) {
            // Return probability of class 1 (at-risk)
            const total = node.value[0] + node.value[1];
            return total > 0 ? node.value[1] / total : 0.5;
        }
        
        // Get feature index
        const featureIndex = modelData.feature_names.indexOf(node.feature);
        const featureValue = features[featureIndex];
        
        // Traverse left or right based on threshold
        if (featureValue <= node.threshold) {
            return traverse(node.left);
        } else {
            return traverse(node.right);
        }
    }
    
    return traverse(tree);
}

// Predict using Random Forest (average of all trees)
function predictRandomForest(features) {
    if (!modelLoaded || !modelData) {
        throw new Error('Model not loaded');
    }
    
    // Scale features
    const scaledFeatures = scaleFeatures(features);
    
    // Get predictions from all trees
    let totalProbability = 0;
    for (const tree of modelData.trees) {
        totalProbability += predictTree(tree, scaledFeatures);
    }
    
    // Average probability across all trees
    const avgProbability = totalProbability / modelData.trees.length;
    
    return avgProbability;
}

// Feature engineering (same as Python)
function engineerFeatures(rawData) {
    // Calculate early warning signals
    const spending_stress = rawData.spend_change < -20 ? 2 : (rawData.spend_change < -10 ? 1 : 0);
    const utilization_risk = rawData.utilisation >= 90 ? 3 : (rawData.utilisation >= 70 ? 2 : (rawData.utilisation >= 50 ? 1 : 0));
    const payment_stress = rawData.min_due_frequency < 20 ? 2 : (rawData.min_due_frequency < 40 ? 1 : 0);
    const cash_stress = rawData.cash_withdrawal >= 20 ? 2 : (rawData.cash_withdrawal >= 10 ? 1 : 0);
    const cash_stress_indicator = cash_stress > 0 ? 1 : 0;
    
    // Composite signals
    const utilization_payment_mismatch = (rawData.utilisation > 70 && rawData.avg_payment_ratio < 60) ? 1 : 0;
    const spending_utilization_stress = (rawData.spend_change < -15 && rawData.utilisation > 60) ? 1 : 0;
    const payment_utilization_critical = (rawData.min_due_frequency < 30 && rawData.utilisation > 70) ? 1 : 0;
    
    // Return features in the same order as Python
    return [
        rawData.utilisation,                    // Utilisation %
        rawData.avg_payment_ratio,              // Avg Payment Ratio
        rawData.min_due_frequency,              // Min Due Paid Frequency
        rawData.merchant_mix,                   // Merchant Mix Index
        rawData.cash_withdrawal,                // Cash Withdrawal %
        rawData.spend_change,                   // Recent Spend Change %
        spending_stress,                        // spending_stress
        utilization_risk,                       // utilization_risk
        payment_stress,                         // payment_stress
        cash_stress_indicator,                  // cash_stress_indicator
        utilization_payment_mismatch,           // utilization_payment_mismatch
        spending_utilization_stress,            // spending_utilization_stress
        payment_utilization_critical            // payment_utilization_critical
    ];
}

// Main prediction function using ML model
async function predictWithMLModel(rawData) {
    // Load model if not already loaded
    const loaded = await loadModel();
    if (!loaded) {
        throw new Error('Could not load ML model');
    }
    
    // Engineer features
    const features = engineerFeatures(rawData);
    
    // Get probability from Random Forest
    const probability = predictRandomForest(features);
    
    // Convert to prediction
    const prediction = probability > 0.5 ? 'At-Risk' : 'Not At-Risk';
    
    return {
        prediction: prediction,
        probability: probability,
        probabilityPercent: (probability * 100).toFixed(1) + '%'
    };
}

