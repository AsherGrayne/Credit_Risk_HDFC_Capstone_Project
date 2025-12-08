// CSV Batch Prediction Script
// Handles CSV upload, prediction via Flask API (ML models only), and result display with pie chart

let uploadedFile = null;
let pieChartInstance = null;
let currentResults = null; // Store current prediction results for download

// Partition switching
function switchPartition(partition) {
    // Hide all partitions
    document.getElementById('csv-upload-partition').style.display = 'none';
    document.getElementById('individual-customer-partition').style.display = 'none';
    
    // Remove active class from all buttons
    document.getElementById('csvUploadBtn').classList.remove('active');
    document.getElementById('individualCustomerBtn').classList.remove('active');
    
    // Show selected partition
    if (partition === 'csv-upload') {
        document.getElementById('csv-upload-partition').style.display = 'block';
        document.getElementById('csvUploadBtn').classList.add('active');
        document.getElementById('csvUploadBtn').style.background = '#8b0000';
        document.getElementById('csvUploadBtn').style.color = 'white';
        document.getElementById('individualCustomerBtn').style.background = '#1e293b';
        document.getElementById('individualCustomerBtn').style.color = '#9ca3af';
    } else {
        document.getElementById('individual-customer-partition').style.display = 'block';
        document.getElementById('individualCustomerBtn').classList.add('active');
        document.getElementById('individualCustomerBtn').style.background = '#8b0000';
        document.getElementById('individualCustomerBtn').style.color = 'white';
        document.getElementById('csvUploadBtn').style.background = '#1e293b';
        document.getElementById('csvUploadBtn').style.color = '#9ca3af';
    }
}

// Handle CSV file upload
function handleCSVUpload(event) {
    const file = event.target.files[0];
    if (file) {
        uploadedFile = file;
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('uploadPredictBtn').disabled = false;
        document.getElementById('uploadPredictBtn').style.background = '#8b0000';
        document.getElementById('uploadPredictBtn').style.color = 'white';
        document.getElementById('uploadPredictBtn').style.cursor = 'pointer';
        
        // Hide previous results and errors
        document.getElementById('csvResultsContainer').style.display = 'none';
        document.getElementById('csvErrorMessage').style.display = 'none';
    }
}

// Process CSV prediction - Flask API only
async function processCSVPrediction() {
    if (!uploadedFile) {
        showCSVError('Please select a CSV file first');
        return;
    }
    
    // Show loading, hide results and errors
    document.getElementById('csvLoading').style.display = 'block';
    document.getElementById('csvResultsContainer').style.display = 'none';
    document.getElementById('csvErrorMessage').style.display = 'none';
    document.getElementById('uploadPredictBtn').disabled = true;
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        // Call Flask API (ML models only)
        const response = await fetch('http://localhost:5000/predict_batch', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `API error: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Prediction failed');
        }
        
        // Display results
        displayCSVResults(result);
        
    } catch (error) {
        console.error('Prediction error:', error);
        
        // Show clear error message
        let errorMessage = 'Prediction failed. ';
        
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            errorMessage += 'Cannot connect to Flask API. Please ensure the Flask API is running on port 5000. ';
            errorMessage += 'Start it by running: python app.py';
        } else {
            errorMessage += error.message;
        }
        
        showCSVError(errorMessage);
    } finally {
        document.getElementById('csvLoading').style.display = 'none';
        document.getElementById('uploadPredictBtn').disabled = false;
    }
}

// Display CSV prediction results
function displayCSVResults(result) {
    // Store results for download
    currentResults = result;
    
    // Hide loading and errors
    document.getElementById('csvLoading').style.display = 'none';
    document.getElementById('csvErrorMessage').style.display = 'none';
    
    // Show results container
    document.getElementById('csvResultsContainer').style.display = 'block';
    
    // Draw pie chart
    drawPieChart(result.risk_counts);
    
    // Display categorized customers
    displayCategorizedCustomers(result.categorized);
}

// Draw pie chart using Canvas API
function drawPieChart(riskCounts) {
    const canvas = document.getElementById('riskPieChart');
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) - 20;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Colors for each risk level
    const colors = {
        'No Risk': '#10b981',
        'Low Risk': '#3b82f6',
        'Medium Risk': '#f59e0b',
        'High Risk': '#ef4444'
    };
    
    // Calculate total
    const total = Object.values(riskCounts).reduce((sum, count) => sum + count, 0);
    
    if (total === 0) {
        ctx.fillStyle = '#9ca3af';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No data available', centerX, centerY);
        return;
    }
    
    // Draw pie slices
    let currentAngle = -Math.PI / 2; // Start at top
    
    Object.entries(riskCounts).forEach(([riskLevel, count]) => {
        if (count === 0) return;
        
        const sliceAngle = (count / total) * 2 * Math.PI;
        
        // Draw slice
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
        ctx.closePath();
        ctx.fillStyle = colors[riskLevel];
        ctx.fill();
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw label
        const labelAngle = currentAngle + sliceAngle / 2;
        const labelX = centerX + Math.cos(labelAngle) * (radius * 0.7);
        const labelY = centerY + Math.sin(labelAngle) * (radius * 0.7);
        
        ctx.fillStyle = '#e5e7eb';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`${riskLevel}`, labelX, labelY - 5);
        ctx.fillText(`(${count})`, labelX, labelY + 15);
        
        currentAngle += sliceAngle;
    });
    
    // Draw center circle for donut effect
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * 0.4, 0, 2 * Math.PI);
    ctx.fillStyle = '#0f172a';
    ctx.fill();
    
    // Draw total in center
    ctx.fillStyle = '#e5e7eb';
    ctx.font = 'bold 20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Total', centerX, centerY - 10);
    ctx.fillText(total.toString(), centerX, centerY + 15);
}

// Display categorized customers
function displayCategorizedCustomers(categorized) {
    const container = document.getElementById('categorizedCustomers');
    container.innerHTML = '';
    
    const riskOrder = ['No Risk', 'Low Risk', 'Medium Risk', 'High Risk'];
    const colors = {
        'No Risk': '#10b981',
        'Low Risk': '#3b82f6',
        'Medium Risk': '#f59e0b',
        'High Risk': '#ef4444'
    };
    
    riskOrder.forEach(riskLevel => {
        const customers = categorized[riskLevel] || [];
        if (customers.length === 0) return;
        
        const card = document.createElement('div');
        card.style.background = '#020617';
        card.style.border = `2px solid ${colors[riskLevel]}`;
        card.style.borderRadius = '0.5rem';
        card.style.padding = '1.5rem';
        
        const title = document.createElement('h3');
        title.style.color = colors[riskLevel];
        title.style.marginBottom = '1rem';
        title.style.fontSize = '1.25rem';
        title.textContent = `${riskLevel} - ${customers.length} customer${customers.length !== 1 ? 's' : ''}`;
        
        const customerList = document.createElement('div');
        customerList.style.maxHeight = '300px';
        customerList.style.overflowY = 'auto';
        customerList.style.color = '#9ca3af';
        customerList.style.fontSize = '0.875rem';
        customerList.style.lineHeight = '1.8';
        
        customers.forEach(customerId => {
            const span = document.createElement('div');
            span.textContent = customerId;
            span.style.padding = '0.25rem 0';
            customerList.appendChild(span);
        });
        
        card.appendChild(title);
        card.appendChild(customerList);
        container.appendChild(card);
    });
}

// Show CSV error message
function showCSVError(message) {
    const errorDiv = document.getElementById('csvErrorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    errorDiv.style.background = '#7f1d1d';
    errorDiv.style.color = '#fca5a5';
    errorDiv.style.padding = '1rem';
    errorDiv.style.borderRadius = '0.5rem';
    errorDiv.style.marginTop = '1rem';
}

// Download risk segmentation CSV
function downloadRiskSegmentationCSV() {
    if (!currentResults || !currentResults.categorized) {
        alert('No results available to download. Please upload and predict a CSV file first.');
        return;
    }
    
    const categorized = currentResults.categorized;
    const riskOrder = ['No Risk', 'Low Risk', 'Medium Risk', 'High Risk'];
    
    // Create CSV content with format: Risk Level, Customer IDs
    let csvContent = 'Risk Level,Customer IDs\n';
    
    riskOrder.forEach(riskLevel => {
        const customers = categorized[riskLevel] || [];
        if (customers.length > 0) {
            // Format: "Medium Risk", "C001, C002, C003"
            const customerList = customers.join(', ');
            csvContent += `"${riskLevel}","${customerList}"\n`;
        }
    });
    
    // Create blob and download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    // Generate filename with timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const filename = `risk_segmentation_${timestamp}.csv`;
    
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.display = 'none';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Clean up
    URL.revokeObjectURL(url);
}

