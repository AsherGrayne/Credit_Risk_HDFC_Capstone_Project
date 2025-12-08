// CSV Batch Prediction Script
// Handles CSV upload, prediction via Flask API (ML models only), and result display with pie chart

// API Configuration
// ============================================
// IMPORTANT: Update this with your Render API URL
// ============================================
// To find your Render URL:
// 1. Go to your Render dashboard
// 2. Click on your web service
// 3. Copy the URL (e.g., https://your-app-name.onrender.com)
// 4. Replace the URL below
// ============================================

const RENDER_API_URL = 'https://credit-risk-hdfc-capstone-project.onrender.com';

const API_BASE_URL = (() => {
    // Check if we're running on localhost (for local development)
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return 'http://localhost:5000';
    }
    // For production, use the Render URL
    return RENDER_API_URL;
})();

let uploadedFile = null;
let pieChartInstance = null;
let currentResults = null; // Store current prediction results for download
let originalCSVData = null; // Store original CSV data (headers and rows)

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

// Parse CSV line handling quoted fields
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    result.push(current.trim());
    
    return result;
}

// Handle CSV file upload
async function handleCSVUpload(event) {
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
        
        // Parse and store original CSV data
        try {
            const text = await file.text();
            const lines = text.split('\n').filter(line => line.trim());
            
            if (lines.length > 0) {
                // Parse header
                const headers = parseCSVLine(lines[0]);
                
                // Parse rows
                const rows = [];
                for (let i = 1; i < lines.length; i++) {
                    if (lines[i].trim()) {
                        const values = parseCSVLine(lines[i]);
                        if (values.length >= headers.length) {
                            rows.push(values);
                        }
                    }
                }
                
                originalCSVData = {
                    headers: headers,
                    rows: rows
                };
            }
        } catch (error) {
            console.error('Error parsing CSV file:', error);
            originalCSVData = null;
        }
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
        const response = await fetch(`${API_BASE_URL}/predict_batch`, {
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
            errorMessage += `Cannot connect to Flask API at ${API_BASE_URL}. `;
            if (API_BASE_URL.includes('localhost')) {
                errorMessage += 'Please ensure the Flask API is running on port 5000. Start it by running: python app.py';
            } else {
                errorMessage += 'Please check if your Render API is deployed and accessible.';
            }
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
        ctx.fillStyle = '#111827';
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
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw label
        const labelAngle = currentAngle + sliceAngle / 2;
        const labelX = centerX + Math.cos(labelAngle) * (radius * 0.7);
        const labelY = centerY + Math.sin(labelAngle) * (radius * 0.7);
        
        ctx.fillStyle = '#111827';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`${riskLevel}`, labelX, labelY - 5);
        ctx.fillText(`(${count})`, labelX, labelY + 15);
        
        currentAngle += sliceAngle;
    });
    
    // Draw center circle for donut effect
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * 0.4, 0, 2 * Math.PI);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    
    // Draw total in center
    ctx.fillStyle = '#111827';
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

// Escape CSV field if it contains commas, quotes, or newlines
function escapeCSVField(field) {
    if (field === null || field === undefined) {
        return '';
    }
    const str = String(field);
    if (str.includes(',') || str.includes('"') || str.includes('\n')) {
        return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
}

// Download risk segmentation CSV with original data plus risk level
function downloadRiskSegmentationCSV() {
    if (!currentResults || !currentResults.predictions) {
        alert('No results available to download. Please upload and predict a CSV file first.');
        return;
    }
    
    if (!originalCSVData || !originalCSVData.headers || !originalCSVData.rows) {
        alert('Original CSV data not available. Please upload the CSV file again.');
        return;
    }
    
    // Create a mapping from Customer ID to Risk Level
    const riskLevelMap = {};
    currentResults.predictions.forEach(prediction => {
        riskLevelMap[prediction['Customer ID']] = prediction['Risk Level'];
    });
    
    // Find Customer ID column index
    const customerIdIndex = originalCSVData.headers.findIndex(
        header => header.trim().toLowerCase() === 'customer id'
    );
    
    if (customerIdIndex === -1) {
        alert('Could not find "Customer ID" column in the uploaded CSV file.');
        return;
    }
    
    // Build CSV content
    // Header: original headers + Risk Level
    const headers = [...originalCSVData.headers, 'Risk Level'];
    const csvRows = [headers.map(escapeCSVField).join(',')];
    
    // Add rows: original data + risk level
    originalCSVData.rows.forEach(row => {
        // Get customer ID from the appropriate column
        const customerId = row[customerIdIndex]?.trim() || '';
        const riskLevel = riskLevelMap[customerId] || 'Unknown';
        
        // Add original row data + risk level
        const newRow = [...row, riskLevel];
        csvRows.push(newRow.map(escapeCSVField).join(','));
    });
    
    const csvContent = csvRows.join('\n');
    
    // Create blob and download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    // Generate filename with timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const filename = `customer_data_with_risk_level_${timestamp}.csv`;
    
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.display = 'none';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Clean up
    URL.revokeObjectURL(url);
}

