"""
Flask API for serving synthetic dataset with pagination and search
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset once at startup
dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_dataset_50000.csv')
df = None

def load_dataset():
    """Load the dataset into memory (limited to first 1000 rows)"""
    global df
    if df is None:
        try:
            # Load only first 1000 rows
            df = pd.read_csv(dataset_path, nrows=1000)
            print(f"Dataset loaded: {len(df)} rows (limited to first 1000)")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            df = pd.DataFrame()
    return df

@app.route('/api/dataset/info', methods=['GET'])
def get_dataset_info():
    """Get dataset metadata"""
    df = load_dataset()
    return jsonify({
        'total_records': len(df),
        'columns': df.columns.tolist(),
        'status': 'success'
    })

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    """Get paginated dataset with optional search"""
    df = load_dataset()
    
    if df.empty:
        return jsonify({
            'error': 'Dataset not loaded',
            'status': 'error'
        }), 500
    
    # Get query parameters
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 25))
    search = request.args.get('search', '').lower()
    
    # Filter data if search term provided
    filtered_df = df.copy()
    if search:
        # Search across all columns
        mask = False
        for col in df.columns:
            mask = mask | df[col].astype(str).str.lower().str.contains(search, na=False)
        filtered_df = df[mask]
    
    # Calculate pagination
    total_records = len(filtered_df)
    total_pages = (total_records + per_page - 1) // per_page
    
    # Get page data
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_data = filtered_df.iloc[start_idx:end_idx]
    
    # Convert to list of dictionaries
    records = page_data.to_dict('records')
    
    # Format numeric values
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, (int, float)):
                # Keep as is, will format in frontend
                pass
    
    return jsonify({
        'data': records,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_records': total_records,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        },
        'status': 'success'
    })

@app.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats():
    """Get dataset statistics"""
    df = load_dataset()
    
    if df.empty:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    stats = {
        'total_records': len(df),
        'columns': df.columns.tolist(),
        'numeric_stats': {}
    }
    
    # Calculate basic stats for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        stats['numeric_stats'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'median': float(df[col].median())
        }
    
    return jsonify(stats)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'dataset-api'})

if __name__ == '__main__':
    # Load dataset on startup
    load_dataset()
    print("Starting Dataset API server...")
    print(f"Dataset path: {dataset_path}")
    print("API endpoints:")
    print("  GET /api/dataset/info - Get dataset metadata")
    print("  GET /api/dataset?page=1&per_page=25&search=term - Get paginated data")
    print("  GET /api/dataset/stats - Get dataset statistics")
    print("  GET /health - Health check")
    app.run(debug=True, host='0.0.0.0', port=5001)

