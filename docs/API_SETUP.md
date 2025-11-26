# Dataset API Setup Guide

This guide explains how to set up and use the Dataset API instead of direct CSV loading.

## Why Use an API?

- **Better Performance**: Only loads the data you need (pagination)
- **Faster Loading**: No need to download entire 50,000 row CSV
- **Server-Side Search**: More efficient filtering
- **Scalability**: Can handle larger datasets
- **Better Error Handling**: Centralized error management

## Setup Instructions

### 1. Install Dependencies

Make sure you have Flask and Flask-CORS installed:

```bash
pip install flask flask-cors pandas
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

Run the dataset API server:

```bash
python src/dataset_api.py
```

The API will start on `http://localhost:5001`

### 3. API Endpoints

#### Get Dataset Info
```
GET /api/dataset/info
```
Returns metadata about the dataset (total records, columns).

#### Get Paginated Data
```
GET /api/dataset?page=1&per_page=25&search=term
```
Parameters:
- `page`: Page number (default: 1)
- `per_page`: Records per page (default: 25)
- `search`: Optional search term

#### Get Dataset Statistics
```
GET /api/dataset/stats
```
Returns statistical information about the dataset.

#### Health Check
```
GET /health
```
Check if the API is running.

### 4. Update Frontend Configuration

The frontend is already configured to use the API. The API base URL is set in `index.html`:

```javascript
const API_BASE_URL = 'http://localhost:5001/api';
```

### 5. For Production Deployment

If deploying to a server, update the API_BASE_URL in `index.html`:

```javascript
const API_BASE_URL = 'https://your-api-domain.com/api';
```

## Running Both Services

### Option 1: Run API Separately

Terminal 1 (API):
```bash
python src/dataset_api.py
```

Terminal 2 (Website):
```bash
python -m http.server 8000
```

Then open `http://localhost:8000`

### Option 2: Use Flask to Serve Both

You can modify the API to also serve the static files, or use a reverse proxy like nginx.

## Troubleshooting

### CORS Errors
If you see CORS errors, make sure Flask-CORS is installed and the API is running.

### Connection Refused
- Check if the API server is running on port 5001
- Verify the API_BASE_URL in index.html matches your server

### Dataset Not Found
- Ensure `data/synthetic_dataset_50000.csv` exists
- Check the file path in `dataset_api.py`

## Benefits Over Direct CSV Loading

1. **Performance**: Only loads 25 rows at a time instead of 50,000
2. **Memory**: Uses less browser memory
3. **Speed**: Faster initial page load
4. **Scalability**: Can handle datasets of any size
5. **Search**: Server-side search is more efficient
6. **Flexibility**: Easy to add features like sorting, filtering, etc.

