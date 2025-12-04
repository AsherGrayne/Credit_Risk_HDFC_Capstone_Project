# Deployment Guide

This project consists of two main components:
1. **Flask API Backend** (`app.py`) - Handles CSV predictions
2. **Frontend** (`index.html` + JavaScript files) - User interface

## Deployment Options

### Option 1: Deploy Everything Together (Recommended)

#### A. Render (Free Tier Available)
**Best for:** Quick deployment with minimal configuration

1. **Create a Render account** at https://render.com

2. **Create a new Web Service:**
   - Connect your GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
   - Environment: Python 3

3. **Update API URL in JavaScript:**
   - Edit `website/csv-batch-predictor.js`
   - Change `http://localhost:5000` to your Render URL (e.g., `https://your-app.onrender.com`)

4. **Deploy static files:**
   - Render can serve static files, or use GitHub Pages for frontend

**Pros:** Free tier, automatic HTTPS, easy setup
**Cons:** Free tier has cold starts (first request may be slow)

---

#### B. Railway (Free Tier Available)
**Best for:** Simple deployment with good performance

1. **Create Railway account** at https://railway.app

2. **Deploy from GitHub:**
   - Connect repository
   - Railway auto-detects Python
   - Set start command: `python app.py`

3. **Add environment variables** (if needed)

4. **Update API URL** in JavaScript to Railway URL

**Pros:** Fast, easy deployment, good free tier
**Cons:** Free tier has usage limits

---

#### C. Heroku (Paid, but has free alternatives)
**Best for:** Established platform with good documentation

1. **Install Heroku CLI**

2. **Create `Procfile`:**
   ```
   web: python app.py
   ```

3. **Deploy:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

**Note:** Heroku removed free tier, but alternatives exist

---

### Option 2: Separate Frontend and Backend

#### Frontend: GitHub Pages (Free)
**Best for:** Static file hosting

1. **Push code to GitHub**

2. **Enable GitHub Pages:**
   - Go to repository Settings → Pages
   - Select branch (usually `main` or `gh-pages`)
   - Select folder (`/root`)

3. **Update API URL** in JavaScript to your backend URL

4. **Access:** `https://yourusername.github.io/repository-name/`

**Pros:** Free, easy, automatic HTTPS
**Cons:** Only serves static files (no Flask API)

---

#### Frontend: Netlify (Free)
**Best for:** Modern static hosting with CI/CD

1. **Create Netlify account** at https://netlify.com

2. **Deploy:**
   - Connect GitHub repository
   - Build command: (leave empty, or `npm install` if needed)
   - Publish directory: `/` (root)

3. **Update API URL** in JavaScript

**Pros:** Free, fast CDN, easy setup
**Cons:** Only static files

---

#### Backend: PythonAnywhere (Free Tier)
**Best for:** Python-focused hosting

1. **Create account** at https://www.pythonanywhere.com

2. **Upload files:**
   - Upload `app.py`, `requirements.txt`, and `models/` folder

3. **Install dependencies:**
   - Open Bash console
   - Run: `pip3.10 install --user -r requirements.txt`

4. **Create Web App:**
   - Go to Web tab
   - Create new web app
   - Set source code to your files
   - Set WSGI file to point to `app.py`

**Pros:** Free tier available, Python-focused
**Cons:** Free tier has limitations

---

#### Backend: Render (Free Tier)
**Best for:** Modern deployment platform

1. **Create Web Service** on Render

2. **Configure:**
   - Build: `pip install -r requirements.txt`
   - Start: `python app.py`
   - Environment: Python 3

3. **Add environment variables** if needed

**Pros:** Free tier, automatic HTTPS, easy
**Cons:** Cold starts on free tier

---

### Option 3: Cloud Platforms

#### AWS (EC2 or Elastic Beanstalk)
**Best for:** Production applications

1. **EC2:**
   - Launch EC2 instance (Ubuntu)
   - Install Python, pip, nginx
   - Deploy Flask app with gunicorn
   - Configure nginx as reverse proxy

2. **Elastic Beanstalk:**
   - Upload application
   - AWS handles deployment

**Pros:** Scalable, production-ready
**Cons:** More complex, costs money

---

#### Google Cloud Platform (Cloud Run)
**Best for:** Containerized deployments

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "app.py"]
   ```

2. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy
   ```

**Pros:** Serverless, scalable
**Cons:** Requires Docker knowledge

---

## Recommended Setup (Easiest)

### Quick Start: Render + GitHub Pages

1. **Deploy Flask API on Render:**
   - Connect GitHub repo
   - Create Web Service
   - Build: `pip install -r requirements.txt`
   - Start: `python app.py`
   - Get URL: `https://your-app.onrender.com`

2. **Update API URL:**
   - Edit `website/csv-batch-predictor.js`
   - Line ~47: Change `http://localhost:5000` to your Render URL

3. **Deploy Frontend on GitHub Pages:**
   - Push code to GitHub
   - Enable GitHub Pages in Settings
   - Your site: `https://username.github.io/repo-name/`

4. **Done!** Frontend calls Render API

---

## Important Files for Deployment

### Required Files:
- `app.py` - Flask API
- `requirements.txt` - Python dependencies
- `models/` folder - ML model files (joblib)
- `index.html` - Frontend
- `website/` folder - JavaScript files
- `data/` folder (optional) - Sample data

### Configuration Updates Needed:

1. **Update API URL in JavaScript:**
   File: `website/csv-batch-predictor.js`
   ```javascript
   // Change this line:
   const response = await fetch('http://localhost:5000/predict_batch', {
   
   // To your deployed API URL:
   const response = await fetch('https://your-api-url.com/predict_batch', {
   ```

2. **CORS Configuration:**
   - `app.py` already has `CORS(app)` enabled
   - This allows frontend to call API from different domain

3. **Environment Variables (Optional):**
   - Create `.env` file for sensitive configs
   - Use `python-dotenv` to load them

---

## Step-by-Step: Render Deployment

### 1. Prepare Repository
```bash
# Ensure all files are committed
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 2. Create Render Account
- Go to https://render.com
- Sign up with GitHub

### 3. Create New Web Service
- Click "New +" → "Web Service"
- Connect your GitHub repository
- Configure:
  - **Name:** credit-card-delinquency-api
  - **Environment:** Python 3
  - **Build Command:** `pip install -r requirements.txt`
  - **Start Command:** `python app.py`
- Click "Create Web Service"

### 4. Wait for Deployment
- Render will build and deploy
- Get your URL: `https://your-app.onrender.com`

### 5. Update Frontend
- Edit `website/csv-batch-predictor.js`
- Update API URL to Render URL
- Commit and push

### 6. Deploy Frontend (GitHub Pages)
- Go to repository Settings → Pages
- Select branch: `main`
- Select folder: `/ (root)`
- Save

### 7. Test
- Visit GitHub Pages URL
- Test CSV upload functionality

---

## Troubleshooting

### API Not Working
- Check API URL is correct in JavaScript
- Verify CORS is enabled in Flask
- Check Render logs for errors
- Ensure model files are uploaded

### CORS Errors
- Ensure `flask-cors` is installed
- Verify `CORS(app)` in `app.py`
- Check browser console for specific errors

### Model Files Missing
- Ensure `models/` folder is in repository
- Check file paths in `app.py`
- Verify joblib files are committed

### Port Issues
- Render/Railway handle ports automatically
- For other platforms, use environment variable:
  ```python
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port)
  ```

---

## Cost Comparison

| Platform | Frontend | Backend | Cost |
|----------|----------|---------|------|
| GitHub Pages + Render | Free | Free (with limits) | $0 |
| Netlify + Railway | Free | Free (with limits) | $0 |
| AWS | Free tier | Pay-as-you-go | ~$5-20/month |
| Heroku | N/A | Paid only | $7+/month |

---

## Recommended for Production

**Best Option:** Render (Backend) + GitHub Pages (Frontend)
- Both free
- Easy setup
- Good documentation
- Automatic HTTPS

**Alternative:** Railway (Both)
- Single platform
- Easy deployment
- Good performance

---

## Next Steps

1. Choose deployment platform
2. Update API URL in JavaScript
3. Deploy backend
4. Deploy frontend
5. Test functionality
6. Share your deployed URL!

Need help with a specific platform? Let me know!

