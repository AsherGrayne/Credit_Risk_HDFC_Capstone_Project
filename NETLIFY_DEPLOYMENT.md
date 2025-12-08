# Netlify Deployment Guide

This guide will help you deploy your Credit Card Delinquency Prediction frontend to Netlify.

## Prerequisites

- Your code is pushed to GitHub
- You have a GitHub account
- Your Render API is already deployed (backend)

## Step-by-Step Deployment

### Method 1: Deploy via Netlify Dashboard (Easiest)

#### Step 1: Sign Up/Login to Netlify

1. Go to [https://www.netlify.com](https://www.netlify.com)
2. Click **"Sign up"** or **"Log in"**
3. Choose **"Sign up with GitHub"** (recommended for easy integration)

#### Step 2: Create New Site

1. Once logged in, click **"Add new site"** → **"Import an existing project"**
2. Click **"Connect to GitHub"** (if not already connected)
3. Authorize Netlify to access your GitHub repositories
4. Select your repository: `Credit_Risk_HDFC_Capstone_Project`

#### Step 3: Configure Build Settings

Netlify will auto-detect settings, but verify:

- **Branch to deploy:** `main`
- **Build command:** (leave empty - no build needed for static site)
- **Publish directory:** `.` (root directory)

**Note:** Since this is a static site, no build command is needed.

#### Step 4: Deploy

1. Click **"Deploy site"**
2. Wait for deployment to complete (usually 1-2 minutes)
3. Your site will be live at: `https://random-name-12345.netlify.app`

#### Step 5: Customize Domain (Optional)

1. Go to **Site settings** → **Domain management**
2. Click **"Add custom domain"**
3. Enter your domain name (if you have one)
4. Follow DNS configuration instructions

---

### Method 2: Deploy via Netlify CLI (Advanced)

#### Step 1: Install Netlify CLI

```bash
# Install globally using npm
npm install -g netlify-cli

# Or using Homebrew (Mac)
brew install netlify-cli
```

#### Step 2: Login to Netlify

```bash
netlify login
```

This will open your browser to authenticate.

#### Step 3: Initialize Site

```bash
# Navigate to your project directory
cd "C:\Users\profe\Desktop\Credit Card Delinquency Pack"

# Initialize Netlify
netlify init
```

Follow the prompts:
- **Create & configure a new site** (or link to existing)
- **Team:** Select your team
- **Site name:** (optional, Netlify will generate one)
- **Build command:** (press Enter - leave empty)
- **Directory to deploy:** `.` (press Enter for root)

#### Step 4: Deploy

```bash
# Deploy to production
netlify deploy --prod

# Or deploy a draft first
netlify deploy
```

---

## Post-Deployment Configuration

### 1. Update API URL (Already Done)

Your API URL is already configured in `website/csv-batch-predictor.js`:
```javascript
const RENDER_API_URL = 'https://credit-risk-hdfc-capstone-project.onrender.com';
```

This should work automatically with Netlify.

### 2. Enable Automatic Deployments

Netlify automatically deploys when you push to GitHub:
- Go to **Site settings** → **Build & deploy**
- Verify **"Deploy settings"** shows your GitHub repository
- **Automatic deploys** should be enabled by default

### 3. Configure Environment Variables (If Needed)

If you need to set environment variables:

1. Go to **Site settings** → **Environment variables**
2. Add variables:
   - `API_URL` = `https://credit-risk-hdfc-capstone-project.onrender.com`
   - (Add any others as needed)

---

## Troubleshooting

### Issue: Site shows 404 or blank page

**Solution:**
- Check that `index.html` is in the root directory
- Verify `netlify.toml` redirects are configured correctly
- Check Netlify build logs for errors

### Issue: API calls failing (CORS errors)

**Solution:**
- Verify your Render API has CORS enabled (already configured in `app.py`)
- Check browser console for specific error messages
- Ensure API URL in `csv-batch-predictor.js` is correct

### Issue: Assets not loading (CSS/JS files)

**Solution:**
- Check file paths in `index.html` (should be relative paths)
- Verify all files are committed to GitHub
- Check Netlify deploy logs for missing files

### Issue: Slow performance

**Solution:**
- Netlify uses CDN, so it should be fast
- Check if large files are being loaded unnecessarily
- Consider lazy loading heavy features (already implemented)

---

## Netlify Features You Can Use

### 1. Form Handling (If Needed)

Netlify can handle form submissions:
- Add `netlify` attribute to forms
- No backend needed for simple forms

### 2. Branch Previews

- Every pull request gets a preview URL
- Test changes before merging

### 3. Split Testing

- Test different versions of your site
- A/B testing capabilities

### 4. Analytics (Optional)

- Enable Netlify Analytics
- Track site visitors and performance

---

## Quick Commands Reference

```bash
# Login
netlify login

# Deploy draft
netlify deploy

# Deploy to production
netlify deploy --prod

# Open site dashboard
netlify open

# View site logs
netlify logs

# Check site status
netlify status
```

---

## Next Steps After Deployment

1. ✅ Test your deployed site
2. ✅ Verify API calls work correctly
3. ✅ Test CSV upload functionality
4. ✅ Share your Netlify URL!

---

## Your Deployment URLs

- **Frontend (Netlify):** `https://your-site-name.netlify.app`
- **Backend (Render):** `https://credit-risk-hdfc-capstone-project.onrender.com`

Both are now live and connected! 🎉

---

## Need Help?

- Netlify Docs: https://docs.netlify.com
- Netlify Community: https://answers.netlify.com
- Check deployment logs in Netlify dashboard

