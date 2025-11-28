# Website Deployment Guide

Complete guide for deploying the Credit Card Delinquency Watch website.

## 🎯 Deployment Status

**✅ READY FOR DEPLOYMENT**

All files are in place and properly configured. The website is a static site that can be deployed to any static hosting service.

## 📋 Pre-Deployment Checklist

- [x] All HTML files created (`index.html`, `workflow.html`, `apply.html`, `interactive-dashboard.html`)
- [x] All CSS files created (`styles.css`, `workflow-styles.css`)
- [x] All JavaScript files created (`workflow-script.js`, `apply-script.js`, `ml-model-predictor.js`, `interactive-dashboard.js`)
- [x] Model file exists (`models/model.json`)
- [x] All visualization images exist (`visualizations/*.png` - 9 files)
- [x] File paths are relative (no absolute paths)
- [x] External dependencies loaded via CDN (Google Fonts, Chart.js)
- [x] No build step required (pure static site)
- [x] Client-side prediction working (no server needed)

## 🚀 Deployment Options

### 1. GitHub Pages (Free & Easy)

**Best for:** Projects already on GitHub

**Steps:**

1. **Ensure files are committed:**
   ```bash
   git add website/
   git commit -m "Add deployable website"
   git push origin main
   ```

2. **Enable GitHub Pages:**
   - Navigate to your repository on GitHub
   - Go to **Settings** → **Pages**
   - Under **Source**, select:
     - Branch: `main`
     - Folder: `/website`
   - Click **Save**

3. **Wait 1-2 minutes** for GitHub to build

4. **Your site will be live at:**
   ```
   https://YOUR_USERNAME.github.io/REPO_NAME/website/
   ```

**Note:** If you want the site at the root URL, you'll need to configure GitHub Pages to serve from the `/website` directory or move files to root.

### 2. Netlify (Recommended - Easiest)

**Best for:** Quick deployment, custom domains

**Method 1: Drag & Drop**

1. Go to [netlify.com](https://www.netlify.com)
2. Sign up/login (free account)
3. Drag the entire `website/` folder onto Netlify dashboard
4. Site is live instantly!
5. Get a custom domain or use `your-site.netlify.app`

**Method 2: Git Integration**

1. Connect your GitHub repository
2. Configure build settings:
   - Build command: (leave empty)
   - Publish directory: `website`
3. Click **Deploy site**
4. Netlify will auto-deploy on every push

**Configuration:** The `netlify.toml` file is already configured in the website folder.

### 3. Vercel (Fast & Modern)

**Best for:** Modern deployments, edge functions

**Steps:**

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Deploy:**
   ```bash
   cd website
   vercel
   ```

3. **Follow prompts:**
   - Link to existing project? No
   - Project name: (press enter for default)
   - Directory: `.` (current directory)
   - Override settings? No

4. **Site is live!** Get URL from terminal

**Configuration:** The `vercel.json` file is already configured.

### 4. Cloudflare Pages (Free & Fast)

**Best for:** Global CDN, fast performance

**Steps:**

1. Go to [pages.cloudflare.com](https://pages.cloudflare.com)
2. Sign up/login
3. Connect GitHub repository
4. Configure:
   - Framework preset: None
   - Build command: (leave empty)
   - Build output directory: `website`
5. Click **Save and Deploy**

### 5. AWS S3 + CloudFront (Enterprise)

**Best for:** Enterprise deployments, custom infrastructure

**Steps:**

1. Create S3 bucket
2. Upload all website files
3. Enable static website hosting
4. Configure CloudFront distribution
5. Set up custom domain (optional)

## 🔧 Configuration Files

### netlify.toml
- Already created in `website/` folder
- Configures headers, caching, redirects
- No changes needed

### vercel.json
- Already created in `website/` folder
- Configures routing and headers
- No changes needed

### .gitignore
- Already created in `website/` folder
- Excludes unnecessary files
- No changes needed

## ✅ Post-Deployment Testing

After deployment, test:

1. **Homepage loads:**
   - Visit your deployed URL
   - Verify `index.html` loads correctly

2. **Navigation works:**
   - Click all tabs
   - Verify pages switch correctly
   - Check links to other pages

3. **Form functionality:**
   - Go to "Apply Here" tab
   - Fill out form with sample data:
     - Customer ID: C001
     - Credit Limit: 165000
     - Utilisation %: 85
     - Avg Payment Ratio: 25
     - Min Due Paid Frequency: 15
     - Merchant Mix Index: 0.5
     - Cash Withdrawal %: 20
     - Recent Spend Change %: -25
   - Submit form
   - Verify prediction displays
   - Check risk flags appear

4. **Visualizations load:**
   - Check all images display
   - Verify no broken image links

5. **Model loads:**
   - Open browser console (F12)
   - Check for "✓ ML Model loaded successfully"
   - Verify no errors

6. **Responsive design:**
   - Test on mobile device
   - Test on tablet
   - Verify layout adapts

## 🐛 Common Issues & Solutions

### Issue: Images Not Loading

**Solution:**
- Verify `visualizations/` folder is deployed
- Check file names match exactly (case-sensitive)
- Ensure images are committed to git

### Issue: Model Not Loading

**Solution:**
- Verify `models/model.json` exists
- Check browser console for CORS errors
- Ensure model file is committed

### Issue: 404 Errors

**Solution:**
- Check file paths are relative (not absolute)
- Verify all files are in correct directories
- Check hosting platform's base path configuration

### Issue: Tabs Not Working

**Solution:**
- Check browser console for JavaScript errors
- Verify all script files are loaded
- Check file paths in HTML

## 📊 Performance Optimization

The website is already optimized:

- ✅ Images are PNG format (optimized)
- ✅ JavaScript is minified-ready
- ✅ CSS is organized and efficient
- ✅ External libraries loaded via CDN
- ✅ No unnecessary dependencies

**Optional Optimizations:**

1. **Image Optimization:**
   - Compress PNG files further
   - Consider WebP format
   - Use image CDN

2. **Code Minification:**
   - Minify JavaScript files
   - Minify CSS files
   - Use build tool (optional)

3. **Caching:**
   - Already configured in `netlify.toml` and `vercel.json`
   - Static assets cached for 1 year

## 🔒 Security

Security headers are configured:

- ✅ X-Frame-Options: DENY
- ✅ X-XSS-Protection: Enabled
- ✅ X-Content-Type-Options: nosniff
- ✅ Referrer-Policy: strict-origin-when-cross-origin

## 📝 Custom Domain Setup

### Netlify:
1. Go to Site settings → Domain management
2. Add custom domain
3. Follow DNS configuration instructions

### Vercel:
1. Go to Project settings → Domains
2. Add domain
3. Configure DNS records

### GitHub Pages:
1. Go to Repository settings → Pages
2. Add custom domain
3. Configure DNS CNAME record

## 🆘 Support

If you encounter issues:

1. Check browser console for errors
2. Verify all files are deployed
3. Test locally first
4. Check hosting platform logs
5. Review this deployment guide

## 📚 Additional Resources

- [GitHub Pages Docs](https://docs.github.com/en/pages)
- [Netlify Docs](https://docs.netlify.com)
- [Vercel Docs](https://vercel.com/docs)
- [Cloudflare Pages Docs](https://developers.cloudflare.com/pages)

---

**Your website is ready to deploy! 🚀**

Choose your preferred platform and follow the steps above. The website will be live in minutes!

