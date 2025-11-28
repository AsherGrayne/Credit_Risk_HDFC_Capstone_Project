# Deployment Readiness Checklist

Use this checklist to verify your website is ready for deployment.

## 📁 File Structure Verification

- [ ] `index.html` exists and is valid HTML
- [ ] `workflow.html` exists and is valid HTML
- [ ] `apply.html` exists and is valid HTML
- [ ] `interactive-dashboard.html` exists and is valid HTML
- [ ] `styles.css` exists
- [ ] `workflow-styles.css` exists
- [ ] `workflow-script.js` exists
- [ ] `apply-script.js` exists
- [ ] `ml-model-predictor.js` exists
- [ ] `interactive-dashboard.js` exists
- [ ] `models/model.json` exists
- [ ] `visualizations/` folder contains 9 PNG files:
  - [ ] `model_comparison.png`
  - [ ] `dataset_comparison.png`
  - [ ] `risk_distribution.png`
  - [ ] `behavioral_patterns.png`
  - [ ] `flag_frequency.png`
  - [ ] `feature_importance.png`
  - [ ] `outreach_strategy.png`
  - [ ] `risk_heatmap.png`
  - [ ] `workflow_diagram.png`

## 🔗 Link Verification

- [ ] All CSS files are linked in HTML files
- [ ] All JavaScript files are linked in HTML files
- [ ] Image paths use relative paths (e.g., `visualizations/image.png`)
- [ ] Model path uses relative path (`models/model.json`)
- [ ] No absolute paths (e.g., `C:/` or `/Users/`)
- [ ] External CDN links work (Google Fonts, Chart.js)

## 🧪 Functionality Testing

### Local Testing (Before Deployment)

- [ ] Open `index.html` in browser - loads without errors
- [ ] All tabs switch correctly
- [ ] "Workflow" tab displays content
- [ ] "Apply Here" tab displays form
- [ ] Form validation works (required fields)
- [ ] Form submission works
- [ ] Prediction results display
- [ ] Risk flags appear
- [ ] All images display correctly
- [ ] Interactive dashboard loads
- [ ] Charts render in dashboard
- [ ] Model loads (check browser console)
- [ ] No JavaScript errors in console
- [ ] No 404 errors in Network tab

### Browser Compatibility

- [ ] Tested in Chrome/Edge
- [ ] Tested in Firefox
- [ ] Tested in Safari (if available)
- [ ] Mobile responsive (test on phone/tablet)

## 📝 Configuration Files

- [ ] `.gitignore` exists (excludes unnecessary files)
- [ ] `README.md` exists (deployment instructions)
- [ ] `DEPLOYMENT.md` exists (detailed guide)
- [ ] `netlify.toml` exists (Netlify config)
- [ ] `vercel.json` exists (Vercel config)

## 🔒 Security & Performance

- [ ] No sensitive data in files
- [ ] No API keys hardcoded
- [ ] External resources use HTTPS
- [ ] Images are optimized (reasonable file sizes)
- [ ] JavaScript files are reasonable size

## 📦 Git Repository

- [ ] All files are committed to git
- [ ] No uncommitted changes
- [ ] `.gitignore` is working (no unnecessary files tracked)
- [ ] Repository is pushed to remote (GitHub/GitLab/etc.)

## 🌐 Deployment Platform Specific

### GitHub Pages
- [ ] Repository is on GitHub
- [ ] GitHub Pages is enabled
- [ ] Source folder set to `/website` (or root if moved)
- [ ] Site URL is accessible

### Netlify
- [ ] Netlify account created
- [ ] Site connected to repository (or drag-drop done)
- [ ] Build settings configured (or using netlify.toml)
- [ ] Custom domain configured (optional)

### Vercel
- [ ] Vercel account created
- [ ] Project connected to repository
- [ ] Build settings configured (or using vercel.json)
- [ ] Custom domain configured (optional)

## ✅ Post-Deployment Verification

After deployment, verify:

- [ ] Site URL is accessible
- [ ] Homepage loads correctly
- [ ] All pages are accessible
- [ ] All images load
- [ ] Form submission works
- [ ] Predictions work
- [ ] No console errors
- [ ] Mobile view works
- [ ] HTTPS is enabled (if available)

## 🐛 Troubleshooting

If something doesn't work:

1. **Check Browser Console:**
   - Open Developer Tools (F12)
   - Check Console tab for errors
   - Check Network tab for failed requests

2. **Verify File Paths:**
   - All paths should be relative
   - Check case sensitivity (Linux servers)
   - Verify file extensions match

3. **Check Hosting Platform:**
   - Review deployment logs
   - Check build status
   - Verify file structure matches

4. **Test Locally First:**
   - If it works locally but not deployed, check:
     - Base path configuration
     - File permissions
     - Case sensitivity

## 📊 Quick Test Data

Use this sample data to test the form:

```
Customer ID: C001
Credit Limit: 165000
Utilisation %: 85
Avg Payment Ratio: 25
Min Due Paid Frequency: 15
Merchant Mix Index: 0.5
Cash Withdrawal %: 20
Recent Spend Change %: -25
```

Expected result: HIGH or CRITICAL risk with multiple flags.

## ✨ Final Steps

Once all items are checked:

1. ✅ **Deploy to chosen platform**
2. ✅ **Test deployed site thoroughly**
3. ✅ **Share URL with stakeholders**
4. ✅ **Monitor for any issues**

---

**Status:** ☐ Not Ready | ☐ Ready for Deployment | ☐ Deployed

**Date:** _______________

**Deployed URL:** _______________

**Notes:** 
_________________________________________________
_________________________________________________
_________________________________________________

