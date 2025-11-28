# Credit Card Delinquency Watch - Website

A professional, deployable website showcasing the Early Risk Signals system for credit card delinquency prediction.

## 🚀 Quick Deploy

This website is ready to deploy to any static hosting service. Choose your preferred platform:

### Option 1: GitHub Pages (Recommended)

1. **Push to GitHub:**
   ```bash
   git add website/
   git commit -m "Add deployable website"
   git push origin main
   ```

2. **Enable GitHub Pages:**
   - Go to repository Settings → Pages
   - Source: Select `/website` folder
   - Branch: `main`
   - Click Save

3. **Your site will be live at:**
   ```
   https://YOUR_USERNAME.github.io/REPO_NAME/website/
   ```

### Option 2: Netlify (Easiest)

1. **Drag & Drop:**
   - Go to [netlify.com](https://www.netlify.com)
   - Drag the `website/` folder onto Netlify
   - Site is live instantly!

2. **Or Git Integration:**
   - Connect GitHub repository
   - Build command: (leave empty)
   - Publish directory: `website`
   - Deploy!

### Option 3: Vercel

```bash
cd website
vercel
```

Follow the prompts - deployed in seconds!

## 📁 File Structure

```
website/
├── index.html                 # Main landing page with tabs
├── workflow.html              # Standalone workflow page
├── apply.html                 # Risk prediction form
├── interactive-dashboard.html # Interactive dashboard
├── styles.css                 # Main stylesheet
├── workflow-styles.css        # Workflow page styles
├── workflow-script.js          # Workflow navigation logic
├── apply-script.js            # Form handling & prediction
├── ml-model-predictor.js      # Client-side ML model
├── interactive-dashboard.js   # Dashboard interactivity
├── models/
│   └── model.json            # Trained Random Forest model
└── visualizations/            # All PNG charts (9 files)
    ├── model_comparison.png
    ├── dataset_comparison.png
    ├── risk_distribution.png
    ├── behavioral_patterns.png
    ├── flag_frequency.png
    ├── feature_importance.png
    ├── outreach_strategy.png
    ├── risk_heatmap.png
    └── workflow_diagram.png
```

## ✅ Pre-Deployment Checklist

- [x] All HTML files created and linked correctly
- [x] CSS files included and referenced
- [x] JavaScript files included and referenced
- [x] Model JSON file in `models/` directory
- [x] All visualization PNGs in `visualizations/` directory
- [x] External fonts loaded (Google Fonts)
- [x] External libraries loaded (Chart.js via CDN)
- [x] Client-side prediction working
- [x] Form validation implemented
- [x] Responsive design tested

## 🌐 Features

### Pages

1. **index.html** - Main page with tabbed interface
   - Workflow visualization
   - Apply Here form
   - Links to dashboard

2. **workflow.html** - Complete workflow breakdown
   - 7-step process visualization
   - Code examples
   - Results and findings

3. **apply.html** - Risk prediction form
   - Customer parameter input
   - Real-time risk prediction
   - Early warning flags display

4. **interactive-dashboard.html** - Interactive analysis
   - Live parameter adjustment
   - Real-time risk calculation
   - Visual charts and graphs

### Functionality

- ✅ **Client-Side ML Prediction** - Runs Random Forest model in browser
- ✅ **Risk Flag Generation** - Deterministic rule-based flags
- ✅ **Interactive Visualizations** - Chart.js powered charts
- ✅ **Responsive Design** - Works on mobile, tablet, desktop
- ✅ **Dark Theme** - Professional dark UI
- ✅ **No Server Required** - Fully static, works offline

## 🔧 Configuration

### For GitHub Pages

If deploying to GitHub Pages, ensure:
- All file paths are relative (they already are)
- `index.html` is in the root of the `website/` folder
- Images are in `visualizations/` subfolder
- Model is in `models/` subfolder

### For Netlify

Create `netlify.toml` in project root:
```toml
[build]
  publish = "website"
  command = ""

[[redirects]]
  from = "/*"
  to = "/website/index.html"
  status = 200
```

### For Vercel

Create `vercel.json` in project root:
```json
{
  "buildCommand": "",
  "outputDirectory": "website",
  "rewrites": [
    { "source": "/(.*)", "destination": "/website/index.html" }
  ]
}
```

## 📊 Testing

Before deploying, test locally:

1. **Open `index.html` in browser**
2. **Test all tabs:**
   - Workflow tab loads correctly
   - Apply Here form submits
   - Dashboard loads
3. **Check console for errors:**
   - No 404 errors for assets
   - JavaScript loads correctly
   - Model loads successfully
4. **Test form submission:**
   - Enter sample data
   - Verify prediction displays
   - Check risk flags appear

## 🐛 Troubleshooting

### Images Not Loading
- Verify `visualizations/` folder exists
- Check file names match exactly (case-sensitive)
- Ensure images are committed to git

### Model Not Loading
- Verify `models/model.json` exists
- Check browser console for fetch errors
- Ensure CORS is enabled if using API

### Tabs Not Working
- Check browser console for JavaScript errors
- Verify all script tags are loaded
- Check file paths are correct

### Form Not Submitting
- Check browser console for errors
- Verify form validation passes
- Check prediction function is called

## 📝 Notes

- **No Build Step Required** - This is a pure static site
- **No Dependencies** - All libraries loaded via CDN
- **Works Offline** - Client-side prediction doesn't need server
- **Mobile Friendly** - Responsive design included

## 🔗 Links

- Main Documentation: `../docs/README_HOSTING.md`
- Hosting Guide: `../docs/HOSTING_GUIDE.md`
- Project README: `../README.md`

## 📄 License

Same as main project.

---

**Ready to deploy! 🚀**

