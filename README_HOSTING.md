# Website Hosting Guide

This website can be hosted on any static hosting service. Here are instructions for popular platforms:

## 🚀 Quick Hosting Options

### Option 1: GitHub Pages (Free & Easy)

1. **Create a GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/credit-risk-website.git
   git push -u origin main
   ```

2. **Enable GitHub Pages**
   - Go to your repository on GitHub
   - Click **Settings** → **Pages**
   - Under **Source**, select **main branch**
   - Click **Save**
   - Your site will be available at: `https://YOUR_USERNAME.github.io/credit-risk-website/`

### Option 2: Netlify (Recommended - Easiest)

1. **Drag & Drop Method**
   - Go to [netlify.com](https://www.netlify.com)
   - Sign up/login (free)
   - Drag the entire folder containing `index.html` and `styles.css` onto Netlify
   - Your site is live instantly!

2. **Git Integration**
   - Connect your GitHub repository
   - Netlify will auto-deploy on every push
   - Get a custom domain or use netlify subdomain

### Option 3: Vercel

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Deploy:
   ```bash
   vercel
   ```

3. Follow the prompts - your site will be live in seconds!

### Option 4: GitHub Pages via GitHub Desktop

1. Download [GitHub Desktop](https://desktop.github.com/)
2. Create a new repository
3. Add all files (index.html, styles.css, and all PNG images)
4. Publish to GitHub
5. Enable GitHub Pages in repository settings

## 📁 Required Files

Make sure these files are in the same directory:

```
website/
├── index.html
├── styles.css
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

## 🔧 Local Testing

Before hosting, test locally:

1. **Using Python** (if installed):
   ```bash
   python -m http.server 8000
   ```
   Then open: http://localhost:8000

2. **Using Node.js** (if installed):
   ```bash
   npx http-server
   ```

3. **Using VS Code**:
   - Install "Live Server" extension
   - Right-click `index.html` → "Open with Live Server"

## 🌐 Custom Domain (Optional)

### Netlify:
1. Go to Site Settings → Domain Management
2. Add your custom domain
3. Update DNS records as instructed

### GitHub Pages:
1. Add `CNAME` file with your domain name
2. Update DNS records:
   - Type: `CNAME`
   - Name: `www` (or `@`)
   - Value: `YOUR_USERNAME.github.io`

## 📱 Mobile Responsive

The website is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones

## 🔒 HTTPS

All hosting platforms provide free HTTPS certificates automatically.

## 📊 Performance Tips

1. **Image Optimization**: Images are already optimized, but you can further compress if needed
2. **CDN**: Netlify and Vercel provide CDN automatically
3. **Caching**: Browser caching is enabled via headers

## 🆘 Troubleshooting

### Images Not Showing:
- Check that all PNG files are in the same directory as `index.html`
- Verify file names match exactly (case-sensitive)
- Check browser console for 404 errors

### Styling Issues:
- Ensure `styles.css` is in the same directory
- Check browser console for CSS loading errors
- Clear browser cache (Ctrl+F5)

### GitHub Pages Not Updating:
- Wait 1-2 minutes after pushing changes
- Clear browser cache
- Check repository settings → Pages

## 📝 Notes

- The website is a static site (no backend required)
- All images are loaded lazily for better performance
- Smooth scrolling navigation included
- Fully responsive design

## 🎨 Customization

To customize the website:

1. **Colors**: Edit CSS variables in `styles.css`:
   ```css
   :root {
       --primary-color: #2563eb;
       --secondary-color: #1e40af;
   }
   ```

2. **Content**: Edit text in `index.html`

3. **Images**: Replace PNG files with your own (keep same names)

## ✅ Checklist Before Hosting

- [ ] All PNG images are in the same folder
- [ ] `index.html` and `styles.css` are present
- [ ] Test locally first
- [ ] Check all images load correctly
- [ ] Test on mobile device
- [ ] Verify navigation links work

---

**Need Help?** Check the hosting platform's documentation or support forums.

