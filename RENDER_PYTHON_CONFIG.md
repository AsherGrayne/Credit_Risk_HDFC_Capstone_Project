# Render Python Version Configuration

Since `runtime.txt` was renamed to prevent Netlify detection, configure Python version in Render's UI:

## Steps to Configure Python Version in Render:

1. **Go to your Render Dashboard**
   - Navigate to your web service

2. **Go to Settings**
   - Click on your service
   - Click "Settings" tab

3. **Set Python Version**
   - Look for "Python Version" or "Environment" section
   - Set to: **Python 3.9** (or Python 3.9.18 if available)
   - Or use the dropdown to select Python 3.9

4. **Alternative: Use Environment Variable**
   - Go to "Environment" section
   - Add environment variable:
     - Key: `PYTHON_VERSION`
     - Value: `3.9.18`

5. **Save and Redeploy**
   - Click "Save Changes"
   - Render will redeploy with the specified Python version

## Note:
- Render will use the Python version you specify in the UI
- The `.runtime.txt.render` file is kept as backup/reference
- Your Render deployment should continue working normally

## If Render Still Needs runtime.txt:

If Render requires `runtime.txt` in the root, you can:
1. Create a build script in Render that copies `.runtime.txt.render` to `runtime.txt`
2. Or add this to Render's build command:
   ```bash
   cp .runtime.txt.render runtime.txt && pip install -r requirements.txt
   ```

But typically, Render's UI Python version setting is sufficient.

