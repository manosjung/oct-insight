# OCT-Insight Deployment Guide

## Prerequisites
- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)
- Your trained model file: `models/baseline_model.pkl` (104 MB)

---

## Step 1: Prepare Your Repository

### 1.1 Create a `.gitignore` file

Create a file named `.gitignore` in your project root with this content:

```
# Data files (too large for GitHub)
data/
*.csv

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Legacy code (optional - you may want to keep this)
code/

# Development logs
GELISTIRME_GUNLUGU.md
```

### 1.2 Initialize Git Repository

Open terminal/command prompt in your project folder:

```bash
git init
git add .
git commit -m "Initial commit: OCT-Insight retinal disease classification system"
```

### 1.3 Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `oct-insight`
3. Description: "AI-powered retinal disease classification from OCT scans using ResNet50 and Grad-CAM explainability"
4. Choose **Public** (so Streamlit Cloud can access it)
5. Click "Create repository"

### 1.4 Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/oct-insight.git
git branch -M main
git push -u origin main
```

**IMPORTANT:** Your model file (`models/baseline_model.pkl`) is 104 MB. GitHub has a 100 MB file size limit, so you have two options:

**Option A: Use Git LFS (Large File Storage) - Recommended**
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add models/baseline_model.pkl
git commit -m "Add trained model with Git LFS"
git push
```

**Option B: Host model elsewhere**
- Upload to Google Drive, Dropbox, or Hugging Face
- Modify `app.py` to download the model on first run
- See "Alternative: External Model Hosting" section below

---

## Step 2: Deploy to Streamlit Cloud

### 2.1 Sign Up for Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Authorize Streamlit to access your repositories

### 2.2 Deploy Your App
1. Click "New app"
2. Select your repository: `YOUR_USERNAME/oct-insight`
3. Branch: `main`
4. Main file path: `app.py`
5. Click "Deploy"

### 2.3 Wait for Deployment
- First deployment takes 5-10 minutes
- Streamlit will install all requirements
- You'll see logs in real-time

### 2.4 Your App is Live!
- You'll get a URL like: `https://oct-insight.streamlit.app`
- Share this link on LinkedIn, CV, and social media

---

## Step 3: Alternative - External Model Hosting

If you can't use Git LFS, host your model externally:

### Host on Google Drive
1. Upload `baseline_model.pkl` to Google Drive
2. Right-click → Share → Get shareable link
3. Copy the file ID from the URL (the long string between `/d/` and `/view`)

### Modify app.py to download model:

```python
import gdown
import os

@st.cache_resource
def load_model():
    """Downloads model from Google Drive if not present, then loads it."""
    model_path = Path('models/baseline_model.pkl')

    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        st.info("Downloading model for first run (this may take a minute)...")

        # Replace FILE_ID with your Google Drive file ID
        file_id = "YOUR_FILE_ID_HERE"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(model_path), quiet=False)

    return load_learner(model_path)
```

Add `gdown` to `requirements.txt`:
```
gdown
```

---

## Step 4: Testing Your Deployment

### Test with sample OCT images:
1. Download test images from Kermany dataset
2. Upload to your deployed app
3. Verify predictions are working
4. Check all 4 classes (CNV, DME, Drusen, Normal)

### Common Issues:

**Issue: "ModuleNotFoundError"**
- Check requirements.txt has all dependencies
- Make sure plotly is included

**Issue: "Model not found"**
- Verify model is in `models/baseline_model.pkl`
- Check Git LFS is working
- Or use Google Drive option

**Issue: "Memory limit exceeded"**
- Streamlit Cloud free tier has 1GB RAM limit
- Consider optimizing model size
- Use model quantization if needed

---

## Step 5: Monitoring and Updates

### View App Logs
- Go to your app on Streamlit Cloud
- Click "Manage app" → "Logs"

### Update Your App
```bash
git add .
git commit -m "Update: [describe changes]"
git push
```
Streamlit will automatically redeploy in 1-2 minutes.

### Analytics
- Streamlit Cloud shows visitor count
- No user data is collected by default

---

## Step 6: Optional Enhancements

### Add a README.md to GitHub
Create a professional README with:
- Project description
- Demo GIF/screenshots
- Installation instructions
- Citation for Kermany dataset

### Add Professional Touches
- Custom domain (requires Streamlit Teams plan)
- Better logo in sidebar (replace React logo)
- Add sample images for users to try

### Security Note
This is a research prototype. For clinical use, you would need:
- HIPAA compliance
- CE/FDA approval
- Secure patient data handling
- Audit logging

---

## Your Deployment Checklist

- [ ] Create `.gitignore` file
- [ ] Initialize Git repository
- [ ] Create GitHub repository
- [ ] Set up Git LFS for model file (or use Google Drive)
- [ ] Push code to GitHub
- [ ] Sign up for Streamlit Cloud
- [ ] Deploy app on Streamlit Cloud
- [ ] Test with sample images
- [ ] Get shareable URL
- [ ] Share on LinkedIn with the post below!

---

## Questions?
If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify all files are on GitHub
3. Test locally first: `streamlit run app.py`
4. Streamlit Community Forum: https://discuss.streamlit.io

Good luck with your deployment! 🚀
