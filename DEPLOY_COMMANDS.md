# 🚀 DEPLOYMENT COMMANDS - Copy & Paste These

Follow these steps in order. Open **Git Bash** or **Command Prompt** in your project folder.

---

## ✅ Step 1: Test Locally (Optional but Recommended)

First, let's make sure everything works on your computer:

```bash
streamlit run app.py
```

- Open the URL it shows (usually http://localhost:8501)
- Upload a test OCT image from your data folder
- Check that prediction works
- **IMPORTANT:** Click on "🔍 See Explanation (Grad-CAM Heatmap)" and verify it shows the heatmap
- If it works, press Ctrl+C to stop the server

---

## ✅ Step 2: Initialize Git Repository

```bash
git init
```

---

## ✅ Step 3: Install Git LFS (For Large Model File)

Your model is 100MB, which is at GitHub's limit. Git LFS helps manage large files.

**Install Git LFS:**

**Windows:**
```bash
# Download and install from: https://git-lfs.github.com/
# After installation, run:
git lfs install
```

**Mac:**
```bash
brew install git-lfs
git lfs install
```

**Linux:**
```bash
sudo apt-get install git-lfs
git lfs install
```

**Track your model file with LFS:**
```bash
git lfs track "*.pkl"
git add .gitattributes
```

---

## ✅ Step 4: Stage All Files

```bash
git add .
```

---

## ✅ Step 5: Create First Commit

```bash
git commit -m "Initial commit: OCT-Insight retinal disease classification with Grad-CAM"
```

---

## ✅ Step 6: Create GitHub Repository

1. Go to: https://github.com/new
2. **Repository name:** `oct-insight`
3. **Description:** `AI-powered retinal disease classification from OCT scans using ResNet50 and Grad-CAM explainability`
4. **Visibility:** ✅ **Public** (required for free Streamlit deployment)
5. **DO NOT** check "Initialize with README" (we already have files)
6. Click **"Create repository"**

---

## ✅ Step 7: Connect to GitHub

**Replace `YOUR_USERNAME` with your actual GitHub username:**

```bash
git remote add origin https://github.com/YOUR_USERNAME/oct-insight.git
git branch -M main
```

---

## ✅ Step 8: Push to GitHub

```bash
git push -u origin main
```

**If it asks for credentials:**
- Username: Your GitHub username
- Password: Use a **Personal Access Token** (not your password)
  - Create token at: https://github.com/settings/tokens
  - Click "Generate new token (classic)"
  - Select scopes: `repo` (all checkboxes)
  - Copy the token and paste when asked for password

**If you get an error about file size:**
- Make sure Git LFS is installed (Step 3)
- Run: `git lfs migrate import --include="*.pkl"`
- Then try pushing again

---

## ✅ Step 9: Deploy to Streamlit Cloud

1. Go to: https://share.streamlit.io
2. Click **"Sign in"** → Sign in with GitHub
3. Authorize Streamlit Cloud to access your repositories
4. Click **"New app"** (or "Create app")
5. Fill in:
   - **Repository:** `YOUR_USERNAME/oct-insight`
   - **Branch:** `main`
   - **Main file path:** `app.py`
6. Click **"Deploy!"**

**Wait 5-10 minutes** for first deployment (it's installing all libraries)

You can watch the logs in real-time.

---

## ✅ Step 10: Get Your Live URL

Once deployed, you'll get a URL like:
```
https://oct-insight.streamlit.app
```

or

```
https://YOUR_USERNAME-oct-insight.streamlit.app
```

**🎉 Your app is now LIVE!**

---

## ✅ Step 11: Test Your Live App

1. Open your Streamlit Cloud URL
2. Upload a test OCT image
3. Verify predictions work
4. **Check Grad-CAM works** - click "See Explanation" and verify heatmap displays
5. Test all 4 disease classes if possible

---

## ✅ Step 12: Share on LinkedIn

1. Open `LINKEDIN_AND_CV_CONTENT.md`
2. Copy the LinkedIn post (use the detailed or short version)
3. Replace `[INSERT YOUR STREAMLIT CLOUD URL HERE]` with your actual URL
4. Replace `[INSERT YOUR GITHUB REPO URL HERE]` with `https://github.com/YOUR_USERNAME/oct-insight`
5. Post on LinkedIn!

**Suggested hashtags:**
#MedicalAI #DeepLearning #Ophthalmology #MachineLearning #HealthTech #ExplainableAI #RetinalImaging #ComputerVision #PyTorch #DataScience #ClinicalResearch #DigitalHealth

---

## 🔧 Troubleshooting

### "Model not found" error on Streamlit Cloud
- Check that `models/baseline_model.pkl` is in your GitHub repo
- Verify Git LFS is tracking it: `git lfs ls-files`
- Try re-pushing: `git lfs push --all origin main`

### "Memory limit exceeded"
- Streamlit Cloud free tier has 1GB RAM
- Your model + app should fit, but if not, consider model optimization

### "Module not found" error
- Check `requirements.txt` has all dependencies
- All dependencies are already listed correctly

### Grad-CAM not working
- Check the Streamlit Cloud logs for errors
- Verify pytorch-grad-cam is installed (it's in requirements.txt)
- Test locally first to debug

### Want to update your app?
```bash
# Make changes to your code
git add .
git commit -m "Update: [describe what you changed]"
git push
```
Streamlit will auto-redeploy in 1-2 minutes!

---

## 📊 Monitoring Your App

- **View logs:** Go to your app on Streamlit Cloud → "Manage app" → "Logs"
- **Restart app:** "Manage app" → "Reboot app"
- **View analytics:** See visitor count on the dashboard

---

## 🎯 Summary Checklist

- [ ] Tested app locally with Grad-CAM
- [ ] Initialized Git
- [ ] Installed Git LFS
- [ ] Committed all files
- [ ] Created GitHub repository (public)
- [ ] Pushed to GitHub
- [ ] Deployed on Streamlit Cloud
- [ ] Tested live app
- [ ] Verified Grad-CAM works online
- [ ] Got shareable URL
- [ ] Posted on LinkedIn
- [ ] Updated CV

---

## 🆘 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Verify all commands completed successfully
3. Check Streamlit Cloud logs
4. Test locally: `streamlit run app.py`
5. Google the specific error message
6. Ask on Streamlit Forum: https://discuss.streamlit.io

---

**Good luck! 🚀**

Your OCT-Insight app with working Grad-CAM is ready to impress!
