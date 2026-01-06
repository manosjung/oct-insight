# 👁️ OCT-Insight: AI-Powered Retinal Disease Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**An interactive deep learning system for automated classification of retinal diseases from Optical Coherence Tomography (OCT) scans, with clinical explainability through Grad-CAM visualization.**

![OCT-Insight Banner](assets/banner.png)
*AI-powered diagnosis of CNV, DME, Drusen, and Normal retinal conditions*

---

## 🎯 Overview

OCT-Insight is a research prototype that demonstrates the application of deep learning to retinal disease diagnosis. Built by a physician with interest in medical AI, this project emphasizes **clinical relevance** and **explainability**—not just raw accuracy.

### Key Features

- 🔬 **4-Class Classification**: CNV, DME, Drusen, Normal
- 🧠 **ResNet50 Architecture**: Transfer learning from ImageNet
- 📊 **84K Training Images**: Kermany OCT Dataset
- 🔍 **Grad-CAM Explainability**: Visual heatmaps showing decision rationale
- 🌐 **Interactive Web App**: Upload and analyze OCT scans in real-time
- 📈 **Clinical Metrics**: Sensitivity-focused evaluation for patient safety

### Clinical Significance

| Disease | Type | Urgency | Treatment |
|---------|------|---------|-----------|
| **CNV** (Choroidal Neovascularization) | Wet AMD | 🚨 URGENT | Anti-VEGF injections |
| **DME** (Diabetic Macular Edema) | Diabetic | High | Steroids/anti-VEGF |
| **Drusen** | Dry AMD | Monitoring | Vitamins, observation |
| **Normal** | Healthy | N/A | None needed |

**Why this matters:** CNV can cause irreversible vision loss within days if missed. Automated triage improves efficiency while Grad-CAM explanations build physician trust.

---

## 🚀 Quick Start

### Try the Live Demo
👉 **[Launch OCT-Insight](YOUR_STREAMLIT_URL_HERE)**

### Run Locally

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/oct-insight.git
cd oct-insight

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open http://localhost:8501 and upload an OCT scan!

---

## 📊 Technical Architecture

### Model

- **Architecture**: ResNet50 (50-layer Residual Network)
- **Pre-training**: ImageNet weights
- **Framework**: PyTorch + FastAI
- **Input**: 224×224 RGB (converted from grayscale OCT)
- **Output**: 4-class softmax probabilities

### Dataset

- **Source**: [Kermany et al. OCT Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- **Total Images**: 84,495 retinal OCT scans
- **Classes**:
  - CNV: 37,205 images (44.5%)
  - NORMAL: 26,315 images (31.5%)
  - DME: 11,348 images (13.6%)
  - DRUSEN: 8,616 images (10.3%)
- **Split**: 83,484 training / 1,000 test (balanced)

### Performance

- **Target Accuracy**: >92% (4-class classification)
- **CNV Sensitivity**: >95% (critical for urgent cases)
- **Normal Specificity**: >90%
- **Evaluation**: Confusion matrix, AUC-ROC, F1-score

### Technology Stack

```
PyTorch ────→ Deep Learning Framework
FastAI ─────→ High-level Training API
Streamlit ──→ Interactive Web Interface
Grad-CAM ───→ Explainability Heatmaps
Plotly ─────→ Interactive Visualizations
OpenCV ─────→ Image Processing
```

---

## 🔍 Grad-CAM Explainability

Unlike black-box AI systems, OCT-Insight shows **which anatomical regions** influenced each diagnosis:

- **CNV**: Highlights subretinal fluid and neovascular membranes
- **DME**: Focuses on intraretinal cysts and diffuse edema
- **Drusen**: Identifies sub-RPE deposits
- **Normal**: Shows uniform attention across healthy retinal layers

This transparency is critical for clinical adoption and physician trust.

---

## 📁 Project Structure

```
oct-insight/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── models/
│   └── baseline_model.pkl      # Trained ResNet50 model (100MB)
├── notebooks/
│   ├── 01_data_exploration.py  # Dataset analysis
│   └── 02_baseline_model.py    # Model training script
├── data/                       # OCT dataset (not in repo)
│   └── raw/OCT2017/
│       ├── train/
│       └── test/
├── DEPLOYMENT_GUIDE.md         # Deployment instructions
├── LINKEDIN_AND_CV_CONTENT.md  # Professional content templates
└── README.md                   # This file
```

---

## ⚠️ Important Disclaimer

**This is a research prototype for educational and demonstration purposes only.**

- ❌ NOT approved for clinical diagnosis (no CE/FDA clearance)
- ❌ NOT validated on independent clinical datasets
- ❌ NOT HIPAA-compliant for patient data
- ✅ Intended for research, learning, and portfolio demonstration

**Always consult qualified ophthalmologists for medical diagnosis and treatment decisions.**

---

## 📚 Citation

### Dataset

```
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018):
Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification.
Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
```

### This Project

```
[Your Name] (2025). OCT-Insight: Explainable AI for Retinal Disease Classification.
GitHub: https://github.com/YOUR_USERNAME/oct-insight
```

---

## 🤝 Contributing

Feedback and contributions are welcome! Areas for improvement:

- [ ] Model optimization (EfficientNet, Vision Transformers)
- [ ] Multi-modal integration (OCT + Fundus photos)
- [ ] Clinical validation on independent datasets
- [ ] Uncertainty quantification
- [ ] Real-time inference optimization

---

## 📧 Contact

**[Your Name]**
Medical Doctor | AI/ML Enthusiast

- 🔗 LinkedIn: [Your Profile]
- 📧 Email: [Your Email]
- 🌐 Portfolio: [Your Website]

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

This project is free to use for research and educational purposes. Commercial use requires approval.

---

## 🙏 Acknowledgments

- **Kermany et al.** for the OCT dataset
- **FastAI team** for democratizing deep learning
- **Streamlit** for making ML deployment accessible
- **PyTorch** and the open-source AI community

---

## 🌟 Star This Project

If you found this project helpful for learning medical AI or building your portfolio, please consider giving it a ⭐ on GitHub!

---

**Built with ❤️ by a physician passionate about translational medical AI**

*Bridging clinical expertise with artificial intelligence for better patient care*
