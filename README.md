# OCT-Insight: Retinal Disease Classification

AI-powered classification of retinal diseases from OCT scans using deep learning. Built with ResNet50 and FastAI, featuring Grad-CAM explainability for clinical transparency.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live Demo:** [oct-insight-farukorman.streamlit.app](https://oct-insight-farukorman.streamlit.app/)

---

## Overview

This project demonstrates automated retinal disease classification using deep learning on Optical Coherence Tomography (OCT) images. The system classifies four conditions:

- **CNV** (Choroidal Neovascularization) - Wet AMD, urgent treatment needed
- **DME** (Diabetic Macular Edema) - Requires monitoring and treatment
- **Drusen** - Early AMD marker, routine monitoring
- **Normal** - Healthy retina

**Key Features:**
- ResNet50 architecture with transfer learning
- Trained on 84,000+ OCT images (Kermany et al. 2018 dataset)
- Grad-CAM visualization showing model attention areas
- Interactive Streamlit web interface
- CPU-optimized for deployment without GPU

---

## Project Structure

```
oct2/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages for deployment
├── LICENSE                         # MIT license
├── models/
│   └── baseline_model.pkl          # Trained ResNet50 model (~100MB)
├── notebooks/
│   ├── 01_data_exploration.py      # Dataset analysis script
│   └── 02_baseline_model.py        # Model training script
└── data/                           # Training data (not in repo - too large)
    └── raw/OCT2017/                # Kermany OCT dataset (5.7 GB)
        ├── train/                  # 83,484 images
        └── test/                   # 1,000 images
```

---

## Quick Start

### Try the Live Demo
Visit **[oct-insight-farukorman.streamlit.app](https://oct-insight-farukorman.streamlit.app/)** to test the model with your own OCT images.

### Run Locally

**Prerequisites:**
- Python 3.10 or higher
- 2 GB RAM minimum

**Installation:**

```bash
# Clone repository
git clone https://github.com/manosjung/oct2.git
cd oct2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`. Upload an OCT scan to get started!

---

## Dataset

**Source:** [Kermany et al. OCT Dataset (2018)](https://data.mendeley.com/datasets/rscbjbr9sj/2)

- **Training:** 83,484 OCT B-scan images across 4 classes
- **Test:** 1,000 images (250 per class, balanced)
- **Format:** JPEG, grayscale, ~500×400 pixels average
- **Device:** Heidelberg Spectralis OCT

**Class Distribution (Training Set):**
```
CNV:     37,205 images (44.5%)
DME:     11,348 images (13.6%)
DRUSEN:   8,616 images (10.3%)
NORMAL:  26,315 images (31.5%)
```

Note: The dataset is **not included** in this repository due to size (5.7 GB). Download it from the link above and place it in `data/raw/OCT2017/`.

---

## Model Architecture

**Base Model:** ResNet50 pretrained on ImageNet

**Training Details:**
- Transfer learning with FastAI
- Input size: 224×224 RGB
- Data augmentation: rotation, flip, zoom, brightness
- Optimizer: Adam with 1cycle policy
- Loss: Cross-entropy

**Explainability:**
- Grad-CAM heatmaps targeting ResNet50's layer4
- Visualizes which image regions influenced the prediction
- Helps build trust and enable clinical validation

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch 2.1.0 (CPU-only) |
| Framework | FastAI 2.7.14 |
| Web App | Streamlit |
| Explainability | Grad-CAM |
| Visualization | Plotly, OpenCV |
| Deployment | Streamlit Cloud + Hugging Face |

**Key Dependencies:**
- `torch==2.1.0` - Deep learning framework
- `fastai==2.7.14` - High-level training API
- `grad-cam>=1.4.8` - Explainability heatmaps
- `numpy<2,>=1.24.0` - Must use 1.x for PyTorch 2.1 compatibility
- `streamlit` - Web interface
- `plotly` - Interactive charts

---

## Training Your Own Model

If you want to retrain from scratch:

1. **Download the dataset** from [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2)
2. **Extract to** `data/raw/OCT2017/`
3. **Run the training script:**

```bash
python notebooks/02_baseline_model.py
```

The trained model will be saved to `models/baseline_model.pkl`. You can adjust hyperparameters in the script (epochs, batch size, learning rate).

---

## Troubleshooting

### "Grad-CAM not available" error

**Cause:** NumPy 2.x compatibility issue with PyTorch 2.1

**Fix:** Force install NumPy 1.x:
```bash
pip install "numpy<2,>=1.24.0"
```

### Model loading fails

**Solution:** The model will auto-download from Hugging Face on first run. If download fails, check your internet connection.

### Out of memory error

**Solution:** The app uses CPU-only inference and should run with 2GB RAM. Close other applications or reduce image size.

---

## Disclaimer

**This is a research prototype for educational purposes only.**

- ❌ NOT a medical device
- ❌ NOT FDA/CE approved
- ❌ NOT for clinical diagnosis or treatment decisions
- ✅ For research, learning, and portfolio demonstration

**Always consult qualified ophthalmologists for medical diagnosis and treatment.**

---

## Citation

If you use this project in your research, please cite:

**Dataset:**
```bibtex
@data{kermany2018oct,
  author    = {Kermany, Daniel and Zhang, Kang and Goldbaum, Michael},
  title     = {Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images},
  year      = {2018},
  publisher = {Mendeley Data},
  version   = {V2},
  doi       = {10.17632/rscbjbr9sj.2}
}
```

**This Project:**
```bibtex
@software{oct_insight_2025,
  author = {Faruk Orman},
  title  = {OCT-Insight: AI-Powered Retinal Disease Classification},
  year   = {2025},
  url    = {https://github.com/manosjung/oct2}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

**Note:** Any clinical or commercial use requires independent validation, regulatory approval, and compliance with healthcare regulations.

---

## Contact

**Faruk Orman**

- 🔗 LinkedIn: [linkedin.com/in/farukorman](https://www.linkedin.com/in/farukorman/)
- 🐙 GitHub: [@manosjung](https://github.com/manosjung)

**Issues & Questions:** [GitHub Issues](https://github.com/manosjung/oct2/issues)

---

## Acknowledgments

- **Kermany et al.** for the OCT dataset and pioneering work
- **FastAI community** for democratizing deep learning
- **PyTorch team** for the framework
- **Streamlit** for easy ML deployment

---

**Built with PyTorch • FastAI • Streamlit • Grad-CAM**
