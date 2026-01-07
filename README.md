# 👁️ OCT-Insight: AI-Powered Retinal Disease Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oct-insight.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![FastAI](https://img.shields.io/badge/FastAI-2.7.14-blue.svg)](https://www.fast.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An interactive deep learning system for automated classification of retinal diseases from Optical Coherence Tomography (OCT) scans, with clinical explainability through Grad-CAM visualization.**

---

## 🎯 Overview

OCT-Insight is a research prototype that demonstrates the application of deep learning to retinal disease diagnosis. Built with medical AI best practices, this project emphasizes **clinical relevance**, **explainability**, and **reproducibility**—not just raw accuracy.

### Key Features

- 🔬 **4-Class Classification**: CNV, DME, Drusen, Normal
- 🧠 **ResNet50 Architecture**: Transfer learning from ImageNet
- 📊 **84K Training Images**: Kermany OCT Dataset (2018)
- 🔍 **Grad-CAM Explainability**: Visual heatmaps showing model attention
- 🌐 **Interactive Web App**: Upload and analyze OCT scans in real-time
- 📈 **Production-Ready**: Deployed on Streamlit Cloud with Hugging Face model hosting
- ⚡ **CPU-Optimized**: Runs efficiently without GPU requirements

### Clinical Significance

| Disease | Full Name | Type | Urgency | Treatment |
|---------|-----------|------|---------|-----------|
| **CNV** | Choroidal Neovascularization | Wet AMD | 🚨 **URGENT** | Anti-VEGF injections |
| **DME** | Diabetic Macular Edema | Diabetic | 🟡 High | Steroids/Anti-VEGF |
| **DRUSEN** | Drusenoid Deposits | Dry AMD | 🟠 Monitoring | Vitamins, observation |
| **NORMAL** | Healthy Retina | N/A | 🟢 None | None needed |

**Why this matters:** CNV can cause irreversible vision loss within days if missed. Automated triage improves efficiency while Grad-CAM explanations build physician trust and enable clinical validation.

---

## 📁 Project Structure

```
oct2/
├── app.py                          # Main Streamlit web application (634 lines)
├── requirements.txt                # Python dependencies (17 packages)
├── packages.txt                    # System packages for Linux deployment
├── .python-version                 # Python version (3.10.11)
├── .gitignore                      # Git ignore rules
├── .gitattributes                  # Git LFS configuration
│
├── .streamlit/                     # Streamlit configuration directory
│   └── config.toml                 # (optional) Custom Streamlit settings
│
├── models/                         # Model weights directory
│   └── baseline_model.pkl          # Trained ResNet50 FastAI model (100MB)
│                                   # Also hosted on Hugging Face: manosjung/oct-insight-model
│
├── notebooks/                      # Jupyter notebooks (development only)
│   ├── 01_data_exploration.py      # Dataset analysis and visualization
│   └── 02_baseline_model.py        # Model training script (FastAI)
│
├── data/                           # Training/test data (NOT in repo - too large)
│   └── raw/OCT2017/                # Kermany et al. OCT dataset (5.7 GB)
│       ├── train/                  # 83,484 training images
│       │   ├── CNV/                # 37,205 images
│       │   ├── DME/                # 11,348 images
│       │   ├── DRUSEN/             # 8,616 images
│       │   └── NORMAL/             # 26,315 images
│       └── test/                   # 1,000 test images (balanced)
│           ├── CNV/                # 250 images
│           ├── DME/                # 250 images
│           ├── DRUSEN/             # 250 images
│           └── NORMAL/             # 250 images
│
├── DEPLOYMENT_GUIDE.md             # Step-by-step deployment instructions
└── README.md                       # This file
```

### File Size Summary
- **Total Repository Size**: ~100 MB (model only)
- **Data Directory**: 5.7 GB (excluded from repo via .gitignore)
- **Main Application**: 23 KB (app.py)
- **Model Weights**: 100 MB (Git LFS or Hugging Face)

---

## 🚀 Quick Start

### Try the Live Demo
👉 **[Launch OCT-Insight](https://oct-insight.streamlit.app/)**

### Run Locally

#### Prerequisites
- Python 3.10 or higher
- 2 GB RAM minimum (CPU-only inference)
- Internet connection (for first-time model download)

#### Installation

```bash
# Clone repository
git clone https://github.com/manosjung/oct2.git
cd oct2

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open http://localhost:8501 and upload an OCT scan!

#### Download Test Images
Get sample OCT images from the [Kermany Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2) to test the application.

---

## 🏗️ Technical Architecture

### Model Architecture

```
ResNet50 (Transfer Learning)
├── Input Layer: 224×224×3 RGB tensor
├── ResNet50 Body (Pre-trained on ImageNet)
│   ├── Conv1: 7×7 conv, 64 filters
│   ├── Layer1: 3× Bottleneck blocks
│   ├── Layer2: 4× Bottleneck blocks
│   ├── Layer3: 6× Bottleneck blocks
│   └── Layer4: 3× Bottleneck blocks (target for Grad-CAM)
├── AdaptiveAvgPool2d: 7×7 → 1×1
├── Custom Head (FastAI)
│   ├── Flatten
│   ├── BatchNorm1d(2048)
│   ├── Dropout(0.5)
│   ├── Linear(2048 → 512)
│   ├── ReLU
│   ├── BatchNorm1d(512)
│   ├── Dropout(0.25)
│   └── Linear(512 → 4) [CNV, DME, DRUSEN, NORMAL]
└── Output: Softmax probabilities
```

**Key Design Decisions:**
- **Why ResNet50?** Proven performance on medical imaging, good balance of accuracy and speed
- **Why Transfer Learning?** Leverages ImageNet features (edges, textures) that generalize to OCT
- **Why CPU-Only?** Enables free deployment on Streamlit Cloud, faster for single-image inference
- **Why FastAI?** Simplified training with best practices (1cycle policy, data augmentation, mixed precision)

### Data Pipeline

#### 1. Image Preprocessing (app.py:241-303)
```python
Input: PIL Image (variable size, grayscale or RGB)
    ↓
1. Convert to RGB (if grayscale)
    ↓
2. Resize to 224×224 (BILINEAR interpolation)
    ↓
3. Convert to PyTorch tensor (manual, no numpy dependency)
   - Extract pixel values via img.getdata()
   - Flatten RGB tuples to list
   - Create torch.tensor(dtype=float32)
    ↓
4. Reshape to (C, H, W) format
   - View as (224, 224, 3)
   - Permute to (3, 224, 224)
    ↓
5. Normalize to [0, 1] (divide by 255.0)
    ↓
6. Apply ImageNet normalization
   - Mean: [0.485, 0.456, 0.406]
   - Std:  [0.229, 0.224, 0.225]
    ↓
7. Add batch dimension → (1, 3, 224, 224)
    ↓
Output: Ready for model inference
```

**Why Manual Tensor Conversion?**
The app uses a custom preprocessing function instead of FastAI's default pipeline to avoid dependency issues in deployment. This ensures:
- No dependency on FastAI's transform pipeline (which can be fragile in production)
- Pure PyTorch operations (reliable, predictable)
- Easier debugging and troubleshooting

#### 2. Model Inference (app.py:287-303)
```python
with torch.no_grad():
    output = learner.model(img_tensor)      # Raw logits
    probs = F.softmax(output, dim=1)[0]     # Convert to probabilities
    pred_idx = torch.argmax(probs).item()   # Get predicted class index
    pred_class = vocab[pred_idx]            # Map to class name
```

### Grad-CAM Implementation (app.py:306-395)

**Gradient-weighted Class Activation Mapping** visualizes which regions of the OCT scan influenced the model's decision.

#### Algorithm
```
1. Forward pass: Get prediction for target class
2. Backward pass: Compute gradients of class score w.r.t. last conv layer (layer4)
3. Global average pooling: Weight each feature map by gradient importance
4. Weighted combination: Sum weighted feature maps
5. ReLU activation: Keep only positive contributions
6. Upsample: Resize heatmap to input image size (224×224)
7. Overlay: Blend heatmap with original image
```

#### Implementation Details
- **Target Layer**: `layer4[-1]` (final convolutional block of ResNet50)
  - Has 2048 feature maps at 7×7 spatial resolution
  - Captures high-level semantic features (lesions, fluid, structures)
- **Library**: `grad-cam` package (PyPI), imported as `pytorch_grad_cam`
- **Fallback Strategy**: If `layer4` not found (due to model wrapping), searches for:
  1. Direct `layer4` attribute
  2. Sequential wrapper `model[0].layer4`
  3. Any module named `layer4` via `named_modules()`
  4. Last `Conv2d` layer as final fallback

#### Error Handling
```python
try:
    gradcam_img = generate_gradcam(learn, img_tensor, pred)
    st.image(gradcam_img, caption="AI Focus Areas")
except Exception as e:
    st.error(f"Could not generate heatmap: {e}")
    st.info("The diagnosis above is still valid.")
```

Even if Grad-CAM fails (rare), the prediction remains available.

---

## 🛠️ Technology Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | 2.1.0 | PyTorch deep learning framework (CPU-only build) |
| **torchvision** | 0.16.0 | Image transformations and pre-trained models |
| **fastai** | 2.7.14 | High-level training API, transfer learning utilities |
| **streamlit** | latest | Interactive web UI framework |
| **numpy** | <2,≥1.24.0 | Numerical operations (heatmap visualization) - Must be 1.x for PyTorch 2.1 compatibility |
| **opencv-python-headless** | latest | Image processing (heatmap overlay) |
| **grad-cam** | ≥1.4.8 | Grad-CAM implementation (explainability) |
| **plotly** | latest | Interactive probability charts |
| **huggingface-hub** | latest | Model downloading from Hugging Face |
| **pandas** | latest | Data manipulation (training phase) |
| **matplotlib** | latest | Plotting (training phase) |
| **scikit-learn** | latest | Metrics (training phase) |

### System Dependencies (packages.txt)
```
libgl1-mesa-glx       # OpenGL for OpenCV
libglib2.0-0          # GLib libraries
```

### Development Tools
- **Git LFS**: For versioning large model files
- **Jupyter**: For notebooks (training, analysis)
- **Python 3.10.11**: Stable, compatible with all dependencies

---

## 🧪 Model Training Details

### Dataset: Kermany et al. OCT (2018)

**Source**: [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2)

**Statistics**:
- **Total Images**: 84,495 retinal OCT B-scans
- **Image Format**: JPEG (grayscale, variable dimensions ~500×400 avg)
- **Acquisition**: Heidelberg Spectralis OCT, multiple clinical sites
- **Annotations**: Physician-labeled by experienced ophthalmologists

**Class Distribution (Training Set)**:
```
CNV:     37,205 images (44.5%) ████████████████████████
DME:     11,348 images (13.6%) ███████
DRUSEN:   8,616 images (10.3%) █████
NORMAL:  26,315 images (31.5%) ████████████████
```

**Imbalance Handling**:
- FastAI automatically applies class weights
- Data augmentation increases effective dataset size
- Evaluation uses balanced test set (250 images per class)

### Training Configuration (notebooks/02_baseline_model.py)

```python
# Data Augmentation
tfms = aug_transforms(
    mult=2.0,              # Random zoom up to 2x
    do_flip=True,          # Horizontal flip (50% chance)
    flip_vert=True,        # Vertical flip (retina orientation invariant)
    max_rotate=10.0,       # ±10° rotation
    max_lighting=0.2,      # Brightness/contrast variation
    max_warp=0.2,          # Perspective distortion
    p_affine=0.75,         # 75% chance of affine transforms
    p_lighting=0.75        # 75% chance of lighting transforms
)

# DataLoader
dls = ImageDataLoaders.from_folder(
    'data/raw/OCT2017/train',
    valid_pct=0.2,         # 20% validation split
    item_tfms=Resize(224), # Resize all images to 224×224
    batch_tfms=tfms,       # Apply augmentation
    bs=64                  # Batch size (adjust based on RAM)
)

# Model
learner = vision_learner(
    dls,
    resnet50,              # Pre-trained ResNet50 architecture
    metrics=[accuracy, error_rate],
    loss_func=CrossEntropyLoss()
)

# Training Strategy (1cycle policy)
learner.fine_tune(
    epochs=10,             # Fine-tune for 10 epochs
    base_lr=1e-3,          # Learning rate for backbone
    freeze_epochs=3        # Freeze backbone for first 3 epochs
)
```

### Performance Metrics (Expected)

Based on similar OCT classification studies with ResNet50:

| Metric | Target | Clinical Significance |
|--------|--------|----------------------|
| **Overall Accuracy** | >92% | General model quality |
| **CNV Sensitivity** | >95% | Critical for urgent triage |
| **CNV Specificity** | >90% | Avoid false alarms |
| **Normal Specificity** | >95% | Don't miss pathology |
| **DME F1-Score** | >88% | Balanced detection |
| **Drusen F1-Score** | >85% | Challenging class |

**Evaluation Methods**:
- Confusion matrix analysis
- Per-class ROC-AUC curves
- Precision-Recall curves (class imbalance)
- Sensitivity analysis for clinical thresholds

---

## 🔍 Code Documentation

### Key Functions in app.py

#### `load_model()` (Lines 150-237)
```python
@st.cache_resource
def load_model() -> Learner
```
**Purpose**: Loads the FastAI model, handling missing modules and deployment quirks.

**Features**:
- Downloads from Hugging Face if `models/baseline_model.pkl` missing
- Creates stub modules for `fasttransform` (deployment compatibility)
- Forces CPU mode (`defaults.device = torch.device('cpu')`)
- Handles missing vocabulary gracefully (hardcoded fallback)

**Returns**: FastAI `Learner` object with loaded ResNet50 model

---

#### `predict_image(learner, img)` (Lines 241-303)
```python
def predict_image(learner: Learner, img: PIL.Image) -> Tuple[str, int, torch.Tensor]
```
**Purpose**: Custom prediction function bypassing FastAI's transform pipeline.

**Args**:
- `learner`: FastAI Learner object
- `img`: PIL Image (any size, RGB or grayscale)

**Returns**:
- `pred_class` (str): Predicted class name (e.g., "CNV")
- `pred_idx` (int): Class index (0-3)
- `probs` (torch.Tensor): Softmax probabilities for all 4 classes

**Why Not Use `learner.predict()`?**
FastAI's default prediction relies on transform pipelines that can break in deployment due to pickle versioning issues. This function:
- Uses pure PyTorch operations (stable)
- Manually applies ImageNet normalization (explicit, verifiable)
- Avoids FastAI's internal state dependencies

---

#### `generate_gradcam(model, img_tensor, pred_class)` (Lines 306-395)
```python
def generate_gradcam(model: Learner, img_tensor: torch.Tensor, pred_class: str) -> np.ndarray
```
**Purpose**: Generates Grad-CAM heatmap for the predicted class.

**Args**:
- `model`: FastAI Learner with ResNet50
- `img_tensor`: Preprocessed image tensor (3, 224, 224)
- `pred_class`: Target class for visualization

**Returns**: NumPy array (224, 224, 3) with heatmap overlay

**Algorithm**:
1. Locate `layer4` (last conv layer) using fallback strategy
2. Create `GradCAM` object with target layer
3. Get class index from vocabulary
4. Generate Grad-CAM heatmap (0-1 normalized)
5. Denormalize input image (reverse ImageNet norm)
6. Overlay heatmap using `show_cam_on_image()`

**Error Handling**: Raises `ValueError` if no suitable conv layer found (extremely rare)

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Could not generate heatmap: Numpy is not available" / "_ARRAY_API not found"

**Cause**: NumPy 2.x compatibility issue with PyTorch 2.1.0

**Error Message**:
```
Failed to initialize NumPy: _ARRAY_API not found
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**Root Cause**:
- PyTorch 2.1.0 was compiled with NumPy 1.x
- NumPy 2.x has breaking C API changes
- Grad-CAM internally calls `.numpy()` which fails with NumPy 2.x

**Fix**:
```diff
# requirements.txt
- numpy>=1.24.0
+ numpy<2,>=1.24.0
+ grad-cam>=1.4.8
```

**Note**:
- The PyPI package name is `grad-cam` (not `pytorch-grad-cam`)
- Must force NumPy 1.x (e.g., 1.26.4) for PyTorch 2.1 compatibility
- This is a common issue in the PyTorch ecosystem

**Status**: ✅ Fixed in current version

---

#### 2. "Model loading failed: No module named 'fasttransform'"

**Cause**: FastAI model was pickled with custom transforms that aren't available in deployment

**Fix**: Already handled in `load_model()` (lines 173-207) via stub modules

**If still occurs**:
```python
import sys, types
fasttransform_module = types.ModuleType('fasttransform')
sys.modules['fasttransform'] = fasttransform_module
```

---

#### 3. Low Prediction Confidence (<70%)

**Possible Causes**:
1. **Model Quality**: Undertrained, poor hyperparameters
2. **Image Quality**: Low resolution, artifacts, incorrect imaging modality
3. **Preprocessing Mismatch**: Training vs. inference preprocessing differs
4. **Domain Shift**: Test images from different OCT device/protocol

**Debugging Steps**:
1. Check if training images match test images (resolution, contrast)
2. Verify ImageNet normalization is consistent
3. Review training logs in `notebooks/02_baseline_model.py`
4. Test with known-good images from Kermany dataset

**Improving Confidence**:
- Train for more epochs (10 → 20)
- Use learning rate finder (`learner.lr_find()`)
- Add temperature scaling (post-training calibration)
- Ensemble multiple models

---

#### 4. Grad-CAM Shows Uniform Heatmap (No Focus)

**Cause**: Targeting wrong layer or model is not confident

**Check**:
```python
# Verify target layer has spatial dimensions
for name, module in model.model.named_modules():
    if 'layer4' in name:
        print(f"{name}: {module}")
```

**Expected**: Layer4 output should be (batch, 2048, 7, 7)

---

#### 5. Streamlit Deployment Fails with Memory Error

**Cause**: Streamlit Cloud free tier has 1 GB RAM limit

**Solutions**:
1. **Use CPU-only PyTorch** (already configured via `--extra-index-url`)
2. **Reduce batch size** in inference (currently 1, already minimal)
3. **Model quantization** (advanced):
   ```python
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```
4. **Upgrade to Streamlit Teams** (paid, more resources)

---

## 🔐 Security & Privacy

### Important Considerations

#### For Research/Portfolio Use:
- ✅ No patient data collection
- ✅ Images processed in-memory only
- ✅ No server-side storage
- ✅ Safe for demonstration purposes

#### For Clinical Deployment (NOT current status):
- ❌ **NOT HIPAA-compliant** (no encryption, audit logs, BAA)
- ❌ **NOT FDA/CE approved** for clinical diagnosis
- ❌ **NOT validated** on independent clinical datasets
- ❌ **NO liability coverage** for medical decisions

**If you adapt this for clinical use:**
1. Implement end-to-end encryption (TLS 1.3+)
2. Add user authentication (RBAC)
3. Log all predictions with timestamps (audit trail)
4. Validate on multi-center datasets
5. Obtain regulatory approval (FDA 510(k), CE mark)
6. Get medical malpractice insurance
7. Implement HIPAA compliance (if US-based)

---

## ⚠️ Medical Disclaimer

**THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

- ❌ NOT a medical device
- ❌ NOT intended for clinical diagnosis or treatment
- ❌ NOT a substitute for professional medical advice
- ❌ NOT validated for real-world clinical use

**Always consult qualified ophthalmologists for:**
- Medical diagnosis and treatment planning
- Interpretation of OCT scans
- Patient care decisions

**Developers and users assume all responsibility for appropriate use.**

---

## 📚 References & Citations

### Dataset
```bibtex
@data{rscbjbr9sj-2,
  author    = {Kermany, Daniel and Zhang, Kang and Goldbaum, Michael},
  title     = {Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification},
  year      = {2018},
  publisher = {Mendeley Data},
  version   = {V2},
  doi       = {10.17632/rscbjbr9sj.2},
  url       = {https://data.mendeley.com/datasets/rscbjbr9sj/2}
}
```

### Related Publications
1. Kermany et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning". *Cell*, 172(5), 1122-1131.
2. Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". *ICCV*.
3. He et al. (2016). "Deep Residual Learning for Image Recognition". *CVPR*.

### This Project
```bibtex
@software{oct_insight_2025,
  author = {Faruk Orman},
  title  = {OCT-Insight: Explainable AI for Retinal Disease Classification},
  year   = {2025},
  url    = {https://github.com/manosjung/oct2}
}
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

### Areas for Improvement
- [ ] **Model optimization**: EfficientNet, Vision Transformers (ViT), Swin Transformer
- [ ] **Multi-modal fusion**: Combine OCT + fundus photos + patient metadata
- [ ] **Uncertainty quantification**: Bayesian neural networks, Monte Carlo dropout
- [ ] **Clinical validation**: Test on independent datasets (different OCT devices)
- [ ] **Real-time inference**: ONNX export, TensorRT optimization
- [ ] **Active learning**: Iterative model improvement with expert feedback
- [ ] **Segmentation**: Retinal layer segmentation before classification
- [ ] **3D analysis**: Process full OCT volumes (currently single B-scans)

### Development Setup
```bash
git clone https://github.com/manosjung/oct2.git
cd oct2
pip install -r requirements.txt
pre-commit install  # (if using pre-commit hooks)
```

---

## 📧 Contact & Support

**Project Maintainer**: Faruk Orman

- 🔗 **LinkedIn**: [linkedin.com/in/farukorman](https://www.linkedin.com/in/farukorman/)
- 🐙 **GitHub**: [@manosjung](https://github.com/manosjung)

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/manosjung/oct2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/manosjung/oct2/discussions)

---

## 📄 License

**MIT License** - see [LICENSE](LICENSE) file for details.

**Summary**:
- ✅ Free for research and educational use
- ✅ Free for portfolio/demonstration purposes
- ⚠️ Commercial clinical use requires approval and regulatory clearance
- ⚠️ Derivative works must cite original dataset (Kermany et al.)

**No Warranty**: This software is provided "as-is" without any warranty. See LICENSE for full terms.

---

## 🙏 Acknowledgments

- **Kermany, Zhang, Goldbaum** for the OCT dataset and pioneering work in medical image AI
- **FastAI team** (Jeremy Howard, Sylvain Gugger) for democratizing deep learning
- **Streamlit** for making ML deployment accessible
- **PyTorch team** for the deep learning framework
- **Grad-CAM authors** (Selvaraju et al.) for explainability methodology
- **Open-source community** for tools, libraries, and shared knowledge

---

## 🌟 Star This Project

If you found this project helpful for:
- Learning medical AI development
- Building your ML portfolio
- Understanding Grad-CAM explainability
- Deploying deep learning models

**Please consider giving it a ⭐ on GitHub!**

Stars help others discover the project and motivate continued development.

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/manosjung/oct2?style=social)
![GitHub forks](https://img.shields.io/github/forks/manosjung/oct2?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/manosjung/oct2?style=social)

**Last Updated**: January 2025
**Version**: 1.0.0
**Status**: Active Development

---

**Built by Faruk Orman**

*Exploring the intersection of medical imaging and deep learning*
