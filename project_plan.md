# OCT-Insight: Clinical Research Prototype
## Explainable Deep Learning for Multi-Class Retinal Disease Classification

---

## 📋 Executive Summary

**Project Name:** OCT-Insight (aligned with UKSH iAuge initiative)
**Domain:** Translational Ophthalmology Research
**Objective:** Build a clinical research prototype for multi-class retinal disease classification from OCT scans using Deep Learning with Explainable AI (XAI)

**Target Audience:** Prof. Roider (UKSH) and medical research community

**Core Value Proposition:**
- **4-Class Classification:** CNV (Wet AMD), DME, Drusen (Dry AMD), Normal
- **Clinical Explainability:** Grad-CAM heatmaps showing biomarker localization
- **Research Prototype:** Demonstrates clinical understanding + AI competency
- **Multimodal Foundation:** Architecture ready for OCT-Fundus registration (iAuge alignment)

---

## 🎯 Project Objectives

### Primary Goals (Clinical Research Focus)
1. **4-Class Classification:** Distinguish between CNV (Wet AMD), DME, Drusen (Dry AMD), and Normal
   - **Why:** CNV requires immediate anti-VEGF injections (urgent), Drusen does not (monitoring)
   - **Clinical Impact:** Triage patients by urgency level

2. **Clinical Explainability:** Grad-CAM visualization of pathological features
   - **CNV:** Subretinal fluid, intraretinal cysts
   - **Drusen:** Sub-RPE deposits
   - **DME:** Intraretinal fluid, cystoid spaces

3. **Research Prototype:** Demonstrate medical + AI competency to UKSH
   - **Not for clinical deployment** (no regulatory approval)
   - **Purpose:** Show understanding of ophthalmology + deep learning

4. **iAuge Alignment:** Foundation for multimodal OCT-Fundus registration
   - **Current:** OCT classification only
   - **Future-Ready:** Architecture supports adding Fundus imaging

### Success Metrics (Doctor-Focused, Not Engineer-Focused)
- **Overall Accuracy:** Target >92% on test set (4-class problem is harder than binary)
- **CNV Sensitivity:** >95% (CRITICAL - missing Wet AMD causes irreversible blindness)
- **Specificity for Normal:** >90% (avoid unnecessary referrals)
- **AUC-ROC per class:** >0.90 for all classes
- **Clinical Validity:** Grad-CAM heatmaps highlight anatomically correct regions
  - Example: CNV heatmaps should show outer retina/sub-RPE, not vitreous
- **Demo Quality:** Streamlit app loads in <5 seconds, predictions in <3 seconds

---

## 🏗️ Technical Architecture (Simplified Research Prototype)

### System Design: Monolithic Streamlit Application

```
┌─────────────────────────────────────────────────────────────┐
│            STREAMLIT WEB APPLICATION                         │
│                  (All-in-One Interface)                      │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  USER INTERFACE LAYER                               │    │
│  │  - Image Upload (OCT)                               │    │
│  │  - Fundus Upload Button (Placeholder - Coming Soon) │    │
│  │  - Prediction Display with Confidence Scores        │    │
│  │  - Grad-CAM Heatmap Visualization                   │    │
│  │  - Clinical Report Generation                       │    │
│  └──────────────────┬──────────────────────────────────┘    │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐    │
│  │  INFERENCE LOGIC (Python Functions)                 │    │
│  │  - preprocess_image()                               │    │
│  │  - predict_disease()                                │    │
│  │  - generate_gradcam()                               │    │
│  │  - create_clinical_report()                         │    │
│  └──────────────────┬──────────────────────────────────┘    │
│                     │                                        │
│       ┌─────────────┴──────────────┐                        │
│       │                            │                        │
│  ┌────▼─────────┐       ┌──────────▼──────────┐            │
│  │ DEEP LEARNING│       │  GRAD-CAM MODULE    │            │
│  │   MODEL      │       │  (Explainability)   │            │
│  │              │       │                     │            │
│  │ - ResNet50   │       │ - Heatmap Generator │            │
│  │ - 4 Classes  │       │ - OCT Overlay       │            │
│  │ - Pre-trained│       │ - Biomarker Viz     │            │
│  └──────────────┘       └─────────────────────┘            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│         OFFLINE TRAINING PIPELINE (Jupyter Notebooks)        │
│  - Data Exploration & Preprocessing                         │
│  - Model Training (PyTorch/FastAI)                          │
│  - Evaluation & Metrics                                     │
│  - Export trained model → models/best_model.pth             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              DATA (Local Folder Structure)                   │
│  - Kermany Dataset: CNV / DME / DRUSEN / NORMAL             │
│  - Train/Val/Test splits (simple CSV files)                 │
└─────────────────────────────────────────────────────────────┘
```

**Why This Architecture?**
- ✅ **Fast to build:** 1 week vs 1 month for microservices
- ✅ **Easy to demo:** Just run `streamlit run app.py`
- ✅ **Suitable for research:** Professors care about results, not DevOps
- ✅ **Deployment:** Can share via Streamlit Cloud (free hosting)

---

## 📁 Folder Structure (Simplified for Research Prototype)

```
oct/                                   # Project root
│
├── README.md                          # Project overview (keep it simple)
├── requirements.txt                   # Python dependencies ONLY
├── .gitignore                         # Git ignore (data/, models/, .ipynb_checkpoints)
│
├── data/                              # Data directory (NOT in git - add to .gitignore)
│   ├── raw/                          # Kermany dataset (downloaded)
│   │   ├── CNV/                      # Wet AMD (Choroidal Neovascularization)
│   │   ├── DME/                      # Diabetic Macular Edema
│   │   ├── DRUSEN/                   # Dry AMD (Drusen deposits)
│   │   └── NORMAL/                   # Normal retina
│   │
│   └── splits/                       # Train/Val/Test split metadata
│       ├── train.csv                 # List of training images + labels
│       ├── val.csv                   # Validation set
│       └── test.csv                  # Test set
│
├── notebooks/                         # Jupyter notebooks (YOUR MAIN WORKSPACE)
│   ├── 01_data_exploration.ipynb     # EDA - understand the dataset
│   ├── 02_baseline_model.ipynb       # Train first ResNet50 model
│   ├── 03_model_optimization.ipynb   # Improve model (augmentation, tuning)
│   ├── 04_evaluation.ipynb           # Metrics, confusion matrix, per-class analysis
│   └── 05_gradcam_analysis.ipynb     # Generate and validate Grad-CAM heatmaps
│
├── src/                               # Reusable Python modules
│   ├── __init__.py
│   ├── dataset.py                    # PyTorch Dataset for OCT images
│   ├── model.py                      # Model definition (ResNet50 wrapper)
│   ├── train.py                      # Training functions
│   ├── gradcam.py                    # Grad-CAM implementation
│   └── utils.py                      # Helper functions (metrics, plotting)
│
├── app.py                             # ⭐ STREAMLIT APPLICATION (Main deliverable)
│                                      # All-in-one web interface
│
├── models/                            # Saved models (NOT in git)
│   └── best_resnet50_4class.pth      # Your trained model weights
│
└── assets/                            # Static files for Streamlit app
    ├── uksh_logo.png                 # UKSH/iAuge branding (optional)
    └── example_oct.jpeg              # Sample image for demo
```

**What We CUT (compared to original plan):**
- ❌ `api/` folder (no separate FastAPI backend)
- ❌ `deployment/` folder (no Docker, Kubernetes, docker-compose)
- ❌ `tests/` folder (skip unit tests for prototype - focus on results)
- ❌ `docs/` folder (documentation goes in README.md + code comments)
- ❌ `experiments/` folder (no MLflow/wandb - just save metrics in notebooks)
- ❌ `configs/` folder (hardcode settings or use simple Python dict)

**What We KEEP:**
- ✅ `notebooks/` - This is where you'll spend 80% of your time
- ✅ `src/` - Clean, reusable code for Streamlit app
- ✅ `app.py` - The single file that demonstrates everything
- ✅ `data/` - Well-organized dataset

---

## 🔧 Technologies & Techniques

### Deep Learning Framework
- **PyTorch** (v2.0+) - Core DL framework
  - Reason: Flexibility, research-friendly, excellent debugging
- **FastAI** (v2.7+) - High-level training API
  - Reason: Built on PyTorch, medical imaging utilities, progressive resizing

### Model Architecture: ResNet50 for 4-Class Classification

```python
# Modified ResNet50 for Retinal Disease Classification
# Input: OCT scan (224x224 or 512x512, grayscale converted to 3-channel)
# Output: 4 classes - [CNV, DME, DRUSEN, NORMAL]

# Architecture:
# 1. ResNet50 backbone (pre-trained on ImageNet)
# 2. Replace final FC layer: 2048 -> 4 (instead of 2048 -> 1000)
# 3. Softmax activation for multi-class probabilities

# Why ResNet50?
# ✅ Strong baseline (proven on medical images)
# ✅ Transfer learning from ImageNet (natural images → retinal images)
# ✅ Residual connections prevent vanishing gradients
# ✅ Fast training (can train on CPU, better with GPU)

# Alternative (if ResNet50 doesn't work well):
# - EfficientNet-B4 (more parameters, potentially better accuracy)
# - We'll start simple and only try this if needed
```

### Explainable AI (XAI)

#### Grad-CAM (Gradient-weighted Class Activation Mapping)
```python
# Visualization technique showing "where the model looks"
# Steps:
# 1. Forward pass through model
# 2. Compute gradients of target class w.r.t. final conv layer
# 3. Weight feature maps by gradients
# 4. Generate heatmap highlighting important regions
# 5. Overlay on original OCT scan
```

### Data Processing Techniques

#### Medical-Grade Augmentations
```python
# Conservative augmentations to preserve medical validity:
# - Horizontal flips (anatomically valid)
# - Small rotations (±10°)
# - Brightness/contrast adjustments
# - Gaussian noise (simulating acquisition variance)
# - NO vertical flips (not anatomically valid for retina)
# - NO aggressive crops (preserve context)
```

#### Preprocessing Pipeline
```python
# 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
#    - Enhance OCT contrast
# 2. Resize to 224x224 or 512x512
# 3. Normalize using ImageNet statistics (for transfer learning)
# 4. Convert to tensor
```

### Training Techniques

#### Transfer Learning
```python
# Use pre-trained weights from ImageNet
# Fine-tune on OCT data
# Reason: Limited medical data, leverage learned features
```

#### Progressive Resizing
```python
# Start: Train at 128x128 (fast iterations)
# Middle: Train at 224x224 (standard)
# Final: Fine-tune at 512x512 (high detail)
# Reason: Faster convergence, better generalization
```

#### Learning Rate Schedule
```python
# 1. Learning Rate Finder (FastAI feature)
# 2. One Cycle Policy (1cycle scheduler)
#    - Gradually increase LR, then decrease
#    - Improves generalization
```

#### Class Imbalance Handling
```python
# Technique 1: Weighted Loss (if AMD/Normal imbalanced)
# Technique 2: Focal Loss (focus on hard examples)
# Technique 3: Oversampling minority class
```

### Evaluation Metrics

```python
# Primary Metrics:
# - Accuracy: Overall correct predictions
# - Sensitivity (Recall): True Positive Rate - CRITICAL (don't miss AMD)
# - Specificity: True Negative Rate
# - AUC-ROC: Area under ROC curve
# - F1-Score: Harmonic mean of precision/recall
# - Confusion Matrix: Visual error analysis
```

### Experiment Tracking
- **MLflow** - Track experiments, parameters, metrics
- **Weights & Biases (wandb)** - Alternative with better visualizations

### API & Deployment
- **FastAPI** - High-performance API framework
- **Pydantic** - Data validation
- **Docker** - Containerization
- **ONNX** - Model export for cross-platform inference

### Frontend
- **Streamlit** - Quick prototyping, Python-native
- **Gradio** - Alternative for ML demos

---

## 📊 Implementation Roadmap (Accelerated for Research Prototype)

### WEEK 1: Data + Baseline Model
**Goal:** Get a working model that makes predictions

**Day 1-2: Setup + Data**
- [ ] Install Python, PyTorch, FastAI, Streamlit
- [ ] Download Kermany OCT dataset (84K images, ~5GB)
- [ ] Create folder structure (data/, notebooks/, src/, models/)
- [ ] Write `data/splits/train.csv` with image paths + labels

**Day 3-5: First Model**
- [ ] Notebook 01: Data exploration (look at images, class distribution)
- [ ] Notebook 02: Train ResNet50 baseline
  - Use FastAI `vision_learner()` for quick setup
  - Train for 5-10 epochs
  - Save `models/baseline.pth`
- [ ] Evaluate: accuracy, confusion matrix
- [ ] **Milestone:** Model predicts 4 classes with >80% accuracy

**Day 6-7: Optimization**
- [ ] Notebook 03: Improve model
  - Add augmentations (horizontal flip, rotation, brightness)
  - Train longer (20-30 epochs)
  - Learning rate finder + 1cycle policy
- [ ] **Target:** >90% accuracy

---

### WEEK 2: Explainability + Streamlit App
**Goal:** Build the demo application

**Day 8-10: Grad-CAM Implementation**
- [ ] Notebook 05: Implement Grad-CAM
  - Use `pytorch-grad-cam` library (don't code from scratch)
  - Generate heatmaps for 20-30 validation images
  - Verify heatmaps highlight correct anatomical regions
- [ ] Save example heatmaps in `assets/` for README

**Day 11-14: Streamlit Application**
- [ ] Create `app.py` with these sections:
  1. Title + Project description
  2. Image uploader (OCT)
  3. "Predict" button
  4. Display: Class prediction + confidence scores
  5. Display: Grad-CAM heatmap overlay
  6. **Bonus:** "Upload Fundus Image (Coming Soon)" placeholder button
- [ ] Test locally: `streamlit run app.py`
- [ ] **Milestone:** Working end-to-end demo

---

### WEEK 3: Polish + Documentation
**Goal:** Make it presentation-ready for Prof. Roider

**Day 15-17: Clinical Validation**
- [ ] Notebook 04: Detailed evaluation
  - Per-class metrics (sensitivity, specificity for each disease)
  - ROC curves for all 4 classes
  - Error analysis: Which images does the model misclassify?
- [ ] Validate Grad-CAM: Do heatmaps make clinical sense?
  - CNV → Should highlight subretinal fluid
  - Drusen → Should highlight sub-RPE deposits

**Day 18-21: Documentation + Deployment**
- [ ] Write README.md:
  - Project summary (link to UKSH iAuge)
  - Medical background (CNV vs Drusen - why it matters)
  - Model performance (include confusion matrix image)
  - Instructions to run locally
- [ ] Deploy to Streamlit Cloud (free, public URL)
- [ ] **Optional:** Record 2-minute video demo
- [ ] **Deliverable:** GitHub repo + Live demo link

---

### Post-Week 3: Contact Prof. Roider
With these deliverables:
1. **GitHub Repository:** Clean code, documented notebooks
2. **Live Demo:** Streamlit Cloud link (anyone can test)
3. **Email to Prof. Roider:**
   - "Dear Prof. Roider, I developed an OCT classification prototype aligned with your iAuge project..."
   - Include demo link
   - Mention: "4-class model with Grad-CAM, foundation for multimodal OCT-Fundus work"

---

## 🧪 Code Explanation Philosophy

### Every Line Will Be Documented

For this project, we will follow **extreme documentation** standards:

```python
# Example of our documentation level:

# Import PyTorch library for deep learning operations
import torch

# Import neural network module from PyTorch
# nn contains building blocks for neural networks (layers, loss functions)
import torch.nn as nn

# Import functional API from PyTorch
# F provides functions for operations like ReLU, softmax, etc.
import torch.nn.functional as F

class OCTClassifier(nn.Module):
    """
    OCT Classification Model for AMD Detection

    Architecture: ResNet50-based binary classifier
    Input: OCT scan image (3 channels, 224x224 pixels)
    Output: Probability of AMD (0 = Normal, 1 = AMD)

    This class inherits from nn.Module, which is the base class
    for all neural network modules in PyTorch. By inheriting,
    we get access to parameter tracking, GPU support, and
    training/evaluation modes.
    """

    def __init__(self, pretrained=True):
        """
        Initialize the OCT Classifier model

        Args:
            pretrained (bool): If True, load ImageNet pre-trained weights
                              This helps with transfer learning by starting
                              with learned features instead of random weights
        """
        # Call the parent class (nn.Module) constructor
        # This is required for PyTorch to properly initialize the model
        super(OCTClassifier, self).__init__()

        # Load ResNet50 architecture from torchvision models
        # ResNet50 has 50 layers with residual connections
        # pretrained=True downloads and loads ImageNet weights (~100MB)
        self.backbone = models.resnet50(pretrained=pretrained)

        # Get the number of input features to the final layer
        # ResNet50's final fully connected layer expects 2048 features
        # We save this to know how to build our custom classifier
        num_features = self.backbone.fc.in_features

        # Replace the final fully connected layer
        # Original: 2048 -> 1000 (ImageNet classes)
        # New: 2048 -> 1 (AMD probability)
        # nn.Linear creates a fully connected layer: y = xW^T + b
        self.backbone.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        """
        Forward pass: defines how data flows through the network

        Args:
            x (torch.Tensor): Input batch of OCT images
                             Shape: (batch_size, 3, 224, 224)
                             - batch_size: number of images processed together
                             - 3: RGB channels (or grayscale replicated 3x)
                             - 224x224: image dimensions in pixels

        Returns:
            torch.Tensor: AMD probability for each image
                         Shape: (batch_size, 1)
                         Values: between 0 and 1 after sigmoid activation
        """
        # Pass input through ResNet50 backbone
        # This applies convolutions, batch norms, ReLU, pooling
        # Output shape: (batch_size, 1) - raw logits (unbounded values)
        logits = self.backbone(x)

        # Apply sigmoid activation function
        # Sigmoid: σ(x) = 1 / (1 + e^(-x))
        # Converts logits to probabilities [0, 1]
        # Values close to 1 = high AMD probability
        # Values close to 0 = low AMD probability (Normal)
        probabilities = torch.sigmoid(logits)

        # Return the probability predictions
        return probabilities
```

**Documentation Standards:**
1. **Line-by-line comments** for complex logic
2. **Docstrings** for all functions/classes (Google style)
3. **Inline explanations** of mathematical operations
4. **Architecture diagrams** in comments
5. **Medical context** where relevant

---

## 🎓 Learning Objectives

As we build this project, you will understand:

### Deep Learning Fundamentals
- How neural networks learn from data
- Backpropagation and gradient descent
- Loss functions and optimization
- Transfer learning principles

### Medical AI Specifics
- Challenges of medical imaging
- Importance of explainability in healthcare
- Handling sensitive patient data
- Clinical validation requirements

### Production ML Engineering
- Model training pipelines
- API development for ML models
- Containerization and deployment
- Monitoring and logging

### PyTorch Ecosystem
- Tensors and automatic differentiation
- Building custom datasets
- Training loops and callbacks
- Model saving and loading

---

## 🔒 Important Considerations

### Medical Ethics & Compliance
- **Not FDA-approved:** This is a prototype for educational purposes
- **Clinical validation required:** Needs prospective clinical trials
- **Supervision required:** Not for autonomous diagnosis
- **Data privacy:** HIPAA/GDPR compliance if using real patient data

### Data Sources
- **Publicly available datasets:**
  - Kermany OCT Dataset (Kaggle)
  - Duke OCT Dataset
  - UCSD OCT Dataset
- **Synthetic data:** For initial development

### Regulatory Considerations
- Label as "Research Prototype"
- Include disclaimers
- Document all validation steps
- Maintain audit trails

---

## 📚 Resources & References

### Academic Papers
1. **Grad-CAM:** Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks"
2. **Medical AI:** De Fauw et al. (2018) - "Clinically applicable deep learning for diagnosis and referral in retinal disease"
3. **OCT Classification:** Lee et al. (2017) - "Deep learning is effective for classifying normal versus age-related macular degeneration OCT images"

### Datasets
- **Kermany Dataset:** https://data.mendeley.com/datasets/rscbjbr9sj/2
- Contains 84,495 OCT images (AMD, DME, Drusen, Normal)

### Tools & Libraries
- PyTorch: https://pytorch.org/
- FastAI: https://docs.fast.ai/
- Grad-CAM PyTorch: https://github.com/jacobgil/pytorch-grad-cam

---

## 🚀 Next Steps

1. **Review this plan** and ask questions
2. **Set up development environment** (Python 3.10+, CUDA if GPU available)
3. **Download OCT dataset** (Kermany recommended)
4. **Create folder structure** as defined above
5. **Start with data exploration** notebook

---

## ✅ Success Criteria (Research Prototype)

This project will be considered successful when:
- ✅ **Model Performance:** >90% accuracy on 4-class test set
- ✅ **CNV Sensitivity:** >95% (critical - Wet AMD is urgent)
- ✅ **Grad-CAM Validity:** Heatmaps highlight anatomically correct regions
- ✅ **Working Demo:** Streamlit app runs locally + deployed to cloud
- ✅ **Code Quality:** Every line explained, clean notebooks
- ✅ **Documentation:** README with medical context + usage instructions
- ✅ **Deliverable:** GitHub repo presentable to Prof. Roider

**What SUCCESS looks like in 3 weeks:**
- Prof. Roider receives your email with a live demo link
- He clicks the link, uploads an OCT scan, sees the prediction + heatmap
- He thinks: "This candidate understands both ophthalmology AND deep learning"

---

## 🚨 Critical Reminders

### What NOT to Do (Avoid These Traps)
1. ❌ **Don't over-engineer:** No Docker, Kubernetes, or microservices
2. ❌ **Don't chase perfection:** 90% accuracy is enough for a prototype
3. ❌ **Don't get stuck on tooling:** Use FastAI's defaults, don't write custom training loops
4. ❌ **Don't ignore clinical context:** Always explain WHY (CNV vs Drusen matters)

### What TO Do (Focus Areas)
1. ✅ **Prioritize working code:** Messy code that works > beautiful code that doesn't
2. ✅ **Document everything:** Every line gets a comment (as shown in examples)
3. ✅ **Show clinical understanding:** Mention anti-VEGF injections, biomarkers
4. ✅ **Make it demo-ready:** The Streamlit app is your "resume"

---

## 📝 Summary: From Plan to Action

**You are building:**
A Streamlit web app that classifies OCT scans into 4 diseases (CNV, DME, Drusen, Normal) and explains predictions using Grad-CAM heatmaps.

**Why this matters:**
- Shows Prof. Roider you understand the iAuge project goals (multimodal retinal imaging)
- Demonstrates clinical knowledge (CNV urgency, biomarker localization)
- Proves technical competency (PyTorch, transfer learning, XAI)

**Timeline:**
- Week 1: Train the model (notebooks)
- Week 2: Build the app (Streamlit + Grad-CAM)
- Week 3: Polish + deploy + contact Prof. Roider

**Next Steps:**
1. Review this plan
2. Ask questions if anything is unclear
3. Start with folder structure creation
4. Jump into notebooks (don't overthink setup)

---

**Ready to build a clinical research prototype that gets you the job!**
