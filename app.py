import streamlit as st
import mimetypes
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

from fastai.vision.all import *
import pathlib
import plotly.express as px
import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- SETUP ---
# Fix for cross-platform Path compatibility (Windows <-> Linux)
import platform
if platform.system() == 'Windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Page Config
st.set_page_config(
    page_title="OCT-Insight | AI Diagnosis",
    page_icon="👁️",
    layout="centered"
)

# --- STYLES ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .confidence-high { color: #28a745; }
    .confidence-med { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    """Loads the FastAI model once and caches it."""
    path = Path('models/baseline_model.pkl')
    if not path.exists():
        st.error(f"Model not found at {path}. Please train the model first.")
        return None
    return load_learner(path)

learn = load_model()

# --- GRAD-CAM FUNCTION ---
def generate_gradcam(model, img_tensor, pred_class):
    """
    Generates Grad-CAM heatmap for the predicted class.

    Args:
        model: FastAI learner object
        img_tensor: Preprocessed image tensor
        pred_class: Predicted class name

    Returns:
        Heatmap overlaid on original image
    """
    # Get the underlying PyTorch model
    pytorch_model = model.model.eval()

    # Find the last convolutional layer in the ResNet50 architecture
    # FastAI wraps models, so we need to navigate the structure
    # The model structure is usually: Sequential(ResNet_body, custom_head)
    # We need to find the last conv layer in the ResNet body

    # Try different possible structures
    target_layers = None

    # Option 1: Direct access to layer4 (standard ResNet)
    if hasattr(pytorch_model, 'layer4'):
        target_layers = [pytorch_model.layer4[-1]]
    # Option 2: Sequential wrapper - access first element
    elif hasattr(pytorch_model, '__getitem__') and hasattr(pytorch_model[0], 'layer4'):
        target_layers = [pytorch_model[0].layer4[-1]]
    # Option 3: Look for any module named layer4
    else:
        for name, module in pytorch_model.named_modules():
            if 'layer4' in name and len(list(module.children())) == 0:
                # Found a leaf module with layer4 in its name
                target_layers = [module]
                break

        # If still not found, use the last convolutional layer we can find
        if target_layers is None:
            conv_layers = []
            for module in pytorch_model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    conv_layers.append(module)
            if conv_layers:
                target_layers = [conv_layers[-1]]

    if target_layers is None:
        raise ValueError("Could not find suitable target layer for Grad-CAM")

    # Create GradCAM object
    cam = GradCAM(model=pytorch_model, target_layers=target_layers)

    # Get class index from class name
    class_idx = model.dls.vocab.o2i[pred_class]
    targets = [ClassifierOutputTarget(class_idx)]

    # Prepare input tensor (add batch dimension if needed)
    input_tensor = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor

    # Generate Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension

    # Convert original image tensor to numpy for visualization
    # FastAI normalizes with ImageNet stats, need to denormalize
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    # Denormalize using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    # Create visualization
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    return visualization

# --- UI LAYOUT ---
st.title("👁️ OCT-Insight")
st.markdown("### Retinal Disease Classification Prototype")
st.markdown("Upload an Optical Coherence Tomography (OCT) scan to detect: **CNV, DME, Drusen, or Normal**.")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/1200px-React-icon.svg.png", width=50) # Placeholder logo
    st.header("About")
    st.info(
        """
        **Model:** ResNet50 (Transfer Learning)
        **Classes:** 4 (CNV, DME, Drusen, Normal)
        **Training Data:** Kermany Dataset (84k images)
        **Status:** Research Prototype
        """
    )
    st.warning("⚠️ For Research Use Only. Not for clinical diagnosis.")

# Main Area
uploaded_file = st.file_uploader("Choose an OCT Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption='Uploaded OCT Scan', use_container_width=True)
    
    with col2:
        if learn:
            with st.spinner("Analyzing retina..."):
                # Make Prediction
                img = PILImage.create(uploaded_file)
                pred, pred_idx, probs = learn.predict(img)
                
                # Get confidence score
                confidence = probs[pred_idx] * 100
                
                # Display Result
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Diagnosis</h3>
                    <h1 style="color: #0066cc;">{pred}</h1>
                    <p>Confidence: <b>{confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Bar Chart for all classes
                chart_data = {
                    "Condition": learn.dls.vocab,
                    "Probability": [p.item() for p in probs]
                }
                fig = px.bar(chart_data, x="Probability", y="Condition", orientation='h', 
                             title="Class Probabilities", text_auto='.2%',
                             color="Probability", color_continuous_scale="Blues")
                fig.update_layout(showlegend=False, height=250)
                st.plotly_chart(fig, use_container_width=True)

                # Grad-CAM Visualization
                with st.expander("🔍 AI Explanation (What the model looked at)", expanded=False):
                    st.markdown("**Heatmap**: Shows which areas of the scan the AI examined to make this diagnosis.")
                    st.markdown("🔴 **Red** = AI focused here | 🔵 **Blue** = Less important")

                    with st.spinner("Generating heatmap..."):
                        try:
                            # Get the tensor from the image for Grad-CAM
                            img_tensor = learn.dls.test_dl([img]).one_batch()[0][0]

                            # Generate Grad-CAM
                            gradcam_img = generate_gradcam(learn, img_tensor, pred)

                            # Display side by side
                            col_orig, col_grad = st.columns(2)
                            with col_orig:
                                st.image(uploaded_file, caption="Original Scan", use_container_width=True)
                            with col_grad:
                                st.image(gradcam_img, caption=f"AI Focus Areas: {pred}", use_container_width=True)

                            # Simplified explanation based on prediction
                            if pred == "NORMAL":
                                st.success("✅ **For Normal scans**: The AI checked these areas to confirm no abnormalities are present.")
                            elif pred == "CNV":
                                st.warning("⚠️ **For CNV**: Red areas likely show subretinal fluid or abnormal blood vessels (signs of wet AMD).")
                            elif pred == "DME":
                                st.warning("⚠️ **For DME**: Red areas likely show intraretinal fluid pockets (diabetic swelling).")
                            elif pred == "DRUSEN":
                                st.info("ℹ️ **For Drusen**: Red areas likely show deposits under the retina (signs of dry AMD).")

                        except Exception as e:
                            st.error(f"Could not generate heatmap: {str(e)}")
                            st.info("The diagnosis above is still valid.")

