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
    page_title="OCT-Insight | AI-Powered Retinal Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL MINIMAL STYLES (shadcn-inspired) ---
st.markdown("""
    <style>
    /* Main background - clean neutral */
    .main {
        background-color: #ffffff;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #18181b;
        color: #fafafa;
    }
    [data-testid="stSidebar"] * {
        color: #fafafa !important;
    }

    /* Clean buttons */
    .stButton>button {
        background-color: #18181b;
        color: white;
        border: 1px solid #27272a;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #27272a;
        border-color: #3f3f46;
    }

    /* Prediction box - minimal card */
    .prediction-box {
        padding: 2rem;
        border-radius: 8px;
        background-color: white;
        border: 1px solid #e4e4e7;
        text-align: center;
        margin: 1rem 0;
    }

    /* Disease info cards */
    .disease-card {
        padding: 1rem;
        border-radius: 6px;
        background-color: #fafafa;
        border: 1px solid #e4e4e7;
        margin: 0.5rem 0;
    }
    .disease-card h4 {
        margin: 0 0 0.5rem 0;
        color: #18181b;
        font-weight: 600;
    }
    .disease-card p {
        margin: 0;
        color: #52525b;
        font-size: 0.9rem;
    }

    /* Header styling */
    .header-container {
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #e4e4e7;
        margin-bottom: 2rem;
    }

    /* Tech badges */
    .tech-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        background-color: #18181b;
        color: white;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Sample image buttons */
    .sample-img-btn {
        padding: 0.5rem;
        border: 1px solid #e4e4e7;
        border-radius: 6px;
        background-color: white;
        cursor: pointer;
        transition: all 0.2s;
    }
    .sample-img-btn:hover {
        border-color: #18181b;
    }

    /* Clean expander */
    .streamlit-expanderHeader {
        background-color: #fafafa;
        border-radius: 6px;
        border: 1px solid #e4e4e7;
    }

    /* Hide Streamlit branding and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    /* Hide anchor links */
    a[href^="#"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    """Loads the FastAI model once and caches it. Downloads from Hugging Face if needed."""
    from huggingface_hub import hf_hub_download

    model_path = Path('models/baseline_model.pkl')

    # Download from Hugging Face if not present
    if not model_path.exists():
        st.info("📥 Downloading model from Hugging Face (first run only, ~100MB)...")
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path = hf_hub_download(
                repo_id="manosjung/oct-insight-model",
                filename="baseline_model.pkl",
                cache_dir="./models"
            )
            # Copy to expected location
            import shutil
            shutil.copy(downloaded_path, model_path)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    return load_learner(model_path)

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

# Professional Header
st.markdown("""
<div style="padding: 2rem; margin-bottom: 2rem; background-color: #18181b; border-radius: 8px;">
    <h1 style="margin:0; font-size: 2.5rem; font-weight: 700; color: #fafafa;">👁️ OCT-Insight</h1>
    <p style="margin: 0.75rem 0 0 0; font-size: 1.1rem; color: #d4d4d8; font-weight: 400;">AI-Powered Retinal Disease Classification</p>
    <div style="margin-top: 1.25rem;">
        <span class="tech-badge" style="background-color: #27272a; border: 1px solid #3f3f46;">ResNet50</span>
        <span class="tech-badge" style="background-color: #27272a; border: 1px solid #3f3f46;">PyTorch</span>
        <span class="tech-badge" style="background-color: #27272a; border: 1px solid #3f3f46;">Grad-CAM</span>
        <span class="tech-badge" style="background-color: #27272a; border: 1px solid #3f3f46;">84K Images</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 About This Project")
    st.markdown("""
    **Model Architecture**
    ResNet50 with transfer learning

    **Training Dataset**
    Kermany OCT Dataset (84,495 images)

    **Disease Classes**
    • CNV (Wet AMD)
    • DME (Diabetic edema)
    • Drusen (Dry AMD)
    • Normal retina

    **Explainability**
    Grad-CAM heatmaps
    """)

    st.markdown("---")

    st.markdown("### ℹ️ Disease Information")

    with st.expander("🔴 CNV (Urgent)"):
        st.markdown("""
        **Choroidal Neovascularization**
        Wet AMD - requires immediate treatment

        • Abnormal blood vessel growth
        • Causes vision loss within days
        • Treatment: Anti-VEGF injections
        """)

    with st.expander("🟡 DME"):
        st.markdown("""
        **Diabetic Macular Edema**
        Diabetic complication

        • Fluid accumulation in retina
        • Caused by diabetes
        • Treatment: Steroids/Anti-VEGF
        """)

    with st.expander("🟠 Drusen"):
        st.markdown("""
        **Drusenoid Deposits**
        Dry AMD - monitoring needed

        • Yellow deposits under retina
        • Not immediately urgent
        • Treatment: Monitoring, vitamins
        """)

    with st.expander("🟢 Normal"):
        st.markdown("""
        **Healthy Retina**
        No pathology detected

        • All retinal layers intact
        • No fluid or deposits
        • No treatment needed
        """)

    st.markdown("---")
    st.warning("⚠️ **Research Prototype Only**  \nNot for clinical diagnosis")

# Main Area
st.markdown("### 📤 Upload OCT Scan")

# Sample images section
with st.expander("💡 Don't have an OCT scan? Try sample images"):
    st.markdown("""
    Sample images are available from the test dataset. To test the app:
    1. Download OCT sample images from [Kermany Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)
    2. Or use your own OCT scan images (JPG, PNG format)

    **What to expect:**
    - **CNV images**: Show subretinal fluid
    - **DME images**: Show intraretinal cysts
    - **Drusen images**: Show deposits
    - **Normal images**: Clear retinal layers
    """)

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
                    <h3 style="color: #71717a; font-weight: 500; margin-bottom: 1rem;">AI Diagnosis</h3>
                    <h1 style="color: #18181b; margin: 0.5rem 0;">{pred}</h1>
                    <p style="color: #52525b; font-size: 1.1rem;">Confidence: <b style="color: #18181b;">{confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Bar Chart for all classes
                st.markdown("#### Class Probabilities")
                chart_data = {
                    "Condition": learn.dls.vocab,
                    "Probability": [p.item() for p in probs]
                }
                fig = px.bar(chart_data, x="Probability", y="Condition", orientation='h',
                             text_auto='.2%',
                             color="Probability", color_continuous_scale=["#fafafa", "#18181b"])
                fig.update_layout(
                    showlegend=False,
                    height=250,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#18181b'),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #71717a;">
    <p style="margin: 0.5rem 0;">
        <b>OCT-Insight</b> | Research Prototype for Retinal Disease Classification
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.9rem;">
        Built with ResNet50 • PyTorch • FastAI • Grad-CAM • Streamlit
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.85rem;">
        Dataset: Kermany et al. OCT Dataset (84,495 images)
    </p>
    <p style="margin: 1rem 0 0 0; font-size: 0.85rem; color: #a1a1aa;">
        ⚠️ For research and educational purposes only • Not for clinical diagnosis
    </p>
</div>
""", unsafe_allow_html=True)
