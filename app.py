import streamlit as st
import mimetypes
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

from fastai.vision.all import *
import pathlib
import plotly.express as px

# --- SETUP ---
# Fix for Windows Path issues when loading FastAI models trained on different OS/Paths
# (Sometimes needed if moving between machines, good practice to have)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

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

                # Explanation (Placeholder for Grad-CAM)
                with st.expander("See Explanation (Grad-CAM)"):
                    st.info("Grad-CAM visualization coming in the next update!")

