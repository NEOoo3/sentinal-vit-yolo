"""
Advanced Pedestrian Detection HUD (Heads-Up Display)
Target Environment: Indian Urban Traffic
Architecture: YOLOv11 + Vision Transformer (ViT) Neck
"""

import streamlit as st
import numpy as np
import time
from PIL import Image

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (THE "VIBE")
# ==========================================
st.set_page_config(
    page_title="Pedestrian Detection HUD | Nammos Techno Labs",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark HUD, Glassmorphism, and Neon Accents
custom_css = """
<style>
    /* Global Theme */
    .stApp {
        background-color: #0b0f19;
        color: #e0e6ed;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Hide Default Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Glassmorphic Panels */
    div[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 255, 204, 0.2);
    }
    
    /* Neon Glowing Metrics */
    div[data-testid="stMetricValue"] {
        color: #00ffcc !important;
        text-shadow: 0px 0px 10px rgba(0, 255, 204, 0.5);
    }
    
    /* Custom High-Risk Metric Override */
    .metric-high-risk {
        color: #ff3366 !important;
        text-shadow: 0px 0px 15px rgba(255, 51, 102, 0.8);
        font-weight: bold;
        font-size: 2rem;
    }

    /* Video Canvas Border */
    .video-canvas {
        border: 2px solid rgba(0, 255, 204, 0.3);
        border-radius: 8px;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.1);
        padding: 10px;
        background: #05080f;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ==========================================
# 2. SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ SYSTEM CONTROLS")
    st.markdown("---")
    
    st.markdown("**Engine Parameters**")
    backbone = st.selectbox("Backbone Architecture", ["YOLOv11-Small + ViT", "YOLOv11-Nano + ViT"])
    
    conf_thresh = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    iou_thresh = st.slider("NMS IoU Threshold", min_value=0.1, max_value=0.9, value=0.50, step=0.05)
    
    st.markdown("---")
    st.markdown("**Environment Simulation (India)**")
    env_condition = st.radio("Current Condition", ["Clear Day", "Delhi Haze (Smog)", "Monsoon (Low Light/Glare)"])
    
    st.markdown("---")
    st.button("🔄 Initialize Edge TensorRT", use_container_width=True, type="primary")


# ==========================================
# 3. MAIN DASHBOARD AREA
# ==========================================
st.markdown("<h2 style='color: #00ffcc; font-weight: 300;'>⚡ Pedestrian Detection HUD</h2>", unsafe_allow_html=True)
st.markdown("Real-time context-aware inference optimized for high-occlusion environments.")

# --- Top Row: Telemetry Metrics ---
col1, col2, col3, col4 = st.columns(4)

# (Mock logic to simulate real-time metrics changing)
mock_fps = np.random.randint(28, 35)
mock_count = np.random.randint(2, 8)
risk_level = "HIGH (Proximity)" if mock_count > 5 else "NORMAL"

col1.metric("Inference Latency", f"{int(1000/mock_fps)} ms")
col2.metric("Processing Speed", f"{mock_fps} FPS")
col3.metric("Pedestrians Tracked", f"{mock_count}")

# Custom rendering for the Risk Level to make it glow red if High
if risk_level == "HIGH (Proximity)":
    col4.markdown(f"<div><span style='font-size: 14px; color: #a0aec0;'>Threat Assessment</span><br><span class='metric-high-risk'>⚠ {risk_level}</span></div>", unsafe_allow_html=True)
else:
    col4.metric("Threat Assessment", risk_level)

st.markdown("<br>", unsafe_allow_html=True)

# --- Middle Row: The Live Feed & Attention Map ---
video_col, map_col = st.columns([2, 1])

with video_col:
    st.markdown("<div class='video-canvas'>", unsafe_allow_html=True)
    st.markdown("**Live Camera Feed (Bbox + Segmentation)**")
    # In a real app, you would use cv2.VideoCapture here.
    # For the UI shell, we will use a blank noisy image to look like a camera waiting for signal.
    dummy_frame = np.random.randint(0, 50, (480, 854, 3), dtype=np.uint8)
    st.image(dummy_frame, use_container_width=True, caption=f"Source: UVH-26 Format | Filter: {env_condition}")
    st.markdown("</div>", unsafe_allow_html=True)

with map_col:
    st.markdown("<div class='video-canvas'>", unsafe_allow_html=True)
    st.markdown("**ViT Global Attention Map**")
    st.caption("Visualizing Transformer 'Global Context' linking.")
    # Mocking a heatmap
    heatmap = np.random.randint(0, 255, (240, 240, 1), dtype=np.uint8)
    # Applying a pseudo-color map (using pure numpy/PIL for demo purposes without cv2)
    colored_heatmap = np.zeros((240, 240, 3), dtype=np.uint8)
    colored_heatmap[:,:,1] = heatmap[:,:,0]  # Green channel active for that "Radar" look
    st.image(colored_heatmap, use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Diagnostics:**")
    st.code("""
Status: ACTIVE
Device: CUDA:0
Fusion: Cross-Attention
Occlusion Handling: TRUE
    """, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Bottom Row: Log Console ---
with st.expander("System Logs (Terminal)", expanded=False):
    st.code("""
[INFO] Loading YOLOv11-Small backbone... OK
[INFO] Injecting Vision Transformer block (depth=3, heads=8)... OK
[INFO] Compiling model with TensorRT optimizations...
[WARN] Heavy occlusion detected in sector 4. Activating Cross-Attention focus.
[INFO] Bounding box stabilized. Confidence: 0.89
    """, language="bash")
