# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time
import pandas as pd

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
CNN_MODEL_PATH = "./models/best_cnn.h5"
TL_MODEL_PATH = "./models/resnet50_final.h5"
LOG_CSV = "./inference_log.csv"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['drowsy', 'notdrowsy']
DEFAULT_THRESHOLD = 0.5

TEAM_MEMBERS = [
    "Safiuddin Fazil Mohammed - 1002188728",
    "Purva Dankhara - 1002260167"
]
# -----------------------------------------------------

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="😴",
    layout="wide"
)

# -----------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    cnn_model, tl_model = None, None
    try:
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
    except Exception as e:
        st.warning(f"Could not load CNN model: {e}")

    try:
        tl_model = tf.keras.models.load_model(TL_MODEL_PATH, compile=False)
    except Exception as e:
        st.warning(f"Could not load Transfer Learning model: {e}")

    return cnn_model, tl_model


# -----------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------
def preprocess_for_cnn(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_for_resnet(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict(model, preprocessed_input):
    return float(model.predict(preprocessed_input, verbose=0).ravel()[0])

def log_inference(entry):
    df = pd.DataFrame([entry])
    if os.path.exists(LOG_CSV):
        df.to_csv(LOG_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_CSV, index=False)


# =====================================================
# PAGE HEADER (ABOVE TABS)
# =====================================================
st.title("Driver Drowsiness Detection Dashboard")
st.markdown(
    """
    Monitor driver alertness from facial images using deep learning models.  
    Upload a frame, select the model, and review predictions in real time.
    """
)
st.write("---")

# =====================================================
# SIDEBAR – GLOBAL CONTROLS
# =====================================================
with st.sidebar:
    st.header("🔧 Controls")
    st.caption("Configure the model and threshold, then upload an image.")

    # Model toggle – segmented control
    model_choice = st.segmented_control(
        "Model selection",
        options=["CNN", "Both", "Transfer Learning"],
        default="Both"
    )
    
    st.markdown("---")
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Driver frame (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write(
        "This tool compares a custom CNN with a ResNet50 transfer‑learning "
        "model for driver drowsiness classification."
    )

# =====================================================
# MAIN TABS (BELOW HEADER)
# =====================================================

tab_pred, tab_info = st.tabs(["🔍 Prediction", "📘 Project Info"])

# =====================================================
# TAB 1 — PREDICTION DASHBOARD
# =====================================================
with tab_pred:
    cnn_model, tl_model = load_models()

    if uploaded_file is None:
        st.info("Upload a driver image from the sidebar to start.")
    else:
        pil_img = Image.open(uploaded_file)

        # Layout: left = image + basic info, right = predictions
        col_left, col_right = st.columns([1.1, 1.2])

        with col_left:
            st.subheader("Input Image")
            st.image(pil_img, caption=uploaded_file.name, width='stretch')

            st.markdown("##### Prediction Settings")
            st.write(f"- **Model mode:** `{model_choice}`")

        with col_right:
            st.subheader("Model Predictions")

            results = []
            now = time.strftime("%Y-%m-%d %H:%M:%S")

            # Compact summary cards
            summary_cols = st.columns(2)

            # CNN
            if model_choice in ("CNN", "Both") and cnn_model:
                x = preprocess_for_cnn(pil_img)
                prob = predict(cnn_model, x)
                p_drowsy = prob
                p_not = 1.0 - p_drowsy

                drowsy_pct = 100 * p_drowsy
                not_pct = 100 * p_not
                
                label = CLASS_NAMES[int(p_not > p_drowsy)]  # 0 = drowsy, 1 = notdrowsy
                
                with summary_cols[0]:
                    st.markdown("**📘 CNN (from scratch)**")
                    with st.expander("Probabilities"):
                        st.metric("P(drowsy)", f"{drowsy_pct:.1f}%")
                        st.metric("P(not drowsy)", f"{not_pct:.1f}%")
                    st.write(f"Predicted label: **{label}**")
                
                results.append(("CNN", p_drowsy, label))
                
            # Transfer learning
            if model_choice in ("Transfer Learning", "Both") and tl_model:
                x = preprocess_for_resnet(pil_img)
                prob = predict(tl_model, x)          # P(drowsy)
                p_drowsy = prob
                p_not = 1.0 - p_drowsy
                
                drowsy_pct = 100 * p_drowsy
                not_pct = 100 * p_not
                
                label = CLASS_NAMES[int(p_not > p_drowsy)]
                
                with summary_cols[1]:
                    st.markdown("**🧠 ResNet50 (TL)**")
                    with st.expander("Probabilities"):
                        st.metric("P(drowsy)", f"{drowsy_pct:.1f}%")
                        st.metric("P(not drowsy)", f"{not_pct:.1f}%")
                    st.write(f"Predicted label: **{label}**")
                
                results.append(("ResNet50", p_drowsy, label))


            if not results:
                st.warning("Selected model(s) are not available. Check that the .h5 files exist.")
            else:
                # Log results
                for model_name, prob, label in results:
                    entry = {
                        "timestamp": now,
                        "filename": uploaded_file.name,
                        "model": model_name,
                        "probability": prob,
                        "pred_label": label
                    }
                    try:
                        log_inference(entry)
                    except Exception:
                        pass

                if len(results) > 1:
                    st.markdown("#### 🔄 Model Comparison")
                    df_res = pd.DataFrame(
                        [
                            {
                                "Model": name,
                                "Drowsy_Prob": f"{p:.3f}",
                                "Pred_Label": lbl
                            }
                            for name, p, lbl in results
                        ]
                    )
                    st.dataframe(df_res, width='stretch')

                st.success("Inference completed and logged.")


# =====================================================
# TAB 2 — PROJECT INFO & RESULTS
# =====================================================
with tab_info:
    st.header("CSE 6367-001 Computer Vision Final Project")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("### 👥 Team Members")
        for member in TEAM_MEMBERS:
            st.markdown(f"- {member}")

        
        st.markdown("### 🧠 Models")
        st.write(
            "- **CNN from scratch** trained on 224×224 RGB frames.\n"
            "- **ResNet50 transfer learning** fine‑tuned on the same dataset.\n"
            "- Unified preprocessing and evaluation for a fair comparison."
        )

    with col_b:
        st.markdown("### 🚗 Project Description")
        st.write(
            """
            Driver fatigue and drowsiness are significant contributors to road accidents worldwide, often resulting in serious injuries and fatalities. Early detection of drowsy states can help prevent accidents by alerting the driver before a critical lapse in attention occurs.
            
            In this project, we aim to build a computer vision system that detects driver states as drowsy or not using deep learning models. Our approach focuses on facial cues (e.g., closed eyes, yawning) captured from images, and the primary goal is to train and evaluate a deep learning classifier on a benchmark dataset, with the potential to extend the model toward real-time applications.
            """
        )

    st.markdown("---")

    st.markdown("### 📊 Training Curves")
    
    col_1, col_2 = st.columns([1, 1])
    with col_1:
        if os.path.exists("./models/best_cnn_confusion_matrix.png"):
            st.image("./models/best_cnn_confusion_matrix.png", caption="CNN Confusion Matrix", width='stretch')
    with col_2:
        if os.path.exists("./models/resnet50_final_confusion_matrix.png"):
            st.image("./models/resnet50_final_confusion_matrix.png", caption="ResNet50 Confusion Matrix", width='stretch')

    st.markdown("---")
    st.markdown("### ⚠️ Drowsiness Alerts & Deployment Ideas")
    st.write(
        "- Integrate with an in‑car camera for continuous monitoring.\n"
        "- Trigger audio alarms, haptic feedback, or dashboard warnings "
        "when drowsy probability stays high over several consecutive frames.\n"
        "- Deploy on embedded hardware (Jetson, Raspberry Pi + Coral) for real‑time use."
    )

    st.markdown("---")
    st.markdown("### 📝 Notes")
    st.write(
        "- This demo focuses on static image classification; video streaming and "
        "temporal smoothing are natural extensions.\n"
        "- The decision threshold can be tuned per deployment to trade off between "
        "false alarms and missed drowsiness events."
    )
