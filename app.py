
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title= DriveGuard AI: AI-Powered Driver Drowsiness Detection & Safety Analytics
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "EfficientNet_B0.keras"
CLASSES = ["Closed", "Open", "no_yawn", "yawn"]
IMG_SIZE = (224, 224)

st.markdown("""
<style>
.block-container{max-width:1250px;padding:1.5rem 2rem 3rem}
.hero{
    padding:32px 36px;border-radius:24px;
    background:linear-gradient(135deg,#0f172a,#1e3a5f,#0f766e);
    color:white;margin-bottom:24px;
}
.hero h1{font-size:2.5rem;margin:0 0 8px}
.hero p{margin:5px 0;color:#dbeafe}
.section{
    padding:22px;border:1px solid #dbe4ef;border-radius:20px;
    background:#ffffff;margin:18px 0;
}
.small-note{color:#64748b;font-size:.9rem}
.result{
    padding:18px;border-radius:16px;background:#f8fafc;
    border:1px solid #e2e8f0;
}
.footer{text-align:center;color:#64748b;padding:25px}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading EfficientNet-B0...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = None
model_error = None
if MODEL_PATH.exists():
    try:
        model = load_model()
    except Exception as e:
        model_error = str(e)

st.markdown("""
<div class="hero">
<h1>🚗 DriveGuard AI</h1>
<p>AI-Powered Driver Drowsiness Detection & Intelligent Road Safety Assistance</p>
<p><b>Detection + Model Performance Comparison — all in one page</b></p>
</div>
""", unsafe_allow_html=True)

# ---------------- DROWSINESS DETECTION ----------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("🔍 Drowsiness Detection")
st.write("Upload an eye/yawn image or use your camera to run the EfficientNet-B0 model.")

if model is None:
    st.error("EfficientNet_B0.keras could not be loaded.")
    if model_error:
        with st.expander("Technical error"):
            st.code(model_error)
    st.info("Keep EfficientNet_B0.keras in the same GitHub folder as app.py.")
else:
    input_col, result_col = st.columns([1, 1], gap="large")

    with input_col:
        source = st.radio(
            "Choose input method",
            ["📁 Upload Image", "📷 Camera"],
            horizontal=True
        )
        image = None

        if source == "📁 Upload Image":
            uploaded = st.file_uploader(
                "Choose JPG, JPEG or PNG",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
        else:
            captured = st.camera_input("Take a picture")
            if captured:
                image = Image.open(captured).convert("RGB")

        if image:
            st.image(image, caption="Input image", use_container_width=True)
        else:
            st.info("👆 Add an image above to start detection.")

    with result_col:
        st.subheader("AI Detection Result")

        if image:
            resized = image.resize(IMG_SIZE)
            x = np.expand_dims(np.asarray(resized, dtype=np.float32), 0)
            probs = model.predict(x, verbose=0)[0]
            idx = int(np.argmax(probs))
            label = CLASSES[idx]
            confidence = float(probs[idx])

            if label in ["Closed", "yawn"]:
                st.error(f"⚠️ Possible drowsiness sign: {label}")
                st.warning(
                    "If you are tired while driving, stop somewhere safe and take a break."
                )
            else:
                st.success(f"✅ Awake-related class predicted: {label}")

            m1, m2 = st.columns(2)
            m1.metric("Prediction", label)
            m2.metric("Confidence", f"{confidence*100:.2f}%")

            st.subheader("Prediction Probabilities")
            for name, p in sorted(
                zip(CLASSES, probs), key=lambda item: item[1], reverse=True
            ):
                st.write(f"**{name}** — {p*100:.2f}%")
                st.progress(float(p))

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MODEL PERFORMANCE ----------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("📊 Model Performance Comparison")
st.write(
    "Comparison of the three models evaluated for the driver drowsiness project."
)

c1, c2, c3 = st.columns(3)
c1.metric("CNN", "72.29%")
c2.metric("MobileNetV3-Small", "84.53%")
c3.metric("🏆 EfficientNet-B0", "90.53%")

performance = {
    "Model": ["CNN", "MobileNetV3-Small", "EfficientNet-B0"],
    "Accuracy": ["72.29%", "84.53%", "90.53%"],
    "Precision": ["72.86%", "85.79%", "91.19%"],
    "Recall": ["72.29%", "84.53%", "90.53%"],
    "F1-Score": ["72.11%", "84.15%", "90.44%"],
}
st.table(performance)

chart_data = {
    "CNN": 72.29,
    "MobileNetV3-Small": 84.53,
    "EfficientNet-B0": 90.53,
}
st.subheader("Accuracy Comparison")
st.bar_chart(chart_data)

st.success("🏆 EfficientNet-B0 achieved the highest reported accuracy: 90.53%.")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PROJECT SUMMARY ----------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("📘 Project Summary")

a, b, c, d = st.columns(4)
a.metric("Best Model", "EfficientNet-B0")
b.metric("Reported Accuracy", "90.53%")
c.metric("Classes", "4")
d.metric("Model Status", "Online" if model is not None else "Offline")

st.write(
    "Recognized classes: **Closed • Open • no_yawn • yawn**"
)
st.info(
    "System flow: Image / Camera → 224×224 preprocessing → EfficientNet-B0 "
    "→ Prediction → Confidence → Safety message"
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer"><b>DriveGuard AI</b> • Driver Drowsiness Detection • Model Performance Comparison • College Mini Project</div>',
    unsafe_allow_html=True
)
