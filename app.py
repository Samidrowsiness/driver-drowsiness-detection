import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# Model and classes
# -----------------------------
MODEL_PATH = "EfficientNet_B0.keras"

CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.title("🚗 AI Driver Drowsiness Detection")
st.subheader("Intelligent Road Safety Assistance System")

st.write(
    "This AI system analyzes eye and yawn images to identify "
    "possible signs of driver drowsiness."
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Project Information")
st.sidebar.write("**Dataset:** Yawn-Eye Dataset")
st.sidebar.write("**Models:** CNN, MobileNetV3-Small, EfficientNet-B0")
st.sidebar.write("**Best Model:** EfficientNet-B0")
st.sidebar.write("**Test Accuracy:** 90.53%")

# -----------------------------
# Main tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["🔍 Detection", "📊 Model Comparison", "📘 About Project"]
)

# -----------------------------
# Detection
# -----------------------------
with tab1:

    st.header("🔍 Drowsiness Detection")

    uploaded_file = st.file_uploader(
        "Upload an eye/yawn image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=350
        )

        # Resize image
        img = image.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(prediction[0][predicted_index]) * 100

        st.divider()

        st.subheader("Prediction Result")

        st.success(
            f"Prediction: **{predicted_class}**"
        )

        st.info(
            f"Confidence: **{confidence:.2f}%**"
        )

        # Drowsiness warning
        if predicted_class in ["Closed", "yawn"]:
            st.warning(
                "⚠️ Possible drowsiness detected. "
                "Please take a break and drive safely."
            )
        else:
            st.success(
                "✅ No strong drowsiness indication detected."
            )

        # Probability table
        st.subheader("📊 Prediction Probabilities")

        for i, class_name in enumerate(CLASS_NAMES):
            probability = float(prediction[0][i]) * 100
            st.write(
                f"{class_name}: {probability:.2f}%"
            )
            st.progress(min(probability / 100, 1.0))


# -----------------------------
# Model Comparison
# -----------------------------
with tab2:

    st.header("🏆 Model Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("CNN", "72.29%")

    with col2:
        st.metric("MobileNetV3-Small", "84.53%")

    with col3:
        st.metric("EfficientNet-B0", "90.53%")

    st.divider()

    st.subheader("Performance Results")

    st.table({
        "Model": [
            "CNN",
            "MobileNetV3-Small",
            "EfficientNet-B0"
        ],
        "Accuracy": [
            "72.29%",
            "84.53%",
            "90.53%"
        ],
        "Precision": [
            "72.86%",
            "85.79%",
            "91.19%"
        ],
        "Recall": [
            "72.29%",
            "84.53%",
            "90.53%"
        ],
        "F1-Score": [
            "72.11%",
            "84.15%",
            "90.44%"
        ]
    })

    st.success(
        "🏆 EfficientNet-B0 achieved the best overall performance."
    )


# -----------------------------
# About
# -----------------------------
with tab3:

    st.header("📘 About the Project")

    st.write("""
    ### Problem Statement

    AI Agent for Driver Drowsiness Detection and
    Intelligent Road Safety Assistance.

    ### Objective

    The objective of this project is to detect signs of
    driver drowsiness using deep learning and provide
    an intelligent safety warning.

    ### Dataset

    Yawn-Eye Dataset containing four classes:

    - Closed
    - Open
    - no_yawn
    - yawn

    ### Deep Learning Models

    Three models were evaluated:

    1. CNN
    2. MobileNetV3-Small
    3. EfficientNet-B0

    ### Final Result

    EfficientNet-B0 achieved the highest test accuracy
    of 90.53%.

    ### Evaluation Metrics

    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - Confusion Matrix
    """)

st.divider()

st.caption(
    "AI Driver Drowsiness Detection | Mini Project"
)
