import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DriveGuard AI",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = Path(__file__).parent / "EfficientNet_B0.keras"

# IMPORTANT:
# These must match the class order used during training.
CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]

IMG_SIZE = (224, 224)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1250px;
    padding: 1.5rem 2rem 3rem;
}

.hero {
    padding: 38px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a5f,
        #0f766e
    );
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 2.5rem;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe;
    font-size: 1.05rem;
}

.section {
    padding: 25px;
    border: 1px solid #dbe4ef;
    border-radius: 20px;
    background: white;
    margin: 20px 0;
}

.result-box {
    padding: 20px;
    border-radius: 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FACE CROP FUNCTION
# ============================================================

def get_largest_face(image, faces):

    if len(faces) == 0:
        return None

    # Select largest detected face
    largest = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    x, y, w, h = largest

    # Add small margin around face
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)

    x2 = min(image.width, x + w + margin_x)
    y2 = min(image.height, y + h + margin_y)

    face_crop = image.crop(
        (x1, y1, x2, y2)
    )

    return face_crop


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_drowsiness(face_image):

    resized = face_image.resize(
        IMG_SIZE
    )

    image_array = np.asarray(
        resized,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    prediction_index = int(
        np.argmax(probabilities)
    )

    prediction = CLASS_NAMES[
        prediction_index
    ]

    confidence = float(
        probabilities[prediction_index]
    )

    return prediction, confidence, probabilities


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<h1>🚗 DriveGuard AI</h1>

<p>
AI Agent for Driver Drowsiness Detection and
Intelligent Road Safety Assistance
</p>

<p>
<b>
EfficientNet-B0 • Face Detection • Drowsiness Analysis
</b>
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PROBLEM STATEMENT
# ============================================================

st.markdown(
    '<div class="section">',
    unsafe_allow_html=True
)

st.header("🎯 Problem Statement")

st.write(
    "**AI Agent for Driver Drowsiness Detection and "
    "Intelligent Road Safety Assistance**"
)

st.write(
    "The objective of this project is to develop an AI-powered "
    "driver monitoring system that identifies visual signs of "
    "drowsiness and provides an early road-safety warning. "
    "The system uses EfficientNet-B0 to analyze driver eye and "
    "yawning patterns."
)

st.subheader("🤖 Model Used")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Deep Learning Model",
    "EfficientNet-B0"
)

col2.metric(
    "Input Resolution",
    "224 × 224"
)

col3.metric(
    "Classes",
    "4"
)

st.subheader("🛡️ Intelligent Road Safety Assistance")

st.write(
    "The system first checks whether a driver face is visible. "
    "If a face is detected, the system analyzes the face region "
    "for drowsiness-related classes. If no face is detected, "
    "the system avoids making an unreliable drowsiness prediction."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL STATUS
# ============================================================

if model is None:

    st.error(
        "❌ EfficientNet_B0.keras could not be loaded."
    )

    st.info(
        "Make sure EfficientNet_B0.keras is in the "
        "same GitHub folder as app.py."
    )

    if model_error:

        with st.expander(
            "Technical error"
        ):

            st.code(model_error)


# ============================================================
# DROWSINESS DETECTION
# ============================================================

st.markdown(
    '<div class="section">',
    unsafe_allow_html=True
)

st.header("🔍 Driver Drowsiness Detection")

st.write(
    "Upload a driver image or use the camera."
)

if model is not None:

    left, right = st.columns(
        [1, 1],
        gap="large"
    )

    with left:

        input_type = st.radio(
            "Select input",
            [
                "📁 Upload Image",
                "📷 Camera"
            ],
            horizontal=True
        )

        image = None

        if input_type == "📁 Upload Image":

            uploaded_file = st.file_uploader(
                "Upload driver image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ]
            )

            if uploaded_file:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

        else:

            camera_image = st.camera_input(
                "Take a driver image"
            )

            if camera_image:

                image = Image.open(
                    camera_image
                ).convert("RGB")

        if image:

            st.image(
                image,
                caption="Input image",
                use_container_width=True
            )

    with right:

        st.subheader(
            "🧠 AI Detection Result"
        )

        if image:

            faces = detect_faces(image)

            # ------------------------------------------------
            # NO FACE
            # ------------------------------------------------

            if len(faces) == 0:

                st.warning(
                    "⚠️ No driver face detected."
                )

                st.info(
                    "Please provide a clear image "
                    "containing the driver's face."
                )

                st.caption(
                    "Traffic signs, roads, vehicles and "
                    "other non-face images are not classified "
                    "as drowsiness."
                )

            # ------------------------------------------------
            # FACE DETECTED
            # ------------------------------------------------

            else:

                face_image = get_largest_face(
                    image,
                    faces
                )

                st.image(
                    face_image,
                    caption="Detected driver face",
                    width=300
                )

                prediction, confidence, probabilities = \
                    predict_drowsiness(
                        face_image
                    )

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                if prediction in [
                    "Closed",
                    "yawn"
                ]:

                    st.error(
                        f"⚠️ Possible drowsiness: "
                        f"{prediction}"
                    )

                    st.warning(
                        "If you feel tired while driving, "
                        "stop safely and take a break."
                    )

                else:

                    st.success(
                        f"✅ Detected state: "
                        f"{prediction}"
                    )

                r1, r2 = st.columns(2)

                r1.metric(
                    "Prediction",
                    prediction
                )

                r2.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.subheader(
                    "📊 Prediction Probabilities"
                )

                for class_name, probability in sorted(
                    zip(
                        CLASS_NAMES,
                        probabilities
                    ),
                    key=lambda x: x[1],
                    reverse=True
                ):

                    st.write(
                        f"**{class_name}** — "
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(
                        float(probability)
                    )

        else:

            st.info(
                "Upload an image or use the camera "
                "to start detection."
            )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section">',
    unsafe_allow_html=True
)

st.header(
    "📊 Model Performance Comparison"
)

st.write(
    "Previously reported evaluation results:"
)

p1, p2, p3 = st.columns(3)

p1.metric(
    "CNN",
    "72.29%"
)

p2.metric(
    "MobileNetV3-Small",
    "84.53%"
)

p3.metric(
    "🏆 EfficientNet-B0",
    "90.53%"
)

performance_data = {

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
}

st.table(
    performance_data
)

st.subheader(
    "Accuracy Comparison"
)

st.bar_chart(
    {
        "CNN": 72.29,
        "MobileNetV3-Small": 84.53,
        "EfficientNet-B0": 90.53
    }
)

st.success(
    "🏆 EfficientNet-B0 is currently the best "
    "reported model with 90.53% accuracy."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ROAD SAFETY
# ============================================================

st.markdown(
    '<div class="section">',
    unsafe_allow_html=True
)

st.header(
    "🛡️ Intelligent Road Safety Assistance"
)

s1, s2, s3 = st.columns(3)

s1.info(
    "👤 **Face Detection**\n\n"
    "Checks whether a driver face is present."
)

s2.info(
    "👁️ **Drowsiness Analysis**\n\n"
    "Analyzes the detected face using EfficientNet-B0."
)

s3.info(
    "⚠️ **Safety Assistance**\n\n"
    "Displays an awareness message when "
    "drowsiness-related patterns are detected."
)

st.warning(
    "This is an AI-assisted demonstration system. "
    "It should not be treated as a replacement for "
    "a certified vehicle safety system."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROJECT SUMMARY
# ============================================================

st.markdown(
    '<div class="section">',
    unsafe_allow_html=True
)

st.header(
    "📘 Project Summary"
)

a, b, c, d = st.columns(4)

a.metric(
    "Model",
    "EfficientNet-B0"
)

b.metric(
    "Reported Accuracy",
    "90.53%"
)

c.metric(
    "Classes",
    "4"
)

d.metric(
    "Status",
    "Online" if model else "Offline"
)

st.write(
    "**Classes:** Closed • Open • no_yawn • yawn"
)

st.write(
    "**System Flow:** "
    "Image / Camera → Face Detection → Face Crop → "
    "EfficientNet-B0 → Prediction → Confidence → "
    "Safety Assistance"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    <b>DriveGuard AI</b><br>
    AI Agent for Driver Drowsiness Detection and
    Intelligent Road Safety Assistance<br>
    Powered by EfficientNet-B0
    </div>
    """,
    unsafe_allow_html=True
)
