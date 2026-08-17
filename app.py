```python
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DriveGuard AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = Path(__file__).parent / "EfficientNet_B0.keras"

IMG_SIZE = (224, 224)

# IMPORTANT:
# These class names MUST match the order used when your
# EfficientNet-B0 model was trained.
CLASS_NAMES = [
    "Closed",
    "Open",
    "no_yawn",
    "yawn"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
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
        font-size: 2.6rem;
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

    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

@st.cache_resource(show_spinner="Loading EfficientNet-B0...")
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


model = None
model_error = None

if MODEL_PATH.exists():

    try:

        model = load_model()

    except Exception as e:

        model_error = str(e)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🚗 DriveGuard AI</h1>

        <p>
        AI Agent for Driver Drowsiness Detection and
        Intelligent Road Safety Assistance
        </p>

        <p>
        <b>
        EfficientNet-B0 • Drowsiness Detection •
        Safety Analytics
        </b>
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


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
    "Driver drowsiness is an important road-safety concern. "
    "The objective of this project is to develop an AI-powered "
    "system that identifies visual signs of driver drowsiness "
    "and provides an early safety warning."
)

st.write(
    "The system uses the EfficientNet-B0 deep-learning model "
    "to analyze visual patterns related to driver eye closure "
    "and yawning."
)

st.subheader("🤖 Model Used")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Deep Learning Model",
    "EfficientNet-B0"
)

col2.metric(
    "Input Size",
    "224 × 224"
)

col3.metric(
    "Output Classes",
    "4"
)

st.subheader("🛡️ Intelligent Road Safety Assistance")

st.write(
    "The system provides an AI-assisted indication of "
    "drowsiness-related visual patterns and displays a "
    "safety-awareness message when a possible drowsiness "
    "state is detected."
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
        "Please make sure that EfficientNet_B0.keras "
        "is present in the same GitHub folder as app.py."
    )

    if model_error:

        with st.expander("Technical error"):

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
    "Upload a driver image or use the camera to run "
    "the EfficientNet-B0 prediction."
)


if model is not None:

    left_column, right_column = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # INPUT
    # ========================================================

    with left_column:

        st.subheader("📷 Input")

        input_type = st.radio(
            "Choose input method",
            [
                "📁 Upload Image",
                "📷 Camera"
            ],
            horizontal=True
        )

        image = None


        if input_type == "📁 Upload Image":

            uploaded_file = st.file_uploader(
                "Upload a JPG, JPEG or PNG image",
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
                caption="Input Image",
                use_container_width=True
            )

        else:

            st.info(
                "👆 Upload an image or use the camera "
                "to start detection."
            )


    # ========================================================
    # PREDICTION
    # ========================================================

    with right_column:

        st.subheader("🧠 AI Detection Result")


        if image:

            # ------------------------------------------------
            # RESIZE IMAGE
            # ------------------------------------------------

            resized_image = image.resize(
                IMG_SIZE
            )


            # ------------------------------------------------
            # CONVERT TO NUMPY
            # ------------------------------------------------

            image_array = np.asarray(
                resized_image,
                dtype=np.float32
            )


            # ------------------------------------------------
            # ADD BATCH DIMENSION
            # ------------------------------------------------

            image_array = np.expand_dims(
                image_array,
                axis=0
            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            probabilities = model.predict(
                image_array,
                verbose=0
            )[0]


            # ------------------------------------------------
            # GET PREDICTED CLASS
            # ------------------------------------------------

            prediction_index = int(
                np.argmax(probabilities)
            )

            prediction = CLASS_NAMES[
                prediction_index
            ]

            confidence = float(
                probabilities[
                    prediction_index
                ]
            )


            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            if confidence < 0.60:

                st.warning(
                    "⚠️ Low-confidence prediction"
                )

                st.write(
                    "Please provide a clearer driver image."
                )


            elif prediction in [
                "Closed",
                "yawn"
            ]:

                st.error(
                    f"⚠️ Possible drowsiness detected: "
                    f"{prediction}"
                )

                st.warning(
                    "If the driver feels tired, stop at "
                    "a safe location and take a break."
                )


            else:

                st.success(
                    f"✅ Detected state: {prediction}"
                )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Prediction",
                prediction
            )

            result_col2.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


            # ------------------------------------------------
            # PROBABILITY BREAKDOWN
            # ------------------------------------------------

            st.subheader(
                "📊 Prediction Probabilities"
            )

            results = sorted(
                zip(
                    CLASS_NAMES,
                    probabilities
                ),
                key=lambda x: x[1],
                reverse=True
            )


            for class_name, probability in results:

                st.write(
                    f"**{class_name}** — "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================

st.markdown(
    '<div class="section">',
    unsafe_allow_html=True
)

st.header(
    "📊 Model Performance Comparison"
)

st.write(
    "Previously reported evaluation results from the project:"
)


performance_col1, performance_col2, performance_col3 = \
    st.columns(3)


performance_col1.metric(
    "CNN",
    "72.29%"
)

performance_col2.metric(
    "MobileNetV3-Small",
    "84.53%"
)

performance_col3.metric(
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
    "📈 Accuracy Comparison"
)

st.bar_chart(
    {
        "CNN": 72.29,
        "MobileNetV3-Small": 84.53,
        "EfficientNet-B0": 90.53
    }
)


st.success(
    "🏆 EfficientNet-B0 currently has the highest "
    "reported accuracy: 90.53%."
)


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INTELLIGENT ROAD SAFETY ASSISTANCE
# ============================================================

st.markdown(
    '<div class="section">',
    unsafe_allow_html=True
)

st.header(
    "🛡️ Intelligent Road Safety Assistance"
)


safety1, safety2, safety3 = st.columns(3)


safety1.info(
    "👁️ **Visual Analysis**\n\n"
    "The model analyzes visual patterns "
    "associated with the trained drowsiness classes."
)


safety2.info(
    "⚠️ **Drowsiness Alert**\n\n"
    "Possible drowsiness states are highlighted "
    "with a safety warning."
)


safety3.info(
    "🛑 **Safety Recommendation**\n\n"
    "Drivers who feel tired should stop safely "
    "and take an appropriate break."
)


st.warning(
    "This is an AI-assisted academic project and "
    "should not be treated as a certified vehicle "
    "safety or emergency system."
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


summary1, summary2, summary3, summary4 = \
    st.columns(4)


summary1.metric(
    "Model",
    "EfficientNet-B0"
)

summary2.metric(
    "Reported Accuracy",
    "90.53%"
)

summary3.metric(
    "Classes",
    "4"
)

summary4.metric(
    "System Status",
    "Online" if model is not None else "Offline"
)


st.write(
    "**Recognized Classes:** "
    "Closed • Open • no_yawn • yawn"
)


st.write(
    "**System Flow:** "
    "Image / Camera → Image Preprocessing → "
    "EfficientNet-B0 → Prediction → "
    "Confidence → Safety Assistance"
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
```

