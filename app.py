# ============================================================
# DROWSINESS DETECTION
# ============================================================

st.markdown(
    '<div class="section">',
    unsafe_allow_html=True
)

st.header("🔍 Driver Drowsiness Detection")

st.write(
    "Upload a clear driver/face image or use the camera. "
    "EfficientNet-B0 will classify the image using the "
    "classes on which the model was trained."
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
                type=["jpg", "jpeg", "png"]
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

        st.subheader("🧠 AI Detection Result")

        if image:

            # ----------------------------------------------
            # PREPROCESS IMAGE
            # ----------------------------------------------

            resized = image.resize(
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

            # ----------------------------------------------
            # PREDICTION
            # ----------------------------------------------

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

            # ----------------------------------------------
            # CONFIDENCE CHECK
            # ----------------------------------------------

            if confidence < 0.60:

                st.warning(
                    "⚠️ Low-confidence prediction"
                )

                st.write(
                    "Please upload a clearer driver/face image."
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
                    "For safety, if the driver feels tired, "
                    "stop at a safe location and take a break."
                )

            else:

                st.success(
                    f"✅ Detected state: {prediction}"
                )

            # ----------------------------------------------
            # METRICS
            # ----------------------------------------------

            r1, r2 = st.columns(2)

            r1.metric(
                "Prediction",
                prediction
            )

            r2.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            # ----------------------------------------------
            # PROBABILITIES
            # ----------------------------------------------

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

        else:

            st.info(
                "Upload an image or use the camera "
                "to start detection."
            )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)
