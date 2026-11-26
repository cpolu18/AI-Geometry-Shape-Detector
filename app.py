
import streamlit as st
import cv2
import numpy as np
from geometry import analyze_image

st.set_page_config(
    page_title="AI Geometry Shape Detector",
    page_icon="📐",
    layout="wide"
)

st.title("📐 AI Geometry Shape Detector")
st.write("Detect geometric shapes using image processing and a live camera.")

mode = st.radio(
    "Choose Detection Mode",
    ["Upload Image", "Live Camera"]
)

if mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Unable to read the uploaded image.")

        else:
            st.subheader("Original Image")

            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            result_image, results = analyze_image(image)

            st.subheader("Detection Results")

            st.image(
                cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            st.subheader("Shape Measurements")

            if results:
                for item in results:
                    st.write(f"**Shape:** {item['shape']}")
                    st.write(f"**Area:** {item['area']} square pixels")
                    st.write(f"**Perimeter:** {item['perimeter']} pixels")
                    st.write("---")

            else:
                st.warning("No geometric shapes detected.")

elif mode == "Live Camera":

    st.subheader("Live Camera Detection")

    camera_image = st.camera_input("Capture an image")

    if camera_image is not None:

        file_bytes = np.asarray(
            bytearray(camera_image.getvalue()),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:

            result_image, results = analyze_image(image)

            st.subheader("Camera Detection Results")

            st.image(
                cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            st.subheader("Shape Measurements")

            if results:
                for item in results:
                    st.write(f"**Shape:** {item['shape']}")
                    st.write(f"**Area:** {item['area']} square pixels")
                    st.write(f"**Perimeter:** {item['perimeter']} pixels")
                    st.write("---")

            else:
                st.warning("No geometric shapes detected.")
