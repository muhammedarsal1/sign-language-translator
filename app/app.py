import streamlit as st
import numpy as np
import cv2
import base64
from predictor import predict_sign
from camera_component import camera_input

st.set_page_config(page_title="Hand Sign Language Translator", layout="wide")

st.title("🤟 Hand Sign Language Translator")
st.write("Use 'Image' for a single sign or 'Live' for real-time translation.")

# ✅ Buttons: Image, Live, Translate, Clear
col1, col2, col3, col4 = st.columns(4)
with col1:
    image_mode = st.button("🖼 Image")
with col2:
    live_mode = st.button("📹 Live")
with col3:
    translate_mode = st.button("🔄 Translate")
with col4:
    clear_mode = st.button("🗑 Clear")

# ✅ Initialize Session State
for key in ["captured_image", "video_frame", "prediction"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ✅ Process Image Data
def process_image(image_data):
    """Convert uploaded image or video frame to a valid format for AI model."""
    try:
        if isinstance(image_data, str):  # Base64 from camera_component
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif isinstance(image_data, bytes):  # Uploaded image
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            raise ValueError("Unsupported image format.")

        if image is None:
            raise ValueError("Invalid image format. Could not decode.")

        image = cv2.resize(image, (64, 64))  # Resize for AI model
        return image
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

# ✅ Image Mode: Capture Image
if image_mode:
    st.write("📸 Upload an image to translate a hand sign.")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        st.session_state.captured_image = uploaded_image.read()
        st.image(uploaded_image, caption="Captured Image", use_container_width=True)

# ✅ Live Mode: Real-time Video Translation
if live_mode:
    st.write("🎥 Showing live video feed...")
    video_frame = camera_input()
    if video_frame:
        st.session_state.video_frame = video_frame
        st.image(video_frame, caption="Live Video Frame", use_container_width=True)

# ✅ Translate Button
if translate_mode:
    st.write("🔍 Checking session state before translation...")
    if st.session_state.captured_image:
        processed_image = process_image(st.session_state.captured_image)
        if processed_image is not None:
            st.session_state.prediction = predict_sign(processed_image)
    elif st.session_state.video_frame:
        processed_image = process_image(st.session_state.video_frame)
        if processed_image is not None:
            st.session_state.prediction = predict_sign(processed_image)
    else:
        st.error("⚠️ No image or video frame captured. Please try again.")

if st.session_state.prediction:
    st.subheader(f"🔠 Translated Sign: **{st.session_state.prediction}**")

# ✅ Clear Button
if clear_mode:
    for key in ["captured_image", "video_frame", "prediction"]:
        st.session_state[key] = None
    st.success("🗑 Data Cleared!")
