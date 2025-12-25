import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os

st.set_page_config(page_title="Potato Disease Detection")

# ------------------ MODEL PATH ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "potatoes.h5")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    # Load model without compiling to avoid loss deserialization issues
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ------------------ CLASS NAMES ------------------
class_names = ["Early Blight", "Late Blight", "Healthy"]

# ------------------ UI ------------------
st.title("🥔 Potato Disease Detection")
st.write("Upload a potato leaf image")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=700)  # Updated here

    # Preprocess the image: resize to 256x256 as per model input size
    img = np.array(image)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display results
    st.subheader("Prediction")
    st.success(predicted_class)
    st.write(f"Confidence: {confidence:.2f}%")
