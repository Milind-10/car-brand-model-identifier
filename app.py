import streamlit as st
from tensorflow.keras.preprocessing import image

from utils.model_loader import load_car_model, load_class_names
from utils.inference import predict_top_k
from utils.detector_inference import CarDetector


# Page Configuration 
st.set_page_config(
    page_title="Car Brand and Model Identifier",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Brand & Model Identifier")
st.write("Upload a car image to identify its **Brand and Model**.")


# Load Models
@st.cache_resource
def load_resources():
    model = load_car_model()
    class_names = load_class_names()
    detector = CarDetector("models/car_not_car_detector.keras")
    return model, class_names, detector

model, class_names, car_detector = load_resources()

st.caption(f"Model trained on {len(class_names)} car models")


# File Upload
uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = image.load_img(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Car / Not-Car Detection
    with st.spinner("Checking if image contains a car..."):
        is_car, prob = car_detector.predict(img)

    if not is_car:
        st.error("❌ No car detected in this image.")
        st.stop()

    st.success("✅ Car detected! Identifying brand and model...")

    # Brand & Model Prediction
    with st.spinner("Analyzing image..."):
        predictions = predict_top_k(img, model, class_names, k=3)

    st.subheader("🔍 Top Predictions")

    for i, pred in enumerate(predictions, start=1):
        st.write(
            f"**{i}. {pred['label']}** — {pred['confidence']*100:.2f}%"
        )
