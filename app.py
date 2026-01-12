import streamlit as st
import numpy as np 
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from ultralytics import YOLO

#Load yolo model

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo




#Car or not verification

def detect_car(img, conf_threshold=0.4):
    results = yolo_model(img)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = yolo_model.names[cls]

            if label == "car" and conf > conf_threshold:
                return True, conf
    return False, None





#Page Configuration

st.set_page_config(page_title="Car Brand and Model identifier",
                   page_icon= "🚗",
                   layout="centered")

st.title("🚗 Car Brand & Model Identifier")
st.write("Upload a car image to identify its **Brand and Model**. ")




#Load the model

@st.cache_resource
def load_car_model():
    return load_model("car_model_efficientnet.keras")

model = load_car_model()




#Load class names

@st.cache_resource
def load_class_names():
    with open("class_names.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

class_names = load_class_names()
NUM_CLASSES = len(class_names)
st.caption(f"Model trained on {NUM_CLASSES} car models")




#Prediction function

def predict_top_k(img, k=3):
    img = img.resize((224,224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    preds = model.predict(img_arr)[0]
    top_idices = preds.argsort()[-k:][::-1]
    results = []

    for idx in top_idices:
        results.append({
             "label": class_names[idx],
             "confidence": float(preds[idx]) 
        })
    return results
    



#File Upload and Prediction UI

uploaded_file = st.file_uploader(
    "Upload a car's image",
    type=["jpg","jpeg","png"] 
)

if uploaded_file:
    img = image.load_img(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        is_car , car_conf = detect_car(img)

    if not is_car:
        st.error("No car detected in this image")
        st.stop()

    st.success(f"Car detected (confidence: {car_conf*100:.2f}%)")

    with st.spinner("Identifying Car..."):
        predictions = predict_top_k(img, k=3)

    st.subheader("🔍 Top Predictions")

    for i, pred in enumerate(predictions,start=1):
       st.write(
             f"**{i}. {pred['label']}** - {pred['confidence']*100:.2f}%"
            )
