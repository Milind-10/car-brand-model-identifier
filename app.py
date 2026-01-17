import streamlit as st
import numpy as np 
from pathlib import Path
#from utils.car_detector import contain_cars
from tensorflow.keras.preprocessing import image
from utils.model_loader import load_car_model, load_class_names
from utils.inference import predict_top_k



#Page Configuration

st.set_page_config(page_title="Car Brand and Model identifier",
                   page_icon= "🚗",
                   layout="centered")

st.title("🚗 Car Brand & Model Identifier")
st.write("Upload a car image to identify its **Brand and Model**. ")


@st.cache_resource
def load_resources():
   model = load_car_model()
   class_names = load_class_names()
   return model, class_names 

model, class_names = load_resources()

st.caption(f"Model trained on {len(class_names)} car models")


#File Upload and Prediction UI

uploaded_file = st.file_uploader(
    "Upload a car's image",
    type=["jpg","jpeg","png"] 
)

if uploaded_file:
    img = image.load_img(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

 

    with st.spinner("Analyzing image..."):
        predictions = predict_top_k(img,model,class_names, k=3)

    st.subheader("🔍 Top Predictions")

    for i, pred in enumerate(predictions,start=1):
       st.write(
             f"**{i}. {pred['label']}** - {pred['confidence']*100:.2f}%"
            )
