# 🚗 Car Brand & Model Identifier

A deep learning web application that identifies the **brand and model of a car** from an uploaded image.

## 🔍 Features
- Classifies **196 car models** (Stanford Cars Dataset)
- Uses **EfficientNet (Transfer Learning)**
- Shows **Top-3 predictions with confidence**
- Rejects non-car images using confidence threshold
- Interactive web interface built with Streamlit

## 🧠 Model Details
- Architecture: EfficientNetB0
- Training Strategy: Transfer Learning + Fine-Tuning
- Dataset: Stanford Cars Dataset
- Input Size: 224×224 RGB images

## 🚀 Live Demo
👉 **[Click here to try the app](YOUR_HF_LINK_HERE)**

## ⚠️ Limitations
- Model is trained only on car images
- For non-car images, predictions are filtered using confidence threshold

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- Hugging Face Spaces
