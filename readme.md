# 🚗 Car Brand & Model Identification System

A production-ready computer vision application that identifies **car brand and model** from images.  
The system is designed with a **two-stage inference pipeline** to improve robustness and real-world usability.

---

## 🔍 Project Overview

This project started as a car brand & model classifier and evolved into a more reliable system by adding a **car / not-car gatekeeper** and **regional fine-tuning for Indian cars**.

The application is deployed using **Streamlit** and supports real-time image uploads.

---

## 🧠 Architecture

### 1️⃣ Car / Not-Car Detection (Gatekeeper)
- **Model:** MobileNetV2
- **Purpose:** Filters out non-car images before classification
- **Benefit:** Reduces false positives and improves user experience

### 2️⃣ Brand & Model Classification
- **Model:** EfficientNet-based classifier
- **Training data:**
  - Stanford Cars Dataset
  - Additional Indian car images (fine-tuned)
- **Output:** Top-K predictions with confidence scores

---
## 🚀 Live Demo
👉 **https://milind-pandya-car-brand-model-identifier.streamlit.app**

## 🇮🇳 Indian Car Fine-Tuning (v2.2)

To improve performance on Indian roads, the classifier was fine-tuned with additional images of popular Indian car brands such as:

- Maruti Suzuki
- Tata
- Mahindra

Only **10–20 clean images per model** were required, leveraging transfer learning and avoiding overfitting.

> Note: Confidence scores may appear low due to a large number of classes; however, correct predictions consistently rank at the top.

---

## 🚀 Features

- ✅ Car / Not-Car detection
- ✅ Brand & model prediction
- ✅ Top-K predictions
- ✅ Streamlit-based UI
- ✅ Modular preprocessing & inference
- ✅ Ready for API / mobile extension

---

## 🛠️ Tech Stack

- Python 3.13
- TensorFlow / Keras
- MobileNetV2
- EfficientNet
- Streamlit
- NumPy, PIL
- icrawler (data collection)

---

## 📌 Version History

- **v1.0** – Initial car brand & model classifier  
- **v2.1** – Added MobileNetV2 car/not-car gatekeeper  
- **v2.2** – Fine-tuned classifier with Indian car images  

