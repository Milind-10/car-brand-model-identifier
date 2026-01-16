# import streamlit as st
# from ultralytics import YOLO


# Yolo_model_name = "yolov8n.pt"

# @st.cache_resource
# def load_yolo_model():
#     return YOLO(Yolo_model_name)

# def contain_cars(image):
#     model = load_yolo_model()
#     results = model(image, conf=0.25, verbose=False)

#     for r in results:
#         for cls in r.boxes.cls:
#             if int(cls) in [2,3,5,7]:
#                 return True
#     return False        