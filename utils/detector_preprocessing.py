import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_car_detector(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    return preprocess_input(img_arr)
