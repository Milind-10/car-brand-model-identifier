import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.detector_preprocessing import preprocess_car_detector

class CarDetector():
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, img, threshold=0.6):
        img_arr = preprocess_car_detector(img)
        prob = self.model.predict(img_arr, verbose=0)[0][0]

        is_car = prob < threshold
        return is_car, prob
