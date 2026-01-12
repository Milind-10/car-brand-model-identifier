from tensorflow.keras.models import load_model

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "car_model_efficientnet.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.txt"


#Load the model

def load_car_model():
    return load_model(MODEL_PATH)



#Load class names

def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


