import numpy as np 
from utils.preprocessing import preprocess_image

def predict_top_k(img, model, class_names, k=3):
    img_arr = preprocess_image(img)

    preds = model.predict(img_arr)[0]
    top_indices = preds.argsort()[-k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "label": class_names[idx],
            "confidence": float(preds[idx])
        })

    return results