import tensorflow as tf
import numpy as np
from utils import load_and_preprocess_image

MODEL_PATH = "digit_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = load_and_preprocess_image(img_path)
    batch = tf.expand_dims(img, axis=0)

    preds = model.predict(batch)[0]
    digit = int(np.argmax(preds))
    confidence = float(np.max(preds))

    print(f"Prediction: {digit} (confidence: {confidence*100:.2f}%)")

    return digit, confidence

if __name__ == "__main__":
    path = input("Enter image path: ")
    predict_image(path)
