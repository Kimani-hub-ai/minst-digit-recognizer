import gradio as gr
import tensorflow as tf
import numpy as np
from utils import prepare_image_for_model

MODEL_PATH = "digit_model"
model = tf.keras.models.load_model(MODEL_PATH)

def predict_digit(img):
    batch = prepare_image_for_model(img)
    preds = model.predict(batch)[0]
    return {
        "Predicted Digit": int(np.argmax(preds)),
        "Confidence (%)": f"{np.max(preds)*100:.2f}%"
    }

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Handwritten Digit Recognizer (Transfer Learning)")
    gr.Markdown("Upload or draw a digit to test the model.")

    with gr.Tab("Upload"):
        img = gr.Image(type="numpy")
        out = gr.JSON()
        img.submit(predict_digit, inputs=img, outputs=out)

    with gr.Tab("Draw"):
        sketch = gr.Sketchpad(shape=(256,256))
        out2 = gr.JSON()
        sketch.submit(predict_digit, inputs=sketch, outputs=out2)

demo.launch()
