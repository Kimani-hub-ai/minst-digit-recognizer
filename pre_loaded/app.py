# app.py - Optimized for Simple CNN
import tensorflow as tf
import gradio as gr
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2

MODEL_PATH = "digit_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img):
    """Preprocess image to match MNIST format exactly."""
    if img is None:
        return None
    
    # Handle dictionary format from ImageEditor
    if isinstance(img, dict):
        if 'composite' in img:
            img = img['composite']
        elif 'background' in img:
            img = img['background']
        elif 'layers' in img and len(img['layers']) > 0:
            img = img['layers'][0]
    
    # Convert to PIL if numpy array
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))
    
    if img is None:
        return None
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Invert if background is white (drawing should be white on black)
    img_array = np.array(img)
    if np.mean(img_array) > 127:
        img = ImageOps.invert(img)
    
    # Convert to numpy for processing
    img_array = np.array(img)
    
    # Find bounding box of the digit
    rows = np.any(img_array > 30, axis=1)
    cols = np.any(img_array > 30, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop to bounding box with small margin
    margin = 5
    rmin = max(0, rmin - margin)
    rmax = min(img_array.shape[0], rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(img_array.shape[1], cmax + margin)
    
    img_cropped = img_array[rmin:rmax+1, cmin:cmax+1]
    
    # Get dimensions
    height, width = img_cropped.shape
    
    # Make it square by padding the shorter dimension
    if height > width:
        # Pad width
        pad_total = height - width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        img_cropped = np.pad(img_cropped, ((0, 0), (pad_left, pad_right)), mode='constant')
    elif width > height:
        # Pad height
        pad_total = width - height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        img_cropped = np.pad(img_cropped, ((pad_top, pad_bottom), (0, 0)), mode='constant')
    
    # Resize to 20x20 (MNIST uses 20x20 for the digit itself)
    img_resized = cv2.resize(img_cropped, (20, 20), interpolation=cv2.INTER_AREA)
    
    # Center in 28x28 image (MNIST standard)
    img_centered = np.zeros((28, 28), dtype=np.uint8)
    img_centered[4:24, 4:24] = img_resized
    
    # Apply slight Gaussian blur (MNIST has this)
    img_final = cv2.GaussianBlur(img_centered, (3, 3), 0)
    
    # Normalize to [0, 1]
    img_final = img_final.astype('float32') / 255.0
    
    # Add channel dimension
    img_final = np.expand_dims(img_final, -1)
    
    return img_final

def predict_digit(img):
    """Predict digit from drawn image."""
    if img is None:
        return "Please draw a digit first!"
    
    processed = preprocess_image(img)
    
    if processed is None:
        return "Please draw a digit first!"
    
    # Add batch dimension and predict
    batch = np.expand_dims(processed, 0)
    pred = model.predict(batch, verbose=0)[0]
    
    # Create prediction dictionary
    result = {str(i): float(pred[i]) for i in range(10)}
    
    return result

# Create Blocks interface with ImageEditor
with gr.Blocks(title="MNIST Digit Recognizer") as demo:
    gr.Markdown("# 🔢 MNIST Digit Recognizer")
    gr.Markdown("### Draw a digit (0-9) clearly in the canvas below")
    
    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.ImageEditor(
                type="pil",
                label="✏️ Draw Here",
                brush=gr.Brush(
                    default_size=20,
                    colors=["#000000"],
                    default_color="#000000"
                ),
                canvas_size=(280, 280),
                sources=[],
                transforms=[]
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")
                submit_btn = gr.Button("🚀 Predict", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output = gr.Label(num_top_classes=5, label="📊 Predictions")
            
            gr.Markdown("""
            ### 💡 Tips for Best Results:
            - Draw digits **large and centered**
            - Use **consistent stroke thickness**
            - Write digits **clearly** (like you normally would)
            - Avoid extra marks or noise
            - If prediction is wrong, try redrawing
            
            ### ⚡ Expected Accuracy: 99%+
            """)
    
    # Button actions
    submit_btn.click(fn=predict_digit, inputs=canvas, outputs=output)
    clear_btn.click(fn=lambda: None, inputs=None, outputs=canvas)
    
    gr.Markdown("---")
    gr.Markdown("Built with TensorFlow & Gradio | Model: Custom CNN trained on MNIST")

demo.launch()