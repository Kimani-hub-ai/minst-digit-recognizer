# utils.py
import tensorflow as tf
import numpy as np
from PIL import Image

def load_and_preprocess_image(img):
    """
    Preprocess a Gradio Sketchpad or uploaded image for MobileNetV2.
    Handles:
      - PIL.Image or NumPy array
      - Grayscale or RGBA
      - Resizes to 96x96
      - Normalizes to [0,1]
    """
    # Handle string path (for predict.py)
    if isinstance(img, str):
        img = Image.open(img)
    
    # If PIL Image, convert to NumPy array
    if isinstance(img, Image.Image):
        # Convert to RGB first if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
    
    # Ensure NumPy array
    img = np.array(img, dtype=np.float32)
    
    # Drop alpha channel if present
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    
    # Convert grayscale to RGB
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    
    # Convert to Tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    
    # Resize to 96x96
    img = tf.image.resize(img, [96, 96])
    
    # Normalize
    img = img / 255.0
    
    return img