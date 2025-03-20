import os
import gdown
import tensorflow as tf
import h5py
import cv2
import numpy as np
import json

# Define paths
MODEL_PATH = "model/sign_model.h5"
LABELS_PATH = "model/labels.json"
DRIVE_FILE_ID = "1dVdZ1nYpd6l8_aV9u0NrDnr1xWDtWquD"

# Ensure model directory exists
if not os.path.exists("model"):
    os.makedirs("model", exist_ok=True)

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    try:
        gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
        print("✅ Model download complete!")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("Please ensure the Google Drive file is accessible ('Anyone with the link' permission).")
        raise

# Load model with error handling
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Load labels
try:
    with open(LABELS_PATH, "r") as file:
        labels = json.load(file)
    print("✅ Labels loaded successfully!")
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    raise

def predict_sign(image):
    """Predict hand sign from processed image."""
    try:
        image = image / 255.0
        image = image.reshape(1, 64, 64, 3)
        predictions = model.predict(image, verbose=0)
        predicted_label = list(labels.keys())[predictions.argmax()]
        return predicted_label
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None