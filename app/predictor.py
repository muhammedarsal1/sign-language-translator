import os
import gdown
import tensorflow as tf
import h5py
import cv2
import numpy as np
import json

# ✅ Google Drive Model Download
MODEL_PATH = "model/sign_model.h5"
LABELS_PATH = "model/labels.json"
DRIVE_FILE_ID = "1dVdZ1nYpd6l8_aV9u0NrDnr1xWDtWquD"  # Your Google Drive ID

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# ✅ Load Model
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Load Labels
with open(LABELS_PATH, "r") as file:
    labels = json.load(file)

def predict_sign(image):
    """Predict hand sign from processed image."""
    image = image / 255.0
    image = image.reshape(1, 64, 64, 3)
    predictions = model.predict(image)
    return labels[str(predictions.argmax())]
