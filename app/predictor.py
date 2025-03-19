import os
import gdown
import tensorflow as tf
import numpy as np
import json

MODEL_PATH = "model/sign_model.h5"
LABELS_PATH = "model/labels.json"
DRIVE_FILE_ID = "1dVdZ1nYpd6l8_aV9u0NrDnr1xWDtWquD"  # Update this if needed

# Ensure the model exists, or download it
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# Load model and labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as file:
    labels = json.load(file)

def predict_sign(image):
    image = image / 255.0
    image = image.reshape(1, 64, 64, 3)
    predictions = model.predict(image, verbose=0)
    return labels[str(predictions.argmax())]
