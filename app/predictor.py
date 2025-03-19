import os
import gdown
import tensorflow as tf
import json
import numpy as np

# ✅ Define paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "sign_model.h5")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "labels.json")

# ✅ Google Drive Model URL
DRIVE_FILE_ID = "1dVdZ1nYpd6l8_aV9u0NrDnr1xWDtWquD"  # Your File ID
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# ✅ Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ✅ Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model from Google Drive...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

# ✅ Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Load labels
with open(LABELS_PATH, "r") as file:
    labels = json.load(file)

def predict_sign(image):
    """Predict the hand sign from the processed image."""
    image = image / 255.0  # Normalize image
    image = image.reshape(1, 64, 64, 3)  # Ensure correct shape

    predictions = model.predict(image)
    predicted_label = labels[str(predictions.argmax())]
    return predicted_label
