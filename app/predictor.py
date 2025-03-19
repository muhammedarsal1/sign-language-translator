import gdown
import os
import tensorflow as tf

# Google Drive File ID
DRIVE_FILE_ID = "1dVdZ1nYpd6l8_aV9u0NrDnr1xWDtWquD"
MODEL_PATH = "model/sign_model.h5"

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Check if model exists, if not, download from Google Drive
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model download complete!")
    except Exception as e:
        print(f"❌ Model download failed: {e}")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Prediction function
def predict_sign(image):
    """Predict the hand sign from an image."""
    image = image / 255.0  # Normalize image
    image = image.reshape(1, 64, 64, 3)  # Ensure shape matches model input
    predictions = model.predict(image, verbose=0)
    predicted_index = predictions.argmax()
    return predicted_index
