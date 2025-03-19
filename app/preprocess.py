import cv2
import numpy as np

def preprocess_image(frame):
    """Convert frame to correct format for model prediction."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (64, 64))
    return frame
