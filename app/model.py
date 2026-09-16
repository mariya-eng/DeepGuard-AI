from transformers import pipeline
from PIL import Image
import numpy as np

MODEL_NAME = "Hemg/Deepfake-Detection"

print("Loading deepfake detection model...")

classifier = pipeline(
    "image-classification",
    model=MODEL_NAME,
    device=-1
)

print("Model loaded successfully!")


def predict_frame(frame):
    """
    Predict whether a frame is Real or Fake.
    Accepts NumPy/OpenCV frame or PIL Image.
    """

    if isinstance(frame, np.ndarray):
        image = Image.fromarray(frame)
    elif isinstance(frame, Image.Image):
        image = frame
    else:
        raise TypeError(
            "Frame must be a NumPy array or PIL Image"
        )

    results = classifier(image)

    return results