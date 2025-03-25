from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load the trained model from .pkl file
with open("model/CNN_Male_Female_Image_Classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define image size based on model requirements
image_size = (150, 150)  # Model expects 150x150 images

# Function to preprocess the image
def preprocess_image(image: Image.Image):
    image = image.resize(image_size)  # Resize to expected size
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# FastAPI endpoint for image prediction
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")  # Ensure RGB format
    processed_image = preprocess_image(image)  # Preprocess image
    prediction = model.predict(processed_image)[0][0]  # Get prediction

    # Convert probability to class
    predicted_label = "Male" if prediction > 0.5 else "Female"

    return {"prediction": predicted_label, "probability": float(prediction)}
