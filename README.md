# Male-Female Image Classifier Using FastAPI & TensorFlow

This project is a **Male-Female Image Classifier** built using **TensorFlow/Keras** and deployed as an API using **FastAPI**. 
The model was trained using a Convolutional Neural Network (CNN). It processes images of size 150x150 pixels and predicts whether the image represents a male or a female. It predicts whether an uploaded image represents a male or a female face.

## 🚀 Features
- **Trained CNN Model**: Built using TensorFlow/Keras.
- **FastAPI Backend**: Provides a REST API for image classification.
- **Image Preprocessing**: Resizes and normalizes images before prediction.
- **Deployment Ready**: Uses a `.pkl` file for model loading.

## 🛠 Tech Stack
- **Machine Learning**: TensorFlow/Keras
- **Backend**: FastAPI
- **Data Processing**: NumPy, PIL (Pillow)

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/male-female-classifier.git
   cd male-female-classifier

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the FastAPI server:
   ```bash
   pip install -r requirements.txt

## 📦 API Useage
- **Endpoint**: /predict_image
- **Method**: POST
- **Parameter**: Upload an image file
- **Response**: 
 ```bash
   {
     "prediction": "Male",
     "probability": 0.78
   }
