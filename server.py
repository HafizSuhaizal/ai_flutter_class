import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
#model_path = "/Users/hafizsuhaizal/Downloads/converted_keras-9/keras_model.h5"
#put your own path at this "model_path" basically .keras is produced after we run the modeltrain . the original code is model.h5 that we can see at the file that we downloaded
model_path = "/Users/hafizsuhaizal/Downloads/converted_keras-8/keras_model.keras"
try:
    model = load_model(model_path, compile=False)
    logging.info(f"Model loaded from {model_path}")
    model.summary()
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Initialize Flask app
app = Flask(__name__)

# Define the class labels exactly as in the training (alphabetical order)
#For the class label , change as your preferences
class_labels = ["Earbuds", "Spectacle", "Watch"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read the image
        img = Image.open(io.BytesIO(file.read()))
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))  # Resize image to match model input
        img_array = np.array(img) / 255.0  # Normalize image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Debug: Check the shape of your input data
        logging.debug(f"Input data shape: {img_array.shape}")

        # Predict
        predictions = model.predict(img_array)
        logging.debug(f"Predictions raw output: {predictions}")

        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]

        predicted_class_name = class_labels[predicted_class_index]

        logging.debug(f"Predicted class index: {predicted_class_index}, Class name: {predicted_class_name}, Confidence: {confidence}")

        return jsonify({
            "class": predicted_class_name,
            "confidence": float(confidence)
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
