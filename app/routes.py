from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import uuid
from PIL import Image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(_name_)

# Configure upload folder
UPLOAD_FOLDER = 'uploads/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure directory exists

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load TensorFlow model
model = tf.keras.models.load_model("models/weapon_detection_model.h5")

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image for model input
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")  # Convert to RGB
    image = image.resize((224, 224))  # Resize (change if needed)
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to parse model predictions (modify as per your model's output format)
def parse_predictions(predictions):
    return {"weapon_detected": bool(predictions[0][0]), "confidence": float(predictions[0][0])}

# Define a route for uploading and detecting weapons
@app.route("/detect", methods=["POST"])
def detect_image():
    # Check if an image was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files["image"]
    
    # Validate file type
    if image_file.filename == "" or not allowed_file(image_file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    # Save uploaded image with unique filename
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}_{secure_filename(image_file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)

    # Preprocess image and make prediction
    processed_image = preprocess_image(filepath)
    predictions = model.predict(processed_image)

    # Return detection results
    return jsonify({
        "message": "Image uploaded and processed",
        "filename": filename,
        "detections": parse_predictions(predictions)
    })

if _name_ == "_main_":
    app.run(debug=True)
