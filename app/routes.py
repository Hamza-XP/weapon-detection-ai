from flask import Flask, request, jsonify
import tensorflow as tf


# Why is routes.py Important?
# It’s the entry point for all user interactions
# Coordinates between:
# User inputs (images/videos)
# Your AI model
# Helper functions (in utils/)
# Determines what your API can do


# Key Functions in Your routes.py
# Function/Purpose	Example Code Snippet
# Start Flask app	app = Flask(__name__)
# Define a route	@app.route("/detect", methods=["POST"])
# Access uploaded files	request.files["image"]
# Return JSON responses	jsonify({"detections": [...]})
# Load TensorFlow model	tf.keras.models.load_model(...)


# Initialize Flask app
app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model("models/weapon_detection_model.h5")

# Define a route for image detection
@app.route("/detect", methods=["POST"])
def detect_image():
    # Step 1: Check if the user uploaded a file
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Step 2: Get the uploaded image
    image_file = request.files["image"]
    
    # Step 3: Preprocess the image (resize, normalize, etc.)
    processed_image = preprocess_image(image_file)  # You'd define this function
    
    # Step 4: Run the AI model
    predictions = model.predict(processed_image)
    
    # Step 5: Return results as JSON
    return jsonify({
        "detections": parse_predictions(predictions)  # Your custom function
    })

if __name__ == "__main__":
    app.run(debug=True)  # Start the Flask server