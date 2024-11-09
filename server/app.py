from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('model/digit_model.keras')

def preprocess_image(image_data):
    try:
        # Remove the data URL prefix
        image_data = image_data.split('base64,')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize
        image_array = image_array.astype('float32') / 255.0
        
        # Reshape for model
        processed_image = image_array.reshape(1, 28, 28, 1)
        
        # Debug: Save processed image
        plt.imsave('debug_processed.png', processed_image[0].reshape(28, 28), cmap='gray')
        
        return processed_image
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
            
        # Preprocess the image
        processed_image = preprocess_image(data['image'])
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Debug: Print prediction details
        print(f"Predictions array: {predictions[0]}")
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence}")
        
        return jsonify({
            "prediction": predicted_digit,
            "confidence": confidence
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)