from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import gdown
import os

app = Flask(__name__)

# Download model from Google Drive if not present
model_path = "model/pneumonia_cnn_model.h5"
if not os.path.exists(model_path):
    os.makedirs("model", exist_ok=True)
    gdown.download("https://drive.google.com/uc?id=1rG2jDLrmFkFwSIqk-JsVdCp_0VhIMp6S", model_path, quiet=False)

# Load model
from manual_model import create_model
model = create_model()
model.load_weights(model_path)

def preprocess_image(image):
    # Convert to RGB in case it's grayscale or RGBA
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    try:
        image = Image.open(file)
        processed = preprocess_image(image)
        prediction = model.predict(processed)

        confidence = float(prediction[0][0])

        if confidence > 0.5:
            result = "PNEUMONIA"
            display_confidence = confidence
        else:
            result = "NORMAL"
            display_confidence = 1 - confidence

        return jsonify({
            'result': result,
            'confidence': round(display_confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)