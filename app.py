from flask import Flask, request, jsonify, render_template_string
import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from model import DeepfakeModel

app = Flask(__name__)

# Load the deepfake model
model = DeepfakeModel()
model.load_state_dict(torch.load("deepfake_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Serve the HTML (embed it directly or load from file)
@app.route('/')
def home():
    # Read and serve aidriven.html
    with open('website/aidriven.html', 'r') as f:
        html_content = f.read()
    return render_template_string(html_content)

# Endpoint for file analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file temporarily
    temp_path = 'temp_upload.' + file.filename.split('.')[-1]
    file.save(temp_path)
    
    try:
        # Load the image
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform for model
        img_tensor = transform(img_rgb).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
        
        label = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else 1 - prob
        
        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        num_faces = len(faces)
        
        result = {
            'prediction': label,
            'confidence': round(confidence * 100, 2),
            'faces_detected': num_faces
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# New endpoint for real-time prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Read the image from file
    img_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Transform
    img_tensor = transform(img_rgb).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    
    label = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob
    
    return jsonify({
        'prediction': label,
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)